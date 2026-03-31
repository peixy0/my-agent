"""
Agent module containing the core agent logic.

The Agent class orchestrates LLM interactions and context management.
Tool registration is handled externally by ToolRegistry (SRP).
System prompt construction is handled by SystemPromptBuilder (SRP).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Protocol, override

import jsonschema

from agent.core.messaging import Channel
from agent.llm.prompt import SystemPromptBuilder
from agent.llm.types import (
    CompletionResponseView,
    MessageView,
    ToolCallView,
    ToolContent,
)
from agent.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class CompletionClient(Protocol):
    async def do_completion(
        self, *args: Any, **kwargs: Any
    ) -> CompletionResponseView: ...


@dataclass
class ToolCallResult:
    tool_id: str
    tool_name: str
    args: dict[str, Any]
    result: ToolContent


class Orchestrator(ABC):
    def __init__(
        self,
        model: str,
        tool_registry: ToolRegistry,
    ):
        self.model = model
        self.tool_registry = tool_registry.clone()

    @abstractmethod
    async def _before_tool_use(self, message: MessageView) -> None:
        """Hook called before tool results are collected. Override for extra behaviour."""
        pass

    @abstractmethod
    async def _on_final_response(self, content: str) -> None:
        """Handle the final (non-tool-call) LLM response."""
        pass

    async def process(
        self, message: MessageView, finish_reason: str
    ) -> list[dict[str, str]]:
        if message.tool_calls:
            await self._before_tool_use(message)
            tool_results = await asyncio.gather(
                *[self._handle_tool_call(tc) for tc in message.tool_calls]
            )
            reply: list[dict[str, Any]] = [
                {
                    "role": "tool",
                    "tool_call_id": tool_result.tool_id,
                    "content": tool_result.result.to_lm_content(),
                }
                for tool_result in tool_results
            ]
            return reply

        if finish_reason != "stop":
            return [{"role": "user", "content": "continue"}]

        await self._on_final_response(message.content or "")
        return []

    async def _handle_tool_call(self, tool_call: ToolCallView) -> ToolCallResult:
        tool_name = tool_call.function.name
        tool_id = tool_call.id
        raw_arguments = tool_call.function.arguments

        handler = self.tool_registry.get_handler(tool_name)
        if handler is None:
            return ToolCallResult(
                tool_id,
                tool_name,
                {},
                ToolContent.from_dict(
                    "error",
                    {"message": f"No such tool named {tool_name}"},
                ),
            )

        logger.debug(f"Executing tool {tool_name} with args: {raw_arguments}")
        args: dict[str, Any] = {}
        tool_content: ToolContent
        try:
            args = json.loads(raw_arguments)
            schema = self.tool_registry.get_schema(tool_name)
            if schema:
                jsonschema.validate(instance=args, schema=schema["parameters"])
            tool_content = await handler(**args)
        except json.JSONDecodeError as e:
            tool_content = ToolContent.from_dict(
                "error",
                {"message": f"Invalid JSON in tool call arguments: {e}"},
            )
        except jsonschema.ValidationError as e:
            tool_content = ToolContent.from_dict(
                "error",
                {"message": f"Invalid tool call arguments: {e.message}"},
            )
        except Exception as e:
            tool_content = ToolContent.from_dict(
                "error",
                {"message": f"Exception occured during tool call: {e}"},
            )

        if tool_content.status == "error":
            logger.error(
                f"Tool call {tool_name} failed: {tool_content.to_lm_content()}"
            )
        else:
            logger.debug(f"Tool call {tool_name} completed successfully")

        return ToolCallResult(tool_id, tool_name, args, tool_content)


def _strip_thought(content: str | None) -> str:
    if not content:
        return ""
    return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()


class SubagentOrchestrator(Orchestrator):
    """Orchestrator for isolated subagent tasks.

    Captures the final text response without forwarding to a Channel.
    Does not receive the agent tool — subagents cannot spawn further subagents.
    """

    def __init__(self, model: str, tool_registry: ToolRegistry) -> None:
        super().__init__(model, tool_registry)
        self.output: str = ""

    @override
    async def _before_tool_use(self, message: MessageView) -> None:
        pass

    @override
    async def _on_final_response(self, content: str) -> None:
        self.output = _strip_thought(content)


def _register_agent_tool(
    prompt_builder: SystemPromptBuilder, registry: ToolRegistry, agent: Agent
) -> None:
    """Register the agent tool on an orchestrator's tool registry.

    Snapshots the registry before the tool is added so the spawned
    SubagentOrchestrator inherits all other tools but not this one,
    preventing recursive subagent spawning.
    """
    subagent_registry = registry.clone()

    async def run_agent(task: str, system_prompt: str) -> ToolContent:
        """
        Run an isolated subagent to handle a specific, self-contained task.

        The subagent has access to the same tools and runs with a fresh conversation.
        It cannot spawn further subagents.
        Use this to delegate well-defined, isolated work units that can be
        completed independently of the current conversation context.
        Returns the final text response from the subagent.
        """
        messages: list[dict[str, Any]] = [{"role": "user", "content": task}]
        subagent_orchestrator = SubagentOrchestrator(agent.model, subagent_registry)
        await agent.run(
            prompt_builder.build_for_subagent(system_prompt),
            messages,
            subagent_orchestrator,
        )
        return ToolContent.from_dict(
            "success", {"output": subagent_orchestrator.output}
        )

    registry.register(
        run_agent,
        {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": (
                        "The task description for the subagent to execute. "
                        "Provide full context as the subagent has no conversation history."
                    ),
                },
                "system_prompt": {
                    "type": "string",
                    "description": "System prompt for the subagent.",
                },
            },
            "required": ["task", "system_prompt"],
        },
        name="agent",
    )


class BackgroundOrchestrator(Orchestrator):
    def __init__(
        self,
        model: str,
        tool_registry: ToolRegistry,
        sender: Channel,
    ) -> None:
        super().__init__(model, tool_registry)
        self.sender = sender

    @override
    async def _before_tool_use(self, message: MessageView) -> None:
        pass

    @override
    async def _on_final_response(self, content: str) -> None:
        content = _strip_thought(content)
        if content and not content.endswith("NO_REPORT"):
            await self.sender.send(content)


class HumanInputOrchestrator(Orchestrator):
    def __init__(
        self,
        model: str,
        tool_registry: ToolRegistry,
        sender: Channel,
    ) -> None:
        super().__init__(model, tool_registry)
        self.sender = sender
        sender.register_tools(self.tool_registry)

    @override
    async def _before_tool_use(self, message: MessageView) -> None:
        content = _strip_thought(message.content)
        if content:
            await self.sender.send(content)

    @override
    async def _on_final_response(self, content: str) -> None:
        content = _strip_thought(content)
        if content:
            await self.sender.send(content)
        else:
            await self.sender.send("No content generated by the agent.")


class Agent:
    """
    Core agent class that manages LLM interactions.

    Responsibilities (SRP):
    - Maintain conversation history
    - Run the LLM conversation loop
    - Validate structured responses against schemas
    """

    def __init__(
        self,
        llm_client: CompletionClient,
        model: str,
        tool_registry: ToolRegistry,
    ):
        self.llm_client = llm_client
        self.model = model
        self.tool_registry = tool_registry

    def _build_system_messages(self, prompt: str) -> list[dict[str, Any]]:
        """Build the system messages list for a given prompt (no shared state)."""
        return [{"role": "system", "content": prompt}]

    async def _chat(
        self,
        messages: list[dict[str, Any]],
        orchestrator: Orchestrator,
        system_messages: list[dict[str, Any]],
        max_tokens: int = 0,
        keep_last: int = 0,
    ) -> CompletionResponseView:
        """
        Sends a chat message to the LLM and handles the response.

        This method will automatically handle tool calls made by the LLM.
        If finish_reason is not "stop" and max_tokens is set, compresses the
        conversation in-place and retries rather than sending a "continue" message.

        Args:
            messages: A list of messages in the chat history.
            orchestrator: The orchestrator handling tool dispatch and responses.
            system_messages: Pre-built system messages; captured locally so
                             concurrent workers cannot overwrite each other.
            max_tokens: When > 0, compress and retry if finish_reason != "stop"
                        and total_tokens >= this threshold.

        Returns:
            The LLM's response.
        """
        while True:
            messages_to_be_sent = list(system_messages)
            messages_to_be_sent.extend(messages)

            response = await self.llm_client.do_completion(
                model=self.model,
                messages=messages_to_be_sent,
                tools=orchestrator.tool_registry.tool_schemas(),
                # temperature=0.6,
                # top_p=0.95,
                # extra_body={"chat_template_kwargs": {"enable_thinking": True}},
            )

            logger.debug(f"LLM Response: {response}")
            choice = response.choices[0]
            message = choice.message
            finish_reason = choice.finish_reason
            messages.append(message.model_dump())

            if (
                finish_reason != "stop"
                and max_tokens
                and response.usage.total_tokens >= max_tokens
            ):
                logger.info(
                    f"finish_reason={finish_reason!r}, compressing "
                    f"({response.usage.total_tokens} tokens)"
                )
                await self.compress(messages, keep_last)
                continue

            reply = await orchestrator.process(message, finish_reason)
            if not reply:
                return response
            messages.extend(reply)

    async def run(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        orchestrator: Orchestrator,
        max_tokens: int = 0,
        keep_last: int = 0,
    ) -> CompletionResponseView:
        """Run a single turn of the agent conversation.

        Args:
            system_prompt: The system prompt for the agent.
            messages: Conversation history (may start with a summary message).
            orchestrator: The orchestrator handling tool dispatch and responses.
            max_tokens: When > 0, compress mid-loop if finish_reason != "stop"
                        and total_tokens >= this threshold.
            keep_last: Number of recent messages to retain verbatim after compression.

        Returns:
            The final LLM response.
        """
        system_messages = self._build_system_messages(system_prompt)
        return await self._chat(
            messages, orchestrator, system_messages, max_tokens, keep_last
        )

    async def compress(
        self, messages: list[dict[str, Any]], keep_last: int = 0
    ) -> None:
        """
        Compress conversation history in-place, retaining the recent tail verbatim.

        Summarises all messages except the last `keep_last`, then replaces the list
        with [summary_message] + retained_tail.

        Args:
            messages: The conversation history (modified in-place).
            keep_last: Number of recent messages to keep verbatim after the summary.
        """
        if not messages:
            return

        if keep_last > 0 and len(messages) <= keep_last:
            return

        to_summarize = messages
        retained = messages[-keep_last:] if keep_last > 0 else []

        compression_prompt = (
            "You are a context compressor for an autonomous AI agent (summarizer).\n"
            "Produce a structured digest of the conversation segment below.\n"
            "Preserve exact file paths, command outputs, error messages, and values — "
            "the agent will use this to resume work without re-reading the full history.\n"
            "Write in third-person past tense. Be dense; omit pleasantries, "
            "exploratory thinking, and errors that were subsequently resolved.\n\n"
            "Format your response with these Markdown sections "
            "(omit any section that would be empty):\n\n"
            "## Active Tasks\n"
            "Ongoing work and immediate next steps.\n\n"
            "## Completed Tasks\n"
            "Finished actions and their outcomes.\n\n"
            "## Established Facts\n"
            "Specific paths, versions, identifiers, system state, and discovered values.\n\n"
            "## Key Files\n"
            "Files created, modified, or referenced with their purpose.\n\n"
            "## Pending Issues\n"
            "Unresolved errors, blockers, or open questions."
        )

        # Pre-build a lookup of tool_call_id → truncated result for call/result pairing.
        tool_results: dict[str, str] = {}
        for msg in to_summarize:
            if msg.get("role") == "tool":
                tc_id = str(msg.get("tool_call_id") or "")
                if tc_id:
                    tool_results[tc_id] = str(msg.get("content") or "")[:1000]

        # Serialize the transcript. Tool calls are immediately followed by their
        # result so the summariser sees action → outcome as a single unit.
        transcript_parts: list[str] = []

        for msg in to_summarize:
            role = msg.get("role", "")
            content = str(msg.get("content") or "")
            tool_calls = msg.get("tool_calls")
            if role not in ("system", "assistant", "user"):
                continue

            if tool_calls:
                for tc in tool_calls:
                    tc_id = str(tc.get("id") or "")
                    name = tc.get("function", {}).get("name", "unknown")
                    raw_args = tc.get("function", {}).get("arguments") or ""
                    try:
                        parsed = json.loads(raw_args) if raw_args else {}
                        args_str = ", ".join(
                            f"{k}={repr(v)[:80]}" for k, v in parsed.items()
                        )
                    except Exception:
                        args_str = raw_args[:200]

                    result = tool_results.get(tc_id, "")
                    if result:
                        transcript_parts.append(f"[TOOL] {name}({args_str})\n{result}")
                    else:
                        transcript_parts.append(f"[TOOL] {name}({args_str})")
            elif content:
                transcript_parts.append(f"[{role.upper()}]\n{content}")

        transcript = "\n\n".join(transcript_parts)

        response = await self.llm_client.do_completion(
            model=self.model,
            messages=[
                {"role": "system", "content": compression_prompt},
                {
                    "role": "user",
                    "content": f"Conversation to summarize:\n\n{transcript}",
                },
            ],
            temperature=0.3,
            top_p=1.0,
        )

        new_summary = (response.choices[0].message.content or "").strip()
        logger.info(f"Conversation compressed to {response.usage.total_tokens} tokens")

        messages.clear()
        messages.append(
            {
                "role": "user",
                "content": f"[PREVIOUS CONVERSATION SUMMARY]\n\n{new_summary}",
            }
        )
        messages.extend(retained)


class OrchestratorFactory(Protocol):
    """Creates per-turn orchestrators for a conversation session."""

    def make_human_input(self, sender: Channel) -> HumanInputOrchestrator: ...
    def make_background(self, sender: Channel) -> BackgroundOrchestrator: ...


class DefaultOrchestratorFactory:
    """Concrete factory wired by the composition root.

    Holds the shared dependencies needed to fully wire each orchestrator:
    model name, prompt builder, tool registry, and agent reference.
    _register_agent_tool is called here so orchestrators remain ignorant
    of Agent and SystemPromptBuilder.
    """

    def __init__(
        self,
        model: str,
        prompt_builder: SystemPromptBuilder,
        tool_registry: ToolRegistry,
        agent: Agent,
    ) -> None:
        self.model = model
        self.prompt_builder = prompt_builder
        self.tool_registry = tool_registry
        self.agent = agent

    def make_human_input(self, sender: Channel) -> HumanInputOrchestrator:
        orch = HumanInputOrchestrator(self.model, self.tool_registry, sender)
        _register_agent_tool(self.prompt_builder, orch.tool_registry, self.agent)
        return orch

    def make_background(self, sender: Channel) -> BackgroundOrchestrator:
        orch = BackgroundOrchestrator(self.model, self.tool_registry, sender)
        _register_agent_tool(self.prompt_builder, orch.tool_registry, self.agent)
        return orch
