"""
Agent module containing the core agent logic.

The Agent class orchestrates LLM interactions and context management.
Tool registration is handled externally by ToolRegistry (SRP).
System prompt construction is handled by SystemPromptBuilder (SRP).
"""

import asyncio
import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Protocol, override

import jsonschema

from agent.core.sender import MessageSender
from agent.llm.types import (
    CompletionResponseView,
    MessageView,
    ToolCallView,
    ToolContent,
)
from agent.tools.registry import ToolRegistry
from agent.tools.toolbox import register_human_input_tools

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


class HeartbeatOrchestrator(Orchestrator):
    def __init__(
        self,
        model: str,
        tool_registry: ToolRegistry,
        sender: MessageSender,
    ) -> None:
        super().__init__(model, tool_registry)
        self._sender = sender

    @override
    async def _before_tool_use(self, message: MessageView) -> None:
        pass

    @override
    async def _on_final_response(self, content: str) -> None:
        content = _strip_thought(content)
        if content and not content.endswith("NO_REPORT"):
            await self._sender.send(content)


class HumanInputOrchestrator(Orchestrator):
    def __init__(
        self,
        model: str,
        tool_registry: ToolRegistry,
        sender: MessageSender,
    ):
        super().__init__(model, tool_registry)
        self.sender = sender
        register_human_input_tools(self.tool_registry, sender)

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
    ) -> Any:
        """
        Sends a chat message to the LLM and handles the response.

        This method will automatically handle tool calls made by the LLM.

        Args:
            messages: A list of messages in the chat history.
            orchestrator: The orchestrator handling tool dispatch and responses.
            system_messages: Pre-built system messages; captured locally so
                             concurrent workers cannot overwrite each other.

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

            reply = await orchestrator.process(message, finish_reason)
            if not reply:
                return response
            messages.extend(reply)

    async def run(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        orchestrator: Orchestrator,
    ) -> Any:
        """Run a single turn of the agent conversation."""
        system_messages = self._build_system_messages(system_prompt)
        return await self._chat(messages, orchestrator, system_messages)

    async def compress(
        self,
        messages: list[dict[str, Any]],
        previous_summary: str = "",
    ) -> str:
        """
        Compress a slice of conversation history into a structured, sectioned summary.

        Tool calls are paired with their results so the summariser LLM sees
        action → outcome units rather than disconnected fragments.  Any
        existing summary from a prior compression pass is prepended, making
        repeated compressions incremental rather than discarding earlier context.

        Returns an empty string when there is nothing to summarise.

        The caller is responsible for:
        - Slicing the messages list (keep the recent tail verbatim via
          context_num_keep_last; pass only the older head to this method).
        - Storing the returned summary in conversation.previous_summary.
        - Replacing conversation.messages with the retained tail.
        """
        if not messages:
            return ""

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
        for msg in messages:
            if msg.get("role") == "tool":
                tc_id = str(msg.get("tool_call_id") or "")
                if tc_id:
                    tool_results[tc_id] = str(msg.get("content") or "")[:1000]

        # Serialize the transcript.  Tool calls are immediately followed by their
        # result so the summariser sees action → outcome as a single unit.
        transcript_parts: list[str] = []
        included_tool_ids: set[str] = set()

        if previous_summary:
            transcript_parts.append(f"[PRIOR SUMMARY]\n{previous_summary}")

        for msg in messages:
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
                    except (json.JSONDecodeError, Exception):
                        args_str = raw_args[:200]

                    result = tool_results.get(tc_id, "")
                    if result:
                        included_tool_ids.add(tc_id)
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

        summary = response.choices[0].message.content or ""
        logger.info(f"Conversation compressed to {response.usage.total_tokens} tokens")
        return summary.strip()
