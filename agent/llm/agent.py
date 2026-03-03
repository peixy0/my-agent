"""
Agent module containing the core agent logic.

The Agent class orchestrates LLM interactions and context management.
Tool registration is handled externally by ToolRegistry (SRP).
System prompt construction is handled by SystemPromptBuilder (SRP).
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, cast, override

import jsonschema

from agent.core.messaging import Messaging
from agent.llm.openai import OpenAIProvider
from agent.tools.tool_registry import ToolRegistry
from agent.tools.toolbox import register_human_input_tools

logger = logging.getLogger(__name__)


@dataclass
class ToolCallResult:
    tool_id: str
    tool_name: str
    args: dict[str, Any]
    result: dict[str, Any]


class Orchestrator(ABC):
    response_label: str = "Agent"

    def __init__(
        self,
        model: str,
        tool_registry: ToolRegistry,
        messaging: Messaging,
    ):
        self.model = model
        self.tool_registry = tool_registry.clone()
        self.messaging = messaging

    def register_tool(
        self,
        func: Any,
        schema: dict[str, Any],
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        """Register a tool specific to this orchestrator instance."""
        self.tool_registry.register(func, schema, name=name, description=description)

    @abstractmethod
    async def _before_tool_use(self, message: Any) -> None:
        """Hook called before tool results are collected. Override for extra behaviour."""
        pass

    @abstractmethod
    async def _on_final_response(self, content: str) -> None:
        """Handle the final (non-tool-call) LLM response."""
        pass

    async def process(self, message: Any, finish_reason: str) -> list[dict[str, str]]:
        if message.tool_calls:
            await self._before_tool_use(message)
            tool_results = await asyncio.gather(
                *[self._handle_tool_call(tc) for tc in message.tool_calls]
            )
            reply: list[dict[str, Any]] = [
                {
                    "role": "tool",
                    "tool_call_id": tool_result.tool_id,
                    "content": json.dumps(tool_result.result, ensure_ascii=False),
                }
                for tool_result in tool_results
            ]
            return reply

        if finish_reason != "stop":
            return [{"role": "user", "content": "continue"}]

        content = message.content or ""
        await self._on_final_response(content)
        return []

    async def _handle_tool_call(self, tool_call: Any) -> ToolCallResult:
        tool_name = tool_call.function.name
        tool_id = tool_call.id

        handler = self.tool_registry.get_handler(tool_name)
        if handler is None:
            return ToolCallResult(
                tool_id,
                tool_name,
                {},
                {
                    "status": "error",
                    "message": f"No such tool named {tool_name}",
                },
            )

        logger.debug(
            f"Executing tool {tool_name} with args: {tool_call.function.arguments}"
        )
        args: dict[str, Any] = {}
        try:
            args = json.loads(tool_call.function.arguments)
            if self.model.startswith("deepseek-ai/"):
                args = self._fix_deepseek_args(args)
            schema = self.tool_registry.get_schema(tool_name)
            if schema:
                jsonschema.validate(instance=args, schema=schema["parameters"])
            result: dict[str, Any] = await handler(**args)
        except json.JSONDecodeError as e:
            result = {
                "status": "error",
                "message": f"Invalid JSON in tool call arguments: {e}",
            }
        except jsonschema.ValidationError as e:
            result = {
                "status": "error",
                "message": f"Invalid tool call arguments: {e.message}",
            }
        except Exception as e:
            result = {
                "status": "error",
                "message": f"Exception occured during tool call: {e}",
            }

        if result.get("status") == "error":
            logger.error(f"Tool call {tool_name} failed: {result}")
        else:
            logger.debug(f"Tool call {tool_name} completed successfully")

        return ToolCallResult(tool_id, tool_name, args, result)

    @staticmethod
    def _fix_deepseek_args(args: dict[str, Any]) -> dict[str, Any]:
        """DeepSeek sometimes double-encodes argument values as JSON strings."""
        fixed: dict[str, Any] = {}
        for key, value in args.items():
            if isinstance(value, str):
                try:
                    fixed[key] = json.loads(value)
                except json.JSONDecodeError:
                    fixed[key] = value
            else:
                fixed[key] = value
        return fixed


class HeartbeatOrchestrator(Orchestrator):
    response_label = "Heartbeat"

    @override
    async def _before_tool_use(self, message: Any) -> None:
        pass

    @override
    async def _on_final_response(self, content: str) -> None:
        content = content.strip()
        if content and not content.endswith("NO_REPORT"):
            await self.messaging.notify(content)


class HumanInputOrchestrator(Orchestrator):
    response_label = "Human Input"

    def __init__(
        self,
        chat_id: str,
        message_id: str,
        model: str,
        tool_registry: ToolRegistry,
        messaging: Messaging,
    ):
        super().__init__(model, tool_registry, messaging)
        self.chat_id = chat_id
        self.message_id = message_id
        register_human_input_tools(self.tool_registry, messaging, chat_id, message_id)

    @override
    async def _before_tool_use(self, message: Any) -> None:
        if not message.content:
            return
        content = message.content.strip()
        if content:
            await self.messaging.send_message(self.chat_id, content)

    @override
    async def _on_final_response(self, content: str) -> None:
        content = content.strip()
        if content:
            await self.messaging.send_message(self.chat_id, content)
        else:
            await self.messaging.send_message(
                self.chat_id, "No content generated by the agent."
            )


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
        llm_client: OpenAIProvider,
        model: str,
        tool_registry: ToolRegistry,
        messaging: Messaging,
    ):
        self.llm_client = llm_client
        self.model = model
        self.tool_registry = tool_registry
        self.messaging = messaging
        self.system_messages = []
        self.system_prompt = ""

    def _set_system_prompt(self, prompt: str) -> None:
        """Set the system prompt and reset conversation history."""
        self.system_prompt = prompt
        self.system_messages = [{"role": "system", "content": self.system_prompt}]

    async def _chat(
        self, messages: list[dict[str, Any]], orchestrator: Orchestrator
    ) -> Any:
        """
        Sends a chat message to the LLM and handles the response.

        This method will automatically handle tool calls made by the LLM.

        Args:
            messages: A list of messages in the chat history.
            orchestrator: The orchestrator handling tool dispatch and responses.

        Returns:
            The LLM's response.
        """
        while True:
            messages_to_be_sent = self.system_messages.copy()
            messages_to_be_sent.extend(messages)

            response = await self.llm_client.do_completion(
                model=self.model,
                messages=messages_to_be_sent,
                tools=orchestrator.tool_registry.tool_schemas(),
                temperature=0.6,
                top_p=0.95,
                extra_body={"chat_template_kwargs": {"enable_thinking": True}},
            )

            logger.debug(f"LLM Response: {response}")
            message = response.choices[0].message
            finish_reason = response.choices[0].finish_reason
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
        self._set_system_prompt(system_prompt)
        return await self._chat(messages, orchestrator)

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
                for tc in cast(list[dict[str, Any]], tool_calls):
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

        summary: str = response.choices[0].message.content or ""
        logger.info(f"Conversation compressed to {response.usage.total_tokens} tokens")
        return summary.strip()
