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
from typing import Any, override

import jsonschema

from agent.core.event_logger import EventLogger
from agent.core.messaging import Messaging
from agent.llm.openai import OpenAIProvider
from agent.tools.tool_registry import ToolRegistry

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
        event_logger: EventLogger,
    ):
        self.model = model
        self.tool_registry = tool_registry
        self.schemas = tool_registry.schemas
        self.handlers = tool_registry.handlers
        self.messaging = messaging
        self.event_logger = event_logger

    def register_instance_tool(
        self,
        func: Any,
        schema: dict[str, Any],
    ) -> None:
        """Register a tool specific to this orchestrator instance."""
        tool_name = func.__name__
        tool_description = func.__doc__

        self.schemas[tool_name] = {
            "name": tool_name,
            "description": tool_description,
            "parameters": schema,
        }
        self.handlers[tool_name] = self.tool_registry.wrap_tool(func, tool_name)

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
            reply: list[dict[str, Any]] = []
            for tool_result in tool_results:
                await self.event_logger.log_tool_use(
                    tool_result.tool_name, tool_result.args, tool_result.result
                )
                reply.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_result.tool_id,
                        "content": json.dumps(tool_result.result, ensure_ascii=False),
                    }
                )
            return reply

        if finish_reason != "stop":
            return [{"role": "user", "content": "continue"}]

        content = message.content or ""
        content = content.strip()
        await self.event_logger.log_agent_response(
            f"{self.response_label} Response:\n\n{message.content}"
        )
        await self._on_final_response(content)
        return []

    async def _handle_tool_call(self, tool_call: Any) -> ToolCallResult:
        tool_name = tool_call.function.name
        tool_id = tool_call.id

        if tool_name not in self.handlers:
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
                for arg_name, arg_value in args.items():
                    try:
                        arg_value_parsed = json.loads(arg_value)
                        args[arg_name] = arg_value_parsed
                    except json.JSONDecodeError:
                        pass
            jsonschema.validate(
                instance=args,
                schema=self.schemas[tool_name]["parameters"],
            )
            result: dict[str, Any] = await self.handlers[tool_name](**args)
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


class HeartbeatOrchestrator(Orchestrator):
    response_label = "Heartbeat"

    @override
    async def _before_tool_use(self, message: Any) -> None:
        pass

    @override
    async def _on_final_response(self, content: str) -> None:
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
        event_logger: EventLogger,
    ):
        super().__init__(model, tool_registry, messaging, event_logger)
        self.chat_id = chat_id
        self.message_id = message_id
        self._register_tools()

    def _register_tools(self) -> None:
        async def add_reaction(emoji: str) -> dict[str, Any]:
            """
            React to the current message with an emoji.
            """
            try:
                await self.messaging.add_reaction(self.message_id, emoji)
                return {
                    "status": "success",
                    "message": f"Added reaction {emoji} to message",
                }
            except Exception as e:
                logger.error(f"Failed to add reaction {emoji}: {e}", exc_info=True)
                return {"status": "error", "message": str(e)}

        self.register_instance_tool(
            add_reaction,
            {
                "type": "object",
                "properties": {
                    "emoji": {
                        "type": "string",
                        "enum": [
                            "OK",
                            "THUMBSUP",
                            "MUSCLE",
                            "LOL",
                            "THINKING",
                            "Shrug",
                            "Fire",
                            "Coffee",
                            "PARTY",
                            "CAKE",
                            "HEART",
                        ],
                        "description": "The emoji type to react with. OK, THUMBSUP, MUSCLE, LOL, THINKING, Shrug, Fire, Coffee, PARTY, CAKE, HEART",
                    }
                },
                "required": ["emoji"],
            },
        )

        async def send_image(image_path: str) -> dict[str, Any]:
            """
            Send an image file to the user. Image file size must be under 10 MiB.
            """
            try:
                await self.messaging.send_image(self.chat_id, image_path)
                return {
                    "status": "success",
                    "message": f"Sent image {image_path} to user",
                }
            except Exception as e:
                return {"status": "error", "message": str(e)}

        self.register_instance_tool(
            send_image,
            {
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "Absolute path to the image file to send.",
                    }
                },
                "required": ["image_path"],
            },
        )

    @override
    async def _before_tool_use(self, message: Any) -> None:
        if message.content:
            await self.messaging.send_message(self.chat_id, message.content)

    @override
    async def _on_final_response(self, content: str) -> None:
        await self.messaging.send_message(self.chat_id, content)


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
        event_logger: EventLogger,
    ):
        self.llm_client = llm_client
        self.model = model
        self.tool_registry = tool_registry
        self.messaging = messaging
        self.event_logger = event_logger
        self.system_messages = []
        self.system_prompt = ""

    def _set_system_prompt(self, prompt: str) -> None:
        """Set the system prompt and reset conversation history."""
        self.system_prompt = prompt
        self.system_messages = [{"role": "system", "content": self.system_prompt}]

    async def _chat(
        self, messages: list[dict[str, str]], orchestrator: Orchestrator
    ) -> Any:
        """
        Sends a chat message to the LLM and handles the response.

        This method will automatically handle tool calls made by the LLM.

        Args:
            messages: A list of messages in the chat history.
            max_iterations: The maximum number of tool call iterations to perform.

        Returns:
            The LLM's response.
        """
        # Use orchestrator's schemas (includes instance tools)
        schemas = orchestrator.schemas

        while True:
            messages_to_be_sent = self.system_messages.copy()
            messages_to_be_sent.extend(messages)

            response = await self.llm_client.do_completion(
                model=self.model,
                messages=messages_to_be_sent,
                tools=[{"type": "function", "function": fn} for fn in schemas.values()],
                temperature=1.0,
                top_p=1.0,
                tool_choice="auto",
                timeout=600,
                extra_body={
                    "chat_template_kwargs": {
                        "enable_thinking": True,
                        "clear_thinking": False,
                    }
                },
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
        messages: list[dict[str, str]],
        orchestrator: Orchestrator,
    ) -> Any:
        """Run a single turn of the agent conversation."""
        self._set_system_prompt(system_prompt)
        return await self._chat(messages, orchestrator)

    async def compress(self, messages: list[dict[str, str]]) -> str:
        """
        Compress conversation history via full LLM summarization.

        Sends all messages to the LLM with a dedicated summarization prompt
        and returns a concise digest preserving key facts, decisions, tool
        results, and ongoing tasks.

        Returns an empty string when there is nothing to summarize.
        The caller is responsible for clearing conversation.messages and
        storing the returned summary in conversation.previous_summary.
        """
        if not messages:
            return ""

        compression_prompt = (
            "You are a conversation summarizer. "
            "Produce a concise but complete summary of the conversation below. "
            "Preserve: key facts, decisions made, tool results, file paths, "
            "ongoing tasks, and any important context the agent will need later. "
            "Write in third-person past tense. Be dense — omit pleasantries."
        )

        # Serialize messages as a readable transcript for the LLM
        transcript_parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if content:
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
