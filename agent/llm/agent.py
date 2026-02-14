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
    def __init__(self, model: str, tool_registry: ToolRegistry):
        self.model = model
        self.tool_registry = tool_registry
        self.tool_schemas = self.tool_registry.schemas
        self.tool_handlers = self.tool_registry.handlers

    @abstractmethod
    async def process(self, message: Any, finish_reason: str) -> list[dict[str, str]]:
        pass

    async def _handle_tool_call(self, tool_call: Any) -> ToolCallResult:
        tool_name = tool_call.function.name
        tool_id = tool_call.id

        if tool_name not in self.tool_handlers:
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
                schema=self.tool_schemas[tool_name]["parameters"],
            )
            result: dict[str, Any] = await self.tool_handlers[tool_name](**args)
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
    def __init__(
        self,
        model: str,
        tool_registry: ToolRegistry,
        messaging: Messaging,
        event_logger: EventLogger,
    ):
        super().__init__(model, tool_registry)
        self.messaging = messaging
        self.event_logger = event_logger

    @override
    async def process(self, message: Any, finish_reason: str) -> list[dict[str, str]]:
        if message.tool_calls:
            tool_results = await asyncio.gather(
                *[self._handle_tool_call(tool_call) for tool_call in message.tool_calls]
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

        await self.event_logger.log_agent_response(
            f"Heartbeat Response:\n\n{message.content}"
        )
        content = message.content.strip()
        if not content.endswith("NO_REPORT"):
            await self.messaging.notify(content)
        return []


class HumanInputOrchestrator(Orchestrator):
    def __init__(
        self,
        session_id: str,
        model: str,
        tool_registry: ToolRegistry,
        messaging: Messaging,
        event_logger: EventLogger,
    ):
        super().__init__(model, tool_registry)
        self.session_id = session_id
        self.messaging = messaging
        self.event_logger = event_logger

    @override
    async def process(self, message: Any, finish_reason: str) -> list[dict[str, str]]:
        if message.tool_calls:
            if message.content:
                await self.messaging.send_message(self.session_id, message.content)
            tool_results = await asyncio.gather(
                *[self._handle_tool_call(tool_call) for tool_call in message.tool_calls]
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

        await self.event_logger.log_agent_response(
            f"Human Input Response:\n\n{message.content}"
        )
        content = message.content.strip()
        await self.messaging.send_message(self.session_id, content)
        return []


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
        self.messages = []
        self.system_prompt = ""

    def set_system_prompt(self, prompt: str) -> None:
        """Set the system prompt and reset conversation history."""
        self.system_prompt = prompt
        self.messages = [{"role": "system", "content": self.system_prompt}]

    async def _chat(
        self, messages: list[dict[str, str]], orchestrator: Orchestrator
    ) -> str:
        """
        Sends a chat message to the LLM and handles the response.

        This method will automatically handle tool calls made by the LLM.

        Args:
            messages: A list of messages in the chat history.
            max_iterations: The maximum number of tool call iterations to perform.

        Returns:
            The final content of the LLM's response.
        """
        schemas = self.tool_registry.schemas

        while True:
            response = await self.llm_client.do_completion(
                model=self.model,
                messages=messages,
                tools=[{"type": "function", "function": fn} for fn in schemas.values()],
                temperature=1.0,
                top_p=1.0,
                tool_choice="auto",
                extra_body={
                    "chat_template_kwargs": {
                        "thinking": True,
                    }
                },
            )

            logger.debug(f"LLM Response: {response}")
            message = response.choices[0].message
            finish_reason = response.choices[0].finish_reason
            messages.append(message.model_dump())

            reply = await orchestrator.process(message, finish_reason)
            if not reply:
                return message.content
            messages.extend(reply)

    async def run(
        self,
        messages: list[dict[str, str]],
        orchestrator: Orchestrator,
    ) -> str:
        """Run a single turn of the agent conversation."""

        self.messages.extend(messages)
        return await self._chat(self.messages, orchestrator)
