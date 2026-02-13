"""
Agent module containing the core agent logic.

The Agent class orchestrates LLM interactions and context management.
Tool registration is handled externally by ToolRegistry (SRP).
System prompt construction is handled by SystemPromptBuilder (SRP).
"""

import asyncio
import json
import logging
from typing import Any

import jsonschema

from agent.core.event_logger import EventLogger
from agent.core.messaging import Messaging
from agent.llm.openai import OpenAIProvider
from agent.tools.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


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
        self,
        session_key: str | None,
        messages: list[dict[str, str]],
        max_iterations: int,
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
        handlers = self.tool_registry.handlers

        current_iteration = 0
        while True:
            current_iteration += 1
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

            if message.tool_calls:
                reasoning = message.content or getattr(message, "reasoning", "")
                if reasoning:
                    logger.info(f"Agent Thought: {reasoning.strip()}")
                    if session_key:
                        await self.messaging.send_message(session_key, reasoning)

                async def _handle_tool_call(
                    tool_call: Any, iteration: int
                ) -> dict[str, str]:
                    tool_name = tool_call.function.name
                    tool_id = tool_call.id
                    if iteration > max_iterations:
                        return {
                            "role": "tool",
                            "tool_call_id": tool_id,
                            "content": "Error: max tool call iterations reached.",
                        }

                    if tool_name not in handlers:
                        return {
                            "role": "tool",
                            "tool_call_id": tool_id,
                            "content": f"No such tool named {tool_name}",
                        }

                    logger.info(
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
                            schema=schemas[tool_name]["parameters"],
                        )
                        result: dict[str, Any] = await handlers[tool_name](**args)
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
                        logger.info(f"Tool call {tool_name} completed successfully")

                    await self.event_logger.log_tool_use(tool_name, args, result)

                    return {
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "content": json.dumps(result, ensure_ascii=False),
                    }

                tool_messages = await asyncio.gather(
                    *[
                        _handle_tool_call(tool_call, current_iteration)
                        for tool_call in message.tool_calls
                    ]
                )

                messages.extend(tool_messages)
                continue

            if finish_reason != "stop":
                messages.append({"role": "user", "content": "continue"})
                continue

            if session_key:
                await self.messaging.send_message(session_key, message.content)
                await self.event_logger.log_agent_response(
                    f"HUMAN INPUT RESPONSE:\n\n{message.content}"
                )
            else:
                await self.event_logger.log_agent_response(
                    f"SYSTEM RESPONSE:\n\n{message.content}"
                )
            return message.content

    async def run(
        self,
        session_key: str | None,
        messages: list[dict[str, str]],
        max_iterations: int = 80,
    ) -> str:
        """Run a single turn of the agent conversation."""

        self.messages.extend(messages)
        return await self._chat(session_key, self.messages, max_iterations)
