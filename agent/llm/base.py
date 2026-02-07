import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any

import jsonschema

from agent.core.event_logger import EventLogger

logger = logging.getLogger(__name__)


class LLMBase(ABC):
    def __init__(self, model: str, event_logger: EventLogger):
        self.model: str = model
        self.functions: dict[str, dict[str, Any]] = {}
        self.handlers: dict[str, Callable[..., Awaitable[dict[str, Any]]]] = {}
        self.event_logger = event_logger

    def register_function(self, parameters: dict[str, Any]):
        """
        A decorator for registering a tool function with the LLM client.

        Args:
            parameters: A JSON schema describing the parameters of the tool function.

        Returns:
            A decorator that registers the tool function.
        """

        def decorator(func: Callable[..., Awaitable[dict[str, Any]]]):
            name = func.__name__
            description = func.__doc__ or ""
            self.functions[name] = {
                "name": name,
                "description": description.strip(),
                "parameters": parameters,
            }
            self.handlers[name] = func
            return func

        return decorator

    @abstractmethod
    async def _do_completion(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    async def chat(self, messages: list[dict[str, str]], max_iterations: int) -> str:
        """
        Sends a chat message to the LLM and handles the response.

        This method will automatically handle tool calls made by the LLM.

        Args:
            messages: A list of messages in the chat history.
            messages: A list of messages in the chat history.

            max_iterations: The maximum number of tool call iterations to perform.

        Returns:
            The final content of the LLM's response.
        """
        current_iteration = 0
        while True:
            current_iteration += 1
            response = await self._do_completion(
                model=self.model,
                messages=messages,
                tools=[
                    {"type": "function", "function": fn}
                    for fn in self.functions.values()
                ],
                temperature=1.0,
                top_p=1.0,
                tool_choice="auto",
                extra_body={
                    "chat_template_kwargs": {
                        "thinking": True,
                    }
                },
            )

            message = response.choices[0].message
            finish_reason = response.choices[0].finish_reason
            messages.append(message.model_dump())

            if message.tool_calls:
                reasoning = message.content or getattr(message, "reasoning", "")
                if reasoning:
                    logger.info(f"Agent Thought: {reasoning.strip()}")

                tool_messages: list[dict[str, str]] = []
                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_id = tool_call.id
                    if current_iteration > max_iterations:
                        tool_messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_id,
                                "content": "Error: max tool call iterations reached.",
                            }
                        )
                        continue

                    if tool_name not in self.handlers:
                        tool_messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_id,
                                "content": f"No such tool named {tool_name}",
                            }
                        )
                        continue

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
                            schema=self.functions[tool_name]["parameters"],
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
                        logger.info(f"Tool call {tool_name} completed successfully")

                    await self.event_logger.log_tool_use(tool_name, args, result)

                    tool_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_id,
                            "content": json.dumps(result, ensure_ascii=False),
                        }
                    )

                messages.extend(tool_messages)
                continue

            if finish_reason != "stop":
                messages.append({"role": "user", "content": "continue"})
                continue

            await self.event_logger.log_agent_response(message.content)
            return message.content
