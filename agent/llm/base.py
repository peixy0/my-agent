import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any

logger = logging.getLogger(__name__)


class LLMBase(ABC):
    def __init__(self, model: str):
        self.model: str = model
        self.functions: dict[str, dict[str, Any]] = {}
        self.handlers: dict[str, Callable[..., Awaitable[dict[str, Any]]]] = {}

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

    async def chat(
        self, messages: list[dict[str, str]], max_iterations: int = 50
    ) -> str:
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
                tool_choice="auto",
            )

            message = response.choices[0].message
            messages.append(message.model_dump())

            if message.tool_calls:
                if message.content:
                    logger.info(f"LLM Thought: {message.content}")

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

                    args: dict[str, Any] = json.loads(tool_call.function.arguments)
                    try:
                        result: dict[str, Any] = await self.handlers[tool_name](**args)
                    except Exception as e:
                        result = {"error": f"Exception occured during tool call: {e}"}

                    tool_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_id,
                            "content": json.dumps(result, ensure_ascii=False),
                        }
                    )

                messages.extend(tool_messages)
                continue

            if message.content:
                return message.content

            raise RuntimeError(f"Unexpected response: {response}")
