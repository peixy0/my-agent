from __future__ import annotations

import asyncio
import json
import logging
import time
from asyncio import Lock
from collections.abc import Awaitable
from typing import Any, Callable

from openai import AsyncOpenAI, InternalServerError, RateLimitError

logger = logging.getLogger(__name__)


class LLMClient:
    """
    A client for interacting with a Large Language Model (LLM) using the OpenAI API.

    This class provides a high-level interface for sending chat messages to an LLM,
    handling tool calls, and managing rate limits.
    """

    def __init__(self, url: str, model: str, api_key: str = "sk-dummy"):
        """
        Initializes the LLMClient.

        Args:
            url: The base URL of the LLM API.
            model: The name of the LLM model to use.
            api_key: The API key for the LLM API.
        """
        self.client: AsyncOpenAI = AsyncOpenAI(
            base_url=url,
            api_key=api_key,
        )
        self.model: str = model
        self.functions: dict[str, dict[str, Any]] = {}
        self.handlers: dict[str, Callable[..., Awaitable[dict[str, Any]]]] = {}

        self._rate_limit_secs: float = 1
        self._rate_limit_lock: Lock = Lock()
        self._next_request_time: float = 0.0

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

    async def _do_completion(self, *args: Any, **kwargs: Any) -> Any:
        """
        Performs a chat completion with rate limiting and retry logic.
        """
        retry_timer = 5
        while True:
            async with self._rate_limit_lock:
                now = time.monotonic()
                next_request_time = max(self._next_request_time, now)
                self._next_request_time = next_request_time + self._rate_limit_secs
                wait_time = now - next_request_time
                if wait_time > 0:
                    await asyncio.sleep(wait_time)

                try:
                    return await self.client.chat.completions.create(*args, **kwargs)
                except (InternalServerError, RateLimitError) as e:
                    logger.warning(
                        f"LLM Completion failure, retrying in {retry_timer}: {e}"
                    )
                    await asyncio.sleep(retry_timer)
                    retry_timer *= 2
                except Exception as e:
                    logger.warning(f"LLM Completion failure: {e}")
                    raise

    async def chat(
        self, messages: list[dict[str, str]], max_iterations: int = 50
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
        for _ in range(max_iterations):
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

            if message.content:
                return message.content

            if message.tool_calls:
                tool_messages: list[dict[str, str]] = []
                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_id = tool_call.id

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

            raise RuntimeError(f"Unexpected response: {response}")

        raise TimeoutError("Exceeded max tool call iterations without final content.")
