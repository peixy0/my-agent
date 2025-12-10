from __future__ import annotations

import asyncio
import json
import logging
import time
from asyncio import Lock
from collections.abc import Awaitable
from typing import Any, Callable

from openai import AsyncOpenAI, InternalServerError, RateLimitError

from .llm_base import LLMBase

logger = logging.getLogger(__name__)


class LLMClient(LLMBase):
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
