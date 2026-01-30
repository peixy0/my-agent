import logging
from typing import Any

from openai import AsyncOpenAI, InternalServerError, RateLimitError
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from typing_extensions import override

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
        super().__init__(model)

    @retry(
        retry=retry_if_exception_type((InternalServerError, RateLimitError)),
        wait=wait_exponential(multiplier=1, min=5, max=60),
        stop=stop_after_attempt(5),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    @override
    async def _do_completion(self, *args: Any, **kwargs: Any) -> Any:
        """
        Performs a chat completion with rate limiting and retry logic using tenacity.
        """
        return await self.client.chat.completions.create(*args, **kwargs)
