import logging
from typing import Any

from openai import AsyncOpenAI, BadRequestError
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_not_exception_type,
    wait_exponential,
)
from typing_extensions import override

from agent.core.event_logger import EventLogger
from agent.llm.base import LLMBase

logger = logging.getLogger(__name__)


class LLMClient(LLMBase):
    """
    A client for interacting with a Large Language Model (LLM) using the OpenAI API.

    This class provides a high-level interface for sending chat messages to an LLM,
    handling tool calls, and managing rate limits.
    """

    def __init__(self, url: str, model: str, api_key: str, event_logger: EventLogger):
        """
        Initializes the LLMClient.

        Args:
            url: The base URL of the LLM API.
            model: The name of the LLM model to use.
            api_key: The API key for the LLM API.
            event_logger: The event logger to use for logging events.
        """
        self.client: AsyncOpenAI = AsyncOpenAI(
            base_url=url,
            api_key=api_key,
        )
        super().__init__(model, event_logger)

    @retry(
        retry=retry_if_not_exception_type(BadRequestError),
        wait=wait_exponential(multiplier=2, min=5, max=300),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    @override
    async def _do_completion(self, *args: Any, **kwargs: Any) -> Any:
        """
        Performs a chat completion with rate limiting and retry logic using tenacity.
        """
        return await self.client.chat.completions.create(*args, **kwargs)
