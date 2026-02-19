import logging
from typing import Any

from openai import AsyncOpenAI, BadRequestError
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_not_exception_type,
    wait_exponential,
)

logger = logging.getLogger(__name__)


class OpenAIProvider:
    def __init__(
        self,
        url: str,
        api_key: str,
    ):
        self.client: AsyncOpenAI = AsyncOpenAI(
            base_url=url,
            api_key=api_key,
        )

    @retry(
        retry=retry_if_not_exception_type(BadRequestError),
        wait=wait_exponential(multiplier=2, min=5, max=300),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def do_completion(self, *args: Any, **kwargs: Any) -> Any:
        """
        Performs a chat completion with rate limiting and retry logic using tenacity.
        """
        response = await self.client.chat.completions.create(*args, **kwargs)
        if not response.choices:
            raise Exception("Invalid response")
        return response
