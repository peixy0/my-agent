import logging
from asyncio.exceptions import CancelledError
from typing import Any

from openai import AsyncOpenAI, BadRequestError, DefaultAioHttpClient
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_not_exception_type,
    wait_exponential,
)

from agent.llm.types import (
    ChoiceView,
    CompletionResponseView,
    MessageView,
    ToolCallFunctionView,
    ToolCallView,
    UsageView,
)

logger = logging.getLogger(__name__)


def _normalize(response: Any) -> CompletionResponseView:
    choices = []
    for index, choice in enumerate(response.choices):
        msg = choice.message
        tool_calls = [
            ToolCallView(
                id=tc.id,
                type=tc.type,
                function=ToolCallFunctionView(
                    name=tc.function.name,
                    arguments=tc.function.arguments,
                ),
            )
            for tc in (msg.tool_calls or [])
        ]
        choices.append(
            ChoiceView(
                index=index,
                finish_reason=choice.finish_reason or "stop",
                message=MessageView(
                    role=msg.role,
                    content=msg.content,
                    tool_calls=tool_calls,
                ),
            )
        )
    usage = response.usage
    return CompletionResponseView(
        choices=choices,
        usage=UsageView(
            total_tokens=usage.total_tokens if usage else 0,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        ),
        model=response.model or "",
    )


class OpenAIProvider:
    def __init__(
        self,
        url: str,
        api_key: str,
        proxy: str = "",
    ):
        http_client = DefaultAioHttpClient(proxy=proxy) if proxy else None
        self.client: AsyncOpenAI = AsyncOpenAI(
            base_url=url,
            api_key=api_key,
            http_client=http_client,
            timeout=600,
        )

    @retry(
        retry=retry_if_not_exception_type((BadRequestError, CancelledError)),
        wait=wait_exponential(multiplier=2, min=5, max=300),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def do_completion(self, *args: Any, **kwargs: Any) -> CompletionResponseView:
        response = await self.client.chat.completions.create(*args, **kwargs)
        if not response.choices:
            raise Exception("Invalid response")
        return _normalize(response)
