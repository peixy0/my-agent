from __future__ import annotations

from agent.core.event_logger import EventLogger
from agent.llm.base import LLMBase
from agent.tools.tool_registry import ToolRegistry


class LLMFactory:
    """
    Factory for creating LLM clients based on the provided configuration.
    """

    @classmethod
    def create(
        cls,
        url: str,
        model: str,
        api_key: str,
        event_logger: EventLogger,
        tool_registry: ToolRegistry,
    ) -> LLMBase:
        """
        Return an LLM client implementation based on the base URL prefix.
        """
        from agent.llm.openai import LLMClient

        return LLMClient(
            url=url,
            model=model,
            api_key=api_key,
            event_logger=event_logger,
            tool_registry=tool_registry,
        )
