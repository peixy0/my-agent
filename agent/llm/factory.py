from __future__ import annotations

from agent.core.event_logger import EventLogger
from agent.llm.openai import OpenAIProvider
from agent.tools.tool_registry import ToolRegistry


class LLMFactory:
    """
    Factory for creating LLM clients based on the provided configuration.
    """

    def __init__(self, settings):
        self.settings = settings

    def create(
        self,
        event_logger: EventLogger,
        tool_registry: ToolRegistry,
    ) -> OpenAIProvider:
        """
        Return an LLM client implementation based on the base URL prefix.
        """
        return OpenAIProvider(
            url=self.settings.openai_base_url,
            model=self.settings.openai_model,
            api_key=self.settings.openai_api_key,
            event_logger=event_logger,
            tool_registry=tool_registry,
        )
