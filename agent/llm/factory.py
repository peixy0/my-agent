from __future__ import annotations

from agent.llm.openai import OpenAIProvider


class LLMFactory:
    """
    Factory for creating LLM clients based on the provided configuration.
    """

    def __init__(self, settings):
        self.settings = settings

    def create(self) -> OpenAIProvider:
        """
        Return an LLM client implementation based on the base URL prefix.
        """
        return OpenAIProvider(
            url=self.settings.openai_base_url,
            api_key=self.settings.openai_api_key,
        )
