from __future__ import annotations

from agent.llm.base import LLMBase


class LLMFactory:
    """
    Factory for creating LLM clients based on the provided configuration.
    """

    @classmethod
    def create(cls, url: str, model: str, api_key: str = "sk-dummy") -> LLMBase:
        """
        Return an LLM client implementation based on the base URL prefix.
        """
        from agent.llm.openai import LLMClient

        return LLMClient(url=url, model=model, api_key=api_key)
