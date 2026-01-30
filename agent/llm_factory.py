from __future__ import annotations

from .llm_base import LLMBase


class LLMFactory:
    """
    Factory for creating LLM clients based on the provided configuration.
    """

    @classmethod
    def create(cls, url: str, model: str, api_key: str = "sk-dummy") -> LLMBase:
        """
        Return an LLM client implementation based on the base URL prefix.
        """
        from .llm_openai import LLMClient

        return LLMClient(url=url, model=model, api_key=api_key)
