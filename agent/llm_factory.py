from __future__ import annotations

from .llm_base import LLMBase


class LLMFactory:
    """
    Factory for creating LLM clients based on the provided configuration.
    """

    @staticmethod
    def _is_zai_base_url(base_url: str) -> bool:
        """
        Check whether the provided base URL should use the Zai client.
        """
        return base_url.startswith("https://api.z.ai")

    @classmethod
    def create(cls, url: str, model: str, api_key: str = "sk-dummy") -> LLMBase:
        """
        Return an LLM client implementation based on the base URL prefix.
        """
        if cls._is_zai_base_url(url):
            from .llm_zai import LLMClient as ZaiLLMClient

            return ZaiLLMClient(url=url, model=model, api_key=api_key)

        from .llm_openai import LLMClient as OpenAILLMClient

        return OpenAILLMClient(url=url, model=model, api_key=api_key)
