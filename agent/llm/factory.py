from __future__ import annotations

from pathlib import Path

from agent.llm.copilot import (
    GitHubCopilotAuthManager,
    GitHubCopilotAuthStore,
    GitHubCopilotProvider,
)
from agent.llm.openai import OpenAIProvider


class LLMFactory:
    """
    Factory for creating LLM clients based on the provided configuration.
    """

    def __init__(self, settings):
        self.settings = settings

    def get_model_name(self) -> str:
        provider = self.settings.llm_provider.lower()
        if provider == "github-copilot":
            return self.settings.github_copilot_model
        return self.settings.openai_model

    async def create(self) -> OpenAIProvider | GitHubCopilotProvider:
        """
        Return an LLM client implementation based on the configured provider.
        For the GitHub Copilot provider, the auth token is acquired here.
        """
        provider = self.settings.llm_provider.lower()
        if provider == "github-copilot":
            auth_store = GitHubCopilotAuthStore(
                state_path=self.settings.github_copilot_state_path,
                workspace_dir=self.settings.workspace_dir,
            )
            auth_manager = GitHubCopilotAuthManager(
                auth_store=auth_store,
                client_id=self.settings.github_copilot_oauth_client_id,
                scope=self.settings.github_copilot_scope,
                device_code_url=self.settings.github_copilot_device_code_url,
                access_token_url=self.settings.github_copilot_access_token_url,
                copilot_token_url=self.settings.github_copilot_token_url,
                proxy=self.settings.proxy,
                user_agent=self.settings.github_copilot_user_agent,
                editor_version=self.settings.github_copilot_editor_version,
                editor_plugin_version=self.settings.github_copilot_editor_plugin_version,
            )
            await auth_manager.ensure_copilot_token()
            provider = GitHubCopilotProvider(
                api_base_url=self.settings.github_copilot_api_base_url,
                auth_manager=auth_manager,
                proxy=self.settings.proxy,
                user_agent=self.settings.github_copilot_user_agent,
                editor_version=self.settings.github_copilot_editor_version,
                editor_plugin_version=self.settings.github_copilot_editor_plugin_version,
                openai_intent=self.settings.github_copilot_openai_intent,
            )
            models_path = (
                Path(self.settings.workspace_dir).expanduser().resolve()
                / self.settings.github_copilot_models_path
            ).resolve()
            await provider.fetch_and_save_models(models_path)
            return provider

        return OpenAIProvider(
            url=self.settings.openai_base_url,
            api_key=self.settings.openai_api_key,
            proxy=self.settings.proxy,
        )
