from typing import ClassVar

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Settings for the autonomous LLM agent.

    The agent runs on the host machine while using a container
    as a workspace environment for command and file operations.
    """

    # LLM settings
    llm_provider: str = "openai"

    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o"
    openai_api_key: str = ""

    github_copilot_model: str = "gpt-4o"
    github_copilot_oauth_client_id: str = "Iv1.b507a08c87ecfe98"
    github_copilot_scope: str = "read:user"
    github_copilot_device_code_url: str = "https://github.com/login/device/code"
    github_copilot_access_token_url: str = "https://github.com/login/oauth/access_token"
    github_copilot_token_url: str = "https://api.github.com/copilot_internal/v2/token"
    github_copilot_api_base_url: str = "https://api.githubcopilot.com"
    github_copilot_state_path: str = ".state/github-copilot.json"
    github_copilot_models_path: str = ".state/github-copilot-models.json"
    github_copilot_user_agent: str = "my-agent/0.1.0"
    github_copilot_editor_version: str = "vscode/1.99.0"
    github_copilot_editor_plugin_version: str = "my-agent/0.1.0"
    github_copilot_openai_intent: str = "conversation-panel"

    # Container settings
    container_name: str = "sys-agent-workspace"
    container_runtime: str = ""  # "docker" or "podman", run on host if empty

    # Agent settings
    tool_timeout: int = 60
    proxy: str = ""
    web_tools_enabled: bool = True

    # Workspace paths
    cwd: str = "./workspace"
    workspace_dir: str = "./"
    skills_dir: str = "./.skills"

    # Autonomous mode settings
    wake_interval_seconds: int = 1800  # 30 minutes

    # Context compression settings
    context_auto_compression_enabled: bool = False
    context_max_tokens: int = 100000
    context_num_keep_last: int = 9

    # Feishu settings
    feishu_app_id: str = ""
    feishu_app_secret: str = ""
    feishu_encrypt_key: str = ""
    feishu_verification_token: str = ""

    # Vision settings
    vision_support: bool = False
    max_image_size_bytes: int = 5 * 1024 * 1024  # 5 MB

    # WebUI server settings
    webui_enabled: bool = True
    webui_host: str = "localhost"
    webui_port: int = 8017

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


def get_settings() -> Settings:
    """Factory function for settings, enables dependency injection in tests."""
    return Settings()
