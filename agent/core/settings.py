from typing import ClassVar

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Settings for the autonomous LLM agent.

    The agent runs on the host machine while using a container
    as a workspace environment for command and file operations.
    """

    # LLM settings
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o"
    openai_api_key: str = ""

    # Container settings
    container_name: str = "sys-agent-workspace"
    container_runtime: str = "podman"

    # Agent settings
    tool_timeout: int = 60
    proxy: str = ""

    # Workspace paths
    cwd: str = "./workspace"
    workspace_dir: str = "./"
    skills_dir: str = "./.skills"

    # Autonomous mode settings
    wake_interval_seconds: int = 1800  # 30 minutes

    # Context compression settings
    enable_compression: bool = True
    context_max_tokens: int = 30000
    context_num_keep_last: int = 10

    # Event streaming settings
    event_api_url: str = ""
    event_api_key: str = ""

    # Feishu settings
    feishu_app_id: str = ""
    feishu_app_secret: str = ""
    feishu_encrypt_key: str = ""
    feishu_verification_token: str = ""
    feishu_notify_chat_id: str = ""

    # API server settings
    api_enabled: bool = False
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


def get_settings() -> Settings:
    """Factory function for settings, enables dependency injection in tests."""
    return Settings()
