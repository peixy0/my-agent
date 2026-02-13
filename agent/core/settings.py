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

    # Workspace paths (relative to host, mapped to /workspace in container)
    workspace_dir: str = "./workspace"
    journal_dir: str = "./workspace/journal"
    event_log_file: str = "./events.jsonl"
    skills_dir: str = "./workspace/.skills"

    # Autonomous mode settings
    wake_interval_seconds: int = 1800  # 30 minutes

    # Event streaming settings
    event_api_url: str = ""
    event_api_key: str = ""

    # Feishu settings
    feishu_app_id: str = ""
    feishu_app_secret: str = ""
    feishu_encrypt_key: str = ""
    feishu_verification_token: str = ""
    feishu_notify_channel_id: str = ""

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
