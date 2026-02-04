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

    # Workspace paths (relative to host, mapped to /workspace in container)
    workspace_dir: str = "./workspace"
    context_file: str = "./workspace/CONTEXT"
    todo_file: str = "./workspace/TODO"
    journal_dir: str = "./workspace/journal"
    event_log_file: str = "./events.jsonl"
    skills_dir: str = "./workspace/.skills"
    wake_count_file: str = "./.wake_count"

    # Autonomous mode settings
    wake_interval_seconds: int = 600  # 10 minutes

    # Event streaming settings
    stream_api_url: str = ""
    stream_api_key: str = ""
    stream_events: bool = False

    # Tool settings
    whitelist_tools: list[str] = []

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8"
    )


def get_settings() -> Settings:
    """Factory function for settings, enables dependency injection in tests."""
    return Settings()


# Default singleton for convenience
settings = get_settings()
