from pathlib import Path
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
    container_runtime: str = ""  # "docker" or "podman", run on host if empty

    # Agent settings
    tool_timeout: int = 60
    max_output_chars: int = 10_000
    web_search_proxy: str = ""

    # Workspace paths
    cwd: str = "./workspace"
    project_dir: str = Path(__file__).parent.parent.parent.resolve().as_posix()
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
