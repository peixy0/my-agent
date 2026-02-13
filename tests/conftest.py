"""Pytest configuration and fixtures."""

import pytest

from agent.core.settings import Settings


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    return Settings(
        openai_base_url="https://api.example.com/v1",
        openai_model="test-model",
        openai_api_key="test-key",
        container_name="test-container",
        container_runtime="podman",
        workspace_dir="/tmp/test-workspace",
        skills_dir="/tmp/test-workspace/.skills",
        wake_interval_seconds=60,
        api_host="127.0.0.1",
        api_port=8999,
    )
