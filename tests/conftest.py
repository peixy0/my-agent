"""Pytest configuration and fixtures."""

import pytest


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    from agent.core.settings import Settings

    return Settings(
        openai_base_url="https://api.example.com/v1",
        openai_model="test-model",
        openai_api_key="test-key",
        container_name="test-container",
        container_runtime="podman",
        workspace_dir="/tmp/test-workspace",
        journal_dir="/tmp/test-workspace/journal",
        event_log_file="/tmp/test-workspace/events.jsonl",
        skills_dir="/tmp/test-workspace/.skills",
        wake_interval_seconds=60,
    )
