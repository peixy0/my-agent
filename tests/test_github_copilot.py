from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from agent.core.settings import Settings
from agent.llm.copilot import (
    GitHubCopilotAuthState,
    GitHubCopilotAuthStore,
    GitHubCopilotProvider,
)
from agent.llm.factory import LLMFactory
from agent.llm.openai import OpenAIProvider


def test_auth_store_round_trip(tmp_path):
    store = GitHubCopilotAuthStore(
        state_path=".state/github-copilot.json",
        workspace_dir=str(tmp_path),
    )
    state = GitHubCopilotAuthState(
        github_access_token="gh-token",
        github_scope="read:user",
        copilot_access_token="copilot-token",
        copilot_token_expires_at="2030-01-01T00:00:00+00:00",
    )

    store.save(state)

    loaded = store.load()
    assert loaded.github_access_token == "gh-token"
    assert loaded.copilot_access_token == "copilot-token"

    payload = json.loads(
        (tmp_path / ".state/github-copilot.json").read_text(encoding="utf-8")
    )
    assert payload["github_scope"] == "read:user"


def test_factory_creates_openai_provider(tmp_path):
    settings = Settings(
        llm_provider="openai",
        openai_base_url="https://api.example.com/v1",
        openai_model="test-model",
        openai_api_key="test-key",
        container_name="test-container",
        container_runtime="podman",
        workspace_dir=str(tmp_path),
        skills_dir=str(tmp_path / ".skills"),
        wake_interval_seconds=60,
        webui_host="127.0.0.1",
        webui_port=8999,
    )

    factory = LLMFactory(settings)
    assert factory.get_model_name() == "test-model"


@pytest.mark.asyncio
async def test_factory_creates_openai_provider_async(tmp_path):
    settings = Settings(
        llm_provider="openai",
        openai_base_url="https://api.example.com/v1",
        openai_model="test-model",
        openai_api_key="test-key",
        container_name="test-container",
        container_runtime="podman",
        workspace_dir=str(tmp_path),
        skills_dir=str(tmp_path / ".skills"),
        wake_interval_seconds=60,
        webui_host="127.0.0.1",
        webui_port=8999,
    )

    factory = LLMFactory(settings)
    provider = await factory.create()

    assert isinstance(provider, OpenAIProvider)


@pytest.mark.asyncio
async def test_factory_creates_github_copilot_provider(tmp_path):
    settings = Settings(
        llm_provider="github-copilot",
        openai_base_url="https://api.example.com/v1",
        openai_model="unused-openai-model",
        openai_api_key="test-key",
        github_copilot_model="copilot-model",
        container_name="test-container",
        container_runtime="podman",
        workspace_dir=str(tmp_path),
        skills_dir=str(tmp_path / ".skills"),
        wake_interval_seconds=60,
        webui_host="127.0.0.1",
        webui_port=8999,
    )

    factory = LLMFactory(settings)
    with patch(
        "agent.llm.copilot.GitHubCopilotAuthManager.ensure_copilot_token",
        new_callable=AsyncMock,
        return_value="mocked-token",
    ):
        provider = await factory.create()

    assert isinstance(provider, GitHubCopilotProvider)
    assert factory.get_model_name() == "copilot-model"


def test_copilot_response_is_normalized():
    response = GitHubCopilotProvider._coerce_response(
        {
            "id": "cmpl-1",
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 7,
                "total_tokens": 18,
            },
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "arguments": '{"path":"README.md"}',
                                },
                            }
                        ],
                    },
                }
            ],
        }
    )

    assert response.usage.prompt_tokens == 11
    assert response.usage.total_tokens == 18
    assert response.choices[0].finish_reason == "tool_calls"
    assert response.choices[0].message.role == "assistant"
    assert response.choices[0].message.tool_calls[0].id == "call_1"
    assert response.choices[0].message.tool_calls[0].function.name == "read_file"


def test_copilot_response_defaults_usage_to_zero():
    response = GitHubCopilotProvider._coerce_response(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "done",
                    }
                }
            ]
        }
    )

    assert response.usage.prompt_tokens == 0
    assert response.usage.completion_tokens == 0
    assert response.usage.total_tokens == 0
