"""Unit tests for Agent.compress()."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.llm.agent import Agent


def _make_agent() -> Agent:
    """Create an Agent with a mocked LLM client."""
    llm_client = MagicMock()
    tool_registry = MagicMock()
    tool_registry.schemas = {}
    tool_registry.handlers = {}
    messaging = MagicMock()
    event_logger = MagicMock()
    return Agent(
        llm_client=llm_client,
        model="test-model",
        tool_registry=tool_registry,
        messaging=messaging,
        event_logger=event_logger,
    )


def _make_completion_response(content: str) -> MagicMock:
    """Build a minimal mock that looks like an OpenAI completion response."""
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


@pytest.mark.asyncio
async def test_compress_empty_messages_returns_empty_string() -> None:
    """compress() should return '' immediately without calling the LLM."""
    agent = _make_agent()
    agent.llm_client.do_completion = AsyncMock()

    result = await agent.compress([])

    assert result == ""
    agent.llm_client.do_completion.assert_not_called()


@pytest.mark.asyncio
async def test_compress_calls_llm_with_transcript() -> None:
    """compress() should call the LLM and return the summary content."""
    agent = _make_agent()
    expected_summary = "User asked about file X. Agent read it and found Y."
    agent.llm_client.do_completion = AsyncMock(
        return_value=_make_completion_response(expected_summary)
    )

    messages = [
        {"role": "user", "content": "What is in file X?"},
        {"role": "assistant", "content": "File X contains Y."},
    ]

    result = await agent.compress(messages)

    assert result == expected_summary
    agent.llm_client.do_completion.assert_awaited_once()

    call_kwargs = agent.llm_client.do_completion.call_args
    sent_messages = call_kwargs.kwargs["messages"]

    # System message should be the compression prompt
    assert sent_messages[0]["role"] == "system"
    assert "summarizer" in sent_messages[0]["content"].lower()

    # User message should contain the transcript
    user_content = sent_messages[1]["content"]
    assert "What is in file X?" in user_content
    assert "File X contains Y." in user_content


@pytest.mark.asyncio
async def test_compress_uses_low_temperature() -> None:
    """compress() should use a low temperature for deterministic summaries."""
    agent = _make_agent()
    agent.llm_client.do_completion = AsyncMock(
        return_value=_make_completion_response("summary")
    )

    await agent.compress([{"role": "user", "content": "hello"}])

    call_kwargs = agent.llm_client.do_completion.call_args.kwargs
    assert call_kwargs["temperature"] <= 0.5


@pytest.mark.asyncio
async def test_compress_strips_whitespace_from_summary() -> None:
    """compress() should strip leading/trailing whitespace from the LLM output."""
    agent = _make_agent()
    agent.llm_client.do_completion = AsyncMock(
        return_value=_make_completion_response("  summary with spaces  \n")
    )

    result = await agent.compress([{"role": "user", "content": "hi"}])

    assert result == "summary with spaces"


@pytest.mark.asyncio
async def test_compress_skips_messages_without_content() -> None:
    """compress() should not include tool-call messages that have no text content."""
    agent = _make_agent()
    agent.llm_client.do_completion = AsyncMock(
        return_value=_make_completion_response("summary")
    )

    messages = [
        {"role": "assistant", "content": None},  # tool_calls only, no text
        {"role": "tool", "content": ""},  # empty tool result
        {"role": "user", "content": "real message"},
    ]

    await agent.compress(messages)

    user_content = agent.llm_client.do_completion.call_args.kwargs["messages"][1][
        "content"
    ]
    assert "real message" in user_content
    # None / empty content should not appear as literal "None" or blank entries
    assert "[ASSISTANT]\nNone" not in user_content
    assert "[TOOL]\n" not in user_content
