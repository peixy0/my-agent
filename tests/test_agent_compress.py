"""Unit tests for Agent.compress()."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.llm.agent import Agent
from agent.llm.types import ChoiceView, CompletionResponseView, MessageView, UsageView


def _make_agent() -> Agent:
    """Create an Agent with a mocked LLM client."""
    llm_client = MagicMock()
    tool_registry = MagicMock()
    tool_registry.schemas = {}
    tool_registry.handlers = {}
    return Agent(
        llm_client=llm_client,
        model="test-model",
        tool_registry=tool_registry,
    )


def _make_completion_response(content: str) -> CompletionResponseView:
    """Build a minimal completion response for testing."""
    return CompletionResponseView(
        choices=[
            ChoiceView(
                message=MessageView(role="assistant", content=content, tool_calls=[]),
                finish_reason="stop",
            )
        ],
        usage=UsageView(),
        model="test-model",
    )


@pytest.mark.asyncio
async def test_compress_empty_messages_is_noop() -> None:
    """compress() should do nothing when given an empty list."""
    agent = _make_agent()
    agent.llm_client.do_completion = AsyncMock()

    messages: list = []
    await agent.compress(messages)

    assert messages == []
    agent.llm_client.do_completion.assert_not_called()


@pytest.mark.asyncio
async def test_compress_replaces_messages_with_summary() -> None:
    """compress() should replace messages in-place with a single summary message."""
    agent = _make_agent()
    expected_summary = "User asked about file X. Agent read it and found Y."
    agent.llm_client.do_completion = AsyncMock(
        return_value=_make_completion_response(expected_summary)
    )

    messages = [
        {"role": "user", "content": "What is in file X?"},
        {"role": "assistant", "content": "File X contains Y."},
    ]

    await agent.compress(messages)

    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert expected_summary in messages[0]["content"]
    agent.llm_client.do_completion.assert_awaited_once()

    # Compression prompt should be the system message
    sent_messages = agent.llm_client.do_completion.call_args.kwargs["messages"]
    assert sent_messages[0]["role"] == "system"
    assert "summarizer" in sent_messages[0]["content"].lower()

    # Transcript should contain the original messages
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

    messages = [{"role": "user", "content": "hi"}]
    await agent.compress(messages)

    assert "summary with spaces" in messages[0]["content"]


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
    assert "[ASSISTANT]\nNone" not in user_content
    assert "[TOOL]\n" not in user_content


@pytest.mark.asyncio
async def test_compress_incremental_via_summary_message() -> None:
    """compress() naturally incorporates a prior summary already in the message list."""
    agent = _make_agent()
    agent.llm_client.do_completion = AsyncMock(
        return_value=_make_completion_response("new summary")
    )

    prior = "Agent previously set up the database."
    messages = [
        {"role": "user", "content": f"[CONVERSATION SUMMARY]\n\n{prior}"},
        {"role": "user", "content": "follow-up question"},
    ]

    await agent.compress(messages)

    user_content = agent.llm_client.do_completion.call_args.kwargs["messages"][1][
        "content"
    ]
    assert prior in user_content
    assert "follow-up question" in user_content


@pytest.mark.asyncio
async def test_compress_pairs_tool_call_with_result() -> None:
    """compress() should emit a single TOOL line with call and result paired."""
    agent = _make_agent()
    agent.llm_client.do_completion = AsyncMock(
        return_value=_make_completion_response("summary")
    )

    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "tc_1",
                    "function": {
                        "name": "run_command",
                        "arguments": '{"command": "ls /workspace"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "tc_1",
            "content": "README.md\nmain.py",
        },
    ]

    await agent.compress(messages)

    user_content = agent.llm_client.do_completion.call_args.kwargs["messages"][1][
        "content"
    ]
    assert "run_command" in user_content
    assert "README.md" in user_content
    assert user_content.count("README.md") == 1


@pytest.mark.asyncio
async def test_compress_keep_last_retains_recent_messages() -> None:
    """compress() with keep_last summarises everything, then appends the retained tail."""
    agent = _make_agent()
    agent.llm_client.do_completion = AsyncMock(
        return_value=_make_completion_response("full summary")
    )

    messages = [
        {"role": "user", "content": "old message 1"},
        {"role": "assistant", "content": "old reply 1"},
        {"role": "user", "content": "recent message"},
    ]

    await agent.compress(messages, keep_last=1)

    # Result: [summary_message, recent_message]
    assert len(messages) == 2
    assert "full summary" in messages[0]["content"]
    assert messages[1]["content"] == "recent message"

    # The whole conversation (including recent message) was sent to the LLM
    transcript = agent.llm_client.do_completion.call_args.kwargs["messages"][1][
        "content"
    ]
    assert "old message 1" in transcript
    assert "recent message" in transcript


@pytest.mark.asyncio
async def test_compress_noop_when_all_messages_retained() -> None:
    """compress() should not call LLM when keep_last >= len(messages)."""
    agent = _make_agent()
    agent.llm_client.do_completion = AsyncMock()

    messages = [
        {"role": "user", "content": "only message"},
    ]
    original = list(messages)

    await agent.compress(messages, keep_last=5)

    agent.llm_client.do_completion.assert_not_called()
    assert messages == original
