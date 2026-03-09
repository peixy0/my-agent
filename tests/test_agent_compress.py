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
    )


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


@pytest.mark.asyncio
async def test_compress_includes_previous_summary_in_transcript() -> None:
    """compress() should prepend the previous summary so compression is incremental."""
    agent = _make_agent()
    agent.llm_client.do_completion = AsyncMock(
        return_value=_make_completion_response("new summary")
    )

    messages = [{"role": "user", "content": "follow-up question"}]
    prior = "Agent previously set up the database."

    await agent.compress(messages, previous_summary=prior)

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
    # Tool call and its result should appear together
    assert "run_command" in user_content
    assert "README.md" in user_content
    # The paired result should not appear again as a standalone TOOL RESULT
    assert user_content.count("README.md") == 1


@pytest.mark.asyncio
async def test_compress_no_llm_call_when_empty_after_keep() -> None:
    """compress() with an empty list never calls the LLM."""
    agent = _make_agent()
    agent.llm_client.do_completion = AsyncMock()

    await agent.compress([])

    agent.llm_client.do_completion.assert_not_called()
