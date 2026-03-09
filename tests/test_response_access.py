from __future__ import annotations

from agent.llm.types import (
    ChoiceView,
    CompletionResponseView,
    MessageView,
    ToolCallFunctionView,
    ToolCallView,
    UsageView,
)


def test_message_view_model_dump_without_tool_calls():
    message = MessageView(role="assistant", content="hello", tool_calls=[])
    dumped = message.model_dump()
    assert dumped == {"role": "assistant", "content": "hello"}


def test_message_view_model_dump_with_tool_calls():
    message = MessageView(
        role="assistant",
        content=None,
        tool_calls=[
            ToolCallView(
                id="tc_1",
                type="function",
                function=ToolCallFunctionView(name="read_file", arguments="{}"),
            )
        ],
    )
    dumped = message.model_dump()
    assert dumped["role"] == "assistant"
    assert dumped["content"] is None
    assert dumped["tool_calls"][0]["id"] == "tc_1"
    assert dumped["tool_calls"][0]["function"]["name"] == "read_file"


def test_completion_response_view_defaults():
    response = CompletionResponseView(
        choices=[
            ChoiceView(
                message=MessageView(role="assistant", content="hi", tool_calls=[]),
                finish_reason="stop",
            )
        ],
        usage=UsageView(),
    )
    assert response.choices[0].message.content == "hi"
    assert response.usage.total_tokens == 0
