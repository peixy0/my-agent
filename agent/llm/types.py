from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(slots=True)
class ToolJsonResult:
    """Plain JSON-serialized content returned by a tool."""

    result: Any


@dataclass(slots=True)
class ToolImageResult:
    """Multimodal image content blocks returned by a tool."""

    blocks: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class ToolContent:
    """
    Typed container for tool call results sent back to the LLM.

    status="success" — the tool completed without errors.
    status="error"   — the tool encountered an error; result is a ToolJsonResult
                       whose text carries the error JSON.
    result is ToolJsonResult for JSON text payloads and ToolImageResult for
    vision content blocks.
    """

    status: Literal["success", "error"]
    result: ToolJsonResult | ToolImageResult

    def to_lm_content(self) -> str | list[dict[str, Any]]:
        """Return the value expected by the OpenAI messages API."""
        if isinstance(self.result, ToolImageResult):
            return self.result.blocks
        return json.dumps(
            {"status": self.status, "result": self.result.result}, ensure_ascii=False
        )

    @staticmethod
    def from_dict(
        status: Literal["success", "error"], data: dict[str, Any]
    ) -> ToolContent:
        """Wrap a plain tool-result dict as a text ToolContent."""
        return ToolContent(
            status=status,
            result=ToolJsonResult(data),
        )

    @staticmethod
    def from_blocks(blocks: list[dict[str, Any]]) -> ToolContent:
        """Wrap multimodal content blocks as an image ToolContent."""
        return ToolContent(status="success", result=ToolImageResult(blocks=blocks))


@dataclass(slots=True)
class ToolCallFunctionView:
    name: str
    arguments: str


@dataclass(slots=True)
class ToolCallView:
    id: str
    function: ToolCallFunctionView
    type: str = "function"


@dataclass(slots=True)
class MessageView:
    role: str
    content: str | None
    tool_calls: list[ToolCallView]

    def model_dump(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in self.tool_calls
            ]
        return payload


@dataclass(slots=True)
class ChoiceView:
    message: MessageView
    finish_reason: str
    index: int = 0


@dataclass(slots=True)
class UsageView:
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass(slots=True)
class CompletionResponseView:
    choices: list[ChoiceView]
    usage: UsageView
    model: str
