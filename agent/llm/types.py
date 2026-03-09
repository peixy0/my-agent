from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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
