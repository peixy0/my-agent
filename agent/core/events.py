import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class OutputType(str, Enum):
    START = "start"
    THOUGHT = "thought"
    TOOL_RESULT = "tool_result"
    APPROVAL_REQUEST = "approval_request"
    RESPONSE = "response"
    ERROR = "error"


class InputType(str, Enum):
    APPROVAL_RESULT = "approval_result"
    USER_INPUT = "user_input"


@dataclass
class OutputEvent:
    type: OutputType
    data: Any = field(default_factory=dict)


@dataclass
class InputEvent:
    type: InputType
    data: Any = field(default_factory=dict)


class AgentEvents:
    def __init__(self):
        self.input: asyncio.Queue[InputEvent] = asyncio.Queue()
        self.output: asyncio.Queue[OutputEvent] = asyncio.Queue()

    async def publish(self, event_type: OutputType, data: Any = None):
        """Publish an event to the output queue."""
        if data is None:
            data = {}
        await self.output.put(OutputEvent(type=event_type, data=data))

    async def request_input(self, event_type: OutputType, data: Any = None) -> Any:
        """
        Publish an event to request input and wait for a response from the input queue.
        """
        await self.publish(event_type, data)
        
        # Wait for the next event in the input queue
        response_event = await self.input.get()
        return response_event.data
