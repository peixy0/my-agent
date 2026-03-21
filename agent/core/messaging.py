from abc import ABC, abstractmethod

from agent.tools.registry import ToolRegistry


class Channel(ABC):
    """Bound reply handle for a single conversation turn.

    Constructed with a ``chat_id`` and ``message_id`` so callers never pass
    routing information at call-time — the channel already knows where to send
    and which message to react to.

    Feature-specific capabilities (reactions, image/file sending) are exposed
    as agent tools rather than abstract methods. Override ``register_tools`` to
    add only the tools your backend actually supports.
    """

    @abstractmethod
    async def send(self, text: str) -> None: ...

    @abstractmethod
    async def start_thinking(self) -> None: ...

    @abstractmethod
    async def end_thinking(self) -> None: ...

    @abstractmethod
    def register_tools(self, registry: ToolRegistry) -> None: ...


class Gateway(ABC):
    """Background task that receives inbound messages and queues events."""

    @abstractmethod
    async def run(self) -> None: ...
