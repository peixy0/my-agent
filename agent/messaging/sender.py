from abc import ABC, abstractmethod


class MessageSender(ABC):
    """Bound reply handle for a single conversation turn.

    Constructed with a ``chat_id`` and ``message_id`` so callers never pass
    routing information at call-time — the sender already knows where to send
    and which message to react to.
    """

    @abstractmethod
    async def send(self, text: str) -> None: ...

    @abstractmethod
    async def send_image(self, image_path: str) -> None: ...

    @abstractmethod
    async def react(self, emoji: str) -> None:
        """React to the bound message_id with *emoji*."""
        ...


class MessageSource(ABC):
    """Background task that receives inbound messages and queues events."""

    @abstractmethod
    async def run(self) -> None: ...


class NullSender(MessageSender):
    async def send(self, text: str) -> None:
        pass

    async def send_image(self, image_path: str) -> None:
        pass

    async def react(self, emoji: str) -> None:
        pass


class NullSource(MessageSource):
    async def run(self) -> None:
        pass
