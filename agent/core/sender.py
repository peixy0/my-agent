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
    async def send_file(self, file_path: str) -> None: ...

    @abstractmethod
    async def react(self, emoji: str) -> None: ...

    @abstractmethod
    async def start_thinking(self) -> None: ...

    @abstractmethod
    async def end_thinking(self) -> None: ...


class MessageSource(ABC):
    """Background task that receives inbound messages and queues events."""

    @abstractmethod
    async def run(self) -> None: ...


class NullSender(MessageSender):
    async def send(self, text: str) -> None:
        pass

    async def send_image(self, image_path: str) -> None:
        pass

    async def send_file(self, file_path: str) -> None:
        pass

    async def react(self, emoji: str) -> None:
        pass

    async def start_thinking(self) -> None:
        pass

    async def end_thinking(self) -> None:
        pass


class NullSource(MessageSource):
    async def run(self) -> None:
        pass
