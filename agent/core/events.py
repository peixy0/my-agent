from dataclasses import dataclass, field

from agent.core.sender import MessageSender


@dataclass
class HeartbeatEvent:
    chat_id: str
    interval_seconds: int
    sender: MessageSender


@dataclass
class NewSessionEvent:
    chat_id: str
    sender: MessageSender


@dataclass
class TextInputEvent:
    chat_id: str
    message_id: str
    message: str
    sender: MessageSender


@dataclass
class ImageInputEvent:
    chat_id: str
    message_id: str
    image_data: bytes
    sender: MessageSender
    mime_type: str = field(default="image/jpeg")
    message: str = field(default="")


@dataclass
class DropSessionEvent:
    chat_id: str
