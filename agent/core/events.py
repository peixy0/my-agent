from dataclasses import dataclass, field

from agent.core.messaging import Channel


@dataclass
class HeartbeatEvent:
    chat_id: str
    interval_seconds: int
    sender: Channel


@dataclass
class CronEvent:
    chat_id: str
    task_name: str
    prompt: str
    sender: Channel


@dataclass
class NewSessionEvent:
    chat_id: str
    sender: Channel


@dataclass
class TextInputEvent:
    chat_id: str
    message_id: str
    message: str
    sender: Channel


@dataclass
class ImageInputEvent:
    chat_id: str
    message_id: str
    image_data: bytes
    sender: Channel
    mime_type: str = field(default="image/jpeg")
    message: str = field(default="")


@dataclass
class DropSessionEvent:
    chat_id: str


AgentEvent = TextInputEvent | ImageInputEvent | DropSessionEvent
WorkerEvent = (
    HeartbeatEvent | CronEvent | NewSessionEvent | TextInputEvent | ImageInputEvent
)
