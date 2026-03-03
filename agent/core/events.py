from dataclasses import dataclass, field


@dataclass
class HeartbeatEvent:
    pass


@dataclass
class TextInputEvent:
    chat_id: str
    message_id: str
    message: str


@dataclass
class ImageInputEvent:
    chat_id: str
    message_id: str
    image_data: bytes
    mime_type: str = field(default="image/jpeg")
    message: str = field(default="")
