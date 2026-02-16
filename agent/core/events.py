from dataclasses import dataclass


@dataclass
class HeartbeatEvent:
    pass


@dataclass
class HumanInputEvent:
    chat_id: str
    message_id: str
    message: str
