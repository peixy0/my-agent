from dataclasses import dataclass


@dataclass
class HeartbeatEvent:
    pass


@dataclass
class HumanInputEvent:
    session_key: str
    message: str
