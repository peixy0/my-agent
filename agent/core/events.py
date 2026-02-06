from dataclasses import dataclass


@dataclass
class HeartbeatEvent:
    pass


@dataclass
class HumanInputEvent:
    content: str
