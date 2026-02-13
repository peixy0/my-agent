import asyncio
from dataclasses import dataclass


@dataclass
class HeartbeatEvent:
    pass


@dataclass
class HumanInputEvent:
    conversation: list[dict[str, str]]
    reply_fut: asyncio.Future[str]
