import logging
from typing import Any, override

from agent.core.sender import MessageSender

logger = logging.getLogger(__name__)


class WebSocketSender(MessageSender):
    """Sends plain-text and image responses over a WebSocket connection."""

    def __init__(self, websocket: Any, chat_id: str, message_id: str = "") -> None:
        self.ws = websocket
        self.chat_id = chat_id

    @override
    async def send(self, text: str) -> None:
        try:
            await self.ws.send_json(
                {"type": "message", "chat_id": self.chat_id, "text": text}
            )
        except Exception as e:
            logger.warning(f"WebSocket send failed for {self.chat_id}: {e}")

    @override
    async def send_image(self, image_path: str) -> None:
        try:
            await self.ws.send_json(
                {"type": "image_path", "chat_id": self.chat_id, "path": image_path}
            )
        except Exception as e:
            logger.warning(f"WebSocket image send failed for {self.chat_id}: {e}")

    @override
    async def send_file(self, file_path: str) -> None:
        pass

    @override
    async def react(self, emoji: str) -> None:
        pass

    @override
    async def start_thinking(self) -> None:
        try:
            await self.ws.send_json({"type": "thinking_start", "chat_id": self.chat_id})
        except Exception as e:
            logger.warning(f"WebSocket thinking_start failed for {self.chat_id}: {e}")

    @override
    async def end_thinking(self) -> None:
        try:
            await self.ws.send_json({"type": "thinking_end", "chat_id": self.chat_id})
        except Exception as e:
            logger.warning(f"WebSocket thinking_end failed for {self.chat_id}: {e}")
