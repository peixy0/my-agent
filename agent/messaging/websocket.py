import logging
from typing import Any

from agent.core.sender import MessageSender

logger = logging.getLogger(__name__)


class WebSocketSender(MessageSender):
    """Sends plain-text and image responses over a WebSocket connection."""

    def __init__(self, websocket: Any, chat_id: str, message_id: str = "") -> None:
        self._ws = websocket
        self._chat_id = chat_id
        self._message_id = message_id

    async def send(self, text: str) -> None:
        try:
            await self._ws.send_json(
                {"type": "message", "chat_id": self._chat_id, "text": text}
            )
        except Exception as e:
            logger.warning("WebSocket send failed for %s: %s", self._chat_id, e)

    async def send_image(self, image_path: str) -> None:
        try:
            await self._ws.send_json(
                {"type": "image_path", "chat_id": self._chat_id, "path": image_path}
            )
        except Exception as e:
            logger.warning("WebSocket image send failed for %s: %s", self._chat_id, e)

    async def send_file(self, file_path: str) -> None:
        pass

    async def react(self, emoji: str) -> None:
        pass

    async def start_thinking(self) -> None:
        try:
            await self._ws.send_json(
                {"type": "thinking_start", "chat_id": self._chat_id}
            )
        except Exception as e:
            logger.warning(
                "WebSocket thinking_start failed for %s: %s", self._chat_id, e
            )

    async def end_thinking(self) -> None:
        try:
            await self._ws.send_json({"type": "thinking_end", "chat_id": self._chat_id})
        except Exception as e:
            logger.warning("WebSocket thinking_end failed for %s: %s", self._chat_id, e)
