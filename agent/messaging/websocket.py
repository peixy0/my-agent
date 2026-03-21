import logging
from typing import Any, override

from agent.core.messaging import Channel
from agent.llm.types import ToolContent
from agent.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class WebSocketChannel(Channel):
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

    @override
    def register_tools(self, registry: ToolRegistry) -> None:
        """Register WebSocket-supported tools: image path sending."""

        async def send_image(image_path: str) -> ToolContent:
            """
            Send an image file to the user. Image file size must be under 10 MiB.
            """
            try:
                await self.ws.send_json(
                    {"type": "image_path", "chat_id": self.chat_id, "path": image_path}
                )
                return ToolContent.from_dict(
                    "success",
                    {"message": f"Sent image {image_path} to user"},
                )
            except Exception as e:
                logger.warning(f"WebSocket image send failed for {self.chat_id}: {e}")
                return ToolContent.from_dict("error", {"message": str(e)})

        registry.register(
            send_image,
            {
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "Absolute path to the image file to send.",
                    }
                },
                "required": ["image_path"],
            },
        )
