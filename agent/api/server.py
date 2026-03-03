"""
FastAPI HTTP server for accepting human input.

The server shares an asyncio.Queue with the Scheduler so that
HTTP requests are translated into events the agent processes.

/api/bot  — WebSocket endpoint: each connection gets its own chat_id session.
            Inbound text:  {"type": "text",  "message": "...", "message_id": "..."}
                           {"message": "...", "message_id": "..."}  (type omitted = text)
            Inbound image: {"type": "image", "data": "<base64>", "mime_type": "image/jpeg",
                            "message_id": "..."}
            Outbound:      {"type": "connected",  "chat_id": "..."}
                           {"type": "message",    "chat_id": "...", "text": "..."}
                           {"type": "image_path", "chat_id": "...", "path": "..."}
"""

import asyncio
import base64
import logging
import uuid
from abc import ABC, abstractmethod
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from agent.core.events import ImageInputEvent, TextInputEvent
from agent.core.messaging import MessagingBus, WebSocketMessaging
from agent.core.settings import Settings

logger = logging.getLogger(__name__)


class HealthResponse:
    """Response body for the health check endpoint."""

    status: str = "ok"


class ApiService(ABC):
    """Abstract base for API service implementations."""

    @abstractmethod
    async def run(self) -> None: ...


class NullApiService(ApiService):
    """No-op API service when API is disabled."""

    async def run(self) -> None:
        pass


class UvicornApiService(ApiService):
    """Uvicorn-based FastAPI service."""

    def __init__(self, app: FastAPI, host: str, port: int):
        self._app = app
        self._host = host
        self._port = port

    async def run(self) -> None:
        config = uvicorn.Config(
            self._app,
            host=self._host,
            port=self._port,
            log_level="info",
        )
        server = uvicorn.Server(config)
        logger.info(f"Starting API server on {self._host}:{self._port}")
        await server.serve()


def create_api(event_queue: asyncio.Queue, messaging_bus: MessagingBus) -> FastAPI:
    """
    Factory that creates and returns the FastAPI application.

    Args:
        event_queue:   Shared queue with the Scheduler; events are placed here.
        messaging_bus: MessagingBus used to register per-session WebSocket backends
                       so the agent can push replies back to the correct client.
    """
    app = FastAPI(
        title="sys-agent API",
        description="HTTP interface for the autonomous LLM agent",
    )

    _chat_html: str | None = None

    def _load_chat_html() -> str:
        nonlocal _chat_html
        if _chat_html is None:
            html_path = (
                Path(__file__).parent.parent.parent / "assets" / "test_chat.html"
            )
            _chat_html = html_path.read_text(encoding="utf-8")
        return _chat_html

    @app.get("/", response_class=HTMLResponse)
    async def chat_ui() -> HTMLResponse:
        """Serve the single-file WebSocket chat test UI."""
        return HTMLResponse(_load_chat_html())

    @app.websocket("/api/bot")
    async def ws_bot(websocket: WebSocket) -> None:
        """WebSocket endpoint: one session per connection."""
        await websocket.accept()
        chat_id = f"ws-{uuid.uuid4().hex}"
        ws_messaging = WebSocketMessaging(websocket, chat_id)
        messaging_bus.register(chat_id, ws_messaging)
        logger.info(f"WebSocket connected: chat_id={chat_id}")

        try:
            await websocket.send_json({"type": "connected", "chat_id": chat_id})
            while True:
                data = await websocket.receive_json()
                msg_type: str = data.get("type", "text")
                message_id: str = data.get("message_id") or uuid.uuid4().hex
                message: str = data.get("message", "")

                if msg_type == "image":
                    raw_data: str = data.get("data", "")
                    mime_type: str = data.get("mime_type", "image/jpeg")
                    if not raw_data:
                        continue
                    image_bytes = base64.b64decode(raw_data)
                    logger.debug(
                        f"[{chat_id}] received image (id={message_id}, "
                        f"mime={mime_type}, size={len(image_bytes)}B)"
                    )
                    await event_queue.put(
                        ImageInputEvent(
                            chat_id=chat_id,
                            message_id=message_id,
                            image_data=image_bytes,
                            mime_type=mime_type,
                            message=message,
                        )
                    )
                else:
                    logger.debug(
                        f"[{chat_id}] received message (id={message_id}): {message[:100]}"
                    )
                    await event_queue.put(
                        TextInputEvent(
                            chat_id=chat_id,
                            message_id=message_id,
                            message=message,
                        )
                    )
        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected: chat_id={chat_id}")
        except Exception as e:
            logger.error(f"WebSocket error [{chat_id}]: {e}", exc_info=True)
        finally:
            messaging_bus.unregister(chat_id)

    @app.get("/api/health")
    async def health_check() -> dict:
        """Health check endpoint."""
        return {"status": "ok"}

    return app


def create_api_service(
    settings: Settings,
    event_queue: asyncio.Queue,
    messaging_bus: MessagingBus,
) -> ApiService:
    """Create the appropriate API service based on settings."""
    if settings.api_enabled:
        app = create_api(event_queue, messaging_bus)
        return UvicornApiService(app, settings.api_host, settings.api_port)
    return NullApiService()
