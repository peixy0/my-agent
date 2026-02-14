"""
FastAPI HTTP server for accepting human input.

The server shares an asyncio.Queue with the Scheduler so that
HTTP requests are translated into events the agent processes.
"""

import asyncio
import logging
from abc import ABC, abstractmethod

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from agent.core.events import HumanInputEvent
from agent.core.settings import Settings

logger = logging.getLogger(__name__)


class HumanInputRequest(BaseModel):
    """Request body for the human input endpoint."""

    session_id: str
    message_id: str
    message: str


class HumanInputResponse(BaseModel):
    """Response body confirming the input was queued."""

    status: str = "queued"


class HealthResponse(BaseModel):
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


def create_api(event_queue: asyncio.Queue) -> FastAPI:
    """
    Factory that creates and returns the FastAPI application.

    Args:
        event_queue: The shared queue with the Scheduler.
                     Human input events are placed here for processing.
    """
    app = FastAPI(
        title="sys-agent API",
        description="HTTP interface for the autonomous LLM agent",
    )

    @app.post("/api/bot", response_model=HumanInputResponse)
    async def submit_input(request: HumanInputRequest) -> HumanInputResponse:
        """Accept human input and queue it for the agent."""
        logger.info(f"Received human input: {request.message[:100]}...")
        await event_queue.put(
            HumanInputEvent(
                session_id=request.session_id,
                message_id=request.message_id,
                message=request.message,
            )
        )
        return HumanInputResponse()

    @app.get("/api/health", response_model=HealthResponse)
    async def health_check() -> HealthResponse:
        """Health check endpoint."""
        return HealthResponse()

    return app


def create_api_service(settings: Settings, event_queue: asyncio.Queue) -> ApiService:
    """Create the appropriate API service based on settings."""
    if settings.api_enabled:
        app = create_api(event_queue)
        return UvicornApiService(app, settings.api_host, settings.api_port)
    return NullApiService()
