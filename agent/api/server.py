"""
FastAPI HTTP server for accepting human input.

The server shares an asyncio.Queue with the Scheduler so that
HTTP requests are translated into events the agent processes.
"""

import asyncio
import logging

from fastapi import FastAPI
from pydantic import BaseModel

from agent.core.events import HumanInputEvent

logger = logging.getLogger(__name__)


class HumanInputRequest(BaseModel):
    """Request body for the human input endpoint."""

    message: str


class HumanInputResponse(BaseModel):
    """Response body confirming the input was queued."""

    status: str = "queued"


class HealthResponse(BaseModel):
    """Response body for the health check endpoint."""

    status: str = "ok"


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

    @app.post("/api/input", response_model=HumanInputResponse)
    async def submit_input(request: HumanInputRequest) -> HumanInputResponse:
        """Accept human input and queue it for the agent."""
        logger.info(f"Received human input: {request.message[:100]}...")
        await event_queue.put(HumanInputEvent(content=request.message))
        return HumanInputResponse()

    @app.get("/api/health", response_model=HealthResponse)
    async def health_check() -> HealthResponse:
        """Health check endpoint."""
        return HealthResponse()

    return app
