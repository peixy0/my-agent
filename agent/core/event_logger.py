import asyncio
import logging
from typing import Any, Final

import aiohttp

from agent.core.settings import settings

logger = logging.getLogger(__name__)


class EventLogger:
    """
    Logs agent events to remote API.
    """

    event_url: Final[str]
    _headers: Final[dict[str, str]]
    _queue: Final[asyncio.Queue[dict[str, Any]]]
    _worker_task: asyncio.Task[None] | None

    def __init__(
        self,
        event_url: str,
        stream_api_key: str,
    ):
        self.event_url = event_url
        self._headers = {
            "x-api-key": stream_api_key,
            "Content-Type": "application/json",
        }
        self._queue = asyncio.Queue()
        self._worker_task = None

    async def run(self):
        self._worker_task = asyncio.create_task(self._worker())

    async def log_tool_use(
        self, tool_name: str, args: dict[str, Any], result: Any
    ) -> None:
        """Log a tool invocation event."""
        event = {
            "type": "tool_use",
            "data": {
                "tool": tool_name,
                "args": args,
                "result": result,
            },
        }
        await self._queue.put(event)

    async def log_agent_response(self, content: str) -> None:
        """Log an LLM response event."""
        event = {
            "type": "agent_response",
            "data": {
                "content": content,
            },
        }
        await self._queue.put(event)

    async def _worker(self) -> None:
        """Background worker to process events from the queue."""
        logger.info("EventLogger running...")
        async with aiohttp.ClientSession() as session:
            while True:
                event = await self._queue.get()
                if event is None:
                    self._queue.task_done()
                    break
                event_batch = [event]
                while True:
                    try:
                        event = self._queue.get_nowait()
                        event_batch.append(event)
                    except asyncio.QueueEmpty:
                        break

                try:
                    await self._process_event(session, event_batch)
                except Exception as e:
                    logger.error(f"Error processing event: {e}")
                finally:
                    self._queue.task_done()

    async def _process_event(
        self, session: aiohttp.ClientSession, event_batch: list[dict[str, Any]]
    ) -> None:
        """Post event to remote API."""
        if not self.event_url:
            return
        for event in event_batch:
            try:
                async with session.post(
                    f"{self.event_url}/bot_internal",
                    json=event,
                    headers=self._headers,
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as response:
                    _ = await response.read()
                if event["type"] == "agent_response":
                    async with session.post(
                        f"{self.event_url}/bot",
                        json=event,
                        headers=self._headers,
                        timeout=aiohttp.ClientTimeout(total=60),
                    ) as response:
                        _ = await response.read()
            except Exception as e:
                logger.error(f"Error posting event to API: {e}")


event_logger = EventLogger(settings.event_api_url, settings.stream_api_key)
