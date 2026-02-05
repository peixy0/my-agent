import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Final

import aiohttp

logger = logging.getLogger(__name__)


class EventLogger:
    """
    Logs agent events to a JSONL file and optionally streams to remote API.
    Only logs tool_use and llm_response events.
    """

    log_file: Final[Path]
    stream_url: Final[str | None]
    stream_api_key: Final[str | None]
    _lock: Final[asyncio.Lock]
    _session: aiohttp.ClientSession | None

    def __init__(
        self,
        log_file: str = "events.jsonl",
        stream_url: str | None = None,
        stream_api_key: str | None = None,
    ):
        self.log_file = Path(log_file)
        self.stream_url = stream_url
        self.stream_api_key = stream_api_key
        self._lock = asyncio.Lock()
        self._session = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def log_tool_use(
        self, tool_name: str, args: dict[str, Any], result: Any
    ) -> None:
        """Log a tool invocation event."""
        event = {
            "timestamp": datetime.now().astimezone().isoformat(),
            "type": "tool_use",
            "data": {
                "tool": tool_name,
                "args": args,
                "result": result,
            },
        }
        await self._write_event(event)

    async def log_llm_response(self, content: str) -> None:
        """Log an LLM response event."""
        event = {
            "timestamp": datetime.now().astimezone().isoformat(),
            "type": "llm_response",
            "data": {
                "content": content,
            },
        }
        await self._write_event(event)

    async def _post_to_api(self, event: dict[str, Any]) -> None:
        """Post event to remote API."""
        if not self.stream_url:
            return
        try:
            session = await self._get_session()

            headers = {
                "x-api-key": self.stream_api_key or "",
                "Content-Type": "application/json",
            }

            async with session.post(
                self.stream_url,
                json=event,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=20),
            ) as response:
                if response.status >= 400:
                    body = await response.text()
                    logger.warning(
                        f"Failed to post event to API: {response.status} - {body}"
                    )
        except asyncio.TimeoutError:
            logger.warning("Timeout posting event to API")
        except Exception as e:
            logger.warning(f"Error posting event to API: {e}")

    async def _write_event(self, event: dict[str, Any]) -> None:
        """Write an event to the log file and post to API."""
        async with self._lock:
            # Write to local file
            try:
                # Ensure parent directory exists
                self.log_file.parent.mkdir(parents=True, exist_ok=True)

                # Append event as JSON line
                with self.log_file.open("a") as f:
                    _ = f.write(json.dumps(event) + "\n")

            except Exception as e:
                logger.error(f"Failed to write event to file: {e}")

            # Post to remote API asynchronously (don't block on this)
            if self.stream_url:
                _ = asyncio.create_task(self._post_to_api(event))
