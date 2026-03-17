"""
Entry point for the autonomous agent.

Runs the Scheduler (event loop) and FastAPI server concurrently.
"""

import asyncio
import logging

try:
    import uvloop  # type: ignore[import]
except ImportError:  # pragma: no cover
    uvloop = None  # type: ignore[assignment]

from agent.core.settings import get_settings
from agent.engine.app import App
from agent.engine.scheduler import Scheduler

logger = logging.getLogger("agent")


async def main() -> None:
    """Entry point: runs Scheduler + FastAPI server concurrently."""
    logger_stream = logging.StreamHandler()
    logger_stream.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
    )
    logger.addHandler(logger_stream)
    logger.setLevel(logging.DEBUG)

    # Start dependent background tasks (messaging source, API server)
    app = App(get_settings())
    await app.run()

    # Create and run scheduler
    scheduler = Scheduler(app)
    logger.info("Starting scheduler...")
    await scheduler.run()


if __name__ == "__main__":
    if uvloop is not None:
        uvloop.run(main())
    else:
        asyncio.run(main())
