"""
Entry point for the autonomous agent.

Runs the Scheduler (event loop) and FastAPI server concurrently.
"""

import asyncio
import logging
from datetime import datetime
from typing import Final

from agent.app import AppWithDependencies
from agent.core.events import HeartbeatEvent, HumanInputEvent
from agent.tools.command_executor import ensure_container_running

logger = logging.getLogger(__name__)

AgentEvent = HeartbeatEvent | HumanInputEvent


class Scheduler:
    """
    Event-driven scheduler that manages agent execution cycles.

    Processes HeartbeatEvents (periodic) and HumanInputEvents (from API).
    """

    app: Final[AppWithDependencies]
    running: bool
    heartbeat_task: asyncio.Task[None] | None

    def __init__(self, app: AppWithDependencies):
        self.app = app
        self.running = True
        self.heartbeat_task = None
        self.conversation = []

    async def _ensure_container(self) -> bool:
        return await ensure_container_running(
            container_name=self.app.settings.container_name,
            runtime=self.app.settings.container_runtime,
            workspace_path=self.app.settings.workspace_dir,
        )

    async def _schedule_heartbeat(self) -> None:
        logger.info(f"Sleeping for {self.app.settings.wake_interval_seconds} seconds")
        await asyncio.sleep(self.app.settings.wake_interval_seconds)
        await self.app.event_queue.put(HeartbeatEvent())

    async def _process_heartbeat(self) -> None:
        prompt = self.app.prompt_builder.build()
        self.app.agent.set_system_prompt(prompt)

        logger.info("Executing agent heartbeat cycle")
        messages = [{"role": "user", "content": "SYSTEM EVENT: Heartbeat"}]
        response = await self.app.agent.run(messages)
        response = response.strip()
        await self.app.event_logger.log_agent_response(
            f"HEARTBEAT RESPONSE:\n\n{response}"
        )
        if not response.endswith("NO_REPORT"):
            await self.app.messaging.notify(response)
        logger.info("Heartbeat cycle completed")

    async def _process_human_input(self, event: HumanInputEvent) -> None:
        prompt = self.app.prompt_builder.build()
        self.app.agent.set_system_prompt(prompt)

        logger.info(
            f"Processing human input: {event.conversation[-1]['content'][:100]}..."
        )
        response = await self.app.agent.run(event.conversation)
        await self.app.event_logger.log_agent_response(
            f"HUMAN INPUT RESPONSE:\n\n{response}"
        )
        event.reply_fut.set_result(response)
        logger.info("Human input processing completed")

    async def run(self) -> None:
        if not await self._ensure_container():
            logger.error("Failed to start workspace container. Exiting.")
            return

        logger.info("Scheduler starting...")
        logger.info(f"Wake interval: {self.app.settings.wake_interval_seconds} seconds")

        # Trigger initial heartbeat
        # await self.app.event_queue.put(HeartbeatEvent())

        while self.running:
            event = await self.app.event_queue.get()

            if not await self._ensure_container():
                logger.error("Container not available, skipping event")
                continue

            try:
                if isinstance(event, HeartbeatEvent):
                    wake_time = datetime.now().astimezone().isoformat()
                    logger.info(f"Wake cycle at {wake_time}")
                    await self._process_heartbeat()
                elif isinstance(event, HumanInputEvent):
                    await self._process_human_input(event)
                else:
                    logger.warning(f"Unknown event type: {type(event)}")
            except Exception as e:
                logger.error(f"Error during event processing: {e}", exc_info=True)

            # Re-schedule next heartbeat
            if self.heartbeat_task:
                _ = self.heartbeat_task.cancel()
            self.heartbeat_task = asyncio.create_task(self._schedule_heartbeat())


async def main() -> None:
    """Entry point: runs Scheduler + FastAPI server concurrently."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Start dependent background tasks (event logger, messaging, API server)
    app = AppWithDependencies()
    await app.run()

    # Create and run scheduler
    scheduler = Scheduler(app)
    logger.info("Starting scheduler...")
    await scheduler.run()


if __name__ == "__main__":
    asyncio.run(main())
