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
from agent.core.settings import get_settings
from agent.llm.agent import HeartbeatOrchestrator, HumanInputOrchestrator

logger = logging.getLogger("agent")

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
        self.sessions: dict[str, list[dict[str, str]]] = {}
        self.messages: dict[str, set[str]] = {}

    async def _schedule_heartbeat(self) -> None:
        logger.info(f"Sleeping for {self.app.settings.wake_interval_seconds} seconds")
        await asyncio.sleep(self.app.settings.wake_interval_seconds)
        await self.app.event_queue.put(HeartbeatEvent())

    async def _process_heartbeat(self) -> None:
        prompt = self.app.prompt_builder.build()
        self.app.agent.set_system_prompt(prompt)

        logger.info("Executing agent heartbeat cycle")
        messages = [{"role": "user", "content": "SYSTEM EVENT: Heartbeat"}]
        orchestrator = HeartbeatOrchestrator(
            self.app.model_name,
            self.app.tool_registry,
            self.app.messaging,
            self.app.event_logger,
        )
        await self.app.agent.run(messages, orchestrator)
        logger.info("Heartbeat cycle completed")

    async def _process_human_input(self, event: HumanInputEvent) -> None:
        if event.message == "/new":
            self.sessions[event.session_id] = []
            self.messages[event.session_id] = set()
            await self.app.messaging.send_message(
                event.session_id, "New session started"
            )
            return

        messages = self.messages.get(event.session_id, set())
        if event.message_id in messages:
            logger.debug(f"Ignoring duplicated message {event.message_id}")
            return
        messages.add(event.message_id)
        self.messages[event.session_id] = messages

        conversation = self.sessions.get(event.session_id, [])
        conversation.append({"role": "user", "content": event.message})
        logger.info(f"Processing human input: {event.message[:100]}...")

        prompt = self.app.prompt_builder.build(session_id=event.session_id)
        self.app.agent.set_system_prompt(prompt)

        orchestrator = HumanInputOrchestrator(
            event.session_id,
            event.message_id,
            self.app.model_name,
            self.app.tool_registry,
            self.app.messaging,
            self.app.event_logger,
        )
        await self.app.agent.run(conversation, orchestrator)
        self.sessions[event.session_id] = conversation
        logger.info("Human input processing completed")

    async def run(self) -> None:
        logger.info("Scheduler starting...")
        logger.info(f"Wake interval: {self.app.settings.wake_interval_seconds} seconds")

        # Trigger initial heartbeat
        # await self.app.event_queue.put(HeartbeatEvent())

        while self.running:
            event = await self.app.event_queue.get()
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
                await self.app.messaging.notify(f"Error during event processing: {e}")

            self.app.event_queue.task_done()
            # Re-schedule next heartbeat
            if self.heartbeat_task:
                _ = self.heartbeat_task.cancel()
            self.heartbeat_task = asyncio.create_task(self._schedule_heartbeat())


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

    # Start dependent background tasks (event logger, messaging, API server)
    app = AppWithDependencies(get_settings())
    await app.run()

    # Create and run scheduler
    scheduler = Scheduler(app)
    logger.info("Starting scheduler...")
    await scheduler.run()


if __name__ == "__main__":
    asyncio.run(main())
