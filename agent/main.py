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
        if event.message == "/new":
            self.sessions[event.session_key] = []
            await self.app.messaging.send_message(
                event.session_key, "New session started"
            )
            return

        prompt = self.app.prompt_builder.build(session_key=event.session_key)
        self.app.agent.set_system_prompt(prompt)

        logger.info(f"Processing human input: {event.message[:100]}...")
        conversation = self.sessions.get(event.session_key, [])
        conversation.append({"role": "user", "content": event.message})
        response = await self.app.agent.run(conversation.copy())
        await self.app.event_logger.log_agent_response(
            f"HUMAN INPUT RESPONSE:\n\n{response}"
        )
        conversation.append({"role": "assistant", "content": response})
        self.sessions[event.session_key] = conversation
        await self.app.messaging.send_message(event.session_key, response)
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
    app = AppWithDependencies()
    await app.run()

    # Create and run scheduler
    scheduler = Scheduler(app)
    logger.info("Starting scheduler...")
    await scheduler.run()


if __name__ == "__main__":
    asyncio.run(main())
