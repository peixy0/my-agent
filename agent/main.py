import asyncio
import logging
from datetime import datetime
from typing import Final

import aiocron

from agent.core.agent import Agent
from agent.core.event_logger import EventLogger
from agent.core.events import HeartbeatEvent, HumanInputEvent
from agent.core.messaging import messaging
from agent.core.settings import settings
from agent.llm.factory import LLMFactory
from agent.tools.command_executor import ensure_container_running

logger = logging.getLogger(__name__)
agent_queue: Final[asyncio.Queue] = asyncio.Queue()


@aiocron.crontab("0 8 * * *")
async def fetch_news():
    """Example of a scheduled task that could fetch news headlines."""
    await agent_queue.put(
        HumanInputEvent(content="Fetch latest tech news headlines and send them to me.")
    )


class Scheduler:
    """
    Event-driven scheduler that manages agent execution cycles.

    The scheduler uses an asyncio queue to process events like HeartbeatEvent,
    triggering agent wakeups at regular intervals.
    """

    agent: Final[Agent]
    event_logger: Final[EventLogger]
    running: bool
    queue: asyncio.Queue
    heartbeat_task: asyncio.Task[None] | None

    def __init__(self, agent: Agent, queue: asyncio.Queue, event_logger: EventLogger):
        self.agent = agent
        self.event_logger = event_logger
        self.running = True
        self.queue = queue
        self.heartbeat_task = None

    async def _ensure_container(self) -> bool:
        """Ensure the workspace container is running."""
        return await ensure_container_running(
            container_name=settings.container_name,
            runtime=settings.container_runtime,
            workspace_path=settings.workspace_dir,
        )

    async def _schedule_heartbeat(self) -> None:
        """Background task to emit heartbeat events."""
        logger.info(f"Sleeping for {settings.wake_interval_seconds} seconds")
        await asyncio.sleep(settings.wake_interval_seconds)
        await self.queue.put(HeartbeatEvent())

    async def _process_heartbeat(self) -> None:
        """Process heartbeat event."""
        self.agent.initialize_system_prompt()

        prompt = (
            "You are awake. "
            "Review your CONTEXT and TODO, then work on your tasks. "
            "Remember to update your journal and context files."
        )

        logger.info(f"Executing agent with prompt: {prompt}")
        await self.agent.run(prompt)
        logger.info("Wake cycle completed")

    async def _process_human_input(self, content: str) -> None:
        """Process human input event."""
        self.agent.initialize_system_prompt()

        prompt = (
            "You are awake. "
            "Process the following human input and update your TODO and CONTEXT as needed:\n\n"
            f"{content}"
        )

        logger.info(f"Executing agent with prompt: {prompt}")
        await self.agent.run(prompt)
        logger.info("Wake cycle completed")

    async def run(self) -> None:
        """
        Main loop: process events from the queue.
        """
        # Ensure container is running before starting
        if not await self._ensure_container():
            logger.error("Failed to start workspace container. Exiting.")
            return

        logger.info("Scheduler starting...")
        logger.info(f"Wake interval: {settings.wake_interval_seconds} seconds")

        await self.queue.put(HeartbeatEvent())

        while self.running:
            event = await self.queue.get()

            if not await self._ensure_container():
                logger.error("Container not available, skipping event")
                continue

            if isinstance(event, HeartbeatEvent):
                wake_time = datetime.now().astimezone().isoformat()
                logger.info(f"Wake cycle at {wake_time}")

                try:
                    await self._process_heartbeat()
                except Exception as e:
                    logger.error(f"Error during wake cycle: {e}", exc_info=True)

            if isinstance(event, HumanInputEvent):
                logger.info(f"Received human input: {event.content}")
                try:
                    await self._process_human_input(event.content)
                except Exception as e:
                    logger.error(f"Error processing human input: {e}", exc_info=True)

            if self.heartbeat_task:
                _ = self.heartbeat_task.cancel()
            self.heartbeat_task = asyncio.create_task(self._schedule_heartbeat())


async def main() -> None:
    """Entry point for autonomous runner."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    _ = asyncio.create_task(messaging.run())

    event_logger = EventLogger(
        log_file=settings.event_log_file,
        stream_url=settings.stream_api_url,
        stream_api_key=settings.stream_api_key,
    )

    llm_client = LLMFactory.create(
        url=settings.openai_base_url,
        model=settings.openai_model,
        api_key=settings.openai_api_key,
    )

    agent = Agent(llm_client, event_logger, settings)
    runner = Scheduler(agent, agent_queue, event_logger)
    await runner.run()


if __name__ == "__main__":
    asyncio.run(main())
