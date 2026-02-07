import asyncio
import json
import logging
from datetime import datetime
from typing import Final

from agent.core.event_logger import event_logger
from agent.core.events import HeartbeatEvent
from agent.core.settings import settings
from agent.llm.agent import Agent
from agent.llm.factory import LLMFactory
from agent.tools.command_executor import ensure_container_running

logger = logging.getLogger(__name__)

agent_queue: asyncio.Queue = asyncio.Queue()


async def schedule_heartbeat():
    """Scheduled heartbeat."""
    await agent_queue.put(HeartbeatEvent())


class Scheduler:
    """
    Event-driven scheduler that manages agent execution cycles.

    The scheduler uses an asyncio queue to process events like HeartbeatEvent,
    triggering agent wakeups at regular intervals.
    """

    agent: Final[Agent]
    running: bool
    queue: asyncio.Queue
    heartbeat_task: asyncio.Task[None] | None

    def __init__(self, agent: Agent, queue: asyncio.Queue):
        self.agent = agent
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

        response_schema = {
            "type": "object",
            "properties": {
                "report_decision": {
                    "type": "boolean",
                    "description": "Whether to send a report to the human user based on the reporting criteria.",
                },
                "report_message": {
                    "type": "string",
                    "description": "The message to report to the user (if report_decision is true) or a summary of activity (if false).",
                },
            },
            "required": ["report_decision", "report_message"],
        }
        prompt = f"You are awake. Please respond with a JSON object matching this schema: {json.dumps(response_schema)}"

        logger.info(f"Executing agent with prompt: {prompt}")
        response = await self.agent.run(prompt, response_schema=response_schema)
        await event_logger.log_agent_response(
            f"REPORT DECISION: {response['report_decision']}\n\n{response['report_message']}"
        )
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

        await schedule_heartbeat()

        while self.running:
            event = await self.queue.get()

            if not await self._ensure_container():
                logger.error("Container not available, skipping event")
                continue

            try:
                if isinstance(event, HeartbeatEvent):
                    wake_time = datetime.now().astimezone().isoformat()
                    logger.info(f"Wake cycle at {wake_time}")
                    await self._process_heartbeat()

            except Exception as e:
                logger.error(f"Error during event processing: {e}", exc_info=True)

            if self.heartbeat_task:
                _ = self.heartbeat_task.cancel()
            self.heartbeat_task = asyncio.create_task(self._schedule_heartbeat())


async def main() -> None:
    """Entry point for autonomous runner."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    _ = asyncio.create_task(event_logger.run())

    llm_client = LLMFactory.create(
        url=settings.openai_base_url,
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        event_logger=event_logger,
    )

    agent = Agent(llm_client, settings)
    runner = Scheduler(agent, agent_queue)
    await runner.run()


if __name__ == "__main__":
    asyncio.run(main())
