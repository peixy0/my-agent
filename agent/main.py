import asyncio
import logging
import signal
from datetime import datetime
from pathlib import Path
from types import FrameType
from typing import Final

from agent.core.agent import Agent
from agent.core.events import AgentEvents
from agent.core.event_logger import EventLogger
from agent.core.settings import settings
from agent.llm.factory import LLMFactory
from agent.tools.command_executor import ensure_container_running


logger = logging.getLogger(__name__)


class AutonomousRunner:
    """
    Long-running autonomous agent that wakes up at regular intervals.
    
    The runner ensures the workspace container is available before
    each wake cycle and handles graceful shutdown.
    """

    agent: Final[Agent]
    event_logger: Final[EventLogger]
    running: bool

    def __init__(self, agent: Agent, event_logger: EventLogger):
        self.agent = agent
        self.event_logger = event_logger
        self.running = True

    def _setup_signal_handlers(self) -> None:
        """Set up graceful shutdown on SIGTERM/SIGINT."""

        def signal_handler(signum: int, frame: FrameType | None) -> None:
            _ = frame
            logger.info(f"Received signal {signum}, shutting down gracefully...")
            self.running = False

        _ = signal.signal(signal.SIGTERM, signal_handler)
        _ = signal.signal(signal.SIGINT, signal_handler)

    async def _ensure_container(self) -> bool:
        """Ensure the workspace container is running."""
        return await ensure_container_running(
            container_name=settings.container_name,
            runtime=settings.container_runtime,
            workspace_path=settings.workspace_dir,
        )

    async def run(self) -> None:
        """Main loop: wake up, execute agent, sleep, repeat."""
        self._setup_signal_handlers()

        # Load wake count from file
        wake_count_file = Path(settings.wake_count_file)
        wake_count = 1
        if wake_count_file.exists():
            try:
                wake_count = int(wake_count_file.read_text().strip())
                logger.info(f"Loaded wake count: {wake_count}")
            except (ValueError, Exception) as e:
                logger.warning(f"Failed to read .wake_count: {e}")

        # Ensure container is running before starting
        if not await self._ensure_container():
            logger.error("Failed to start workspace container. Exiting.")
            return

        logger.info("Autonomous agent starting...")
        logger.info(f"Wake interval: {settings.wake_interval_seconds} seconds")
        logger.info(f"Container: {settings.container_name}")

        while self.running:
            wake_time = datetime.now().astimezone().isoformat()

            logger.info(f"=== Wake cycle #{wake_count} at {wake_time} ===")

            try:
                # Ensure container is still running
                if not await self._ensure_container():
                    logger.error("Container not available, skipping cycle")
                    continue

                # Reinitialize system prompt to reload context files
                self.agent.initialize_system_prompt()

                # Create autonomous wake prompt
                prompt = (
                    f"You are awake (wake cycle #{wake_count}). "
                    "Review your CONTEXT and TODO, then work on your tasks. "
                    "Remember to update your journal and context files."
                )

                logger.info(f"Executing agent with prompt: {prompt}")
                await self.agent.run(prompt)

                # Save wake count
                wake_count += 1
                try:
                    _ = wake_count_file.write_text(str(wake_count))
                except Exception as e:
                    logger.error(f"Failed to save .wake_count: {e}")

                logger.info(f"Wake cycle #{wake_count} completed")

            except Exception as e:
                logger.error(f"Error during wake cycle #{wake_count}: {e}", exc_info=True)

            # Sleep until next wake cycle
            if self.running:
                logger.info(f"Sleeping for {settings.wake_interval_seconds} seconds...")
                for _ in range(settings.wake_interval_seconds):
                    if not self.running:
                        break
                    await asyncio.sleep(1)

        logger.info("Autonomous agent stopped")


async def main() -> None:
    """Entry point for autonomous runner."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create components
    agent_events = AgentEvents()
    event_logger = EventLogger(
        settings.event_log_file,
        stream_url=settings.stream_api_url if settings.stream_events else None,
        stream_api_key=settings.stream_api_key,
        stream_enabled=settings.stream_events,
    )

    llm_client = LLMFactory.create(
        url=settings.openai_base_url,
        model=settings.openai_model,
        api_key=settings.openai_api_key,
    )

    agent = Agent(llm_client, agent_events, event_logger)

    runner = AutonomousRunner(agent, event_logger)

    try:
        await runner.run()
    finally:
        await event_logger.close()


if __name__ == "__main__":
    asyncio.run(main())
