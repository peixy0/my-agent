"""
Entry point for the autonomous agent.

Runs the Scheduler (event loop) and FastAPI server concurrently.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Final

import uvloop

from agent.app import AppWithDependencies
from agent.core.events import HeartbeatEvent, HumanInputEvent
from agent.core.settings import get_settings
from agent.llm.agent import HeartbeatOrchestrator, HumanInputOrchestrator

logger = logging.getLogger("agent")

AgentEvent = HeartbeatEvent | HumanInputEvent


@dataclass
class Conversation:
    chat_id: str
    messages: list[dict[str, str]]
    message_ids: set[str]
    total_tokens: int
    previous_summary: str

    def __init__(self, chat_id: str):
        self.chat_id = chat_id
        self.messages = []
        self.message_ids = set()
        self.total_tokens = 0
        self.previous_summary = ""


class Scheduler:
    """
    Event-driven scheduler that manages agent execution cycles.

    Processes HeartbeatEvents (periodic) and HumanInputEvents (from API).
    """

    app: Final[AppWithDependencies]
    running: bool
    heartbeat_task: asyncio.Task[None] | None
    conversations: dict[str, Conversation]

    def __init__(self, app: AppWithDependencies):
        self.app = app
        self.running = True
        self.heartbeat_task = None
        self.conversations = {}

    async def _schedule_heartbeat(self) -> None:
        logger.info(f"Sleeping for {self.app.settings.wake_interval_seconds} seconds")
        await asyncio.sleep(self.app.settings.wake_interval_seconds)
        await self.app.event_queue.put(HeartbeatEvent())

    async def _process_heartbeat(self) -> None:
        logger.info("Executing agent heartbeat cycle")
        prompt = self.app.prompt_builder.build()
        now = datetime.now().astimezone()
        current_datetime = now.strftime("%Y-%m-%d %H:%M:%S %Z%z")
        messages = [
            {
                "role": "user",
                "content": f"""Current Time: {current_datetime}
Timezone: {now.tzinfo}
SYSTEM EVENT: Heartbeat""",
            }
        ]
        orchestrator = HeartbeatOrchestrator(
            self.app.model_name,
            self.app.tool_registry,
            self.app.messaging,
            self.app.event_logger,
        )
        await self.app.agent.run(prompt, messages, orchestrator)
        logger.info("Heartbeat cycle completed")

    async def _compress_conversation(self, conversation: Conversation) -> None:
        """Compress all conversation messages into a summary stored in the system prompt."""
        logger.info("Compressing conversation context")
        await self.app.messaging.send_message(
            conversation.chat_id, "Max token reached, compressing conversation"
        )
        conversation.previous_summary = await self.app.agent.compress(
            conversation.messages
        )
        conversation.messages = []
        conversation.total_tokens = 0
        await self.app.messaging.send_message(
            conversation.chat_id, "Conversation compressed"
        )

    async def _process_human_input(self, event: HumanInputEvent) -> None:
        if event.message == "/new":
            self.conversations[event.chat_id] = Conversation(event.chat_id)
            await self.app.messaging.send_message(event.chat_id, "New session started")
            return
        if event.message == "/heartbeat":
            await self.app.event_queue.put(HeartbeatEvent())
            await self.app.messaging.send_message(
                event.chat_id, "New heartbeat started"
            )
            return
        if event.message == "/compress":
            conversation = self.conversations.get(
                event.chat_id, Conversation(event.chat_id)
            )
            if conversation.total_tokens < self.app.settings.context_max_tokens:
                await self.app.messaging.send_message(
                    event.chat_id,
                    f"No need to compress, total tokens: {conversation.total_tokens}",
                )
                return
            await self._compress_conversation(conversation)
            self.conversations[event.chat_id] = conversation
            return

        conversation = self.conversations.get(
            event.chat_id, Conversation(event.chat_id)
        )
        if event.message_id in conversation.message_ids:
            logger.debug(f"Ignoring duplicated message {event.message_id}")
            return
        conversation.message_ids.add(event.message_id)

        now = datetime.now().astimezone()
        current_datetime = now.strftime("%Y-%m-%d %H:%M:%S %Z%z")
        conversation.messages.append(
            {
                "role": "user",
                "content": f"""Message Time: {current_datetime}
Timezone: {now.tzinfo}

{event.message}""",
            }
        )
        logger.info(f"Processing human input: {event.message[:100]}...")

        prompt = self.app.prompt_builder.build(conversation.previous_summary)
        orchestrator = HumanInputOrchestrator(
            event.chat_id,
            event.message_id,
            self.app.model_name,
            self.app.tool_registry,
            self.app.messaging,
            self.app.event_logger,
        )
        response = await self.app.agent.run(prompt, conversation.messages, orchestrator)
        conversation.total_tokens = response.usage.total_tokens
        self.conversations[event.chat_id] = conversation
        logger.info("Human input processing completed")

    async def _dispatch(self, event: AgentEvent) -> None:
        """Dispatch a single event as a self-contained async task."""
        try:
            if self.heartbeat_task:
                self.heartbeat_task.cancel()
            if isinstance(event, HeartbeatEvent):
                await self._process_heartbeat()
            elif isinstance(event, HumanInputEvent):
                await self._process_human_input(event)
            else:
                logger.warning(f"Unknown event type: {type(event)}")
        except Exception as e:
            logger.error(f"Error during event processing: {e}", exc_info=True)
            await self.app.messaging.notify(f"Error during event processing: {e}")
        finally:
            self.heartbeat_task = asyncio.create_task(self._schedule_heartbeat())

    async def run(self) -> None:
        logger.info("Scheduler starting...")
        logger.info(f"Wake interval: {self.app.settings.wake_interval_seconds} seconds")

        # Trigger initial heartbeat
        # await self.app.event_queue.put(HeartbeatEvent())

        while self.running:
            event = await self.app.event_queue.get()
            await self._dispatch(event)
            self.app.event_queue.task_done()


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
    uvloop.run(main())
