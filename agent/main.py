"""
Entry point for the autonomous agent.

Runs the Scheduler (event loop) and FastAPI server concurrently.
"""

import asyncio
import base64
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Final

try:
    import uvloop  # type: ignore[import]
except ImportError:  # pragma: no cover
    uvloop = None  # type: ignore[assignment]

from agent.app import AppWithDependencies
from agent.core.events import (
    HeartbeatEvent,
    ImageInputEvent,
    NewSessionEvent,
    TextInputEvent,
)
from agent.core.settings import get_settings
from agent.llm.agent import HeartbeatOrchestrator, HumanInputOrchestrator
from agent.messaging.sender import MessageSender

logger = logging.getLogger("agent")

AgentEvent = TextInputEvent | ImageInputEvent
WorkerEvent = HeartbeatEvent | NewSessionEvent | TextInputEvent | ImageInputEvent


@dataclass
class Conversation:
    messages: list[dict[str, Any]]
    message_ids: set[str]
    total_tokens: int
    previous_summary: str

    def __init__(self) -> None:
        self.messages = []
        self.message_ids = set()
        self.total_tokens = 0
        self.previous_summary = ""


class ConversationWorker:
    """
    Processes human-input events for a single chat conversation.

    Each worker owns an asyncio.Queue and runs as a background task, ensuring
    messages for the same conversation are processed sequentially while
    conversations for different chat_ids run concurrently.
    """

    app: Final[AppWithDependencies]
    queue: asyncio.Queue[WorkerEvent]
    conversation: Conversation

    def __init__(self, app: AppWithDependencies) -> None:
        self.app = app
        self.queue = asyncio.Queue()
        self.conversation = Conversation()

    async def _compress_conversation(self, sender: MessageSender) -> None:
        """
        Compress old messages into a structured summary, retaining the recent tail.

        The last context_num_keep_last messages are kept verbatim so the agent
        retains immediate context.  The prior summary is passed to compress() so
        each compression is incremental — earlier history is never silently lost.
        """
        keep_last = self.app.settings.context_num_keep_last
        if keep_last > 0 and len(self.conversation.messages) > keep_last:
            to_summarize = self.conversation.messages[:-keep_last]
            retained = self.conversation.messages[-keep_last:]
        else:
            to_summarize = self.conversation.messages
            retained = []

        logger.info(
            f"Compressing {len(to_summarize)} messages, retaining {len(retained)}"
        )
        await sender.send("Context window full, compressing conversation…")
        self.conversation.previous_summary = await self.app.agent.compress(
            to_summarize,
            previous_summary=self.conversation.previous_summary,
        )
        self.conversation.messages = retained
        self.conversation.total_tokens = 0
        await sender.send("Conversation compressed")

    async def _process_new_session(self, event: NewSessionEvent) -> None:
        self.conversation = Conversation()
        await event.sender.send("New session started")

    async def _process_heartbeat(self, event: HeartbeatEvent) -> None:
        prompt = self.app.prompt_builder.build_for_heartbeat()
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
            event.sender,
        )
        await self.app.agent.run(prompt, messages, orchestrator)
        logger.info("Heartbeat cycle completed")

    async def _process_text_input(self, event: TextInputEvent) -> None:
        if event.message_id in self.conversation.message_ids:
            logger.debug(f"Ignoring duplicated message {event.message_id}")
            return
        self.conversation.message_ids.add(event.message_id)

        if (
            self.app.settings.enable_context_auto_compression
            and self.conversation.messages
            and self.conversation.total_tokens >= self.app.settings.context_max_tokens
        ):
            await self._compress_conversation(event.sender)

        now = datetime.now().astimezone()
        current_datetime = now.strftime("%Y-%m-%d %H:%M:%S %Z%z")
        self.conversation.messages.append(
            {
                "role": "user",
                "content": f"""Message Time: {current_datetime}
Timezone: {now.tzinfo}

{event.message}""",
            }
        )
        logger.info(f"Processing text input: {event.message[:100]}...")

        prompt = self.app.prompt_builder.build_with_previous_summary(
            self.conversation.previous_summary
        )
        orchestrator = HumanInputOrchestrator(
            self.app.model_name,
            self.app.tool_registry,
            event.sender,
        )
        response = await self.app.agent.run(
            prompt, self.conversation.messages, orchestrator
        )
        self.conversation.total_tokens = response.usage.total_tokens
        logger.info("Text input processing completed")

    async def _process_image_input(self, event: ImageInputEvent) -> None:
        if not self.app.settings.vision_support:
            logger.debug("Vision support disabled, ignoring ImageInputEvent")
            await event.sender.send(
                "Received image input but vision support is disabled."
            )
            return

        if event.message_id in self.conversation.message_ids:
            logger.debug(f"Ignoring duplicated image message {event.message_id}")
            return
        self.conversation.message_ids.add(event.message_id)

        if (
            self.app.settings.enable_context_auto_compression
            and self.conversation.messages
            and self.conversation.total_tokens >= self.app.settings.context_max_tokens
        ):
            await self._compress_conversation(event.sender)

        now = datetime.now().astimezone()
        current_datetime = now.strftime("%Y-%m-%d %H:%M:%S %Z%z")
        image_b64 = base64.b64encode(event.image_data).decode()
        self.conversation.messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""Message Time: {current_datetime}
Timezone: {now.tzinfo}

{event.message}
""",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{event.mime_type};base64,{image_b64}",
                            "detail": "auto",
                        },
                    },
                ],
            }
        )
        logger.info("Processing image input")

        prompt = self.app.prompt_builder.build_with_previous_summary(
            self.conversation.previous_summary
        )
        orchestrator = HumanInputOrchestrator(
            self.app.model_name,
            self.app.tool_registry,
            event.sender,
        )
        response = await self.app.agent.run(
            prompt, self.conversation.messages, orchestrator
        )
        self.conversation.total_tokens = response.usage.total_tokens
        logger.info("Image input processing completed")

    async def run(self) -> None:
        """Process events from this worker's queue until cancelled."""
        logger.info("Conversation worker started")
        while True:
            event = await self.queue.get()
            try:
                if isinstance(event, HeartbeatEvent):
                    await self._process_heartbeat(event)
                elif isinstance(event, NewSessionEvent):
                    await self._process_new_session(event)
                elif isinstance(event, TextInputEvent):
                    await self._process_text_input(event)
                elif isinstance(event, ImageInputEvent):
                    await self._process_image_input(event)
                else:
                    logger.warning(f"Unexpected event type in worker: {type(event)}")
            except Exception as e:
                logger.error(f"Error in worker {event.chat_id}: {e}", exc_info=True)
                await event.sender.send(f"Error during processing: {e}")
            finally:
                self.queue.task_done()


class Scheduler:
    """
    Event-driven scheduler that dispatches inbound events to per-chat workers.

    Commands (/heartbeat, /new) are translated to typed internal events here
    so ConversationWorkers never need to parse raw message text.

    A repeating heartbeat loop is started only when /heartbeat is received,
    bound to the sender that triggered it.  A new /heartbeat replaces any
    existing loop.
    """

    app: Final[AppWithDependencies]
    running: bool
    _workers: dict[str, tuple[ConversationWorker, asyncio.Task[None]]]
    _heartbeat_task: asyncio.Task[None] | None

    def __init__(self, app: AppWithDependencies):
        self.app = app
        self.running = True
        self._workers = {}
        self._heartbeat_task = None

    def _get_or_create_worker(self, chat_id: str) -> ConversationWorker:
        """Return the existing worker for *chat_id* or create and start a new one."""
        if chat_id not in self._workers:
            worker = ConversationWorker(self.app)
            self._workers[chat_id] = (worker, asyncio.create_task(worker.run()))

            logger.info(f"Started conversation worker for chat_id={chat_id}")
        worker, _ = self._workers[chat_id]
        return worker

    async def _schedule_heartbeat(
        self, chat_id: str, sender: MessageSender, interval_seconds: int
    ) -> None:
        """Sleep then fire a heartbeat to *chat_id* via *sender*, then repeat."""
        worker = self._get_or_create_worker(chat_id)
        await worker.queue.put(HeartbeatEvent(chat_id=chat_id, sender=sender))
        logger.info(f"Heartbeat dispatched to chat_id={chat_id}")
        if interval_seconds <= 0:
            return
        while True:
            await asyncio.sleep(interval_seconds)
            worker = self._get_or_create_worker(chat_id)
            await worker.queue.put(HeartbeatEvent(chat_id=chat_id, sender=sender))
            logger.info(f"Heartbeat dispatched to chat_id={chat_id}")

    async def _dispatch(self, event: AgentEvent) -> None:
        """Translate commands and route every event to the appropriate worker."""
        try:
            if isinstance(event, TextInputEvent) and event.message.startswith(
                "/heartbeat"
            ):
                param = event.message[len("/heartbeat") :].strip()
                try:
                    interval_seconds = int(param)
                except ValueError:
                    interval_seconds = self.app.settings.wake_interval_seconds
                if self._heartbeat_task:
                    self._heartbeat_task.cancel()
                self._heartbeat_task = asyncio.create_task(
                    self._schedule_heartbeat(
                        event.chat_id, event.sender, interval_seconds
                    )
                )
            elif isinstance(event, TextInputEvent) and event.message.startswith("/new"):
                await self._get_or_create_worker(event.chat_id).queue.put(
                    NewSessionEvent(chat_id=event.chat_id, sender=event.sender)
                )
            else:
                worker = self._get_or_create_worker(event.chat_id)
                await worker.queue.put(event)
        except Exception as e:
            logger.error(f"Error during event dispatch: {e}", exc_info=True)

    async def run(self) -> None:
        logger.info("Scheduler starting...")

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

    # Start dependent background tasks (messaging source, API server)
    app = AppWithDependencies(get_settings())
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
