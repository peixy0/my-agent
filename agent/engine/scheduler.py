"""
Event-driven scheduler and per-conversation worker.

The Scheduler consumes from the shared event queue (populated by the API
server and messaging sources) and dispatches each event to the appropriate
ConversationWorker.

Each ConversationWorker owns a private asyncio.Queue and processes its
events sequentially, so conversations are isolated from one another while
still running concurrently.
"""

import asyncio
import base64
import contextlib
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Final, Protocol, runtime_checkable

from agent.core.events import (
    DropSessionEvent,
    HeartbeatEvent,
    ImageInputEvent,
    NewSessionEvent,
    TextInputEvent,
)
from agent.core.sender import MessageSender
from agent.core.settings import Settings
from agent.llm.agent import (
    Agent,
    HeartbeatOrchestrator,
    HumanInputOrchestrator,
)
from agent.llm.prompt import SystemPromptBuilder
from agent.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


@runtime_checkable
class SchedulerContext(Protocol):
    """Read-only view of the application dependencies used by the scheduler."""

    settings: Settings
    model_name: str
    agent: Agent
    tool_registry: ToolRegistry
    prompt: SystemPromptBuilder
    event_queue: asyncio.Queue  # type: ignore[type-arg]


AgentEvent = TextInputEvent | ImageInputEvent | DropSessionEvent
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

    app: Final[SchedulerContext]
    queue: asyncio.Queue[WorkerEvent]
    conversation: Conversation
    _heartbeat_event: HeartbeatEvent | None
    _heartbeat_task: asyncio.Task[None] | None

    def __init__(self, app: SchedulerContext) -> None:
        self.app = app
        self.queue = asyncio.Queue()
        self.conversation = Conversation()
        self._heartbeat_event = None
        self._heartbeat_task = None

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
        self.conversation.message_ids = set()
        await sender.send("Conversation compressed")

    async def _process_new_session(self, event: NewSessionEvent) -> None:
        self.conversation = Conversation()
        await event.sender.send("New session started")

    async def _maybe_schedule_next_heartbeat(self) -> None:
        if not self._heartbeat_event:
            return
        await asyncio.sleep(self._heartbeat_event.interval_seconds)
        await self.queue.put(self._heartbeat_event)

    async def _process_heartbeat(self, event: HeartbeatEvent) -> None:
        if event.interval_seconds <= 0:
            return
        logger.info("Processing heartbeat")
        prompt = self.app.prompt.build_for_heartbeat()
        now = datetime.now().astimezone()
        current_datetime = now.strftime("%Y-%m-%d %H:%M:%S %Z%z")
        self.conversation = Conversation()
        self.conversation.messages = [
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
        await self.app.agent.run(prompt, self.conversation.messages, orchestrator)
        logger.info("Heartbeat cycle completed")

    async def _process_text_input(self, event: TextInputEvent) -> None:
        if event.message_id in self.conversation.message_ids:
            logger.debug(f"Ignoring duplicated message {event.message_id}")
            return
        self.conversation.message_ids.add(event.message_id)

        if (
            self.app.settings.context_auto_compression_enabled
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
        await event.sender.start_thinking()

        prompt = self.app.prompt.build_with_previous_summary(
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
        await event.sender.end_thinking()
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
            self.app.settings.context_auto_compression_enabled
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
        await event.sender.start_thinking()

        prompt = self.app.prompt.build_with_previous_summary(
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
        await event.sender.end_thinking()
        logger.info("Image input processing completed")

    async def run(self) -> None:
        """Process events from this worker's queue until cancelled."""
        logger.info("Conversation worker started")
        while True:
            event = await self.queue.get()
            if self._heartbeat_task:
                self._heartbeat_task.cancel()
            try:
                if isinstance(event, HeartbeatEvent):
                    self._heartbeat_event = event
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
                self._heartbeat_task = asyncio.create_task(
                    self._maybe_schedule_next_heartbeat()
                )
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

    app: Final[SchedulerContext]
    running: bool
    _workers: dict[str, tuple[ConversationWorker, asyncio.Task[None]]]

    def __init__(self, app: SchedulerContext) -> None:
        self.app = app
        self.running = True
        self._workers = {}

    def _get_or_create_worker(self, chat_id: str) -> ConversationWorker:
        """Return the existing worker for *chat_id* or create and start a new one."""
        if chat_id not in self._workers:
            worker = ConversationWorker(self.app)
            self._workers[chat_id] = (worker, asyncio.create_task(worker.run()))
            logger.info(f"Started conversation worker for chat_id={chat_id}")
        worker, _ = self._workers[chat_id]
        return worker

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
                await self._get_or_create_worker(event.chat_id).queue.put(
                    HeartbeatEvent(
                        chat_id=event.chat_id,
                        interval_seconds=interval_seconds,
                        sender=event.sender,
                    )
                )
                await event.sender.send(
                    f"Heartbeat started: interval {interval_seconds}"
                )
            elif isinstance(event, TextInputEvent) and event.message.startswith("/new"):
                await self._get_or_create_worker(event.chat_id).queue.put(
                    NewSessionEvent(chat_id=event.chat_id, sender=event.sender)
                )
            elif isinstance(event, TextInputEvent) and event.message.startswith(
                "/drop"
            ):
                await event.sender.send(f"Dropping session chat_id={event.chat_id}")
                await self.app.event_queue.put(DropSessionEvent(chat_id=event.chat_id))
            elif isinstance(event, DropSessionEvent):
                if event.chat_id in self._workers:
                    worker, task = self._workers.pop(event.chat_id)
                    if worker._heartbeat_task:
                        worker._heartbeat_task.cancel()
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
                    logger.info(f"Dropped session worker for chat_id={event.chat_id}")
                else:
                    logger.debug(
                        f"DropSessionEvent for unknown chat_id={event.chat_id}"
                    )
            else:
                worker = self._get_or_create_worker(event.chat_id)
                await worker.queue.put(event)
        except Exception as e:
            logger.error(f"Error during event dispatch: {e}", exc_info=True)

    async def run(self) -> None:
        """Consume from the shared event queue until stopped."""
        logger.info("Scheduler starting...")
        while self.running:
            event = await self.app.event_queue.get()
            await self._dispatch(event)
            self.app.event_queue.task_done()
