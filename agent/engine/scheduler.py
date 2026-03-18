"""
Event-driven scheduler and per-conversation worker.

The Scheduler consumes from the shared event queue (populated by the API
server and messaging sources) and dispatches each event to the appropriate
ConversationWorker.

Each ConversationWorker owns a private asyncio.Queue and processes its
events sequentially, so conversations are isolated from one another while
still running concurrently.

Cron jobs are managed by CronWorker — one instance per chat session — which
wraps aiocron and enqueues CronEvents onto the session's ConversationWorker.
"""

import asyncio
import base64
import contextlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

import aiocron

from agent.core.events import (
    AgentEvent,
    CronEvent,
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
from agent.tools.cron import CronJobDef, CronLoader
from agent.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class SchedulerContext(Protocol):
    """Read-only view of the application dependencies used by the scheduler."""

    settings: Settings
    model_name: str
    agent: Agent
    tool_registry: ToolRegistry
    prompt: SystemPromptBuilder
    event_queue: asyncio.Queue[AgentEvent]


@dataclass
class Conversation:
    messages: list[dict[str, Any]] = field(default_factory=list)
    message_ids: set[str] = field(default_factory=set)
    total_tokens: int = 0
    previous_summary: str = ""


def _format_current_datetime() -> tuple[datetime, str]:
    """Return the current localtime datetime object and a formatted string."""
    now = datetime.now().astimezone()
    return now, now.strftime("%Y-%m-%d %H:%M:%S %Z%z")


class ConversationWorker:
    """
    Processes human-input events for a single chat conversation.

    Each worker owns an asyncio.Queue and runs as a background task, ensuring
    messages for the same conversation are processed sequentially while
    conversations for different chat_ids run concurrently.
    """

    def __init__(self, app: SchedulerContext) -> None:
        self.app = app
        self.queue = asyncio.Queue()
        self.conversation = Conversation()
        self.heartbeat_event: HeartbeatEvent | None = None
        self.heartbeat_task: asyncio.Task[None] | None = None

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

    async def _check_dedup_and_compress(
        self, message_id: str, sender: MessageSender
    ) -> bool:
        """Return False if duplicate; register and optionally compress, then return True."""
        if message_id in self.conversation.message_ids:
            logger.debug(f"Ignoring duplicated message {message_id}")
            return False
        self.conversation.message_ids.add(message_id)
        if (
            self.app.settings.context_auto_compression_enabled
            and self.conversation.messages
            and self.conversation.total_tokens >= self.app.settings.context_max_tokens
        ):
            await self._compress_conversation(sender)
        return True

    async def _process_new_session(self, event: NewSessionEvent) -> None:
        self.conversation = Conversation()
        await event.sender.send("New session started")

    async def _maybe_schedule_next_heartbeat(self) -> None:
        if not self.heartbeat_event:
            return
        await asyncio.sleep(self.heartbeat_event.interval_seconds)
        await self.queue.put(self.heartbeat_event)

    async def _process_heartbeat(self, event: HeartbeatEvent) -> None:
        if event.interval_seconds <= 0:
            return
        logger.info("Processing heartbeat")
        prompt = self.app.prompt.build_for_heartbeat()
        now, current_datetime = _format_current_datetime()
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

    async def _process_cron(self, event: CronEvent) -> None:
        logger.info(f"Processing cron task: {event.task_name}")
        prompt = self.app.prompt.build_for_cron()
        now, current_datetime = _format_current_datetime()
        self.conversation = Conversation()
        self.conversation.messages = [
            {
                "role": "user",
                "content": f"""Current Time: {current_datetime}
Timezone: {now.tzinfo}

SYSTEM EVENT: Scheduled task '{event.task_name}'

{event.prompt}""",
            }
        ]
        orchestrator = HeartbeatOrchestrator(
            self.app.model_name,
            self.app.tool_registry,
            event.sender,
        )
        await self.app.agent.run(prompt, self.conversation.messages, orchestrator)
        logger.info(f"Cron task '{event.task_name}' completed")

    async def _process_text_input(self, event: TextInputEvent) -> None:
        if not await self._check_dedup_and_compress(event.message_id, event.sender):
            return

        now, current_datetime = _format_current_datetime()
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

        if not await self._check_dedup_and_compress(event.message_id, event.sender):
            return

        now, current_datetime = _format_current_datetime()
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
            if self.heartbeat_task:
                self.heartbeat_task.cancel()
            try:
                if isinstance(event, HeartbeatEvent):
                    self.heartbeat_event = event
                    await self._process_heartbeat(event)
                elif isinstance(event, CronEvent):
                    await self._process_cron(event)
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
                self.heartbeat_task = asyncio.create_task(
                    self._maybe_schedule_next_heartbeat()
                )
                self.queue.task_done()


class CronWorker:
    """Manages time-based cron jobs for a single chat session.

    Encapsulates aiocron lifecycle (start / stop). One instance lives per
    active chat session and is discarded when the session is dropped.
    """

    def __init__(
        self,
        chat_id: str,
        worker: ConversationWorker,
        loader: CronLoader,
    ) -> None:
        self.chat_id = chat_id
        self.worker = worker
        self.loader = loader
        self.jobs: dict[str, list[Any]] = {}

    def load(self, job_name: str, sender: MessageSender) -> list[CronJobDef]:
        """Load (or reload) a job group. Returns the definitions that were scheduled."""
        job_defs = self.loader.load_job(job_name)
        if not job_defs:
            return []

        self._stop_job(job_name)

        cron_objects: list[Any] = []
        for job_def in job_defs:

            def make_callback(
                _chat_id: str = self.chat_id,
                _task_name: str = job_def.task_name,
                _prompt: str = job_def.prompt,
                _sender: MessageSender = sender,
                _worker: ConversationWorker = self.worker,
            ) -> Any:
                async def callback() -> None:
                    await _worker.queue.put(
                        CronEvent(
                            chat_id=_chat_id,
                            task_name=_task_name,
                            prompt=_prompt,
                            sender=_sender,
                        )
                    )

                return callback

            cron_objects.append(
                aiocron.crontab(job_def.cron_expr, func=make_callback())
            )
            logger.info(
                f"Scheduled cron task '{job_def.task_name}' [{job_def.cron_expr}]"
                f" for chat_id={self.chat_id}"
            )

        self.jobs[job_name] = cron_objects
        return job_defs

    def unload(self, job_name: str) -> bool:
        """Stop and remove a job group. Returns True if the group was loaded."""
        return self._stop_job(job_name)

    def unload_all(self) -> None:
        """Stop and remove all loaded job groups."""
        for job_name in list(self.jobs):
            self._stop_job(job_name)

    def loaded_jobs(self) -> list[str]:
        """Return the names of currently active job groups."""
        return list(self.jobs)

    def _stop_job(self, job_name: str) -> bool:
        cron_objs = self.jobs.pop(job_name, None)
        if cron_objs is None:
            return False
        for obj in cron_objs:
            obj.stop()
        logger.info(f"Unloaded cron job '{job_name}' for chat_id={self.chat_id}")
        return True


class Scheduler:
    """
    Event-driven scheduler that dispatches inbound events to per-chat workers.

    Slash commands are parsed here and translated to typed internal events so
    ConversationWorkers never need to handle raw message text.

    Supported commands:
      /heartbeat [seconds]   — start recurring autonomous wake cycle
      /cron load <job>     — load and start cron jobs from .cron/<job>/
      /cron unload <job>   — stop a loaded cron job
      /cron ls               — list available and loaded cron jobs
      /new                   — reset conversation history
      /drop                  — tear down the session worker
    """

    def __init__(self, app: SchedulerContext) -> None:
        self.app = app
        self.running = True
        self.workers = {}
        self.cron_workers = {}
        self.cron_loader = CronLoader(app.settings.crons_dir)

    def _get_or_create_worker(self, chat_id: str) -> ConversationWorker:
        """Return the existing worker for *chat_id* or create and start a new one."""
        if chat_id not in self.workers:
            worker = ConversationWorker(self.app)
            self.workers[chat_id] = (worker, asyncio.create_task(worker.run()))
            logger.info(f"Started conversation worker for chat_id={chat_id}")
        worker, _ = self.workers[chat_id]
        return worker

    def _get_or_create_cron_worker(self, chat_id: str) -> CronWorker:
        """Return the CronWorker for *chat_id*, creating one if needed."""
        if chat_id not in self.cron_workers:
            worker = self._get_or_create_worker(chat_id)
            self.cron_workers[chat_id] = CronWorker(chat_id, worker, self.cron_loader)
        return self.cron_workers[chat_id]

    async def _cmd_heartbeat(self, event: TextInputEvent) -> None:
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
        await event.sender.send(f"Heartbeat started: interval {interval_seconds}")

    async def _cmd_cron(self, event: TextInputEvent) -> None:
        msg = event.message
        if msg.startswith("/cron load"):
            job_name = msg[len("/cron load") :].strip()
            if not job_name:
                await event.sender.send("Usage: /cron load job-name")
                return
            job_defs = self._get_or_create_cron_worker(event.chat_id).load(
                job_name, event.sender
            )
            if not job_defs:
                await event.sender.send(f"No cron tasks found for job '{job_name}'")
            else:
                task_lines = "\n\n".join(
                    f"- {j.task_name} ({j.cron_expr})" for j in job_defs
                )
                await event.sender.send(
                    f"Cron job '{job_name}' loaded"
                    f" with {len(job_defs)} task(s):\n\n{task_lines}"
                )
        elif msg.startswith("/cron unload"):
            job_name = msg[len("/cron unload") :].strip()
            if not job_name:
                await event.sender.send("Usage: /cron unload job-name")
                return
            cron = self._get_or_create_cron_worker(event.chat_id)
            if cron.unload(job_name):
                await event.sender.send(f"Cron job '{job_name}' unloaded")
            else:
                await event.sender.send(f"Cron job '{job_name}' was not loaded")
        elif msg.startswith("/cron ls"):
            available = self.cron_loader.list_jobs()
            if not available:
                await event.sender.send("No cron jobs found in .cron/")
                return
            loaded = (
                set(self.cron_workers[event.chat_id].loaded_jobs())
                if event.chat_id in self.cron_workers
                else set()
            )
            lines: list[str] = []
            for job in available:
                tasks = self.cron_loader.load_job(job)
                status = " [loaded]" if job in loaded else ""
                task_lines = "\n\n".join(
                    f"  - {t.task_name} ({t.cron_expr})" for t in tasks
                )
                lines.append(f"- {job}{status}\n{task_lines}")
            await event.sender.send("Available cron jobs:\n" + "\n\n".join(lines))
        else:
            await event.sender.send(
                "Usage: /cron load <name> | /cron unload <name> | /cron ls"
            )

    async def _cmd_new(self, event: TextInputEvent) -> None:
        await self._get_or_create_worker(event.chat_id).queue.put(
            NewSessionEvent(chat_id=event.chat_id, sender=event.sender)
        )

    async def _cmd_drop(self, event: TextInputEvent) -> None:
        await event.sender.send(f"Dropping session chat_id={event.chat_id}")
        await self.app.event_queue.put(DropSessionEvent(chat_id=event.chat_id))

    async def _handle_drop_session(self, event: DropSessionEvent) -> None:
        if event.chat_id in self.workers:
            worker, task = self.workers.pop(event.chat_id)
            if worker.heartbeat_task:
                worker.heartbeat_task.cancel()
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            logger.info(f"Dropped session worker for chat_id={event.chat_id}")
        else:
            logger.debug(f"DropSessionEvent for unknown chat_id={event.chat_id}")
        cron = self.cron_workers.pop(event.chat_id, None)
        if cron:
            cron.unload_all()

    async def _dispatch_text(self, event: TextInputEvent) -> None:
        """Route a TextInputEvent: intercept slash commands, forward the rest."""
        msg = event.message
        if msg.startswith("/heartbeat"):
            await self._cmd_heartbeat(event)
        elif msg.startswith("/cron"):
            await self._cmd_cron(event)
        elif msg.startswith("/new"):
            await self._cmd_new(event)
        elif msg.startswith("/drop"):
            await self._cmd_drop(event)
        else:
            await self._get_or_create_worker(event.chat_id).queue.put(event)

    async def _dispatch(self, event: AgentEvent) -> None:
        """Route an inbound event to the appropriate handler or worker queue."""
        try:
            if isinstance(event, TextInputEvent):
                await self._dispatch_text(event)
            elif isinstance(event, DropSessionEvent):
                await self._handle_drop_session(event)
            else:
                await self._get_or_create_worker(event.chat_id).queue.put(event)
        except Exception as e:
            logger.error(f"Error during event dispatch: {e}", exc_info=True)

    async def run(self) -> None:
        """Consume from the shared event queue until stopped."""
        logger.info("Scheduler starting...")
        while self.running:
            event = await self.app.event_queue.get()
            await self._dispatch(event)
            self.app.event_queue.task_done()
