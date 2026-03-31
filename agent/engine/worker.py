"""
Per-conversation and cron worker classes.

ConversationWorker owns a private asyncio.Queue and processes its events
sequentially, ensuring messages for the same conversation are handled in
order while different chats run concurrently.

CronWorker manages aiocron lifecycle for one chat session — one instance
per active session — and enqueues CronEvents onto the conversation queue.
"""

import asyncio
import base64
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import aiocron

from agent.core.events import (
    CronEvent,
    HeartbeatEvent,
    ImageInputEvent,
    NewSessionEvent,
    TextInputEvent,
    WorkerEvent,
)
from agent.core.messaging import Channel
from agent.core.settings import Settings
from agent.llm.agent import (
    Agent,
    OrchestratorFactory,
)
from agent.llm.prompt import SystemPromptBuilder
from agent.tools.cron import CronJobDef, CronLoader

logger = logging.getLogger(__name__)


@dataclass
class Conversation:
    messages: list[dict[str, Any]] = field(default_factory=list)
    message_ids: set[str] = field(default_factory=set)
    total_tokens: int = 0


def _format_current_datetime() -> tuple[datetime, str]:
    """Return the current localtime datetime object and a formatted string."""
    now = datetime.now().astimezone()
    return now, now.strftime("%Y-%m-%d %H:%M:%S %Z%z")


class ConversationWorker:
    """
    Processes events for a single chat conversation.

    Each worker owns an asyncio.Queue and runs as a background task, ensuring
    messages for the same conversation are processed sequentially while
    conversations for different chat_ids run concurrently.
    """

    def __init__(
        self,
        settings: Settings,
        agent: Agent,
        prompt_builder: SystemPromptBuilder,
        orchestrator_factory: OrchestratorFactory,
    ) -> None:
        self.settings = settings
        self.agent = agent
        self.prompt_builder = prompt_builder
        self.orchestrator_factory = orchestrator_factory
        self.queue: asyncio.Queue[WorkerEvent] = asyncio.Queue()
        self.conversation = Conversation()
        self.heartbeat_event: HeartbeatEvent | None = None
        self.heartbeat_task: asyncio.Task[None] | None = None
        self.event_handlers: dict[type, Callable[..., Awaitable[None]]] = {
            HeartbeatEvent: self._process_heartbeat,
            CronEvent: self._process_cron,
            NewSessionEvent: self._process_new_session,
            TextInputEvent: self._process_text_input,
            ImageInputEvent: self._process_image_input,
        }

    async def _compress_conversation(self, sender: Channel) -> None:
        """Compress conversation history. Agent handles all message manipulation."""
        logger.info(f"Compressing {len(self.conversation.messages)} messages")
        await sender.send("Context window full, compressing conversation…")
        await self.agent.compress(
            self.conversation.messages, self.settings.context_num_keep_last
        )
        self.conversation.total_tokens = 0
        self.conversation.message_ids = set()
        await sender.send("Conversation compressed")

    async def _check_dedup_and_compress(self, message_id: str, sender: Channel) -> bool:
        """Return False if duplicate; register and optionally compress, then return True."""
        if message_id in self.conversation.message_ids:
            logger.debug(f"Ignoring duplicated message {message_id}")
            return False
        self.conversation.message_ids.add(message_id)
        if (
            self.settings.context_auto_compression_enabled
            and self.conversation.messages
            and self.conversation.total_tokens >= self.settings.context_max_tokens
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
        self.heartbeat_event = event
        logger.info("Processing heartbeat")
        prompt = self.prompt_builder.build_with_context(["HEARTBEAT.md"])
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
        orchestrator = self.orchestrator_factory.make_background(event.sender)
        await self.agent.run(prompt, self.conversation.messages, orchestrator)
        logger.info("Heartbeat cycle completed")

    async def _process_cron(self, event: CronEvent) -> None:
        logger.info(f"Processing cron task: {event.task_name}")
        prompt = self.prompt_builder.build_with_context(["CRON.md"])
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
        orchestrator = self.orchestrator_factory.make_background(event.sender)
        await self.agent.run(prompt, self.conversation.messages, orchestrator)
        logger.info(f"Cron task '{event.task_name}' completed")

    async def _run_user_turn(self, message: dict[str, Any], sender: Channel) -> None:
        """Shared path for text and image input: append, run agent, track tokens."""
        self.conversation.messages.append(message)
        await sender.start_thinking()

        prompt = self.prompt_builder.build()
        orchestrator = self.orchestrator_factory.make_human_input(sender)
        max_tokens = (
            self.settings.context_max_tokens
            if self.settings.context_auto_compression_enabled
            else 0
        )
        response = await self.agent.run(
            prompt,
            self.conversation.messages,
            orchestrator,
            max_tokens,
            self.settings.context_num_keep_last,
        )
        self.conversation.total_tokens = response.usage.total_tokens
        await sender.end_thinking()

    async def _process_text_input(self, event: TextInputEvent) -> None:
        if not await self._check_dedup_and_compress(event.message_id, event.sender):
            return

        now, current_datetime = _format_current_datetime()
        message: dict[str, Any] = {
            "role": "user",
            "content": f"""Message Time: {current_datetime}
Timezone: {now.tzinfo}

{event.message}""",
        }
        logger.info(f"Processing text input: {event.message[:100]}...")
        await self._run_user_turn(message, event.sender)
        logger.info("Text input processing completed")

    async def _process_image_input(self, event: ImageInputEvent) -> None:
        if not self.settings.vision_support:
            logger.debug("Vision support disabled, ignoring ImageInputEvent")
            await event.sender.send(
                "Received image input but vision support is disabled."
            )
            return

        if not await self._check_dedup_and_compress(event.message_id, event.sender):
            return

        now, current_datetime = _format_current_datetime()
        image_b64 = base64.b64encode(event.image_data).decode()
        message: dict[str, Any] = {
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
        logger.info("Processing image input")
        await self._run_user_turn(message, event.sender)
        logger.info("Image input processing completed")

    async def run(self) -> None:
        """Process events from this worker's queue until cancelled."""
        logger.info("Conversation worker started")
        while True:
            event = await self.queue.get()
            if self.heartbeat_task:
                self.heartbeat_task.cancel()
            try:
                handler = self.event_handlers.get(type(event))
                if handler:
                    await handler(event)
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
        queue: asyncio.Queue[WorkerEvent],
        loader: CronLoader,
    ) -> None:
        self.chat_id = chat_id
        self.queue = queue
        self.loader = loader
        self.jobs: dict[str, list[Any]] = {}

    def load(self, job_name: str, sender: Channel) -> list[CronJobDef]:
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
                _sender: Channel = sender,
                _queue: asyncio.Queue[WorkerEvent] = self.queue,
            ) -> Any:
                async def callback() -> None:
                    await _queue.put(
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
