"""
Event-driven scheduler.

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
import contextlib
import logging
from typing import Protocol

from agent.core.events import (
    AgentEvent,
    DropSessionEvent,
    HeartbeatEvent,
    NewSessionEvent,
    TextInputEvent,
)
from agent.core.settings import Settings
from agent.engine.worker import ConversationWorker, CronWorker
from agent.llm.agent import Agent
from agent.llm.prompt import SystemPromptBuilder
from agent.tools.cron import CronLoader
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
            worker = ConversationWorker(
                settings=self.app.settings,
                model_name=self.app.model_name,
                agent=self.app.agent,
                tool_registry=self.app.tool_registry,
                prompt=self.app.prompt,
            )
            self.workers[chat_id] = (worker, asyncio.create_task(worker.run()))
            logger.info(f"Started conversation worker for chat_id={chat_id}")
        worker, _ = self.workers[chat_id]
        return worker

    def _get_or_create_cron_worker(self, chat_id: str) -> CronWorker:
        """Return the CronWorker for *chat_id*, creating one if needed."""
        if chat_id not in self.cron_workers:
            worker = self._get_or_create_worker(chat_id)
            self.cron_workers[chat_id] = CronWorker(
                chat_id, worker.queue, self.cron_loader
            )
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
