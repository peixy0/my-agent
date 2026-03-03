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
from agent.core.events import HeartbeatEvent, ImageInputEvent, TextInputEvent
from agent.core.settings import get_settings
from agent.llm.agent import HeartbeatOrchestrator, HumanInputOrchestrator

logger = logging.getLogger("agent")

AgentEvent = HeartbeatEvent | TextInputEvent | ImageInputEvent


@dataclass
class Conversation:
    chat_id: str
    messages: list[dict[str, Any]]
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

    Processes HeartbeatEvents (periodic), TextInputEvents, and ImageInputEvents.
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
        if self.app.settings.wake_interval_seconds <= 0:
            return
        logger.info(f"Sleeping for {self.app.settings.wake_interval_seconds} seconds")
        await asyncio.sleep(self.app.settings.wake_interval_seconds)
        await self.app.event_queue.put(HeartbeatEvent())

    async def _process_heartbeat(self) -> None:
        logger.info("Executing agent heartbeat cycle")
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
            self.app.messaging,
        )
        await self.app.agent.run(prompt, messages, orchestrator)
        logger.info("Heartbeat cycle completed")

    async def _compress_conversation(self, conversation: Conversation) -> None:
        """
        Compress old messages into a structured summary, retaining the recent tail.

        The last context_num_keep_last messages are kept verbatim so the agent
        retains immediate context.  The prior summary is passed to compress() so
        each compression is incremental — earlier history is never silently lost.
        """
        keep_last = self.app.settings.context_num_keep_last
        if keep_last > 0 and len(conversation.messages) > keep_last:
            to_summarize = conversation.messages[:-keep_last]
            retained = conversation.messages[-keep_last:]
        else:
            to_summarize = conversation.messages
            retained = []

        logger.info(
            f"Compressing {len(to_summarize)} messages, retaining {len(retained)}"
        )
        await self.app.messaging.send_message(
            conversation.chat_id, "Context window full, compressing conversation…"
        )
        conversation.previous_summary = await self.app.agent.compress(
            to_summarize,
            previous_summary=conversation.previous_summary,
        )
        conversation.messages = retained
        conversation.total_tokens = 0
        await self.app.messaging.send_message(
            conversation.chat_id, "Conversation compressed"
        )

    async def _process_text_input(self, event: TextInputEvent) -> None:
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

        conversation = self.conversations.get(
            event.chat_id, Conversation(event.chat_id)
        )
        if event.message_id in conversation.message_ids:
            logger.debug(f"Ignoring duplicated message {event.message_id}")
            return
        conversation.message_ids.add(event.message_id)

        if (
            self.app.settings.enable_context_auto_compression
            and conversation.messages
            and conversation.total_tokens >= self.app.settings.context_max_tokens
        ):
            await self._compress_conversation(conversation)

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
        logger.info(f"Processing text input: {event.message[:100]}...")

        prompt = self.app.prompt_builder.build_with_previous_summary(
            conversation.previous_summary
        )
        orchestrator = HumanInputOrchestrator(
            event.chat_id,
            event.message_id,
            self.app.model_name,
            self.app.tool_registry,
            self.app.messaging,
        )
        response = await self.app.agent.run(prompt, conversation.messages, orchestrator)
        conversation.total_tokens = response.usage.total_tokens
        self.conversations[event.chat_id] = conversation
        logger.info("Text input processing completed")

    async def _process_image_input(self, event: ImageInputEvent) -> None:
        if not self.app.settings.vision_support:
            logger.debug("Vision support disabled, ignoring ImageInputEvent")
            await self.app.messaging.send_message(
                event.chat_id, "Received image input but vision support is disabled."
            )
            return

        conversation = self.conversations.get(
            event.chat_id, Conversation(event.chat_id)
        )
        if event.message_id in conversation.message_ids:
            logger.debug(f"Ignoring duplicated image message {event.message_id}")
            return
        conversation.message_ids.add(event.message_id)

        if (
            self.app.settings.enable_context_auto_compression
            and conversation.messages
            and conversation.total_tokens >= self.app.settings.context_max_tokens
        ):
            await self._compress_conversation(conversation)

        now = datetime.now().astimezone()
        current_datetime = now.strftime("%Y-%m-%d %H:%M:%S %Z%z")
        image_b64 = base64.b64encode(event.image_data).decode()
        conversation.messages.append(
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
            conversation.previous_summary
        )
        orchestrator = HumanInputOrchestrator(
            event.chat_id,
            event.message_id,
            self.app.model_name,
            self.app.tool_registry,
            self.app.messaging,
        )
        response = await self.app.agent.run(prompt, conversation.messages, orchestrator)
        conversation.total_tokens = response.usage.total_tokens
        self.conversations[event.chat_id] = conversation
        logger.info("Image input processing completed")

    async def _dispatch(self, event: AgentEvent) -> None:
        """Dispatch a single event as a self-contained async task."""
        try:
            if self.heartbeat_task:
                self.heartbeat_task.cancel()
            if isinstance(event, HeartbeatEvent):
                await self._process_heartbeat()
            elif isinstance(event, TextInputEvent):
                await self._process_text_input(event)
            elif isinstance(event, ImageInputEvent):
                await self._process_image_input(event)
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

    # Start dependent background tasks (messaging, API server)
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
