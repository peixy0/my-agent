"""
Entry point for the autonomous agent.

Runs the Scheduler (event loop) and FastAPI server concurrently.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Final

import uvicorn

from agent.app import Application
from agent.core.events import HeartbeatEvent, HumanInputEvent
from agent.tools.command_executor import ensure_container_running

logger = logging.getLogger(__name__)

AgentEvent = HeartbeatEvent | HumanInputEvent

HEARTBEAT_RESPONSE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "report_decision": {
            "type": "boolean",
            "description": "Whether to send a report to the human user based on the reporting criteria.",
        },
        "report_message": {
            "type": "string",
            "description": "The message to report to the user with `report-styler` enforced (if report_decision is true) or a summary of activity (if false).",
        },
    },
    "required": ["report_decision", "report_message"],
}

HUMAN_INPUT_RESPONSE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "message": {
            "type": "string",
            "description": "The response to the human's message.",
        },
    },
    "required": ["message"],
}


class Scheduler:
    """
    Event-driven scheduler that manages agent execution cycles.

    Processes HeartbeatEvents (periodic) and HumanInputEvents (from API).
    """

    app: Final[Application]
    running: bool
    heartbeat_task: asyncio.Task[None] | None

    def __init__(self, app: Application):
        self.app = app
        self.running = True
        self.heartbeat_task = None

    async def _ensure_container(self) -> bool:
        return await ensure_container_running(
            container_name=self.app.settings.container_name,
            runtime=self.app.settings.container_runtime,
            workspace_path=self.app.settings.workspace_dir,
        )

    async def _schedule_heartbeat(self) -> None:
        logger.info(f"Sleeping for {self.app.settings.wake_interval_seconds} seconds")
        await asyncio.sleep(self.app.settings.wake_interval_seconds)
        await self.app.event_queue.put(HeartbeatEvent())

    async def _process_heartbeat(self) -> None:
        prompt = self.app.prompt_builder.build()
        self.app.agent.set_system_prompt(prompt)

        schema_str = json.dumps(HEARTBEAT_RESPONSE_SCHEMA)
        user_prompt = f"You are awake. Please respond with a JSON object matching this schema: {schema_str}"

        logger.info("Executing agent heartbeat cycle")
        response = await self.app.agent.run(
            user_prompt, response_schema=HEARTBEAT_RESPONSE_SCHEMA
        )
        await self.app.event_logger.log_agent_response(
            f"REPORT DECISION: {response['report_decision']}\n\n{response['report_message']}"
        )
        logger.info("Heartbeat cycle completed")

    async def _process_human_input(self, event: HumanInputEvent) -> None:
        prompt = self.app.prompt_builder.build()
        self.app.agent.set_system_prompt(prompt)

        user_prompt = (
            f"A human user has sent you the following message. Process it and respond.\n\n"
            f"Human message: {event.content}\n\n"
            f"Respond with a JSON object matching this schema: {json.dumps(HUMAN_INPUT_RESPONSE_SCHEMA)}"
        )

        logger.info(f"Processing human input: {event.content[:100]}...")
        response = await self.app.agent.run(
            user_prompt, response_schema=HUMAN_INPUT_RESPONSE_SCHEMA
        )
        await self.app.event_logger.log_agent_response(
            f"HUMAN INPUT RESPONSE:\n\n{response['report_message']}"
        )
        if response.get("report_decision") and not self.app.settings.mute_agent:
            await self.app.messaging.send_message(response["report_message"])
        logger.info("Human input processing completed")

    async def run(self) -> None:
        if not await self._ensure_container():
            logger.error("Failed to start workspace container. Exiting.")
            return

        logger.info("Scheduler starting...")
        logger.info(f"Wake interval: {self.app.settings.wake_interval_seconds} seconds")

        # Trigger initial heartbeat
        await self.app.event_queue.put(HeartbeatEvent())

        while self.running:
            event = await self.app.event_queue.get()

            if not await self._ensure_container():
                logger.error("Container not available, skipping event")
                continue

            try:
                if isinstance(event, HeartbeatEvent):
                    wake_time = datetime.now().astimezone().isoformat()
                    logger.info(f"Wake cycle at {wake_time}")
                    await self._process_heartbeat()
                elif isinstance(event, HumanInputEvent):
                    await self._process_human_input(event)
                else:
                    logger.warning(f"Unknown event type: {type(event)}")
            except Exception as e:
                logger.error(f"Error during event processing: {e}", exc_info=True)

            # Re-schedule next heartbeat
            if self.heartbeat_task:
                _ = self.heartbeat_task.cancel()
            self.heartbeat_task = asyncio.create_task(self._schedule_heartbeat())


async def main() -> None:
    """Entry point: runs Scheduler + FastAPI server concurrently."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    app = Application()

    # Start event logger background worker
    _ = asyncio.create_task(app.event_logger.run())

    # Start messaging background worker
    _ = asyncio.create_task(app.messaging.run())

    # Create scheduler
    scheduler = Scheduler(app)

    # Create uvicorn server config
    config = uvicorn.Config(
        app.api,
        host=app.settings.api_host,
        port=app.settings.api_port,
        log_level="info",
    )
    server = uvicorn.Server(config)

    logger.info(
        f"Starting API server on {app.settings.api_host}:{app.settings.api_port}"
    )
    logger.info("Starting scheduler...")

    # Run both concurrently
    await asyncio.gather(
        scheduler.run(),
        server.serve(),
    )


if __name__ == "__main__":
    asyncio.run(main())
