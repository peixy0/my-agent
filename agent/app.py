"""
Composition root — the single place where all dependencies are wired.

This replaces the scattered module-level singletons with explicit
construction, making the dependency graph visible and testable.
"""

import asyncio
import logging
import os

from agent.api.server import ApiService, create_api_service
from agent.core.event_logger import EventLogger
from agent.core.messaging import create_messaging
from agent.core.settings import Settings
from agent.llm.agent import Agent
from agent.llm.factory import LLMFactory
from agent.llm.prompt_builder import SystemPromptBuilder
from agent.tools.command_executor import ContainerCommandExecutor, HostCommandExecutor
from agent.tools.skill_loader import SkillLoader
from agent.tools.tool_registry import ToolRegistry
from agent.tools.toolbox import register_default_tools

logger = logging.getLogger(__name__)


class AppWithDependencies:
    """
    Holds the fully-wired object graph and manages background tasks.

    Created once in main(), passed to the Scheduler and API server.
    Call run() to start dependent background tasks (event logger, messaging).
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.event_queue: asyncio.Queue = asyncio.Queue()

        # Event logger
        self.event_logger = EventLogger(
            event_url=self.settings.event_api_url,
            event_api_key=self.settings.event_api_key,
        )

        # Tool infrastructure
        self.executor = (
            ContainerCommandExecutor(
                container_name=self.settings.container_name,
                runtime=self.settings.container_runtime,
            )
            if self.settings.container_runtime
            else HostCommandExecutor()
        )
        self.skill_loader = SkillLoader(self.settings.skills_dir)
        self.tool_registry = ToolRegistry(
            tool_timeout=self.settings.tool_timeout,
        )
        self.messaging = create_messaging(self.settings, self.event_queue)
        register_default_tools(self.tool_registry, self.executor, self.skill_loader)

        # LLM client
        self.llm_client = LLMFactory(self.settings).create()

        # Agent
        self.model_name = self.settings.openai_model
        self.agent = Agent(
            self.llm_client,
            self.settings.openai_model,
            self.tool_registry,
            self.messaging,
            self.event_logger,
        )
        self.prompt_builder = SystemPromptBuilder(self.settings, self.skill_loader)

        # API Service
        self.api_service: ApiService = create_api_service(
            self.settings, self.event_queue
        )

        self._background_tasks: list[asyncio.Task] = []

    async def run(self) -> None:
        """Start dependent background tasks (event logger, messaging, API server)."""
        os.chdir(self.settings.cwd)

        self._background_tasks = [
            asyncio.create_task(self.event_logger.run()),
            asyncio.create_task(self.messaging.run()),
            asyncio.create_task(self.api_service.run()),
        ]
