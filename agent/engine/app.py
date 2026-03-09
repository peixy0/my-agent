"""
Composition root — the single place where all dependencies are wired.

This replaces the scattered module-level singletons with explicit
construction, making the dependency graph visible and testable.
"""

import asyncio
import logging
import os

from agent.api.server import ApiService, create_api_service
from agent.core.runtime import ContainerRuntime, HostRuntime
from agent.core.sender import MessageSource
from agent.core.settings import Settings
from agent.llm.agent import Agent
from agent.llm.factory import LLMFactory
from agent.llm.prompt import SystemPromptBuilder
from agent.messaging.source import create_message_source
from agent.tools.registry import ToolRegistry
from agent.tools.skill import SkillLoader
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

        # Tool infrastructure
        self.runtime = (
            ContainerRuntime(
                container_name=self.settings.container_name,
                runtime=self.settings.container_runtime,
            )
            if self.settings.container_runtime
            else HostRuntime()
        )
        self.skill = SkillLoader(self.settings.skills_dir)
        self.tool_registry = ToolRegistry(
            tool_timeout=self.settings.tool_timeout,
        )
        register_default_tools(
            self.tool_registry, self.runtime, self.skill, self.settings
        )

        # Message source (inbound)
        self.message_source: MessageSource = create_message_source(
            self.settings, self.event_queue, self.runtime
        )

        # API Service
        self.api_service: ApiService = create_api_service(
            self.settings, self.event_queue
        )

        self.prompt = SystemPromptBuilder(self.settings, self.skill)
        self._background_tasks: list[asyncio.Task] = []

    async def run(self) -> None:
        """Acquire LLM credentials, then start dependent background tasks."""
        llm_factory = LLMFactory(self.settings)
        self.llm_client = await llm_factory.create()
        self.model_name = llm_factory.get_model_name()
        self.agent = Agent(
            self.llm_client,
            self.model_name,
            self.tool_registry,
        )

        os.chdir(self.settings.cwd)

        self._background_tasks = [
            asyncio.create_task(self.message_source.run()),
            asyncio.create_task(self.api_service.run()),
        ]
