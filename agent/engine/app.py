"""
Composition root — the single place where all dependencies are wired.

This replaces the scattered module-level singletons with explicit
construction, making the dependency graph visible and testable.
"""

import asyncio
import logging
import os

from agent.api.server import ApiService, create_api_service
from agent.core.events import AgentEvent
from agent.core.runtime import ContainerRuntime, HostRuntime
from agent.core.sender import MessageSource
from agent.core.settings import Settings
from agent.llm.agent import Agent
from agent.llm.openai import OpenAIProvider
from agent.llm.prompt import SystemPromptBuilder
from agent.messaging.source import create_message_source
from agent.tools.registry import ToolRegistry
from agent.tools.skill import SkillLoader
from agent.tools.toolbox import register_default_tools

logger = logging.getLogger(__name__)


class App:
    """
    Holds the fully-wired object graph and manages background tasks.

    Created once in main(), passed to the Scheduler and API server.
    Call run() to start dependent background tasks (event logger, messaging).
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.event_queue: asyncio.Queue[AgentEvent] = asyncio.Queue()

        # Tool infrastructure
        self.runtime = (
            ContainerRuntime(
                container_name=self.settings.container_name,
                runtime=self.settings.container_runtime,
                max_output_chars=self.settings.max_output_chars,
            )
            if self.settings.container_runtime
            else HostRuntime(max_output_chars=self.settings.max_output_chars)
        )
        self.skill = SkillLoader(self.settings.skills_dir)
        self.tool_registry = ToolRegistry()
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

        self.prompt = SystemPromptBuilder(self.skill)

        self.llm_client = OpenAIProvider(
            url=self.settings.openai_base_url,
            api_key=self.settings.openai_api_key,
        )
        self.model_name = self.settings.openai_model
        self.agent = Agent(
            self.llm_client,
            self.model_name,
            self.tool_registry,
        )

        self.background_tasks: list[asyncio.Task] = []

    async def run(self) -> None:
        """Start all background tasks."""
        os.chdir(self.settings.cwd)

        self.background_tasks = [
            asyncio.create_task(self.message_source.run()),
            asyncio.create_task(self.api_service.run()),
        ]
