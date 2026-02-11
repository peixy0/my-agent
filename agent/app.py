"""
Composition root — the single place where all dependencies are wired.

This replaces the scattered module-level singletons with explicit
construction, making the dependency graph visible and testable.
"""

import asyncio
import logging

from agent.api.server import create_api
from agent.core.event_logger import EventLogger
from agent.core.messaging import (
    Messaging,
    NullMessaging,
    WXMessaging,
    WXMessagingConfig,
)
from agent.core.settings import Settings, get_settings
from agent.llm.agent import Agent
from agent.llm.factory import LLMFactory
from agent.llm.prompt_builder import SystemPromptBuilder
from agent.tools.command_executor import ContainerCommandExecutor
from agent.tools.skill_loader import SkillLoader
from agent.tools.tool_registry import ToolRegistry
from agent.tools.toolbox import register_default_tools

logger = logging.getLogger(__name__)


class Application:
    """
    Holds the fully-wired object graph.

    Created once in main(), passed to the Scheduler and API server.
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.event_queue: asyncio.Queue = asyncio.Queue()

        # Event logger
        self.event_logger = EventLogger(
            event_url=self.settings.event_api_url,
            event_api_key=self.settings.event_api_key,
        )

        # Tool infrastructure
        self.executor = ContainerCommandExecutor(
            container_name=self.settings.container_name,
            runtime=self.settings.container_runtime,
        )
        self.skill_loader = SkillLoader(self.settings.skills_dir)
        self.tool_registry = ToolRegistry(
            tool_timeout=self.settings.tool_timeout,
        )
        register_default_tools(self.tool_registry, self.executor, self.skill_loader)

        # LLM client
        self.llm_client = LLMFactory.create(
            url=self.settings.openai_base_url,
            model=self.settings.openai_model,
            api_key=self.settings.openai_api_key,
            event_logger=self.event_logger,
            tool_registry=self.tool_registry,
        )

        # Agent
        self.agent = Agent(self.llm_client)
        self.prompt_builder = SystemPromptBuilder(self.settings, self.skill_loader)

        # Messaging
        self.messaging: Messaging = self._create_messaging()

        # FastAPI
        self.api = create_api(self.event_queue)

    def _create_messaging(self) -> Messaging:
        """Create the appropriate messaging backend."""
        if self.settings.wechat_corpid and self.settings.wechat_corpsecret:
            config = WXMessagingConfig(
                corpid=self.settings.wechat_corpid,
                corpsecret=self.settings.wechat_corpsecret,
                agentid=self.settings.wechat_agentid,
                touser=self.settings.wechat_touser,
                token_refresh_interval=self.settings.wechat_token_refresh_interval,
            )
            return WXMessaging(config)
        return NullMessaging()
