"""
Agent module containing the core agent logic.

The Agent class orchestrates LLM interactions, tool registration,
and context management for the autonomous agent.
"""

import asyncio
import datetime
import logging
import platform
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final

from agent.core.event_logger import EventLogger
from agent.core.settings import Settings, settings
from agent.llm.base import LLMBase
from agent.tools import toolbox
from agent.tools.skill_loader import SkillLoader

logger = logging.getLogger(__name__)


class Agent:
    """
    Core agent class that manages LLM interactions and tool execution.

    The agent runs on the host machine and delegates command/file operations
    to a workspace container via the toolbox functions.
    """

    llm_client: Final[LLMBase]

    event_logger: Final[EventLogger | None]
    settings: Final[Settings]
    messages: list[dict[str, str]]
    system_prompt: str

    def __init__(
        self,
        llm_client: LLMBase,
        event_logger: EventLogger | None = None,
        agent_settings: Settings | None = None,
    ):
        self.llm_client = llm_client

        self.event_logger = event_logger
        self.settings = agent_settings or settings
        self.messages = []
        self._register_default_tools()
        self.system_prompt = ""  # Initialized in initialize_system_prompt
        self.initialize_system_prompt()

    def _register_default_tools(self) -> None:
        """Register the default tools from the toolbox."""

        def register(func: Callable[..., Any], schema: dict[str, Any]) -> None:
            """Helper to register a tool with logging wrapper."""

            async def wrapped_tool(**kwargs: Any) -> Any:
                tool_name = func.__name__
                logger.info(f"Executing tool: {tool_name} with args: {kwargs}")

                try:
                    result = await asyncio.wait_for(
                        func(**kwargs), timeout=settings.tool_timeout
                    )
                    logger.info(f"Tool {tool_name} completed successfully")

                    if self.event_logger:
                        await self.event_logger.log_tool_use(tool_name, kwargs, result)

                    return result
                except asyncio.TimeoutError:
                    logger.error(
                        f"Tool {tool_name} timed out after {settings.tool_timeout}s"
                    )
                    error_result = {
                        "status": "error",
                        "message": f"Tool {tool_name} timed out after {settings.tool_timeout}s",
                    }

                    if self.event_logger:
                        await self.event_logger.log_tool_use(
                            tool_name, kwargs, error_result
                        )

                    return error_result
                except Exception as e:
                    logger.error(f"Tool {tool_name} failed: {e}")
                    error_result = {"status": "error", "message": str(e)}

                    if self.event_logger:
                        await self.event_logger.log_tool_use(
                            tool_name, kwargs, error_result
                        )

                    return error_result

            wrapped_tool.__name__ = func.__name__
            wrapped_tool.__doc__ = func.__doc__

            self.llm_client.functions[func.__name__] = {
                "name": func.__name__,
                "description": (func.__doc__ or "").strip(),
                "parameters": schema,
            }
            self.llm_client.handlers[func.__name__] = wrapped_tool

        # Register tools with their parameter schemas
        register(
            toolbox.run_command,
            {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute in the workspace container.",
                    }
                },
                "required": ["command"],
            },
        )

        register(
            toolbox.web_search,
            {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."}
                },
                "required": ["query"],
            },
        )

        register(
            toolbox.fetch,
            {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL of the web page to fetch.",
                    }
                },
                "required": ["url"],
            },
        )

        register(
            toolbox.write_file,
            {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Path to the file (relative to /workspace or absolute).",
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write.",
                    },
                },
                "required": ["filename", "content"],
            },
        )

        register(
            toolbox.read_file,
            {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Path to the file (relative to /workspace or absolute).",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "The line number to start reading from (default: 1). Use this for pagination.",
                    },
                },
                "required": ["filename"],
            },
        )

        register(
            toolbox.edit_file,
            {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Path to the file (relative to /workspace or absolute).",
                    },
                    "original": {
                        "type": "string",
                        "description": "The exact text to find and replace.",
                    },
                    "replaced": {
                        "type": "string",
                        "description": "The replacement text.",
                    },
                },
                "required": ["filename", "original", "replaced"],
            },
        )

        register(
            toolbox.use_skill,
            {
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "The name of the skill to load.",
                    }
                },
                "required": ["skill_name"],
            },
        )

    def _load_context_file(self, filepath: str, default_content: str = "") -> str:
        """Load content from a context file on the host filesystem."""
        path = Path(filepath)
        try:
            with path.open() as f:
                content = f.read().strip()
                return content if content else default_content
        except FileNotFoundError:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("w") as f:
                    _ = f.write(default_content)
                logger.info(f"Created context file: {filepath}")
                return default_content

            except Exception as e:
                logger.warning(f"Could not create context file {filepath}: {e}")
                return default_content
        except Exception as e:
            logger.warning(f"Error reading context file {filepath}: {e}")
            return default_content

    def initialize_system_prompt(self) -> None:
        """Initialize the system prompt with context and instructions."""
        now = datetime.datetime.now().astimezone()
        current_datetime = now.strftime("%Y-%m-%d %H:%M:%S %Z%z")
        current_date = now.strftime("%Y-%m-%d")
        operating_system = platform.system()

        # Load context files from host
        context = self._load_context_file(
            self.settings.context_file,
            "This is your persistent context. Record important information here.",
        )
        todo = self._load_context_file(
            self.settings.todo_file,
            "# TODO\n\n- Explore your environment\n- Keep a daily journal",
        )

        # Load skills
        skill_loader = SkillLoader(self.settings.skills_dir)
        skill_summaries = skill_loader.discover_skills()
        skills_text = ""
        if skill_summaries:
            skills_text = "Available specialized skills:\n"
            for s in skill_summaries:
                skills_text += f"- {s.name}: {s.description}\n"
            skills_text += "\nUse the `use_skill` tool for detailed instructions."

        self.system_prompt = f"""You are an autonomous agent. Current datetime: {current_datetime}. Host OS: {operating_system}.
You wake up periodically to perform tasks.

## Your Workspace Files
You cwd is /workspace folder. Please organize your files in this folder.
The workspace is persisted between runs. Maintain these files as your long-term memory between wakeups.

### /workspace/CONTEXT.md
{context}
Instructions: Update this file frequently. Refactor and summarize instead of just appending. Keep it relevant to your long-term goals.

### /workspace/TODO.md
{todo}
Instructions: Keep this list active and manageable. Mark completed tasks, remove outdated ones, and prioritize new findings.

## Daily Journal
Keep a daily journal at `/workspace/journal/{current_date}.md`. Document what you did, learned, and your next steps.

## Goals
1. Work on your TODO list.
2. Maintain CONTEXT.md and TODO.md: iterate and refine them frequently to keep them concise and relatively short. Do not append indefinitely; refactor when they grow too large.
3. Keep your daily journal updated.
4. Be autonomous - find ways to improve and discover new information and opportunities.
5. Learn, explore, and expand your knowledge base.

{skills_text}

Use your tools extensively. Be proactive and productive."""

        self.messages = [{"role": "system", "content": self.system_prompt}]

    async def run(self, user_prompt: str) -> None:
        """Run a single turn of the agent conversation."""
        self.messages.append({"role": "user", "content": user_prompt})
        response = ""
        try:
            response = await self.llm_client.chat(self.messages)

            if self.event_logger:
                await self.event_logger.log_llm_response(response)
        except Exception as e:
            logger.error(f"Error in agent run: {e}")
            response = f"Error: {e}"

        finally:
            self.messages.append({"role": "assistant", "content": response})

    def clear_history(self) -> None:
        """Clear conversation history, keeping only the system prompt."""
        self.messages = [{"role": "system", "content": self.system_prompt}]
