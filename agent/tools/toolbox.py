"""
Toolbox module containing tool functions for the agent.

All file and command operations are delegated to the Runtime,
which executes them inside the workspace container.
"""

import logging
from typing import Any, cast

import trafilatura
from ddgs import DDGS

from agent.core.messaging import Messaging
from agent.core.runtime import Runtime
from agent.core.settings import Settings
from agent.tools.skill_loader import SkillLoader
from agent.tools.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


def register_default_tools(
    registry: ToolRegistry,
    runtime: Runtime,
    skill_loader: SkillLoader,
    settings: Settings,
) -> None:
    """Declaratively register all default tools into the registry."""

    async def run_command(command: str) -> dict[str, Any]:
        """
        Executes a shell command in the workspace container.

        Use this tool to explore the filesystem, run scripts, or execute
        any shell command. The command runs in /workspace inside the container.
        """
        return await runtime.execute(command)

    async def web_search(query: str) -> dict[str, Any]:
        """
        Performs a web search using DuckDuckGo.

        Returns a list of search results with titles, URLs, and snippets.
        """
        try:
            proxy = None
            if settings.proxy:
                proxy = settings.proxy
            with cast(Any, DDGS(proxy=proxy, timeout=60)) as ddgs:  # pyright: ignore[reportCallIssue]
                results = [r for r in ddgs.text(query, max_results=7)]
                return {"status": "success", "results": results}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def fetch(url: str) -> dict[str, Any]:
        """
        Fetches and extracts the main content from a web page.

        Returns the extracted text content from the URL.
        """
        try:
            downloaded = trafilatura.fetch_url(url)
            output = trafilatura.extract(downloaded)
            return {"status": "success", "output": output}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def write_file(filename: str, content: str) -> dict[str, Any]:
        """
        Write content to a file in the workspace container.

        The filename should be relative to /workspace or an absolute path.
        Parent directories will be created if they don't exist.
        """
        return await runtime.write_file(filename, content)

    async def read_file(
        filename: str, start_line: int = 1, limit: int = 200
    ) -> dict[str, Any]:
        """
        Read content from a file in the workspace container.

        The filename should be relative to /workspace or an absolute path.
        Returns max 200 lines. Use start_line to read further.
        """
        return await runtime.read_file(filename, start_line=start_line, limit=limit)

    async def edit_file(filename: str, edits: list[dict[str, str]]) -> dict[str, Any]:
        """
        Surgically edit a file by replacing specific blocks of text. Use this for precise code modifications.

        Rules:
        1. SEARCH block must match the file exactly (including indentation).
        2. Provide just enough context in SEARCH to be unique.
        3. If multiple changes are needed, provide multiple edit blocks.
        """
        return await runtime.edit_file(filename, edits)

    async def use_skill(skill_name: str) -> dict[str, Any]:
        """
        Load instructions for a specialized skill.

        Use this when you identify a relevant skill from your available skills list.
        Skills provide detailed instructions for specific tasks.
        """
        skill = skill_loader.load_skill(skill_name)
        if not skill:
            return {"status": "error", "message": f"Skill '{skill_name}' not found"}
        return {
            "status": "success",
            "skill": {
                "name": skill.name,
                "skill_dir": skill.skill_dir,
                "description": skill.description,
                "instructions": skill.instructions,
            },
        }

    registry.register(
        run_command,
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

    registry.register(
        web_search,
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."}
            },
            "required": ["query"],
        },
    )

    registry.register(
        fetch,
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

    registry.register(
        write_file,
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

    registry.register(
        read_file,
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
                "limit": {
                    "type": "integer",
                    "description": "The maximum number of lines to read (default: 200).",
                },
            },
            "required": ["filename"],
        },
    )

    registry.register(
        edit_file,
        {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Path to the file (relative to /workspace or absolute).",
                },
                "edits": {
                    "type": "array",
                    "description": "A list of one or more search-and-replace operations to apply sequentially.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "search": {
                                "type": "string",
                                "description": "The exact snippet of code to look for. Must be a literal match, including whitespace and comments.",
                            },
                            "replace": {
                                "type": "string",
                                "description": "The new code to put in place of the search block.",
                            },
                        },
                        "required": ["search", "replace"],
                    },
                },
            },
            "required": ["filename", "edits"],
        },
    )

    registry.register(
        use_skill,
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


def register_human_input_tools(
    registry: ToolRegistry,
    messaging: Messaging,
    chat_id: str,
    message_id: str,
) -> None:
    """Register tools that are only available during human-input interactions."""

    async def add_reaction(emoji: str) -> dict[str, Any]:
        """
        React to the current message with an emoji.
        """
        try:
            await messaging.add_reaction(message_id, emoji)
            return {
                "status": "success",
                "message": f"Added reaction {emoji} to message",
            }
        except Exception as e:
            logger.error(f"Failed to add reaction {emoji}: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    registry.register(
        add_reaction,
        {
            "type": "object",
            "properties": {
                "emoji": {
                    "type": "string",
                    "enum": [
                        "OK",
                        "THUMBSUP",
                        "MUSCLE",
                        "LOL",
                        "THINKING",
                        "Shrug",
                        "Fire",
                        "Coffee",
                        "PARTY",
                        "CAKE",
                        "HEART",
                    ],
                    "description": "The emoji type to react with. OK, THUMBSUP, MUSCLE, LOL, THINKING, Shrug, Fire, Coffee, PARTY, CAKE, HEART",
                }
            },
            "required": ["emoji"],
        },
    )

    async def send_image(image_path: str) -> dict[str, Any]:
        """
        Send an image file to the user. Image file size must be under 10 MiB.
        """
        try:
            await messaging.send_image(chat_id, image_path)
            return {
                "status": "success",
                "message": f"Sent image {image_path} to user",
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    registry.register(
        send_image,
        {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Absolute path to the image file to send.",
                }
            },
            "required": ["image_path"],
        },
    )
