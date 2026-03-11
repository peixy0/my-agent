"""
Toolbox module containing tool functions for the agent.

All file and command operations are delegated to the Runtime,
which executes them inside the workspace container.
"""

import asyncio
import base64
import logging
from pathlib import Path
from typing import Any, cast

import trafilatura
from ddgs import DDGS

from agent.core.runtime import Runtime
from agent.core.sender import MessageSender
from agent.core.settings import Settings
from agent.llm.types import ToolContent
from agent.tools.registry import ToolRegistry
from agent.tools.skill import SkillLoader

logger = logging.getLogger(__name__)


_MIME_TYPES: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


def register_default_tools(
    registry: ToolRegistry,
    runtime: Runtime,
    skill: SkillLoader,
    settings: Settings,
) -> None:
    """Declaratively register all default tools into the registry."""

    async def web_search(query: str) -> ToolContent:
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
                return ToolContent.from_dict("success", {"results": results})
        except Exception as e:
            return ToolContent.from_dict("error", {"message": str(e)})

    async def fetch(url: str) -> ToolContent:
        """
        Fetches and extracts the main content from a web page.

        Returns the extracted text content from the URL.
        """
        try:
            downloaded = trafilatura.fetch_url(url)
            output = trafilatura.extract(downloaded)
            return ToolContent.from_dict("success", {"output": output})
        except Exception as e:
            return ToolContent.from_dict("error", {"message": str(e)})

    async def run_command(command: str, timeout: int = 60) -> ToolContent:
        """
        Executes a shell command in the workspace container.

        Use this tool to explore the filesystem, run scripts, or execute
        any shell command. The command runs inside the container.
        """
        try:
            result = await asyncio.wait_for(runtime.execute(command), timeout=timeout)
            return ToolContent.from_dict("success", result)
        except asyncio.TimeoutError:
            return ToolContent.from_dict(
                "error", {"message": f"Command timed out after {timeout}s"}
            )
        except Exception as e:
            return ToolContent.from_dict("error", {"message": str(e)})

    async def write_file(filename: str, content: str) -> ToolContent:
        """
        Write content to a file in the workspace container.

        Parent directories will be created if they don't exist.
        """
        try:
            return ToolContent.from_dict(
                "success", await runtime.write_file(filename, content)
            )
        except Exception as e:
            return ToolContent.from_dict("error", {"message": str(e)})

    async def read_file(
        filename: str, start_line: int = 1, limit: int = 200
    ) -> ToolContent:
        """
        Read content from a file in the workspace container.

        Returns max 200 lines. Use start_line to read further.
        """
        try:
            return ToolContent.from_dict(
                "success",
                await runtime.read_file(filename, start_line=start_line, limit=limit),
            )
        except Exception as e:
            return ToolContent.from_dict("error", {"message": str(e)})

    async def edit_file(filename: str, edits: list[dict[str, str]]) -> ToolContent:
        """
        Surgically edit a file by replacing specific blocks of text. Use this for precise code modifications.

        Rules:
        1. SEARCH block must match the file exactly (including indentation).
        2. Provide just enough context in SEARCH to be unique.
        3. If multiple changes are needed, provide multiple edit blocks.
        """
        try:
            return ToolContent.from_dict(
                "success", await runtime.edit_file(filename, edits)
            )
        except Exception as e:
            return ToolContent.from_dict("error", {"message": str(e)})

    async def read_image(filename: str) -> ToolContent:
        """
        Read an image file.

        Supported formats: PNG, JPEG, GIF, WebP.
        The image is returned as a vision content block.
        """

        ext = Path(filename).suffix.lower()
        mime = _MIME_TYPES.get(ext)
        if mime is None:
            return ToolContent.from_dict(
                "error",
                {
                    "message": f"Unsupported image format '{ext}'. Supported: {', '.join(_MIME_TYPES)}"
                },
            )
        try:
            data = await runtime.read_file_internal(filename)
            if len(data) > settings.max_image_size_bytes:
                mb = settings.max_image_size_bytes / (1024 * 1024)
                return ToolContent.from_dict(
                    "error",
                    {
                        "message": f"Image exceeds size limit of {mb:.0f} MB ({len(data)} bytes)."
                    },
                )
            b64 = base64.b64encode(data).decode("ascii")
            return ToolContent.from_blocks(
                [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                        "detail": "low",
                    }
                ]
            )
        except Exception as e:
            return ToolContent.from_dict("error", {"message": str(e)})

    async def use_skill(skill_name: str) -> ToolContent:
        """
        Load instructions for a specialized skill.

        Use this when you identify a relevant skill from your available skills list.
        Skills provide detailed instructions for specific tasks.
        """
        loaded_skill = skill.load_skill(skill_name)
        if not loaded_skill:
            return ToolContent.from_dict(
                "error", {"message": f"Skill '{skill_name}' not found"}
            )
        return ToolContent.from_dict(
            "success",
            {
                "skill": {
                    "name": loaded_skill.name,
                    "skill_dir": loaded_skill.skill_dir,
                    "description": loaded_skill.description,
                    "instructions": loaded_skill.instructions,
                },
            },
        )

    if settings.web_tools_enabled:
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
        run_command,
        {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute in the workspace container.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default: 60).",
                },
            },
            "required": ["command"],
        },
    )

    registry.register(
        write_file,
        {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Path to the file.",
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
                    "description": "Path to the file.",
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
                    "description": "Path to the file.",
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

    if settings.vision_support:
        registry.register(
            read_image,
            {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Path to the image file. Supported formats: PNG, JPEG, GIF, WebP.",
                    }
                },
                "required": ["filename"],
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
    sender: MessageSender,
) -> None:
    """Register tools that are only available during human-input interactions."""

    async def add_reaction(emoji: str) -> ToolContent:
        """
        React to the current message with an emoji.
        """
        try:
            await sender.react(emoji)
            return ToolContent.from_dict(
                "success", {"message": f"Added reaction {emoji} to message"}
            )
        except Exception as e:
            logger.error(f"Failed to add reaction {emoji}: {e}", exc_info=True)
            return ToolContent.from_dict("error", {"message": str(e)})

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

    async def send_image(image_path: str) -> ToolContent:
        """
        Send an image file to the user. Image file size must be under 10 MiB.
        """
        try:
            await sender.send_image(image_path)
            return ToolContent.from_dict(
                "success",
                {
                    "message": f"Sent image {image_path} to user",
                },
            )
        except Exception as e:
            return ToolContent.from_dict("error", {"message": str(e)})

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

    async def send_file(file_path: str) -> ToolContent:
        """
        Send a file to the user. File size must be under 20 MiB.
        Send only when explicitly asked.
        """
        try:
            await sender.send_file(file_path)
            return ToolContent.from_dict(
                "success",
                {
                    "message": f"Sent file {file_path} to user",
                },
            )
        except Exception as e:
            return ToolContent.from_dict("error", {"message": str(e)})

    registry.register(
        send_file,
        {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute path to the file to send.",
                }
            },
            "required": ["file_path"],
        },
    )
