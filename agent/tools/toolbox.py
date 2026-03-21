"""
Toolbox module containing tool functions for the agent.

All file and command operations are delegated to the Runtime,
which executes them inside the workspace container.
"""

import asyncio
import base64
import logging
import shlex
from pathlib import Path
from typing import Any, cast

import trafilatura
from ddgs import DDGS

from agent.core.runtime import Runtime
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
            proxy = settings.web_search_proxy if settings.web_search_proxy else None

            def _do_search() -> list[Any]:
                with cast(Any, DDGS(proxy=proxy, timeout=60)) as ddgs:  # pyright: ignore[reportCallIssue]
                    return [
                        r for r in ddgs.text(query, max_results=7, backend="google")
                    ]

            results = await asyncio.to_thread(_do_search)
            return ToolContent.from_dict("success", {"results": results})
        except Exception as e:
            return ToolContent.from_dict("error", {"message": str(e)})

    async def fetch(url: str) -> ToolContent:
        """
        Fetches and extracts the main content from a web page.

        Returns the extracted text content from the URL.
        """
        try:
            downloaded = await asyncio.to_thread(trafilatura.fetch_url, url)
            output = await asyncio.to_thread(trafilatura.extract, downloaded)
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
        filename: str, start_line: int = 1, limit: int = 500
    ) -> ToolContent:
        """
        Read content from a file in the workspace container.

        Returns max 500 lines. Use start_line to read further.
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

    async def grep(
        pattern: str,
        path: str = ".",
        surrounding_lines: int = 2,
        include: str = "",
        case_sensitive: bool = True,
    ) -> ToolContent:
        """
        Search for a regex pattern across files in the workspace container.

        Returns matching lines with file path and line number.
        Use include to restrict to specific file types (e.g. "*.py").
        """
        try:
            args = [
                "grep",
                "-rn",
                "-a",
                "--color=never",
                "-E",
                f"-C{surrounding_lines}",
            ]
            if not case_sensitive:
                args.append("-i")
            if include:
                args.append(f"--include={shlex.quote(include)}")
            args.append(shlex.quote(pattern))
            args.append(shlex.quote(path))
            result = await runtime.execute(" ".join(args))
            return ToolContent.from_dict("success", result)
        except Exception as e:
            return ToolContent.from_dict("error", {"message": str(e)})

    async def glob(pattern: str) -> ToolContent:
        """
        List files in the workspace container matching a glob pattern.

        Supports recursive patterns with ** (e.g. "src/**/*.py").
        Returns a sorted list of matching paths.
        """
        try:
            python_code = "import glob, sys; print('\\n'.join(sorted(glob.glob(sys.argv[1], recursive=True))))"
            command = f"python3 -c {shlex.quote(python_code)} {shlex.quote(pattern)}"
            result = await runtime.execute(command)
            return ToolContent.from_dict("success", result)
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
            data = await runtime.read_raw_bytes(filename)
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
                    "description": "The maximum number of lines to read (default: 500).",
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

    registry.register(
        grep,
        {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The regex pattern to search for.",
                },
                "path": {
                    "type": "string",
                    "description": 'Directory or file path to search in (default: ".").',
                },
                "surrounding_lines": {
                    "type": "integer",
                    "description": "Number of surrounding lines to include for context (default: 2).",
                },
                "include": {
                    "type": "string",
                    "description": 'Restrict search to files matching this glob pattern (e.g. "*.py").',
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether the search is case-sensitive (default: true).",
                },
            },
            "required": ["pattern"],
        },
    )

    registry.register(
        glob,
        {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": 'Glob pattern to match files (e.g. "**/*.py", "src/*.ts"). Supports ** for recursive matching.',
                }
            },
            "required": ["pattern"],
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
