import subprocess
from typing import Any, cast

import trafilatura
from ddgs import DDGS
from rich.console import Console
from rich.prompt import Confirm, Prompt

from .llm_base import LLMBase
from .settings import settings
from .skill_loader import SkillLoader


skill_loader = SkillLoader(settings.skills_dir)


def register_tools(llm_client: LLMBase, console: Console, whitelist: list[str]):
    """
    Registers the tools with the LLM client.

    Args:
        llm_client: The LLM client to register the tools with.
    """

    def approve_tool(tool: str, intention: str) -> tuple[bool, str]:
        if tool in whitelist:
            console.print(f"Request approval to {intention} \\[auto-approved]")
            return (True, "Auto-approved")

        if Confirm.ask(f"Request approval to {intention}", default=True):
            return (True, "User-approved")
        reason = Prompt.ask("Why")
        return (False, reason)

    @llm_client.register_function(
        {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute.",
                }
            },
            "required": ["command"],
        }
    )
    async def run_command(command: str) -> dict[str, Any]:
        """
        Executes a shell command.
        This tool can be used to explore the local filesystem when needed, e.g. use `ls -R` or `dir` to list files in a directory.
        """

        allow, reason = approve_tool(
            run_command.__name__, f"run command [bold]{command}[/bold]"
        )
        if not allow:
            return {
                "status": "cancelled",
                "reason": reason,
            }

        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True,
        )
        return {"status": "success", "output": result.stdout}

    @llm_client.register_function(
        {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query.",
                }
            },
            "required": ["query"],
        }
    )
    async def web_search(query: str) -> dict[str, Any]:
        """
        Performs a web search.
        """

        allow, reason = approve_tool(
            web_search.__name__, f"search web [bold]{query}[/bold]"
        )
        if not allow:
            return {
                "status": "cancelled",
                "reason": reason,
            }

        with cast(Any, DDGS()) as ddgs:
            results = [r for r in ddgs.text(query, max_results=7)]
            return {"status": "success", "results": results}

    @llm_client.register_function(
        {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The url to be fetched.",
                }
            },
            "required": ["url"],
        }
    )
    async def fetch(url: str) -> dict[str, Any]:
        """
        Performs a web page fetch.
        """

        allow, reason = approve_tool(fetch.__name__, f"fetch [bold]{url}[/bold]")
        if not allow:
            return {"status": "cancelled", "reason": reason}

        downloaded = trafilatura.fetch_url(url)
        output = trafilatura.extract(downloaded)
        return {"status": "success", "output": output}

    @llm_client.register_function(
        {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "The name of the file.",
                },
                "content": {
                    "type": "string",
                    "description": "The content to be written.",
                },
            },
            "required": ["filename", "content"],
        }
    )
    async def write_file(filename: str, content: str) -> dict[str, Any]:
        """
        Write content to a file.
        """

        allow, reason = approve_tool(
            write_file.__name__, f"write to file [bold]{filename}[/bold]"
        )
        if not allow:
            return {"status": "cancelled", "reason": reason}

        with open(filename, "w") as output:
            _ = output.write(content)
        return {"status": "success"}

    @llm_client.register_function(
        {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "The name of the file.",
                },
            },
            "required": ["filename"],
        }
    )
    async def read_file(filename: str) -> dict[str, Any]:
        """
        Read content from a file.
        """

        allow, reason = approve_tool(
            read_file.__name__, f"read from file [bold]{filename}[/bold]"
        )
        if not allow:
            return {"status": "cancelled", "reason": reason}

        with open(filename, "r") as input:
            content = input.read()
            return {"status": "success", "content": content}

    @llm_client.register_function(
        {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "The name of the file.",
                },
                "original": {
                    "type": "string",
                    "description": "The original content in the file to be replaced.",
                },
                "replaced": {
                    "type": "string",
                    "description": "The new content to be written over the replaced text.",
                },
            },
            "required": ["filename", "original", "replaced"],
        }
    )
    async def edit_file(filename: str, original: str, replaced: str) -> dict[str, Any]:
        """
        Edit content of a file.
        """

        allow, reason = approve_tool(
            edit_file.__name__, f"edit file [bold]{filename}[/bold]"
        )
        if not allow:
            return {"status": "cancelled", "reason": reason}

        with open(filename, "rw") as input:
            content = input.read()
            if content.find(original) == -1:
                return {
                    "status": "failed",
                    "error": f"original content not found, please use {read_file.__name__} tool to confirm the content.",
                }
            content = content.replace(original, replaced)
            _ = input.write(content)
            return {"status": "success"}

    @llm_client.register_function(
        {
            "type": "object",
            "properties": {
                "skill_name": {
                    "type": "string",
                    "description": "The name of the skill to use.",
                }
            },
            "required": ["skill_name"],
        }
    )
    async def use_skill(skill_name: str) -> dict[str, Any]:
        """
        Gain knowledge of a specific skill.
        Use this when you identify a relevant skill from your available skills list.
        """
        allow, reason = approve_tool(
            use_skill.__name__, f"use skill [bold]{skill_name}[/bold]"
        )
        if not allow:
            return {"status": "cancelled", "reason": reason}

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
