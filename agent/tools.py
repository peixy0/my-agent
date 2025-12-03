import subprocess
from typing import Any

import trafilatura
from ddgs import DDGS
from rich.console import Console
from rich.prompt import Confirm, Prompt

from .llm import LLMClient


def register_tools(llm_client: LLMClient, console: Console, whitelist: list[str]):
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

        with DDGS() as ddgs:  # pyright: ignore[reportGeneralTypeIssues, reportUnknownVariableType]
            results = [r for r in ddgs.text(query, max_results=7)]  # pyright: ignore[reportUnknownVariableType]
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
