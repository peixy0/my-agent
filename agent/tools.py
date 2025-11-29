import subprocess
from typing import Any

import trafilatura
from ddgs import DDGS
from rich.prompt import Confirm, Prompt

from .llm import LLMClient


def register_tools(llm_client: LLMClient):
    """
    Registers the tools with the LLM client.

    Args:
        llm_client: The LLM client to register the tools with.
    """

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
    async def run_command(command: str) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """
        Executes a shell command.
        """

        if not Confirm.ask(
            f"Agent wants to run command: [bold]{command}[/bold]", default=True
        ):
            reason = Prompt.ask("Why")
            return {"status": "cancelled", "reason": reason, "tool": "run_command"}

        try:
            result = subprocess.run(
                command,
                shell=True,
                check=True,
                capture_output=True,
                text=True,
            )
            return {"status": "success", "output": result.stdout, "tool": "run_command"}
        except subprocess.CalledProcessError as e:
            return {
                "status": "error",
                "message": f"Command failed with error: {e.stderr}",
                "tool": "run_command",
            }

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
    async def web_search(query: str) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """
        Performs a web search.
        """

        if not Confirm.ask(
            f"Agent wants to perform web search: [bold]{query}[/bold]", default=True
        ):
            reason = Prompt.ask("Why")
            return {"status": "cancelled", "reason": reason, "tool": "web_search"}

        with DDGS() as ddgs:  # pyright: ignore[reportGeneralTypeIssues, reportUnknownVariableType]
            results = [r for r in ddgs.text(query, max_results=7)]  # pyright: ignore[reportUnknownVariableType]
            return {"status": "success", "results": results, "tool": "web_search"}

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
    async def fetch(url: str) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """
        Performs a web page fetch.
        """

        if not Confirm.ask(
            f"Agent wants to fetch web page: [bold]{url}[/bold]", default=True
        ):
            reason = Prompt.ask("Why")
            return {"status": "cancelled", "reason": reason, "tool": "fetch"}

        downloaded = trafilatura.fetch_url(url)
        output = trafilatura.extract(downloaded)
        return {"status": "success", "output": output, "tool": "fetch"}
