"""
Toolbox module containing tool functions for the agent.

All file and command operations are delegated to the CommandExecutor,
which executes them inside the workspace container.
"""

from typing import Any, cast

import trafilatura
from ddgs import DDGS

from agent.tools.skill_loader import SkillLoader
from agent.tools.command_executor import CommandExecutor, ContainerCommandExecutor
from agent.core.settings import settings


# Module-level executor instance (initialized lazily)
_executor: CommandExecutor | None = None


def get_executor() -> CommandExecutor:
    """Get or create the command executor instance."""
    global _executor
    if _executor is None:
        _executor = ContainerCommandExecutor(
            container_name=settings.container_name,
            runtime=settings.container_runtime,
        )
    return _executor


def set_executor(executor: CommandExecutor) -> None:
    """Set a custom executor (useful for testing)."""
    global _executor
    _executor = executor


async def run_command(command: str) -> dict[str, Any]:
    """
    Executes a shell command in the workspace container.
    
    Use this tool to explore the filesystem, run scripts, or execute
    any shell command. The command runs in /workspace inside the container.
    """
    executor = get_executor()
    return await executor.execute(command)


async def web_search(query: str) -> dict[str, Any]:
    """
    Performs a web search using DuckDuckGo.
    
    Returns a list of search results with titles, URLs, and snippets.
    """
    try:
        with cast(Any, DDGS()) as ddgs:
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
    executor = get_executor()
    return await executor.write_file(filename, content)


async def read_file(filename: str, start_line: int = 1) -> dict[str, Any]:
    """
    Read content from a file in the workspace container.
    
    The filename should be relative to /workspace or an absolute path.
    Returns max 200 lines. Use start_line to read further.
    """
    executor = get_executor()
    return await executor.read_file(filename, start_line=start_line)


async def edit_file(filename: str, original: str, replaced: str) -> dict[str, Any]:
    """
    Edit content of a file by replacing text.
    
    Finds the 'original' text in the file and replaces it with 'replaced'.
    Use read_file first to verify the exact content to replace.
    """
    executor = get_executor()
    return await executor.edit_file(filename, original, replaced)


async def use_skill(skill_name: str) -> dict[str, Any]:
    """
    Load instructions for a specialized skill.
    
    Use this when you identify a relevant skill from your available skills list.
    Skills provide detailed instructions for specific tasks.
    """
    skill_loader = SkillLoader(settings.skills_dir)
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
