"""
System prompt builder (SRP).

Constructs the agent's system prompt from settings, skills, and runtime context.
Separated from Agent to keep each class focused on one responsibility.
"""

import platform
from dataclasses import dataclass
from pathlib import Path

from agent.core.settings import Settings
from agent.tools.skill import SkillLoader


@dataclass
class _CachedFile:
    content: str
    mtime: float


class SystemPromptBuilder:
    """Builds the system prompt for the agent."""

    def __init__(self, settings: Settings, skill: SkillLoader):
        self._settings = settings
        self._skill = skill
        self._file_cache: dict[str, _CachedFile] = {}

    def _load_file_cached(self, path: Path) -> str | None:
        """Return file content, using a cached value when mtime hasn't changed."""
        try:
            mtime = path.stat().st_mtime
            key = str(path)
            cached = self._file_cache.get(key)
            if cached is not None and cached.mtime == mtime:
                return cached.content
            content = path.read_text(encoding="utf-8")
            self._file_cache[key] = _CachedFile(content, mtime)
            return content or None
        except FileNotFoundError:
            return None

    def _load_workspace_files(self, files: list[str]) -> str:
        bootstrap_context = ""
        for filename in files:
            content = self._load_file_cached(Path(filename))
            if content:
                bootstrap_context += f"# {filename}\n\n{content}\n\n"
        return bootstrap_context

    def build(self) -> str:
        """Build the full system prompt with current datetime and skills."""
        operating_system = platform.system()
        bootstrap_files = ["IDENTITY.md", "USER.md", "MEMORY.md", "CONTEXT.md"]
        bootstrap_context = self._load_workspace_files(bootstrap_files)

        skill_summaries = self._skill.discover_skills()
        skills_text = ""
        if skill_summaries:
            skills_text = "Available specialized skills:\n"
            for s in skill_summaries:
                skills_text += f"- {s.name}: {s.description}\n"
            skills_text += "\nUse the `use_skill` tool for detailed instructions."

        return f"""
You are an autonomous agent acting as a personal assistant.

**Host Environment:** {operating_system}

You are provided with a set of tools and skills to help you with your tasks. Use them wisely and proactively to achieve the best results for the user.

# Skills

{skills_text}

# Workspace

Treat your current working directory as the single global workspace for file operations unless explicitly instructed otherwise.

{bootstrap_context}
"""

    def build_with_previous_summary(self, previous_summary: str) -> str:
        default_prompt = self.build()
        summary_section = ""
        if previous_summary:
            summary_section = f"""# Conversation Summary

The following is a compressed summary of the conversation history so far:

{previous_summary}
"""
        return f"""{default_prompt}

{summary_section}
"""

    def build_for_heartbeat(self) -> str:
        default_prompt = self.build()
        heartbeat_context = self._load_workspace_files(["HEARTBEAT.md"])
        return f"""{default_prompt}

{heartbeat_context}
"""

    def build_for_cron(self) -> str:
        default_prompt = self.build()
        cron_context = self._load_workspace_files(["CRON.md"])
        return f"""{default_prompt}

{cron_context}
"""
