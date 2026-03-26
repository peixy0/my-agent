import platform
from dataclasses import dataclass
from pathlib import Path

from agent.tools.skill import SkillLoader


@dataclass
class _CachedFile:
    content: str
    mtime: float


class SystemPromptBuilder:
    """Builds the system prompt for the agent."""

    def __init__(self, skill: SkillLoader):
        self.skill = skill
        self.file_cache: dict[str, _CachedFile] = {}

    def _load_file_cached(self, path: Path) -> str | None:
        """Return file content, using a cached value when mtime hasn't changed."""
        try:
            mtime = path.stat().st_mtime
            key = str(path)
            cached = self.file_cache.get(key)
            if cached is not None and cached.mtime == mtime:
                return cached.content
            content = path.read_text(encoding="utf-8")
            self.file_cache[key] = _CachedFile(content, mtime)
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

    def _build_minimum(self) -> str:
        """Build a minimal system prompt without workspace context."""
        operating_system = platform.system()
        skill_summaries = self.skill.discover_skills()
        skills_text = ""
        if skill_summaries:
            skills_text = "# SKILLS\n\nAvailable specialized skills:\n\n"
            for s in skill_summaries:
                skills_text += f"- {s.name}: {s.description}\n"
            skills_text += "\nUse the `use_skill` tool for detailed instructions."

        return f"""**Host Environment:** {operating_system}

You are provided with a set of tools and skills to help you with your tasks. Use them wisely and proactively to achieve the best results for the user.

{skills_text}
"""

    def build(self) -> str:
        """Build the full system prompt with current datetime and skills."""
        bootstrap_files = ["IDENTITY.md", "USER.md", "MEMORY.md", "CONTEXT.md"]
        bootstrap_context = self._load_workspace_files(bootstrap_files)

        minimum_prompt = self._build_minimum()

        return f"""You are an autonomous agent acting as a personal assistant.

{minimum_prompt}

{bootstrap_context}
"""

    def build_with_conversation_summary(self, previous_summary: str) -> str:
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

    def build_with_context(self, context_files: list[str]) -> str:
        extra_context = self._load_workspace_files(context_files)
        return f"""{self.build()}

{extra_context}
"""

    def build_for_subagent(self, system_prompt: str) -> str:
        return f"""{system_prompt}

{self.build()}
"""
