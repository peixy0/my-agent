"""
System prompt builder (SRP).

Constructs the agent's system prompt from settings, skills, and runtime context.
Separated from Agent to keep each class focused on one responsibility.
"""

import platform
from pathlib import Path

from agent.core.settings import Settings
from agent.tools.skill_loader import SkillLoader


class SystemPromptBuilder:
    """Builds the system prompt for the agent."""

    def __init__(self, settings: Settings, skill_loader: SkillLoader):
        self._settings = settings
        self._skill_loader = skill_loader

    def build(self, previous_summary: str = "") -> str:
        """Build the full system prompt with current datetime and skills."""
        operating_system = platform.system()
        bootstrap_files = ["IDENTITY.md", "USER.md", "MEMORY.md", "CONTEXT.md"]

        bootstrap_context = ""
        for filename in bootstrap_files:
            try:
                with Path(f"{self._settings.workspace_dir}/{filename}").open() as f:
                    content = f.read()
                    if content:
                        bootstrap_context += f"# {filename}\n\n{content}\n\n"
            except FileNotFoundError:
                pass

        skill_summaries = self._skill_loader.discover_skills()
        skills_text = ""
        if skill_summaries:
            skills_text = "Available specialized skills:\n"
            for s in skill_summaries:
                skills_text += f"- {s.name}: {s.description}\n"
            skills_text += "\nUse the `use_skill` tool for detailed instructions."

        summary_section = ""
        if previous_summary:
            summary_section = f"""# Conversation Summary

The following is a compressed summary of the conversation history so far:

{previous_summary}
"""

        return f"""
You are an autonomous agent acting as a personal assistant.

**Host Environment:** {operating_system}
**Directory:** `/workspace`

You are provided with a set of tools and skills to help you with your tasks.
You can use them to interact with the world or guide yourself to perform actions.

# Skills

{skills_text}

# Workspace

Your working directory is `/workspace`.
Treat this directory as the single global workspace for file operations unless explicitly instructed otherwise.

{bootstrap_context}

{summary_section}

# Silent Replies

If you are woken up because of a heartbeat, and there is nothing that needs attention, respond with content ends with: NO_REPORT

Rules:
- System treats response ending with NO_REPORT as "no need to report" and will not send it to human user.
- NO_REPORT must be at the end
- Never append it to an actual response (never include NO_REPORT in real replies)
- Never wrap it in markdown or code blocks

Wrong: NO_REPORT There's nothing to report
Wrong: There's nothing to report... `NO_REPORT`
Wrong: "NO_REPORT"
Wrong: I need to bring this up with the user... NO_REPORT
Right: NO_REPORT
Right: Nothing needs human attention because... NO_REPORT
Right: Something happened...
"""
