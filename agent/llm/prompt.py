"""
System prompt builder (SRP).

Constructs the agent's system prompt from settings, skills, and runtime context.
Separated from Agent to keep each class focused on one responsibility.
"""

import platform
from pathlib import Path

from agent.core.settings import Settings
from agent.tools.skill import SkillLoader


class SystemPromptBuilder:
    """Builds the system prompt for the agent."""

    def __init__(self, settings: Settings, skill: SkillLoader):
        self._settings = settings
        self._skill = skill

    def build(self) -> str:
        """Build the full system prompt with current datetime and skills."""
        operating_system = platform.system()
        bootstrap_files = ["IDENTITY.md", "USER.md", "MEMORY.md", "CONTEXT.md"]

        bootstrap_context = ""
        for filename in bootstrap_files:
            try:
                with Path(f"{self._settings.workspace_dir}/{filename}").open(
                    encoding="utf-8"
                ) as f:
                    content = f.read()
                    if content:
                        bootstrap_context += f"# {filename}\n\n{content}\n\n"
            except FileNotFoundError:
                pass

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
        return f"""{default_prompt}

# Silent Replies

If you are woken up because of a heartbeat, you will follow the protocol of a standard heartbeat session.
You will need to decide if you have anything new to report to the user since the last heartbeat.
If you don't have anything new to report, you will reply with your reasoning ending by a single token NO_REPORT to indicate that there is no need to report to the user.
If you do have something new to report, you will reply with the new information without including "NO_REPORT" in your response.

Rules:
- Use NO_REPORT only during system events
- System treats response ending with NO_REPORT as "no need to report" and will NOT send to human user.
- NO_REPORT must be at the end, appended after the reason why report is not needed
- Never append it to an actual response (never include NO_REPORT in real replies)
- Never wrap it in markdown or code blocks

Wrong: NO_REPORT There's nothing to report
Wrong: There's nothing to report... `NO_REPORT`
Wrong: "NO_REPORT"
Wrong: I need to bring this up with the user... NO_REPORT
Wrong: NO_REPORT
Right: Nothing new is happening because... NO_REPORT
Right: Something happened...
Right: I've descovered something user may be interested...
"""
