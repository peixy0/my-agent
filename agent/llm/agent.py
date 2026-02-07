"""
Agent module containing the core agent logic.

The Agent class orchestrates LLM interactions, tool registration,
and context management for the autonomous agent.
"""

import asyncio
import datetime
import json
import logging
import platform
from collections.abc import Callable
from typing import Any, Final

from jsonschema import ValidationError, validate

from agent.core.settings import Settings
from agent.llm.base import LLMBase
from agent.tools import toolbox
from agent.tools.skill_loader import SkillLoader

logger = logging.getLogger(__name__)


class Agent:
    """
    Core agent class that manages LLM interactions and tool execution.

    The agent runs on the host machine and delegates command/file operations
    to a workspace container via the toolbox functions.
    """

    llm_client: Final[LLMBase]

    settings: Final[Settings]
    messages: list[dict[str, str]]
    system_prompt: str

    def __init__(
        self,
        llm_client: LLMBase,
        agent_settings: Settings,
    ):
        self.llm_client = llm_client
        self.settings = agent_settings
        self.messages = []
        self._register_default_tools()
        self.system_prompt = ""

    def _register_default_tools(self) -> None:
        """Register the default tools from the toolbox."""

        def register(func: Callable[..., Any], schema: dict[str, Any]) -> None:
            """Helper to register a tool with logging wrapper."""

            async def wrapped_tool(**kwargs: Any) -> Any:
                tool_name = func.__name__
                try:
                    result = await asyncio.wait_for(
                        func(**kwargs), timeout=self.settings.tool_timeout
                    )
                    return result
                except asyncio.TimeoutError:
                    error_result = {
                        "status": "error",
                        "message": f"Tool {tool_name} timed out after {self.settings.tool_timeout}s",
                    }
                    return error_result
                except Exception as e:
                    error_result = {"status": "error", "message": str(e)}
                    return error_result

            wrapped_tool.__name__ = func.__name__
            wrapped_tool.__doc__ = func.__doc__

            self.llm_client.functions[func.__name__] = {
                "name": func.__name__,
                "description": (func.__doc__ or "").strip(),
                "parameters": schema,
            }
            self.llm_client.handlers[func.__name__] = wrapped_tool

        # Register tools with their parameter schemas
        register(
            toolbox.run_command,
            {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute in the workspace container.",
                    }
                },
                "required": ["command"],
            },
        )

        register(
            toolbox.web_search,
            {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."}
                },
                "required": ["query"],
            },
        )

        register(
            toolbox.fetch,
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

        register(
            toolbox.write_file,
            {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Path to the file (relative to /workspace or absolute).",
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write.",
                    },
                },
                "required": ["filename", "content"],
            },
        )

        register(
            toolbox.read_file,
            {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Path to the file (relative to /workspace or absolute).",
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

        register(
            toolbox.edit_file,
            {
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "Path to the file (relative to /workspace or absolute).",
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
                "required": ["filepath", "edits"],
            },
        )

        register(
            toolbox.use_skill,
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

    def initialize_system_prompt(self) -> None:
        """Initialize the system prompt with context and instructions."""
        now = datetime.datetime.now().astimezone()
        current_datetime = now.strftime("%Y-%m-%d %H:%M:%S %Z%z")
        current_date = now.strftime("%Y-%m-%d")
        operating_system = platform.system()

        # Load skills
        skill_loader = SkillLoader(self.settings.skills_dir)
        skill_summaries = skill_loader.discover_skills()
        skills_text = ""
        if skill_summaries:
            skills_text = "Available specialized skills:\n"
            for s in skill_summaries:
                skills_text += f"- {s.name}: {s.description}\n"
            skills_text += "\nUse the `use_skill` tool for detailed instructions."

        self.system_prompt = f"""You are an autonomous agent. Current datetime: {current_datetime}. Host OS: {operating_system}.
You wake up periodically to perform tasks.

## OPERATIONAL PHILOSOPHY: THE "SILENT GUARDIAN"
Your primary goal is to work autonomously without disturbing the user.
- **Internal Monologue:** You may think and log as much as needed in your files.
- **External Communication:** Only report if there is a significant update, a completed goal, or a critical blocker.
- **If nothing significant has changed, do NOT report.** A successful session is one where you make progress silently.
- **Be autonomous:** find ways to improve and discover new information and opportunities.
- **Use your tools extensively:** Be proactive and productive.
- **Learn, explore, and expand your knowledge base.**

## YOUR WORKSPACE (Persistent Memory)
Current working directory: `/workspace`

### 1. /workspace/CONTEXT.md (The "Map")
- Purpose: High-level goals and project state.
- Instruction: Refactor and summarize frequently. Do not let it grow into a messy log.

### 2. /workspace/TODO.md (The "Queue")
- Purpose: Active tasks and priorities.
- Instruction: Update after every session. Prioritize new discoveries.

### 3. /workspace/USER.md (The Interest Profile)
- **Purpose:** The source of truth for what the user wants you to monitor, research, or ignore.
- **Structure:** Use a structured format (e.g., Markdown Task List or Table) containing:
  - `[Interest Topic]`
  - `(Priority: 1-5)`
  - `> User Feedback/Context`
- **The Feedback Loop (CRITICAL):**
  1. **READ:** Every wakeup, check this file for new comments or edits made by the User.
  2. **ADJUST:** If the User adds a comment like "Not interested in this anymore" or "Focus on the financial aspect," you must strictly update the Priority or scope of that interest immediately.
  3. **CURATE:** Proactively add new related topics you think the User might like based on the existing list, but mark them as `(New/Pending Review)`.
  4. **CLEAN:** Once you have processed a specific user comment and updated your internal logic/context, you may simplify the comment or merge it into the topic description to keep the file readable.

### 4. /workspace/TRACK.md (The "Monitor")
- Purpose: Tracking external data (News, Website changes, API statuses, etc.).
- Structure: Record the "Last Known State" and "Timestamp" for everything you monitor.
- Logic: Compare current findings with the data in this file. If the data is the same, simply update the "Last Checked" timestamp. If the data has changed, record the "Delta" (the difference) and evaluate if it's worth a report.

### 5. /workspace/journal/{current_date}.md (The "Logs")
- Purpose: Internal audit trail.
- This is where you log ALL activity, thoughts, and minor updates.
- This file is for YOU. It effectively acts as your internal monologue/logs.
- **Format:** Chronological Append-only.
- **Structure:**
  - `## [HH:mm]`
  - **Action:** Briefly describe what you did.
  - **Outcome:** Success, Failure, or "No change detected."
- **Note:** Do not rewrite previous entries. Just append the new timestamped section.

### 5. /workspace/tmp/ (The "Trash")
- Purpose: Temporary files.
- This is where you store temporary files that you no longer need.
- Delete them when you're done.

## REPORTING CRITERIA
You must only report if:
1. **Significant Delta:** Information in `TRACK.md` changed in a way that impacts your primary goals.
2. **Task Completion:** A major item in `TODO.md` was finished.
3. **Action Required:** You are stuck and require human intervention (rare).
4. **Insight:** You discovered a new opportunity or risk that wasn't in your instructions.

*Note: Routine checks, minor file cleanup, or "still working" status updates are NOT reportable.*

## EXECUTION STEPS
1. **RECON:** Read `CONTEXT.md`, `TODO.md`, and `TRACK.md`.
2. **PROFILE SYNC:** Read `USER.md`. Parse any new comments from the User. Update your internal search parameters and the `USER.md` structure accordingly.
3. **MONITOR:** Check external websites/sources listed in `TRACK.md`. Update the file with new data.
4. **WORK:** Execute the tasks in your `TODO.md`.
5. **LOG:** Update your Journal and refine your **RECON** files.

{skills_text}
"""

        self.messages = [{"role": "system", "content": self.system_prompt}]

    async def run(
        self,
        user_prompt: str,
        response_schema: dict[str, Any],
        max_iterations: int = 80,
    ) -> dict[str, Any]:
        """Run a single turn of the agent conversation."""

        self.messages.append({"role": "user", "content": user_prompt})

        while True:
            response = await self.llm_client.chat(self.messages, max_iterations)

            try:
                if response.startswith("```json"):
                    response = response[7:-3]
                elif response.endswith("```"):
                    response = response[:-3]

                data = json.loads(response)
                validate(instance=data, schema=response_schema)
                self.messages.append({"role": "assistant", "content": response})
                return data
            except (json.JSONDecodeError, ValidationError) as e:
                logger.warning(f"Response validation failed: {e}")
                self.messages.append({"role": "assistant", "content": response})
                self.messages.append(
                    {
                        "role": "user",
                        "content": f"Your previous response failed validation: {e}. Please correct it and respond only with valid JSON matching the schema.",
                    }
                )

    def clear_history(self) -> None:
        """Clear conversation history, keeping only the system prompt."""
        self.messages = [{"role": "system", "content": self.system_prompt}]
