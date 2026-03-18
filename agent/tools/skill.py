import logging
from dataclasses import dataclass
from pathlib import Path

from agent.tools.markdown import parse_frontmatter

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SkillSummary:
    """Brief skill info for progressive disclosure."""

    name: str
    description: str


@dataclass(frozen=True)
class Skill:
    """Full skill with complete instructions."""

    name: str
    skill_dir: str
    description: str
    instructions: str


class SkillLoader:
    """Discovers and loads skills from a directory."""

    def __init__(self, skills_dir: str = ".skills"):
        self.skills_dir = Path(skills_dir)

    def discover_skills(self) -> list[SkillSummary]:
        """Return brief summaries of all available skills."""
        summaries: list[SkillSummary] = []
        if not self.skills_dir.exists():
            logger.warning(f"Skills directory {self.skills_dir} does not exist.")
            return summaries

        for skill_file in self.skills_dir.glob("*/SKILL.md"):
            try:
                content = skill_file.read_text(encoding="utf-8")
                data, _ = parse_frontmatter(content)
                name = data.get("name")
                if name:
                    summaries.append(
                        SkillSummary(name=name, description=data.get("description", ""))
                    )
            except Exception as e:
                logger.error(f"Failed to parse skill at {skill_file}: {e}")

        return summaries

    def load_skill(self, name: str) -> Skill | None:
        """Load full skill instructions by name."""
        # Find the directory that contains a SKILL.md with this name
        for skill_file in self.skills_dir.glob("*/SKILL.md"):
            try:
                content = skill_file.read_text(encoding="utf-8")
                data, instructions = parse_frontmatter(content)
                if data.get("name") == name:
                    return Skill(
                        name=name,
                        skill_dir=str(skill_file.parent),
                        description=data.get("description", ""),
                        instructions=instructions,
                    )
            except Exception as e:
                logger.error(f"Failed to load skill {name} from {skill_file}: {e}")

        return None
