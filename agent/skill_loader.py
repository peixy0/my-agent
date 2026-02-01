import logging
import re
from dataclasses import dataclass
from pathlib import Path


logger = logging.getLogger(__name__)


@dataclass
class SkillSummary:
    """Brief skill info for progressive disclosure."""

    name: str
    description: str


@dataclass
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
        self._cache: dict[str, Skill] = {}

    def discover_skills(self) -> list[SkillSummary]:
        """Return brief summaries of all available skills."""
        summaries: list[SkillSummary] = []
        if not self.skills_dir.exists():
            logger.warning(f"Skills directory {self.skills_dir} does not exist.")
            return summaries

        for skill_file in self.skills_dir.glob("*/SKILL.md"):
            try:
                content = skill_file.read_text(encoding="utf-8")
                data = self._parse_frontmatter(content)
                if data.get("name"):
                    summaries.append(
                        SkillSummary(
                            name=data["name"],
                            description=data.get("description", ""),
                        )
                    )
            except Exception as e:
                logger.error(f"Failed to parse skill at {skill_file}: {e}")

        return summaries

    def load_skill(self, name: str) -> Skill | None:
        """Load full skill instructions by name."""
        if name in self._cache:
            return self._cache[name]

        # Find the directory that contains a SKILL.md with this name
        for skill_file in self.skills_dir.glob("*/SKILL.md"):
            try:
                content = skill_file.read_text(encoding="utf-8")
                data = self._parse_frontmatter(content)
                if data.get("name") == name:
                    skill = Skill(
                        name=name,
                        skill_dir=str(skill_file.parent),
                        description=data.get("description", ""),
                        instructions=content,
                    )
                    self._cache[name] = skill
                    return skill
            except Exception as e:
                logger.error(f"Failed to load skill {name} from {skill_file}: {e}")

        return None

    def _parse_frontmatter(self, content: str) -> dict[str, str | list[str]]:
        """Simple regex-based YAML frontmatter parser."""
        match = re.search(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
        if not match:
            return {}

        yaml_text = match.group(1)
        data = {}
        for line in yaml_text.splitlines():
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            value = value.strip('"').strip("'")
            data[key] = value
        return data
