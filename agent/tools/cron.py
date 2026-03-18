"""Cron job loader.

Discovers and loads cron job definitions from .cron/<group>/*.md files.
Each .md file must have a YAML frontmatter block with 'cron' (expression) and
optionally 'name' (task label, defaults to the filename stem).
The body of the file is the prompt sent to the agent when the job fires.

Example .cron/daily-check/status.md:
    ---
    cron: "0 9 * * *"
    name: "Morning status check"
    ---
    Review the workspace journal and summarise any incomplete tasks.
"""

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CronJobDef:
    """Definition of a single cron task loaded from a .md file."""

    task_name: str
    cron_expr: str
    prompt: str


def _parse_frontmatter(content: str) -> tuple[dict[str, str], str]:
    """Extract YAML frontmatter from markdown content.

    Returns (frontmatter_dict, body). If no frontmatter is found returns
    ({}, original_content).  Only simple 'key: value' pairs are supported;
    quoting is stripped from values.
    """
    match = re.match(
        r"^---[ \t]*\n(.*?)^---[ \t]*\n(.*)",
        content,
        re.DOTALL | re.MULTILINE,
    )
    if not match:
        return {}, content

    fm_text = match.group(1)
    body = match.group(2).strip()

    fm: dict[str, str] = {}
    for line in fm_text.splitlines():
        m = re.match(r'^(\w+)\s*:\s*"?(.*?)"?\s*$', line)
        if m:
            fm[m.group(1)] = m.group(2)
    return fm, body


class CronLoader:
    """Loads cron job definitions from a directory structure.

    Expected layout:
        <crons_dir>/
            <group-name>/
                task-a.md
                task-b.md
                ...
    """

    def __init__(self, crons_dir: str) -> None:
        self._crons_dir = Path(crons_dir)

    def list_groups(self) -> list[str]:
        """Return names of all job groups (subdirectories) in the crons directory."""
        if not self._crons_dir.is_dir():
            return []
        return sorted(p.name for p in self._crons_dir.iterdir() if p.is_dir())

    def load_group(self, group_name: str) -> list[CronJobDef]:
        """Load all task definitions from <crons_dir>/<group_name>/.

        Files without a 'cron' frontmatter key are silently skipped.
        Files are sorted lexicographically so load order is deterministic.
        """
        group_path = self._crons_dir / group_name
        if not group_path.is_dir():
            return []

        jobs: list[CronJobDef] = []
        for md_file in sorted(group_path.glob("*.md")):
            content = md_file.read_text(encoding="utf-8")
            fm, body = _parse_frontmatter(content)

            cron_expr = fm.get("cron", "").strip()
            if not cron_expr:
                continue

            task_name = fm.get("name", md_file.stem).strip()
            jobs.append(
                CronJobDef(task_name=task_name, cron_expr=cron_expr, prompt=body)
            )

        return jobs
