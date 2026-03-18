"""Cron job loader.

Discovers and loads cron job definitions from .cron/<job>/*.md files.
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

from dataclasses import dataclass
from pathlib import Path

from agent.tools.markdown import parse_frontmatter


@dataclass
class CronJobDef:
    """Definition of a single cron task loaded from a .md file."""

    task_name: str
    cron_expr: str
    prompt: str


class CronLoader:
    """Loads cron job definitions from a directory structure.

    Expected layout:
        <crons_dir>/
            <job-name>/
                task-a.md
                task-b.md
                ...
    """

    def __init__(self, crons_dir: str) -> None:
        self.crons_dir = Path(crons_dir)

    def list_jobs(self) -> list[str]:
        """Return names of all job groups (subdirectories) in the crons directory."""
        if not self.crons_dir.is_dir():
            return []
        return sorted(p.name for p in self.crons_dir.iterdir() if p.is_dir())

    def load_job(self, job_name: str) -> list[CronJobDef]:
        """Load all task definitions from <crons_dir>/<job_name>/.

        Files without a 'cron' frontmatter key are silently skipped.
        Files are sorted lexicographically so load order is deterministic.
        """
        job_path = self.crons_dir / job_name
        if not job_path.is_dir():
            return []

        jobs: list[CronJobDef] = []
        for md_file in sorted(job_path.glob("*.md")):
            content = md_file.read_text(encoding="utf-8")
            fm, body = parse_frontmatter(content)

            cron_expr = fm.get("cron", "").strip()
            if not cron_expr:
                continue

            task_name = fm.get("name", md_file.stem).strip()
            jobs.append(
                CronJobDef(task_name=task_name, cron_expr=cron_expr, prompt=body)
            )

        return jobs
