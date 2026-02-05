"""
Command executor module providing abstraction for executing commands.

Implements the Strategy pattern for command execution, enabling
the agent to execute commands in different environments (container, local).
"""

import asyncio
import logging
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Final, override

from agent.core.settings import settings

logger = logging.getLogger(__name__)


class CommandExecutor(ABC):
    """Abstract base class for command execution (Strategy pattern)."""

    @abstractmethod
    async def execute(self, command: str) -> dict[str, Any]:
        """Execute a command and return the result."""
        ...

    @abstractmethod
    async def read_file(
        self, filepath: str, start_line: int = 1, limit: int = 200
    ) -> dict[str, Any]:
        """Read content from a file with pagination."""
        ...

    @abstractmethod
    async def write_file(self, filepath: str, content: str) -> dict[str, Any]:
        """Write content to a file."""
        ...

    @abstractmethod
    async def edit_file(
        self, filepath: str, original: str, replaced: str
    ) -> dict[str, Any]:
        """Edit content in a file by replacing original with replaced."""
        ...


class ContainerCommandExecutor(CommandExecutor):
    """
    Executes commands inside a container.

    This executor delegates all operations to a running container,
    allowing the agent to work in an isolated workspace environment.
    """

    container_name: Final[str]
    runtime: Final[str]
    workdir: Final[str]

    def __init__(
        self,
        container_name: str,
        runtime: str = "podman",
        workdir: str = "/workspace",
    ):
        self.container_name = container_name
        self.runtime = runtime
        self.workdir = workdir
        self._validate_runtime()

    def _validate_runtime(self) -> None:
        """Validate that the container runtime is available."""
        if not shutil.which(self.runtime):
            raise RuntimeError(f"Container runtime '{self.runtime}' not found in PATH")

    async def _exec_in_container(
        self, command: str, timeout: int = settings.command_timeout
    ) -> tuple[str, str, int]:
        """Execute a command in the container and return stdout, stderr, returncode."""
        full_command = [
            self.runtime,
            "exec",
            "-w",
            self.workdir,
            self.container_name,
            "bash",
            "-c",
            command,
        ]

        logger.debug(f"Executing in container: {command}")

        process = await asyncio.create_subprocess_exec(
            *full_command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )
            return (
                stdout.decode("utf-8", errors="replace"),
                stderr.decode("utf-8", errors="replace"),
                process.returncode or 0,
            )
        except (asyncio.TimeoutError, TimeoutError):
            logger.warning(f"Command timed out after {timeout}s, detaching: {command}")
            raise

    @override
    async def execute(self, command: str) -> dict[str, Any]:
        """Execute a shell command in the container."""
        try:
            stdout, stderr, returncode = await self._exec_in_container(command)

            if returncode != 0:
                return {
                    "status": "error",
                    "returncode": returncode,
                    "stdout": stdout,
                    "stderr": stderr,
                }

            return {
                "status": "success",
                "stdout": stdout,
                "stderr": stderr,
            }

        except (asyncio.TimeoutError, TimeoutError):
            return {
                "status": "timeout",
                "message": "Command execution timed out.",
            }
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return {"status": "error", "message": str(e)}

    @override
    async def read_file(
        self, filepath: str, start_line: int = 1, limit: int = 200
    ) -> dict[str, Any]:
        """Read content from a file in the container with pagination."""
        escaped_path = filepath.replace("'", "'\"'\"'")

        # Get total lines first
        total_cmd = f"wc -l < '{escaped_path}'"
        stdout, stderr, returncode = await self._exec_in_container(total_cmd)

        if returncode != 0:
            return {"status": "error", "message": stderr.strip() or "File not found"}

        try:
            total_lines = int(stdout.strip())
        except ValueError:
            total_lines = 0

        # Read specific range using sed
        # sed -n 'start,end p'
        start = max(1, start_line)
        end = start + limit - 1
        read_cmd = f"sed -n '{start},{end}p' '{escaped_path}'"

        stdout, stderr, returncode = await self._exec_in_container(read_cmd)

        if returncode != 0:
            return {"status": "error", "message": stderr.strip()}

        content = stdout
        returned_lines = len(content.splitlines())

        return {
            "status": "success",
            "content": content,
            "total_lines": total_lines,
            "start_line": start,
            "returned_lines": returned_lines,
        }

    @override
    async def write_file(self, filepath: str, content: str) -> dict[str, Any]:
        """Write content to a file in the container."""
        # Use heredoc to write content safely
        escaped_path = filepath.replace("'", "'\"'\"'")
        # Ensure parent directory exists
        mkdir_cmd = f"mkdir -p \"$(dirname '{escaped_path}')\""
        _ = await self._exec_in_container(mkdir_cmd)

        # Use base64 encoding to safely transfer content
        import base64

        encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
        command = f"echo '{encoded}' | base64 -d > '{escaped_path}'"

        _, stderr, returncode = await self._exec_in_container(command)

        if returncode != 0:
            return {"status": "error", "message": stderr.strip()}

        return {"status": "success"}

    @override
    async def edit_file(
        self, filepath: str, original: str, replaced: str
    ) -> dict[str, Any]:
        """Edit content in a file by replacing original with replaced."""
        # First read the file
        read_result = await self.read_file(filepath)
        if read_result["status"] != "success":
            return read_result

        content: str = read_result["content"]

        if original not in content:
            return {
                "status": "error",
                "message": "Original content not found in file. Use read_file to verify content.",
            }

        # Replace and write back
        new_content = content.replace(original, replaced)
        return await self.write_file(filepath, new_content)


async def ensure_container_running(
    container_name: str,
    runtime: str = "podman",
    image: str = "sys-agent-workspace:latest",
    workspace_path: str = "./workspace",
) -> bool:
    """
    Ensure the workspace container is running.

    Returns True if container is running (or was started), False on error.
    """

    # Check if container exists and is running
    check_cmd = [runtime, "ps", "-q", "-f", f"name=^{container_name}$"]
    process = await asyncio.create_subprocess_exec(
        *check_cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await process.communicate()

    if stdout.strip():
        logger.info(f"Container '{container_name}' is already running")
        return True

    # Check if container exists but is stopped
    check_all_cmd = [runtime, "ps", "-aq", "-f", f"name=^{container_name}$"]
    process = await asyncio.create_subprocess_exec(
        *check_all_cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await process.communicate()

    if stdout.strip():
        # Container exists but stopped, start it
        logger.info(f"Starting stopped container '{container_name}'")
        start_cmd = [runtime, "start", container_name]
        process = await asyncio.create_subprocess_exec(*start_cmd)
        _ = await process.wait()
        return process.returncode == 0

    # Container doesn't exist, create it
    abs_workspace = Path(workspace_path).resolve()
    abs_workspace.mkdir(parents=True, exist_ok=True)

    logger.info(f"Creating container '{container_name}'")
    run_cmd = [
        runtime,
        "run",
        "-d",
        "--name",
        container_name,
        "-v",
        f"{abs_workspace}:/workspace",
        "-i",  # Keep stdin open for bash
        image,
    ]

    process = await asyncio.create_subprocess_exec(
        *run_cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await process.communicate()

    if process.returncode != 0:
        logger.error(f"Failed to create container: {stderr.decode()}")
        return False

    logger.info(f"Container '{container_name}' created and running")
    return True
