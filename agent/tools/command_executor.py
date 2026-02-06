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
        self, filepath: str, edits: list[dict[str, str]]
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

    async def _exec_in_container(self, command: str) -> tuple[str, str, int]:
        """
        Execute a command in the container and wait for it to complete.
        If you need long running command, consider running it in background and use `run_command` to check its status.
        Returns stdout, stderr, returncode.
        """
        full_command = [
            self.runtime,
            "exec",
            "-w",
            self.workdir,
            self.container_name,
            "bash",
            "-l",
            "-c",
            command,
        ]

        logger.debug(f"Executing in container: {command}")

        process = await asyncio.create_subprocess_exec(
            *full_command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await process.communicate()
        return (
            stdout.decode("utf-8", errors="replace"),
            stderr.decode("utf-8", errors="replace"),
            process.returncode or 0,
        )

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

        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return {"status": "error", "message": str(e)}

    async def _read_whole_file(self, filepath: str) -> dict[str, Any]:
        """Read entire content from a file in the container."""
        escaped_path = filepath.replace("'", "'\"'\"'")
        read_cmd = f"cat '{escaped_path}'"
        stdout, stderr, returncode = await self._exec_in_container(read_cmd)

        if returncode != 0:
            return {"status": "error", "message": stderr.strip()}

        content = stdout

        return {
            "status": "success",
            "content": content,
        }

    @override
    async def read_file(
        self, filepath: str, start_line: int = 1, limit: int = 200
    ) -> dict[str, Any]:
        """Read content from a file in the container with pagination."""
        escaped_path = filepath.replace("'", "'\"'\"'")

        # Get total lines first
        total_cmd = f"sed -n '$=' '{escaped_path}'"
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
        self, filepath: str, edits: list[dict[str, str]]
    ) -> dict[str, Any]:
        """
        Edit a file by replacing specific blocks of text.
        """
        read_result = await self._read_whole_file(filepath)
        if read_result["status"] != "success":
            return read_result

        content = read_result["content"]
        lines = content.splitlines(keepends=True)

        for edit in edits:
            search_block = edit["search"]
            replace_block = edit["replace"]

            # 1. Try exact match first
            if search_block in content:
                # Ensure it only occurs once to avoid ambiguity
                if content.count(search_block) > 1:
                    return {
                        "status": "error",
                        "message": f"Multiple occurrences of search block found in {filepath}. "
                        "Please include more surrounding context to make it unique.",
                    }
                content = content.replace(search_block, replace_block, 1)
                continue

            # 2. Try 'flexible' matching (ignoring minor whitespace/indentation differences)
            search_lines = search_block.splitlines()
            found_index = -1

            # Simple sliding window search
            for i in range(len(lines) - len(search_lines) + 1):
                window = lines[i : i + len(search_lines)]
                # Compare normalized versions (stripped of trailing whitespace)
                if all(
                    w.rstrip() == s.rstrip()
                    for w, s in zip(window, search_lines, strict=True)
                ):
                    if found_index != -1:
                        return {
                            "status": "error",
                            "message": "Search block is not unique (even with flexible matching).",
                        }
                    found_index = i

            if found_index != -1:
                # Replace lines at found_index
                new_lines = (
                    lines[:found_index]
                    + [
                        replace_block
                        + ("\n" if not replace_block.endswith("\n") else "")
                    ]
                    + lines[found_index + len(search_lines) :]
                )
                content = "".join(new_lines)
            else:
                # 3. If it fails, provide a helpful diff of why it failed
                return {
                    "status": "error",
                    "message": f"Could not find exact match for search block in {filepath}. "
                    "Ensure your SEARCH block is a literal copy of the file content.",
                }

        return await self.write_file(filepath, content)


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
