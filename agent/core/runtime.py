"""
Command runtime module providing abstraction for executing commands.

Implements the Strategy pattern for command execution, enabling
the agent to execute commands in different environments (container, local).
"""

import asyncio
import base64
import logging
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Final, override

logger = logging.getLogger(__name__)


class AgentRuntimeException(Exception):
    """Exception for agent runtime errors."""

    pass


class Runtime(ABC):
    """Abstract base class for command execution (Strategy pattern)."""

    @abstractmethod
    async def execute(self, command: str) -> dict[str, Any]:
        """Execute a command and return stdout, stderr, and return_code.

        Raises AgentRuntimeException if the process cannot be spawned.
        """
        ...

    @abstractmethod
    async def read_file_internal(self, filename: str) -> bytes:
        """Read the full raw content of a file. Raises AgentRuntimeException on error."""
        ...

    @abstractmethod
    async def read_file(
        self, filename: str, start_line: int = 1, limit: int = 500
    ) -> dict[str, Any]:
        """Read a paginated slice of a file. Raises AgentRuntimeException on error."""
        ...

    @abstractmethod
    async def write_file(self, filename: str, content: str) -> dict[str, Any]:
        """Write content to a file. Raises AgentRuntimeException on error."""
        ...

    @abstractmethod
    async def edit_file(
        self, filename: str, edits: list[dict[str, str]]
    ) -> dict[str, Any]:
        """Apply search-and-replace edits to a file. Raises AgentRuntimeException on error."""
        ...


class ContainerRuntime(Runtime):
    """
    Executes commands inside a container.

    This runtime delegates all operations to a running container,
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
            raise AgentRuntimeException(
                f"Container runtime '{self.runtime}' not found in PATH"
            )

    async def _exec_in_container(
        self, command: str, input_data: bytes | None = None
    ) -> tuple[str, str, int]:
        """
        Execute a command in the container and wait for it to complete.
        If you need long running command, consider running it in background and use `run_command` to check its status.
        Returns stdout, stderr, return_code.
        """
        full_command = [self.runtime, "exec"]

        if input_data is not None:
            full_command.append("-i")

        full_command.extend(
            [
                "-w",
                self.workdir,
                self.container_name,
                "bash",
                "-l",
                "-c",
                command,
            ]
        )

        logger.debug(f"Executing in container: {command}")

        process = await asyncio.create_subprocess_exec(
            *full_command,
            stdin=asyncio.subprocess.PIPE if input_data is not None else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await process.communicate(input=input_data)
        return (
            stdout.decode("utf-8", errors="replace"),
            stderr.decode("utf-8", errors="replace"),
            process.returncode or 0,
        )

    @override
    async def execute(self, command: str) -> dict[str, Any]:
        """Execute a shell command in the container."""
        try:
            stdout, stderr, return_code = await self._exec_in_container(command)
        except Exception as e:
            raise AgentRuntimeException(f"Command execution failed: {e}") from e
        if len(stdout) > 10000:
            stdout = (
                stdout[:10000]
                + "\n\n(truncated: output is too long, try saving to a temporary file and read section by section)"
            )
        if len(stderr) > 10000:
            stderr = (
                stderr[:10000]
                + "\n\n(truncated: output is too long, try saving to a temporary file and read section by section)"
            )
        return {"stdout": stdout, "stderr": stderr, "return_code": return_code}

    @override
    async def read_file_internal(self, filename: str) -> bytes:
        """Read entire content from a file in the container from host by base64 and convert."""
        try:
            base64_cmd = f"base64 '{filename}'"
            stdout, stderr, return_code = await self._exec_in_container(base64_cmd)
            if return_code != 0:
                raise Exception(stderr.strip())
            return base64.b64decode(stdout)
        except Exception as e:
            logger.error(f"Failed to read file {filename}: {e}")
            raise

    @override
    async def read_file(
        self, filename: str, start_line: int = 1, limit: int = 500
    ) -> dict[str, Any]:
        """Read content from a file in the container with pagination."""
        total_cmd = f"sed -n '$=' '{filename}'"
        stdout, stderr, return_code = await self._exec_in_container(total_cmd)
        if return_code != 0:
            raise AgentRuntimeException(stderr.strip() or "File not found")
        try:
            total_lines = int(stdout.strip())
        except ValueError:
            total_lines = 0
        start = max(1, start_line)
        end = start + limit - 1
        read_cmd = f"sed -n '{start},{end}p' '{filename}'"
        stdout, stderr, return_code = await self._exec_in_container(read_cmd)
        if return_code != 0:
            raise AgentRuntimeException(stderr.strip())
        content = stdout
        return {
            "content": content,
            "total_lines": total_lines,
            "start_line": start,
            "returned_lines": len(content.splitlines()),
        }

    @override
    async def write_file(self, filename: str, content: str) -> dict[str, Any]:
        """Write content to a file in the container."""
        mkdir_cmd = f"mkdir -p \"$(dirname '{filename}')\""
        _ = await self._exec_in_container(mkdir_cmd)
        encoded_bytes = base64.b64encode(content.encode("utf-8"))
        command = f"base64 -d > '{filename}'"
        _, stderr, return_code = await self._exec_in_container(
            command, input_data=encoded_bytes
        )
        if return_code != 0:
            raise AgentRuntimeException(stderr.strip())
        return {"message": f"Content saved to {filename}"}

    @override
    async def edit_file(
        self, filename: str, edits: list[dict[str, str]]
    ) -> dict[str, Any]:
        """
        Edit a file by replacing specific blocks of text.
        """
        content_bytes = await self.read_file_internal(filename)
        content = content_bytes.decode("utf-8", errors="replace")
        for edit in edits:
            search_block = edit["search"]
            replace_block = edit["replace"]
            if search_block not in content:
                raise AgentRuntimeException(
                    f"Could not find exact match in {filename} for search block\n\n{search_block}\n\n"
                    "Ensure your SEARCH block is a literal copy of the file content. The file is left unmodified."
                )
            if content.count(search_block) > 1:
                raise AgentRuntimeException(
                    f"Multiple occurrences of search block found in {filename}. "
                    "Please include more surrounding context to make it unique."
                )
            content = content.replace(search_block, replace_block, 1)
        return await self.write_file(filename, content)


class HostRuntime(Runtime):
    """
    Execute commands on the host machine.
    """

    @override
    async def execute(self, command: str) -> dict[str, Any]:
        """
        Execute a command and wait for it to complete.
        """
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            out_channel, err_channel = await process.communicate()
            stdout = out_channel.decode("utf-8", errors="replace")
            stderr = err_channel.decode("utf-8", errors="replace")
            if len(stdout) > 10000:
                stdout = (
                    stdout[:10000]
                    + "\n\n(truncated: output is too long, try saving to a temporary file and read section by section)"
                )
            if len(stderr) > 10000:
                stderr = (
                    stderr[:10000]
                    + "\n\n(truncated: output is too long, try saving to a temporary file and read section by section)"
                )
            return_code = process.returncode
            return {"stdout": stdout, "stderr": stderr, "return_code": return_code}
        except Exception as e:
            raise AgentRuntimeException(f"Command execution failed: {e}") from e

    def _read_file_internal_sync(self, filename: str) -> bytes:
        with Path(filename).open("rb") as f:
            return f.read()

    @override
    async def read_file_internal(self, filename: str) -> bytes:
        """Read entire content from a file in the host"""
        try:
            return await asyncio.to_thread(self._read_file_internal_sync, filename)
        except Exception as e:
            logger.error(f"Failed to read file {filename}: {e}")
            raise

    def _read_file_sync(
        self, filename: str, start_line: int, limit: int
    ) -> dict[str, Any]:
        path = Path(filename)
        if not path.exists():
            raise AgentRuntimeException("File not found")
        with path.open("r", encoding="utf-8") as f:
            lines = f.readlines()
        total_lines = len(lines)
        start = max(1, start_line)
        end = start + limit - 1
        content = "".join(lines[start - 1 : end])
        return {
            "content": content,
            "total_lines": total_lines,
            "start_line": start,
            "returned_lines": len(content.splitlines()),
        }

    @override
    async def read_file(
        self, filename: str, start_line: int = 1, limit: int = 500
    ) -> dict[str, Any]:
        """
        Read content from a file
        """
        return await asyncio.to_thread(
            self._read_file_sync, filename, start_line, limit
        )

    def _write_file_sync(self, filename: str, content: str) -> dict[str, Any]:
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            f.write(content)
        return {"message": f"Content saved to {filename}"}

    @override
    async def write_file(self, filename: str, content: str) -> dict[str, Any]:
        """
        Write content to a file
        """
        return await asyncio.to_thread(self._write_file_sync, filename, content)

    @override
    async def edit_file(
        self, filename: str, edits: list[dict[str, str]]
    ) -> dict[str, Any]:
        """
        Edit a file by replacing specific blocks of text.
        """
        path = Path(filename)
        if not path.exists():
            raise AgentRuntimeException("File not found")
        content = path.read_text(encoding="utf-8")
        for edit in edits:
            search_block = edit["search"]
            replace_block = edit["replace"]
            if search_block not in content:
                raise AgentRuntimeException(
                    f"Could not find exact match in {filename} for search block\n\n{search_block}\n\n"
                    "Ensure your SEARCH block is a literal copy of the file content. The file is left unmodified"
                )
            if content.count(search_block) > 1:
                raise AgentRuntimeException(
                    f"Multiple occurrences of search block found in {filename}. "
                    "Please include more surrounding context to make it unique."
                )
            content = content.replace(search_block, replace_block, 1)
        return await self.write_file(filename, content)
