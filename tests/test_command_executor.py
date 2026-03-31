"""Tests for Runtime implementations."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from agent.core.runtime import (
    AgentRuntimeException,
    ContainerRuntime,
    Runtime,
)


class TestContainerRuntime:
    """Tests for ContainerRuntime."""

    def test_implements_protocol(self):
        """Verify ContainerRuntime satisfies Runtime protocol."""
        with patch("shutil.which", return_value="/usr/bin/podman"):
            runtime = ContainerRuntime("test-container")
            assert isinstance(runtime, Runtime)

    def test_init_validates_runtime(self):
        """Verify runtime validation on init."""
        with (
            patch("shutil.which", return_value=None),
            pytest.raises(AgentRuntimeException, match="not found in PATH"),
        ):
            ContainerRuntime("test-container", runtime="nonexistent")

    @pytest.mark.asyncio
    async def test_execute_success(self):
        """Test successful command execution."""
        with patch("shutil.which", return_value="/usr/bin/podman"):
            runtime = ContainerRuntime("test-container")

        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"output\n", b"")
        mock_process.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await runtime.execute("ls -la")

        assert result["stdout"] == "output\n"
        assert result["return_code"] == 0

    @pytest.mark.asyncio
    async def test_execute_failure(self):
        """Test command execution failure."""
        with patch("shutil.which", return_value="/usr/bin/podman"):
            runtime = ContainerRuntime("test-container")

        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"", b"error message")
        mock_process.returncode = 1

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await runtime.execute("bad-command")

        assert result["return_code"] == 1
        assert "error message" in result["stderr"]

    @pytest.mark.asyncio
    async def test_read_file_success(self):
        """Test successful file read with pagination metadata.

        read_file now uses ``wc -l && sed`` inside the container, so the
        mock stdout is the line-count on the first line followed by the
        requested content.
        """
        with patch("shutil.which", return_value="/usr/bin/podman"):
            runtime = ContainerRuntime("test-container")

        wc_sed_output = "10\nline 1\nline 2\n"

        mock_process = AsyncMock()
        mock_process.communicate.return_value = (wc_sed_output.encode(), b"")
        mock_process.returncode = 0

        with patch(
            "asyncio.create_subprocess_exec",
            return_value=mock_process,
        ):
            result = await runtime.read_file("test.txt", start_line=1, limit=2)

        assert result["content"] == "line 1\nline 2\n"
        assert result["total_lines"] == 10
        assert result["start_line"] == 1
        assert result["returned_lines"] == 2

    @pytest.mark.asyncio
    async def test_write_file_success(self):
        """Test successful file write (single exec call with mkdir+base64)."""
        with patch("shutil.which", return_value="/usr/bin/podman"):
            runtime = ContainerRuntime("test-container")

        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"", b"")
        mock_process.returncode = 0

        with patch(
            "asyncio.create_subprocess_exec", return_value=mock_process
        ) as mock_create:
            result = await runtime.write_file("test.txt", "content")

        assert result["message"] == "Content saved to test.txt"
        assert mock_create.call_count == 1
        assert mock_create.call_args.kwargs["stdin"] == asyncio.subprocess.PIPE
        assert mock_process.communicate.call_args.kwargs["input"] is not None

    @pytest.mark.asyncio
    async def test_edit_file_not_found(self):
        """Test edit_file when original content not found."""
        with patch("shutil.which", return_value="/usr/bin/podman"):
            runtime = ContainerRuntime("test-container")

        # Mock read (base64) returning different content
        mock_process_base64 = AsyncMock()
        mock_process_base64.communicate.return_value = (
            b"ZGlmZmVyZW50IGNvbnRlbnQ=",
            b"",
        )  # "different content" in base64
        mock_process_base64.returncode = 0

        with (
            patch(
                "asyncio.create_subprocess_exec",
                side_effect=[mock_process_base64],
            ),
            pytest.raises(AgentRuntimeException, match="Could not find exact match"),
        ):
            await runtime.edit_file(
                "test.txt", [{"search": "original", "replace": "replaced"}]
            )
