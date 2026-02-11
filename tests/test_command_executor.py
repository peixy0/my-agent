"""Tests for CommandExecutor implementations."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from agent.tools.command_executor import (
    CommandExecutor,
    ContainerCommandExecutor,
)


class TestContainerCommandExecutor:
    """Tests for ContainerCommandExecutor."""

    def test_implements_protocol(self):
        """Verify ContainerCommandExecutor satisfies CommandExecutor protocol."""
        with patch("shutil.which", return_value="/usr/bin/podman"):
            executor = ContainerCommandExecutor("test-container")
            assert isinstance(executor, CommandExecutor)

    def test_init_validates_runtime(self):
        """Verify runtime validation on init."""
        with (
            patch("shutil.which", return_value=None),
            pytest.raises(RuntimeError, match="not found in PATH"),
        ):
            ContainerCommandExecutor("test-container", runtime="nonexistent")

    @pytest.mark.asyncio
    async def test_execute_success(self):
        """Test successful command execution."""
        with patch("shutil.which", return_value="/usr/bin/podman"):
            executor = ContainerCommandExecutor("test-container")

        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"output\n", b"")
        mock_process.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await executor.execute("ls -la")

        assert result["status"] == "success"
        assert result["stdout"] == "output\n"

    @pytest.mark.asyncio
    async def test_execute_failure(self):
        """Test command execution failure."""
        with patch("shutil.which", return_value="/usr/bin/podman"):
            executor = ContainerCommandExecutor("test-container")

        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"", b"error message")
        mock_process.returncode = 1

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await executor.execute("bad-command")

        assert result["status"] == "error"
        assert result["returncode"] == 1
        assert "error message" in result["stderr"]

    @pytest.mark.asyncio
    async def test_read_file_success(self):
        """Test successful file read with pagination metadata."""
        with patch("shutil.which", return_value="/usr/bin/podman"):
            executor = ContainerCommandExecutor("test-container")

        # Mock first call (wc -l) and second call (sed)
        mock_process_wc = AsyncMock()
        mock_process_wc.communicate.return_value = (b"10\n", b"")
        mock_process_wc.returncode = 0

        mock_process_sed = AsyncMock()
        mock_process_sed.communicate.return_value = (b"line 1\nline 2\n", b"")
        mock_process_sed.returncode = 0

        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=[mock_process_wc, mock_process_sed],
        ):
            result = await executor.read_file("test.txt", start_line=1, limit=2)

        assert result["status"] == "success"
        assert result["content"] == "line 1\nline 2\n"
        assert result["total_lines"] == 10
        assert result["start_line"] == 1
        assert result["returned_lines"] == 2

    @pytest.mark.asyncio
    async def test_write_file_success(self):
        """Test successful file write."""
        with patch("shutil.which", return_value="/usr/bin/podman"):
            executor = ContainerCommandExecutor("test-container")

        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"", b"")
        mock_process.returncode = 0

        with patch(
            "asyncio.create_subprocess_exec", return_value=mock_process
        ) as mock_create:
            result = await executor.write_file("test.txt", "content")

        assert result["status"] == "success"
        assert mock_create.call_count == 2
        # Verify second call used stdin
        assert mock_create.call_args_list[1].kwargs["stdin"] == asyncio.subprocess.PIPE
        # Verify input was passed to communicate
        assert mock_process.communicate.call_count == 2
        assert mock_process.communicate.call_args_list[1].kwargs["input"] is not None

    @pytest.mark.asyncio
    async def test_edit_file_not_found(self):
        """Test edit_file when original content not found."""
        with patch("shutil.which", return_value="/usr/bin/podman"):
            executor = ContainerCommandExecutor("test-container")

        # Mock read (wc then sed) returning different content
        mock_process_wc = AsyncMock()
        mock_process_wc.communicate.return_value = (b"1\n", b"")
        mock_process_wc.returncode = 0

        mock_process_sed = AsyncMock()
        mock_process_sed.communicate.return_value = (b"different content", b"")
        mock_process_sed.returncode = 0

        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=[mock_process_wc, mock_process_sed],
        ):
            result = await executor.edit_file(
                "test.txt", [{"search": "original", "replace": "replaced"}]
            )

        assert result["status"] == "error"
        assert "could not find exact match" in result["message"].lower()
