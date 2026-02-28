"""Tests for ToolRegistry."""

import asyncio

import pytest

from agent.tools.tool_registry import ToolRegistry


class TestToolRegistry:
    """Tests for ToolRegistry."""

    def test_register_and_retrieve(self):
        """Test registering a tool and retrieving its schema and handler."""
        registry = ToolRegistry(tool_timeout=5)

        async def my_tool(x: int) -> dict:
            """A test tool."""
            return {"result": x * 2}

        schema = {
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
        }

        registry.register(my_tool, schema)

        assert registry.get_schema("my_tool") is not None
        assert registry.get_schema("my_tool")["name"] == "my_tool"  # type: ignore[index]
        assert registry.get_schema("my_tool")["description"] == "A test tool."  # type: ignore[index]
        assert registry.get_handler("my_tool") is not None

    def test_register_with_custom_name(self):
        """Test registering with a custom name and description."""
        registry = ToolRegistry()

        async def func() -> dict:
            return {}

        registry.register(func, {}, name="custom_name", description="Custom desc")

        schema = registry.get_schema("custom_name")
        assert schema is not None
        assert schema["description"] == "Custom desc"

    @pytest.mark.asyncio
    async def test_handler_timeout(self):
        """Test that the timeout wrapper works."""
        registry = ToolRegistry(tool_timeout=1)

        async def slow_tool() -> dict:
            await asyncio.sleep(10)
            return {"status": "success"}

        registry.register(slow_tool, {})

        handler = registry.get_handler("slow_tool")
        assert handler is not None
        result = await handler()
        assert result["status"] == "error"
        assert "timed out" in result["message"]

    @pytest.mark.asyncio
    async def test_handler_error_wrapping(self):
        """Test that exceptions are wrapped into error dicts."""
        registry = ToolRegistry()

        async def bad_tool() -> dict:
            raise ValueError("something broke")

        registry.register(bad_tool, {})

        handler = registry.get_handler("bad_tool")
        assert handler is not None
        result = await handler()
        assert result["status"] == "error"
        assert "something broke" in result["message"]

    @pytest.mark.asyncio
    async def test_handler_success(self):
        """Test successful tool execution through the wrapper."""
        registry = ToolRegistry()

        async def good_tool(value: str) -> dict:
            return {"status": "success", "value": value}

        registry.register(good_tool, {})

        handler = registry.get_handler("good_tool")
        assert handler is not None
        result = await handler(value="hello")
        assert result["status"] == "success"
        assert result["value"] == "hello"
