"""Tests for Channel ABC and the register_tools pattern."""

import pytest

from agent.core.messaging import Channel
from agent.llm.types import ToolContent
from agent.tools.registry import ToolRegistry


class _NoOpChannel(Channel):
    """Minimal Channel with no backend-specific tools."""

    async def send(self, text: str) -> None:
        pass

    async def start_thinking(self) -> None:
        pass

    async def end_thinking(self) -> None:
        pass

    def register_tools(self, registry: ToolRegistry) -> None:
        pass


class _RichChannel(Channel):
    """Channel that advertises a custom tool via register_tools."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def send(self, text: str) -> None:
        pass

    async def start_thinking(self) -> None:
        pass

    async def end_thinking(self) -> None:
        pass

    def register_tools(self, registry: ToolRegistry) -> None:
        async def ping() -> ToolContent:
            """Ping the backend."""
            return ToolContent.from_dict("success", {"pong": True})

        registry.register(ping, {"type": "object", "properties": {}, "required": []})


class TestChannelDefaultRegisterTools:
    """Default register_tools implementation must be a no-op."""

    def test_no_tools_registered_by_default(self) -> None:
        registry = ToolRegistry()
        channel = _NoOpChannel()
        channel.register_tools(registry)
        assert registry.tool_schemas() == [], (
            "Default register_tools should add nothing"
        )


class TestChannelCustomRegisterTools:
    """Subclass override of register_tools must surface tools in the registry."""

    def test_custom_tool_appears_in_registry(self) -> None:
        registry = ToolRegistry()
        channel = _RichChannel()
        channel.register_tools(registry)
        schemas = registry.tool_schemas()
        names = [s["function"]["name"] for s in schemas]
        assert "ping" in names

    @pytest.mark.asyncio
    async def test_custom_tool_callable(self) -> None:
        registry = ToolRegistry()
        channel = _RichChannel()
        channel.register_tools(registry)
        handler = registry.get_handler("ping")
        assert handler is not None
        result = await handler()
        assert result.status == "success"  # type: ignore[union-attr]


class TestHumanInputOrchestratorRegistersChannelTools:
    """HumanInputOrchestrator must call channel.register_tools at construction."""

    def test_channel_tools_in_orchestrator_registry(self) -> None:
        from agent.llm.agent import HumanInputOrchestrator
        from agent.tools.registry import ToolRegistry

        base_registry = ToolRegistry()
        channel = _RichChannel()

        orchestrator = HumanInputOrchestrator(
            model="test-model",
            tool_registry=base_registry,
            sender=channel,
        )

        names = [
            s["function"]["name"] for s in orchestrator.tool_registry.tool_schemas()
        ]
        assert "ping" in names, (
            "Channel tools must appear in the orchestrator's registry"
        )

    def test_no_channel_tools_for_no_op_channel(self) -> None:
        from agent.llm.agent import HumanInputOrchestrator
        from agent.tools.registry import ToolRegistry

        base_registry = ToolRegistry()
        channel = _NoOpChannel()

        orchestrator = HumanInputOrchestrator(
            model="test-model",
            tool_registry=base_registry,
            sender=channel,
        )

        assert orchestrator.tool_registry.tool_schemas() == []
