"""
Tool registry providing declarative tool registration (OCP).

New tools are added by calling `registry.register()` — no existing
code needs to be modified.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Holds tool name -> (schema, handler) mappings.

    Wraps each handler with timeout and error handling so callers
    don't have to worry about it.
    """

    def __init__(self, tool_timeout: int = 60):
        self._tool_timeout = tool_timeout
        self._schemas: dict[str, dict[str, Any]] = {}
        self._handlers: dict[str, Callable[..., Awaitable[Any]]] = {}

    def _wrap_tool(
        self, func: Callable[..., Awaitable[Any]], tool_name: str
    ) -> Callable[..., Awaitable[Any]]:
        """Wrap a tool with timeout and error handling."""

        async def wrapped_tool(**kwargs: Any) -> Any:
            try:
                return await asyncio.wait_for(
                    func(**kwargs), timeout=self._tool_timeout
                )
            except asyncio.TimeoutError:
                return {
                    "status": "error",
                    "message": f"Tool {tool_name} timed out after {self._tool_timeout}s",
                }
            except Exception as e:
                return {"status": "error", "message": str(e)}

        return wrapped_tool

    def register(
        self,
        func: Callable[..., Awaitable[Any]],
        schema: dict[str, Any],
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        """Register a tool with its parameter schema."""
        tool_name = name or func.__name__
        tool_description = description or (func.__doc__ or "").strip()

        self._schemas[tool_name] = {
            "name": tool_name,
            "description": tool_description,
            "parameters": schema,
        }
        self._handlers[tool_name] = self._wrap_tool(func, tool_name)

    def get_schema(self, tool_name: str) -> dict[str, Any] | None:
        return self._schemas.get(tool_name)

    def get_handler(self, tool_name: str) -> Callable[..., Awaitable[Any]] | None:
        return self._handlers.get(tool_name)

    def tool_schemas(self) -> list[dict[str, Any]]:
        """Return all schemas formatted for the OpenAI tools API."""
        return [{"type": "function", "function": fn} for fn in self._schemas.values()]

    def clone(self) -> "ToolRegistry":
        """Return a shallow copy with independent tool mappings."""
        copy = ToolRegistry(self._tool_timeout)
        copy._schemas = dict(self._schemas)
        copy._handlers = dict(self._handlers)
        return copy
