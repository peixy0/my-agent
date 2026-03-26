"""
Tool registry providing declarative tool registration (OCP).

New tools are added by calling `registry.register()` — no existing
code needs to be modified.
"""

import logging
from collections.abc import Awaitable, Callable
from typing import Any

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Holds tool name -> (schema, handler) mappings.
    """

    def __init__(self):
        self.schemas: dict[str, dict[str, Any]] = {}
        self.handlers: dict[str, Callable[..., Awaitable[Any]]] = {}

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

        self.schemas[tool_name] = {
            "name": tool_name,
            "description": tool_description,
            "parameters": schema,
        }
        self.handlers[tool_name] = func

    def get_schema(self, tool_name: str) -> dict[str, Any] | None:
        return self.schemas.get(tool_name)

    def get_handler(self, tool_name: str) -> Callable[..., Awaitable[Any]] | None:
        return self.handlers.get(tool_name)

    def tool_schemas(self) -> list[dict[str, Any]]:
        """Return all schemas formatted for the OpenAI tools API."""
        return [{"type": "function", "function": fn} for fn in self.schemas.values()]

    def clone(self) -> "ToolRegistry":
        """Return a shallow copy with independent tool mappings."""
        copy = ToolRegistry()
        copy.schemas = dict(self.schemas)
        copy.handlers = dict(self.handlers)
        return copy
