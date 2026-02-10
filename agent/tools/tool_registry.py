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

    Wraps each handler with timeout and error handling so the
    Agent and LLMBase don't have to.
    """

    def __init__(self, tool_timeout: int = 60):
        self._tool_timeout = tool_timeout
        self._schemas: dict[str, dict[str, Any]] = {}
        self._handlers: dict[str, Callable[..., Awaitable[Any]]] = {}

    @property
    def schemas(self) -> dict[str, dict[str, Any]]:
        return dict(self._schemas)

    @property
    def handlers(self) -> dict[str, Callable[..., Awaitable[Any]]]:
        return dict(self._handlers)

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

        async def wrapped_tool(**kwargs: Any) -> Any:
            try:
                result = await asyncio.wait_for(
                    func(**kwargs), timeout=self._tool_timeout
                )
                return result
            except asyncio.TimeoutError:
                return {
                    "status": "error",
                    "message": f"Tool {tool_name} timed out after {self._tool_timeout}s",
                }
            except Exception as e:
                return {"status": "error", "message": str(e)}

        self._schemas[tool_name] = {
            "name": tool_name,
            "description": tool_description,
            "parameters": schema,
        }
        self._handlers[tool_name] = wrapped_tool
