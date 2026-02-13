"""
Agent module containing the core agent logic.

The Agent class orchestrates LLM interactions and context management.
Tool registration is handled externally by ToolRegistry (SRP).
System prompt construction is handled by SystemPromptBuilder (SRP).
"""

import logging
from typing import Final

from agent.llm.openai import OpenAIProvider

logger = logging.getLogger(__name__)


class Agent:
    """
    Core agent class that manages LLM interactions.

    Responsibilities (SRP):
    - Maintain conversation history
    - Run the LLM conversation loop
    - Validate structured responses against schemas
    """

    llm_client: Final[OpenAIProvider]
    messages: list[dict[str, str]]
    system_prompt: str

    def __init__(self, llm_client: OpenAIProvider):
        self.llm_client = llm_client
        self.messages = []
        self.system_prompt = ""

    def set_system_prompt(self, prompt: str) -> None:
        """Set the system prompt and reset conversation history."""
        self.system_prompt = prompt
        self.messages = [{"role": "system", "content": self.system_prompt}]

    async def run(
        self,
        messages: list[dict[str, str]],
        max_iterations: int = 80,
    ) -> str:
        """Run a single turn of the agent conversation."""

        self.messages.extend(messages)
        return await self.llm_client.chat(self.messages, max_iterations)
