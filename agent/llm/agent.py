"""
Agent module containing the core agent logic.

The Agent class orchestrates LLM interactions and context management.
Tool registration is handled externally by ToolRegistry (SRP).
System prompt construction is handled by SystemPromptBuilder (SRP).
"""

import json
import logging
from typing import Any, Final

from jsonschema import ValidationError, validate

from agent.llm.base import LLMBase

logger = logging.getLogger(__name__)


class Agent:
    """
    Core agent class that manages LLM interactions.

    Responsibilities (SRP):
    - Maintain conversation history
    - Run the LLM conversation loop
    - Validate structured responses against schemas
    """

    llm_client: Final[LLMBase]
    messages: list[dict[str, str]]
    system_prompt: str

    def __init__(self, llm_client: LLMBase):
        self.llm_client = llm_client
        self.messages = []
        self.system_prompt = ""

    def set_system_prompt(self, prompt: str) -> None:
        """Set the system prompt and reset conversation history."""
        self.system_prompt = prompt
        self.messages = [{"role": "system", "content": self.system_prompt}]

    async def run(
        self,
        user_prompt: str,
        response_schema: dict[str, Any],
        max_iterations: int = 80,
    ) -> dict[str, Any]:
        """Run a single turn of the agent conversation."""

        self.messages.append({"role": "user", "content": user_prompt})

        while True:
            response = await self.llm_client.chat(self.messages, max_iterations)

            try:
                if response.startswith("```json"):
                    response = response[7:-3]
                elif response.endswith("```"):
                    response = response[:-3]

                data = json.loads(response)
                validate(instance=data, schema=response_schema)
                self.messages.append({"role": "assistant", "content": response})
                return data
            except (json.JSONDecodeError, ValidationError) as e:
                logger.warning(f"Response validation failed: {e}")
                self.messages.append({"role": "assistant", "content": response})
                self.messages.append(
                    {
                        "role": "user",
                        "content": f"Your previous response failed validation: {e}. Please correct it and respond only with valid JSON matching the schema.",
                    }
                )

    def clear_history(self) -> None:
        """Clear conversation history, keeping only the system prompt."""
        self.messages = [{"role": "system", "content": self.system_prompt}]
