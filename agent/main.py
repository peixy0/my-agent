import asyncio
import datetime
import logging
import platform

from dotenv import load_dotenv
from rich.console import Console

from . import tools
from .llm import LLMClient
from .settings import settings

_ = load_dotenv()

logger = logging.getLogger(__name__)


def cli():
    """
    The main command-line interface for the system-level LLM agent.
    """
    asyncio.run(main())


async def main():
    """
    The main asynchronous function for the system-level LLM agent.
    """
    console = Console()
    console.print("[bold green]System-Level LLM Agent[/bold green]")
    console.print("Type 'exit' to end the conversation.")

    llm_client = LLMClient(
        url=settings.openai_base_url,
        model=settings.openai_model,
        api_key=settings.openai_api_key,
    )
    tools.register_tools(llm_client)

    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    operating_system = platform.system()

    system_prompt = (
        f"You are a helpful system-level LLM agent. "
        f"Current datetime: {current_datetime}. "
        f"Operating system: {operating_system}."
    )

    messages = [{"role": "system", "content": system_prompt}]

    while True:
        try:
            prompt = console.input("> ")
            if prompt.lower() == "exit":
                break

            messages.append({"role": "user", "content": prompt})
            response = await llm_client.chat(messages)
            console.print(f"[bold blue]Agent:[/bold blue] {response}")
            messages.append({"role": "assistant", "content": response})

        except (KeyboardInterrupt, EOFError):
            break
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            console.print(f"[bold red]Error:[/bold red] {e}")


if __name__ == "__main__":
    cli()
