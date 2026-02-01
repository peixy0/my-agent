import asyncio
import datetime
import logging
import platform

from dotenv import load_dotenv
from rich.console import Console

from . import tools
from .audio import TTS
from .llm_factory import LLMFactory
from .settings import settings
from .skill_loader import SkillLoader

_ = load_dotenv()

logger = logging.getLogger(__name__)


def cli():
    """
    The main command-line interface for the LLM agent.
    """
    asyncio.run(main())


def handle_slash_command(
    prompt: str, console: Console, system_prompt: str, messages: list[dict[str, str]]
) -> tuple[bool, bool, list[dict[str, str]]]:
    """
    Handle commands starting with '/'.

    Returns a tuple:
        (handled, should_exit, updated_messages)
    """
    if not prompt.startswith("/"):
        return False, False, messages

    command, *_ = prompt[1:].split(maxsplit=1)
    command = command.lower()

    if command == "exit":
        return True, True, messages
    if command == "clear":
        # Reset conversation history and clear the console output
        messages = [{"role": "system", "content": system_prompt}]
        console.clear()
        console.print(
            "[bold green]History cleared. You are starting a new conversation.[/bold green]"
        )
        return True, False, messages

    console.print(
        f"[bold yellow]Unknown command:[/bold yellow] {prompt}. "
        + "Supported commands: /exit, /clear"
    )
    return True, False, messages


async def main():
    """
    The main asynchronous function for the LLM agent.
    """
    console = Console()
    console.print("[bold green]LLM Agent[/bold green]")
    console.print("Type '/exit' to end the conversation, '/clear' to clear history.")

    audio = TTS(settings.tts_model_path).create_audio()
    llm_client = LLMFactory.create(
        url=settings.openai_base_url,
        model=settings.openai_model,
        api_key=settings.openai_api_key,
    )
    tools.register_tools(llm_client, console, settings.whitelist_tools)

    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    operating_system = platform.system()

    skill_loader = SkillLoader(settings.skills_dir)
    skill_summaries = skill_loader.discover_skills()
    skills_text = ""
    if skill_summaries:
        skills_text = "Available specialized skills:\n"
        for s in skill_summaries:
            skills_text += f"- {s.name}: {s.description}\n"
        skills_text += "\nUse the `use_skill` tool to gain access to specific domain knowledge. You can use it at any time for any skills that are related to the task description."
        skills_text += "Once you have the skill path, you are encouraged to explore referenced files in the directory to perform the task more effectively."

    system_prompt = (
        f"You are a helpful LLM agent. "
        f"Current datetime: {current_datetime}. "
        f"Operating system: {operating_system}. "
        "You are encouraged to use tools extensively and perform research to answer user queries comprehensively. "
        "Don't hesitate to use multiple tool calls or take multiple steps to solve a problem. \n\n"
        f"{skills_text} \n\n"
        "Generate a helpful, thorough, and precise response optimized for reading aloud."
        "Avoid complex sentence structures or visual formatting that cannot be spoken. Output as a single block of plain text."
        "Do not use Markdown, bolding, headers, bullet points, or numbered lists."
    )

    messages = [{"role": "system", "content": system_prompt}]

    try:
        while True:
            prompt = console.input("> ")
            handled, should_exit, messages = handle_slash_command(
                prompt, console, system_prompt, messages
            )
            if should_exit:
                break
            if handled:
                continue

            messages.append({"role": "user", "content": prompt})
            response = ""
            try:
                response = await llm_client.chat(messages, console)
                console.print(f"[bold blue]Agent:[/bold blue] {response}")
            except Exception as e:
                response = f"An unexpected error occurred: {e}"
                logger.error(response)
                console.print(f"[bold red]Error:[/bold red] {e}")
            finally:
                messages.append({"role": "assistant", "content": response})
            audio.feed(response)
    except (KeyboardInterrupt, EOFError):
        pass
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        console.print(f"[bold red]Error:[/bold red] {e}")
    finally:
        console.print("\n[bold red]Exiting...[/bold red]")
        audio.stop()
        audio.wait_until_done()


if __name__ == "__main__":
    cli()
