# System-Level LLM Agent: Specification

## 1. Introduction

This document outlines the design and specification for a system-level Large Language Model (LLM) agent. The agent is designed to assist users with running commands and solving problems on their local system through a command-line interface (CLI). It will leverage a powerful LLM to understand natural language queries, execute relevant commands, and provide intelligent responses.

## 2. Core Components

The agent consists of the following core components:

*   **LLM Client:** A component responsible for interacting with the LLM API. It uses a factory pattern to create specific clients (e.g., OpenAI) based on an abstract base class.
*   **Tool Registry:** A mechanism for registering and managing tools that the LLM can use to interact with the system.
*   **Skill Loader:** A component that discovers and loads specialized "skills" from a dedicated directory. Skills provide detailed instructions for specific tasks.
*   **Audio System (TTS):** A Text-to-Speech component that converts agent responses into spoken audio using the Piper TTS engine.
*   **CLI Application:** The entry point of the agent, responsible for initializing the LLM client, registering tools, loading skills, and handling user interactions in the terminal, including slash commands.
*   **Configuration:** A component for managing application settings and secrets using `pydantic-settings`.

## 3. Features

*   **Natural Language Interaction:** Users interact with the agent using natural language in a CLI.
*   **Command Execution:** The agent can execute shell commands on the user's system.
*   **Web Search & Fetching:** The agent can perform web searches using DuckDuckGo and fetch web page content.
*   **File Management:** The agent can read, write, and edit files on the local filesystem.
*   **Specialized Skills:** The agent can discover and use "skills" which are Markdown files containing specialized instructions for complex tasks. Discoverable skills are injected into the system prompt.
*   **Audio Feedback:** Agent responses are read aloud using local TTS, optimized for spoken output (plain text, no Markdown).
*   **Slash Commands:** Support for terminal-only commands like `/exit` and `/clear`.
*   **User Confirmation & Whitelisting:** The agent prompts for confirmation before executing tools with side effects. Specific tools can be whitelisted for auto-approval.
*   **Extensibility:** New tools and skills can be easily added.
*   **Secure Configuration:** Configuration is loaded from a `.env` file using `pydantic-settings`.

## 4. High-Level Architecture

1.  **Initialization:** The CLI initializes settings, the LLM client, the audio system, and the skill loader.
2.  **Skill Discovery:** The skill loader scans the `.skills` directory for `SKILL.md` files and provides summaries to the LLM.
3.  **Interaction Loop:**
    a.  The user enters a query or a slash command.
    b.  Slash commands are handled directly by the CLI.
    c.  Natural language queries are sent to the LLM along with the conversation history and a system prompt containing skill summaries and system info (OS, time).
    d.  The LLM processes the query and may call one or more tools.
    e.  The CLI prompts for tool approval (unless whitelisted).
    f.  Tool results (or cancellation reasons) are sent back to the LLM.
    g.  The LLM generates a final response optimized for speech (plain text).
    h.  The response is displayed in the CLI and sent to the audio system for playback.

## 5. Tool Definition

Tools are Python functions registered with the LLM client via a JSON schema.

Current tools include:

*   `run_command`: Executes a shell command.
*   `web_search`: Performs a web search using DuckDuckGo.
*   `fetch`: Fetches the content of a web page.
*   `read_file`: Reads content from a file.
*   `write_file`: Writes content to a file.
*   `edit_file`: Replaces a specific block of text in a file.
*   `use_skill`: Loads the full instructions for a specialized skill.

## 6. Configuration

The `Settings` class (via `pydantic-settings`) manages the following:

*   `openai_base_url`: LLM API base URL.
*   `openai_model`: LLM model name.
*   `openai_api_key`: API key for the LLM.
*   `tts_model_path`: Path to the Piper TTS model file.
*   `whitelist_tools`: List of tool names that don't require user confirmation.
*   `skills_dir`: Directory where specialized skills are stored (default: `.skills`).

## 7. Implementation Details

*   **Language:** Python 3.10+
*   **LLM Interaction:** `openai` library with a custom abstraction layer (`LLMBase`) and `tenacity` for robust error handling and retries.
*   **CLI:** `rich` for formatting and `Prompt` for user input.
*   **Audio:** `pyaudio` for playback and `piper-tts` for synthesis.
*   **Web Tools:** `ddgs` (DuckDuckGo Search) and `trafilatura` (content extraction).
*   **Project Management:** `uv` for dependencies and environment.
*   **Code Quality:** `ruff` for linting and `pyright` for type checking.

The project structure:
*   `agent/`: Source code modules.
*   `docs/`: Documentation.
*   `.skills/`: Specialized skill definitions.
*   `workspace/`: Default area for agent operations.