# LLM Agent

This project provides a system-level LLM agent that can assist you with running commands, managing files, and solving problems on your local system. The agent features a modular architecture with specialized skills and audio feedback.

## Features

*   **Natural Language Interaction:** Interact with the agent using natural language in a powerful CLI.
*   **Command Execution:** The agent can execute shell commands on your system.
*   **Web Research:** Perform web searches using DuckDuckGo and fetch content from web pages.
*   **File Management:** Read, write, and edit files on the local filesystem.
*   **Specialized Skills:** Extensible "skills" system for complex task instructions.
*   **Audio Feedback:** Responses are read aloud using local TTS (optimized for speech).
*   **User Confirmation:** Optional tool whitelisting for auto-approval; otherwise, prompts for confirmation before side effects.
*   **Slash Commands:** Terminal commands like `/exit` and `/clear`.

## Getting Started

### Prerequisites

*   Python 3.10 or later
*   `uv` for project management
*   System libraries for `pyaudio` (e.g., `portaudio`)
*   A Piper TTS model file (for audio feedback)

### Installation

1.  Clone the repository.
2.  Install the dependencies using `uv`:

    ```bash
    uv pip install -e .
    ```

3.  Create a `.env` file in the root of the project and add the following:

    ```
    OPENAI_BASE_URL=your_url
    OPENAI_MODEL=your_model
    OPENAI_API_KEY=your_key
    TTS_MODEL_PATH=path/to/voice.onnx
    ```

### Usage

To start the agent:

```bash
python -m agent.main
```

Type `/exit` to end or `/clear` to reset history.

## Tools

Built-in tools include:

*   **`run_command`**: Executes a shell command.
*   **`web_search`**: Performs a web search.
*   **`fetch`**: Fetches content from a URL.
*   **`read_file`**: Reads content from a file.
*   **`write_file`**: Writes content to a file.
*   **`edit_file`**: Edits specific parts of a file.
*   **`use_skill`**: Loads instructions for specialized skills.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
