# LLM Agent

This project provides a LLM agent that can assist you with running commands and solving problems on your local system. The agent is designed to be extensible, allowing you to easily add new tools and capabilities.

## Features

*   **Natural Language Interaction:** Interact with the agent using natural language.
*   **Command Execution:** The agent can execute shell commands on your system.
*   **Web Search:** The agent can perform web searches using DuckDuckGo.
*   **Web Page Fetching:** The agent can fetch the content of a web page.
*   **User Confirmation:** The agent will ask for confirmation before executing any tool that may have side effects.
*   **Extensibility:** Easily add new tools to the agent.
*   **Secure Configuration:** Uses `pydantic-settings` to manage configuration and secrets.

## Getting Started

### Prerequisites

*   Python 3.10 or later
*   `uv` for project management

### Installation

1.  Clone the repository:

2.  Install the dependencies using `uv`:

    ```bash
    uv pip install -e .
    ```

3.  Create a `.env` file in the root of the project and add the following environment variables:

    ```
    OPENAI_BASE_URL=your_url
    OPENAI_MODEL=your_model
    OPENAI_API_KEY=your_key
    ```

### Usage

To start the agent, run the following command:

```bash
python -m agent.main
```

This will start an interactive command-line interface where you can chat with the agent.

## Tools

The agent comes with the following built-in tools:

*   **`run_command`:** Executes a shell command.
*   **`web_search`:** Performs a web search using DuckDuckGo.
*   **`fetch`:** Fetches the content of a web page.
*   **`write_file`:** Writes content to a file.

You can add new tools by creating a Python function and registering it with the LLM client.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have any ideas for improving the agent.
