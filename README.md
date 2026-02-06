# Autonomous LLM Agent

An autonomous LLM agent that runs on the host machine and uses a container as an isolated workspace environment.

## Architecture

- **Agent runs on host**: The Python agent runs on your machine
- **Container as workspace**: Commands and file operations execute in a container at `/workspace`
- **Persistent storage**: Workspace is bind-mounted for persistence

## Quick Start

### Prerequisites

- Python 3.10+
- `uv` for dependency management
- Podman (or Docker)

### Setup

```bash
# Clone and install dependencies
cd sys-agent
uv sync --group dev

# Create .env file
cp .env.example .env
# Edit .env with your OpenAI API key

# Build the workspace container
./build-container.sh

# Start the workspace container
./run-container.sh
```

### Run the Agent

```bash
# Run the autonomous agent
uv run python -m agent.main
```

The agent will:
1. Ensure the workspace container is running
2. Wake up every 15 minutes (configurable)
3. Work on tasks from its TODO list
4. Maintain context across wake cycles
5. Keep a daily journal

## Configuration

Create a `.env` file:

```env
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o
OPENAI_API_KEY=your_api_key_here

# Optional: Container settings
CONTAINER_NAME=sys-agent-workspace
CONTAINER_RUNTIME=podman

# Optional: Adjust wake interval (seconds)
WAKE_INTERVAL_SECONDS=900

# Optional: WeChat Work integration for notifications
WECHAT_CORPID=your_corp_id
WECHAT_CORPSECRET=your_corp_secret
WECHAT_AGENTID=your_agent_id
WECHAT_TOUSER=@all
```

### WeChat Work Integration

The agent can send notifications via WeChat Work (企业微信). To enable:

1. Get credentials from WeChat Work admin panel:
   - `WECHAT_CORPID`: Your enterprise ID
   - `WECHAT_CORPSECRET`: Application secret
   - `WECHAT_AGENTID`: Application agent ID
   - `WECHAT_TOUSER`: Target users (default: `@all`)

2. Add credentials to your `.env` file

3. Use the `发送应用消息` tool to send messages

## Workspace

The workspace persists in `./workspace/` and is mounted into the container:

```
workspace/
├── CONTEXT          # Persistent context
├── TODO             # Task list
├── journal/         # Daily journals
│   └── 2026-02-04.md
├── .skills/         # Custom skills
└── events.jsonl     # Event log
```

## Tools

| Tool | Description |
|------|-------------|
| `run_command` | Execute shell commands |
| `read_file` | Read file content |
| `write_file` | Write to files |
| `edit_file` | Edit file content |
| `web_search` | Search the web |
| `fetch` | Fetch web pages |
| `发送应用消息` | Send WeChat Work message |
| `use_skill` | Load skill instructions |

## Skills

Add custom skills to `workspace/.skills/`:

```
workspace/.skills/my-skill/
└── SKILL.md
```

Format:
```markdown
---
name: my-skill
description: What this skill does
---

# Instructions
...
```

## Development

```bash
# Install dev dependencies
uv sync --group dev

# Run tests
uv run pytest tests/ -v

# Type checking
uv run pyright agent/

# Linting
uv run ruff check agent/
```

## Container Commands

```bash
# Start workspace container
./run-container.sh

# Execute commands in container
podman exec -it sys-agent-workspace bash

# Stop container
podman stop sys-agent-workspace

# Remove container
podman rm -f sys-agent-workspace
```

## License

MIT License
