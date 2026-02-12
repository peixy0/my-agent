# Autonomous LLM Agent

A system-level autonomous LLM agent that runs continuously on your host machine, using an isolated container workspace for safe command execution. The agent can work autonomously on scheduled tasks, respond to human input via HTTP API, and send notifications through WeChat Work or Feishu.

## Key Features

- **Autonomous Mode**: Wakes up periodically to work on tasks, maintains context across sessions
- **Container Isolation**: All commands execute safely in a containerized workspace
- **HTTP API**: Accept human input via REST endpoint for interactive conversations
- **Messaging Integration**: Send notifications via WeChat Work (企业微信) or Feishu (飞书)
- **Extensible Skills**: Add custom skills as Markdown files to extend agent capabilities
- **Event Logging**: Optional remote event streaming for monitoring and debugging

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

Create a `.env` file with your settings:

```env
# LLM Configuration (Required)
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o
OPENAI_API_KEY=your_api_key_here

# Container Settings
CONTAINER_NAME=sys-agent-workspace
CONTAINER_RUNTIME=podman  # or docker

# Agent Behavior
WAKE_INTERVAL_SECONDS=1800  # 30 minutes (default)
TOOL_TIMEOUT=60  # seconds

# HTTP API (Optional)
API_ENABLED=true
API_HOST=0.0.0.0
API_PORT=8000

# WeChat Work Integration (Optional)
WECHAT_CORPID=your_corp_id
WECHAT_CORPSECRET=your_corp_secret
WECHAT_AGENTID=your_agent_id
WECHAT_TOUSER=@all

# Feishu Integration (Optional)
FEISHU_APP_ID=your_app_id
FEISHU_APP_SECRET=your_app_secret
FEISHU_ENCRYPT_KEY=your_encrypt_key
FEISHU_VERIFICATION_TOKEN=your_verification_token

# Event Logging (Optional)
EVENT_API_URL=https://your-event-api.com/events
EVENT_API_KEY=your_event_api_key
```

### HTTP API

Enable the HTTP API to send messages to the agent:

```bash
# Send a message to the agent
curl -X POST http://localhost:8000/api/bot \
  -H "Content-Type: application/json" \
  -d '{"message": "What tasks are you working on?"}'

# Health check
curl http://localhost:8000/api/health
```

### Messaging Integrations

**WeChat Work (企业微信)**:
1. Get credentials from WeChat Work admin panel
2. Add `WECHAT_*` variables to `.env`
3. Agent can send notifications automatically

**Feishu (飞书)**:
1. Create a Feishu app and get credentials
2. Add `FEISHU_*` variables to `.env`
3. Agent responds to messages and sends notifications

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

For developers extending or contributing to the project, see [AGENT.md](AGENT.md) for architecture details and coding standards.

```bash
# Install dev dependencies
uv sync --group dev

# Run tests
uv run pytest tests/ -v

# Run all checks (format, lint, type check, test)
uv run ruff format . && uv run ruff check . && uv run basedpyright && uv run pytest
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
