# Autonomous LLM Agent: Specification

## 1. Overview

This document specifies a system-level autonomous LLM agent. The agent runs on the **host machine** and uses a **container as a workspace environment** for executing commands and file operations.

## 2. Architecture

```
┌─────────────────────────────────────────────────┐
│                  Host Machine                   │
│  ┌───────────────────────────────────────────┐  │
│  │           Autonomous Agent                │  │
│  │  ┌─────────┐  ┌──────────┐  ┌──────────┐  │  │
│  │  │  Agent  │──│   LLM    │  │  Event   │  │  │
│  │  │  Core   │  │  Client  │  │  Logger  │  │  │
│  │  └────┬────┘  └──────────┘  └──────────┘  │  │
│  │       │                                   │  │
│  │  ┌────▼────────────────────────────────┐  │  │
│  │  │         Command Executor            │  │  │
│  │  │   (ContainerCommandExecutor)        │  │  │
│  │  └────┬────────────────────────────────┘  │  │
│  └───────│───────────────────────────────────┘  │
│          │ podman exec                          │
│  ┌───────▼───────────────────────────────────┐  │
│  │           Workspace Container             │  │
│  │  ┌─────────────────────────────────────┐  │  │
│  │  │  /workspace (mounted from host)     │  │  │
│  │  │  ├── CONTEXT                        │  │  │
│  │  │  ├── TODO                           │  │  │
│  │  │  ├── journal/                       │  │  │
│  │  │  └── events.jsonl                   │  │  │
│  │  └─────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

## 3. Core Components

### Agent (`agent/core/agent.py`)
- Orchestrates LLM interactions and tool execution
- Manages conversation state and context
- Registers tools with the LLM client
- Injects context files into system prompt

### LLM Client (`agent/llm/`)
- Factory pattern for LLM client creation
- OpenAI-compatible API with retry logic
- Tool call handling and response parsing

### Command Executor (`agent/tools/command_executor.py`)
- **Strategy pattern** for command execution
- `ContainerCommandExecutor` executes commands via `podman exec`
- All file operations (read/write/edit) also execute in container
- Dependency injection enables testing

### Event Logger (`agent/core/event_logger.py`)
- Logs tool usage and LLM responses to JSONL
- Optional remote streaming to external API

### Autonomous Runner (`agent/autonomous_runner.py`)
- Wake/sleep cycle management
- Container lifecycle management
- Signal handling for graceful shutdown

## 4. Tools

| Tool | Description |
|------|-------------|
| `run_command` | Execute shell commands in container |
| `read_file` | Read file content in container |
| `write_file` | Write content to file in container |
| `edit_file` | Replace text in a file |
| `web_search` | Search the web via DuckDuckGo |
| `fetch` | Fetch and extract web page content |
| `use_skill` | Load specialized skill instructions |

## 5. Configuration

Settings are managed via `pydantic-settings` and loaded from `.env`:

| Setting | Default | Description |
|---------|---------|-------------|
| `openai_base_url` | `https://api.openai.com/v1` | LLM API endpoint |
| `openai_model` | `gpt-4o` | Model to use |
| `openai_api_key` | - | API key |
| `container_name` | `sys-agent-workspace` | Workspace container name |
| `container_runtime` | `podman` | Container runtime (`podman`/`docker`) |
| `workspace_dir` | `./workspace` | Host workspace path |
| `wake_interval_seconds` | `900` | Wake cycle interval |

## 6. Skills

Skills are Markdown files in `.skills/*/SKILL.md` with YAML frontmatter:

```yaml
---
name: skill-name
description: What this skill does
---

# Skill Instructions
...
```

Skills are discovered at startup and injected into the system prompt.

## 7. Project Structure

```
sys-agent/
├── agent/
│   ├── core/
│   │   ├── agent.py         # Agent class
│   │   ├── events.py        # Event types
│   │   ├── event_logger.py  # JSONL logging
│   │   └── settings.py      # Configuration
│   ├── llm/
│   │   ├── base.py          # Abstract LLM client
│   │   ├── factory.py       # Client factory
│   │   └── openai.py        # OpenAI implementation
│   ├── tools/
│   │   ├── command_executor.py  # Container executor
│   │   ├── toolbox.py       # Tool implementations
│   │   └── skill_loader.py  # Skill discovery
│   ├── autonomous_runner.py # Wake/sleep loop
│   └── main.py              # Entry point
├── tests/                   # Unit tests
├── workspace/               # Persisted workspace
├── Containerfile            # Workspace container image
└── run-container.sh         # Container management
```

## 8. Design Principles

- **Single Responsibility**: Each module has one clear purpose
- **Open/Closed**: Extensible via new tools, skills, LLM clients
- **Liskov Substitution**: CommandExecutor implementations are interchangeable
- **Interface Segregation**: CommandExecutor protocol is focused
- **Dependency Inversion**: Core depends on abstractions, not concretions