# Autonomous LLM Agent: Specification

## 1. Overview

This document specifies a system-level autonomous LLM agent. The agent runs on the **host machine** and uses a **container as a workspace environment** for executing commands and file operations. It accepts human input via a **FastAPI HTTP endpoint**.

## 2. Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      Host Machine                       │
│  ┌───────────────────────────────────────────────────┐  │
│  │          AppWithDependencies (app.py)             │  │
│  │         Composition Root / DI Container           │  │
│  │                                                   │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────────┐│  │
│  │  │  Agent   │  │   LLM    │  │  ToolRegistry    ││  │
│  │  │  (conv   │──│  Client  │  │  (OCP: add tools ││  │
│  │  │   loop)  │  │          │  │   without edits) ││  │
│  │  └──────────┘  └──────────┘  └──────────────────┘│  │
│  │                                                   │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────────┐│  │
│  │  │Scheduler │  │  Event   │  │  ApiService      ││  │
│  │  │ (events) │◄─│  Queue   │◄─│  POST /api/bot   ││  │
│  │  └────┬─────┘  └──────────┘  └──────────────────┘│  │
│  │       │                                           │  │
│  │  ┌────▼──────────────────────────────────────┐   │  │
│  │  │         Command Executor                  │   │  │
│  │  │   (ContainerCommandExecutor)              │   │  │
│  │  └────┬──────────────────────────────────────┘   │  │
│  └───────│───────────────────────────────────────────┘  │
│          │ podman exec                                  │
│  ┌───────▼───────────────────────────────────────────┐  │
│  │           Workspace Container                     │  │
│  │  ┌─────────────────────────────────────────────┐  │  │
│  │  │  /workspace (mounted from host)             │  │  │
│  │  │  ├── CONTEXT.md                             │  │  │
│  │  │  ├── TODO.md                                │  │  │
│  │  │  ├── journal/                               │  │  │
│  │  │  └── events.jsonl                           │  │  │
│  │  └─────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## 3. Core Components

### AppWithDependencies (`agent/app.py`)
- **Composition root** — single place where all dependencies are wired
- Creates: Settings → EventLogger → Executor → ToolRegistry → OpenAIProvider → Agent → Messaging → ApiService
- `run()` method starts all background tasks (event logger, messaging, API server)
- Replaces scattered module-level singletons with explicit construction

### Agent (`agent/llm/agent.py`)
- Manages conversation history and LLM interaction loop
- Validates structured responses against JSON schemas
- **Does NOT** handle tool registration or system prompt construction (SRP)

### SystemPromptBuilder (`agent/llm/prompt_builder.py`)
- Builds the system prompt from settings, skills, and runtime context
- Separated from Agent for single responsibility

### ToolRegistry (`agent/tools/tool_registry.py`)
- Holds tool name → (handler, schema) mappings
- Wraps handlers with timeout and error handling
- Adding new tools requires only a `register()` call (OCP)

### LLM Client (`agent/llm/`)
- Factory pattern for LLM client creation
- OpenAI-compatible API with retry logic
- Reads tool definitions from `ToolRegistry`

### Command Executor (`agent/tools/command_executor.py`)
- **Strategy pattern** for command execution
- `ContainerCommandExecutor` executes commands via `podman exec`
- All file operations (read/write/edit) also execute in container
- Dependency injection enables testing

### Event Logger (`agent/core/event_logger.py`)
- Logs tool usage and LLM responses
- Optional remote streaming to external API
- Instantiated in composition root, not as module-level singleton

### Scheduler (`agent/main.py`)
- Event-driven loop processing `HeartbeatEvent` and `HumanInputEvent`
- Dispatches to appropriate handler based on event type
- Container lifecycle management

### ApiService (`agent/api/server.py`)
- `ApiService` ABC with `NullApiService` and `UvicornApiService` implementations
- `UvicornApiService` wraps FastAPI with uvicorn server
- `POST /api/bot` — accepts human input, queues `HumanInputEvent`
- `GET /api/health` — health check
- Shares `asyncio.Queue` with Scheduler for event delivery

### Messaging (`agent/core/messaging.py`)
- `Messaging` ABC with `NullMessaging` and `WXMessaging` implementations
- `WXMessaging` accepts explicit `WXMessagingConfig` (DIP — no global settings)

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
| `container_runtime` | `podman` | Container runtime |
| `workspace_dir` | `./workspace` | Host workspace path |
| `wake_interval_seconds` | `1800` | Wake cycle interval |
| `api_host` | `0.0.0.0` | API server bind address |
| `api_port` | `8000` | API server port |

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
│   ├── api/
│   │   ├── __init__.py
│   │   └── server.py          # ApiService ABC + FastAPI endpoints
│   ├── core/
│   │   ├── events.py          # Event types
│   │   ├── event_logger.py    # Event logging
│   │   ├── messaging.py       # Messaging ABC + implementations
│   │   └── settings.py        # Configuration
│   ├── llm/
│   │   ├── agent.py           # Agent conversation loop
│   │   ├── base.py            # Abstract LLM client
│   │   ├── factory.py         # Client factory
│   │   ├── openai.py          # OpenAI implementation
│   │   └── prompt_builder.py  # System prompt construction
│   ├── tools/
│   │   ├── command_executor.py  # Container executor
│   │   ├── tool_registry.py  # Tool registration (OCP)
│   │   ├── toolbox.py        # Tool implementations
│   │   └── skill_loader.py   # Skill discovery
│   ├── app.py                # Composition root (DI)
│   └── main.py               # Entry point + Scheduler
├── tests/
│   ├── test_api.py           # API endpoint tests
│   ├── test_command_executor.py
│   ├── test_skill_loader.py
│   └── test_tool_registry.py # ToolRegistry tests
├── workspace/                # Persisted workspace
├── Containerfile             # Workspace container image
└── run-container.sh          # Container management
```

## 8. Design Principles

- **Single Responsibility**: Each class has one clear purpose (Agent ≠ ToolRegistry ≠ PromptBuilder)
- **Open/Closed**: New tools added via `ToolRegistry.register()` — no existing code changes
- **Liskov Substitution**: `CommandExecutor` implementations are interchangeable; `Messaging` implementations are interchangeable
- **Interface Segregation**: `CommandExecutor` protocol is focused; `Messaging` ABC is minimal
- **Dependency Inversion**: Core depends on abstractions (`OpenAIProvider`, `CommandExecutor`, `Messaging`, `ApiService`), not concretions. No module-level singletons — everything wired via composition root.

## 9. HTTP API

### POST /api/bot
Accept human input for agent processing.

**Request:**
```json
{"message": "Your message to the agent"}
```

**Response:**
```json
{"status": "queued"}
```

### GET /api/health
Health check endpoint.

**Response:**
```json
{"status": "ok"}
```