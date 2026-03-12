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
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │  │
│  │  │  Agent   │  │   LLM    │  │  ToolRegistry    │ │  │
│  │  │  (conv   │──│  Client  │  │  (OCP: add tools │ │  │
│  │  │   loop)  │  │          │  │   without edits) │ │  │
│  │  └──────────┘  └──────────┘  └──────────────────┘ │  │
│  │                                                   │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │  │
│  │  │Scheduler │  │  Event   │  │  ApiService      │ │  │
│  │  │ (events) │◄─│  Queue   │◄─│  POST /api/bot   │ │  │
│  │  └────┬─────┘  └──────────┘  └──────────────────┘ │  │
│  │       │                                           │  │
│  │  ┌────▼──────────────────────────────────────┐    │  │
│  │  │          Command runtime                  │    │  │
│  │  │          (ContainerRuntime)               │    │  │
│  │  └────┬──────────────────────────────────────┘    │  │
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

### AppWithDependencies (`agent/engine/app.py`)
- **Composition root** — single place where all dependencies are wired
- Creates: Settings → runtime → ToolRegistry → OpenAIProvider → Agent → Messaging → ApiService
- `run()` method starts all background tasks (messaging, API server)
- Replaces scattered module-level singletons with explicit construction

### Agent (`agent/llm/agent.py`)
- Manages conversation history and LLM interaction loop
- Validates structured responses against JSON schemas
- **Does NOT** handle tool registration or system prompt construction (SRP)

### SystemPromptBuilder (`agent/llm/prompt.py`)
- Builds the system prompt from settings, skills, and runtime context
- Separated from Agent for single responsibility

### ToolRegistry (`agent/tools/registry.py`)
- Holds tool name → (handler, schema) mappings
- Wraps handlers with timeout and error handling
- Adding new tools requires only a `register()` call (OCP)

### LLM Client (`agent/llm/`)
- Factory pattern for LLM client creation
- OpenAI-compatible API with retry logic
- Reads tool definitions from `ToolRegistry`

### Command runtime (`agent/core/runtime.py`)
- **Strategy pattern** for command execution
- `ContainerRuntime` executes commands via `podman exec`
- All file operations (read/write/edit) also execute in container
- Dependency injection enables testing

### Scheduler (`agent/engine/scheduler.py`)
- `Scheduler` consumes from the shared event queue and dispatches to per-chat `ConversationWorker` tasks
- `ConversationWorker` owns a private queue, processes its events sequentially; conversations run concurrently
- `SchedulerContext` Protocol decouples the scheduler from `AppWithDependencies` — no upward import needed
- Handles `TextInputEvent`, `ImageInputEvent`, `HeartbeatEvent`, `NewSessionEvent`, `DropSessionEvent`
- `/heartbeat [seconds]` and `/new` commands are translated to typed events inside `Scheduler._dispatch`
- `DropSessionEvent` tears down a worker (e.g. on WebSocket disconnect)

### ApiService (`agent/api/server.py`)
- `ApiService` ABC with `NullApiService` and `UvicornApiService` implementations
- `UvicornApiService` wraps FastAPI with uvicorn server
- `WS /api/bot` — WebSocket endpoint; each connection gets its own `chat_id` session
- `GET /api/health` — health check
- Queues `TextInputEvent` / `ImageInputEvent` on message; queues `DropSessionEvent` on disconnect
- Shares `asyncio.Queue` with Scheduler for event delivery

### Sender abstractions (`agent/core/sender.py`)
- `MessageSender` ABC — bound reply handle for a single conversation turn
- `MessageSource` ABC — background task that receives inbound messages
- `NullSender` / `NullSource` — no-op implementations
- Lives in `core/` so events and scheduler can depend on it without reaching into `messaging/`

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
| `project_dir` | `./workspace` | Host workspace path |
| `wake_interval_seconds` | `1800` | Wake cycle interval |
| `webui_host` | `0.0.0.0` | API server bind address |
| `webui_port` | `8000` | API server port |

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
│   ├── main.py                   # Entry point only
│   ├── engine/
│   │   ├── app.py                # Composition root (AppWithDependencies)
│   │   └── scheduler.py          # Scheduler, ConversationWorker, SchedulerContext
│   ├── api/
│   │   └── server.py             # ApiService ABC + FastAPI + WebSocket
│   ├── core/
│   │   ├── sender.py             # MessageSender/MessageSource abstractions
│   │   ├── events.py             # Event types
│   │   ├── runtime.py            # Command runtime
│   │   └── settings.py           # Configuration
│   ├── llm/
│   │   ├── agent.py              # Agent conversation loop
│   │   ├── factory.py            # Client factory
│   │   ├── openai.py             # OpenAI implementation
│   │   └── prompt.py             # System prompt construction
│   ├── messaging/
│   │   ├── feishu.py             # Feishu source + sender
│   │   ├── source.py             # MessageSource factory
│   │   └── websocket.py          # WebSocketSender
│   └── tools/
│       ├── registry.py           # Tool registration (OCP)
│       ├── toolbox.py            # Tool implementations
│       └── skill.py              # Skill discovery
├── tests/
│   ├── test_api.py
│   ├── test_agent_compress.py
│   ├── test_command_executor.py
│   ├── test_skill_loader.py
│   └── test_tool_registry.py
├── workspace/                    # Persisted workspace
├── Containerfile                 # Workspace container image
└── run-container.sh              # Container management
```

## 8. Dependency Graph

One-way, no cycles:

```
core/           sender, events, settings, runtime   (no agent imports)
  ↑
tools/          → core/
llm/            → core/, tools/
messaging/      → core/
api/            → core/, messaging/websocket
  ↑
engine/         → core/, llm/, tools/, messaging/, api/
  ↑
main.py         → engine/, core/settings
```

## 9. Design Principles

- **Single Responsibility**: Each class has one clear purpose (Agent ≠ ToolRegistry ≠ PromptBuilder)
- **Open/Closed**: New tools added via `ToolRegistry.register()` — no existing code changes
- **Liskov Substitution**: `Runtime` implementations are interchangeable; `MessageSource` implementations are interchangeable
- **Interface Segregation**: `Runtime` protocol is focused; `MessageSender` ABC is minimal
- **Dependency Inversion**: Core depends on abstractions, not concretions. No module-level singletons — everything wired via composition root.

## 10. HTTP / WebSocket API

### WS /api/bot
WebSocket endpoint. Each connection creates a new `chat_id` session.

Inbound frames:
```json
{"type": "text",  "message": "...", "message_id": "..."}
{"type": "image", "data": "<base64>", "mime_type": "image/jpeg", "message_id": "..."}
```

Outbound frames:
```json
{"type": "connected",  "chat_id": "..."}
{"type": "message",    "chat_id": "...", "text": "..."}
{"type": "image_path", "chat_id": "...", "path": "..."}
```

On disconnect a `DropSessionEvent` is queued, tearing down the worker.

### GET /api/health
Health check.

**Response:** `{"status": "ok"}`