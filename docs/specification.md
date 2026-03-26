# Autonomous LLM Agent: Specification

## 1. Overview

This document specifies a system-level autonomous LLM agent. The agent runs on the **host machine** and optionally uses a **container as a workspace environment** for executing commands and file operations. It accepts human input via a **FastAPI HTTP endpoint** (WebSocket) and optionally via **Feishu** messaging.

## 2. Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      Host Machine                       │
│  ┌───────────────────────────────────────────────────┐  │
│  │                App (app.py)                       │  │
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
│  │  │ (events) │◄─│  Queue   │◄─│  WS /api/bot     │ │  │
│  │  └────┬─────┘  └──────────┘  └──────────────────┘ │  │
│  │       │                                           │  │
│  │  ┌────▼──────────────────────────────────────┐    │  │
│  │  │          Runtime (Strategy)               │    │  │
│  │  │  ContainerRuntime or HostRuntime          │    │  │
│  │  └────┬──────────────────────────────────────┘    │  │
│  └───────│───────────────────────────────────────────┘  │
│          │ podman exec  (ContainerRuntime only)         │
│  ┌───────▼───────────────────────────────────────────┐  │
│  │           Workspace Container (optional)          │  │
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

### App (`agent/engine/app.py`)
- **Composition root** — single place where all dependencies are wired
- Creates: Settings → runtime → ToolRegistry → OpenAIProvider → Agent → Gateway → ApiService
- `run()` method: changes to `cwd`, starts background tasks (`gateway`, `api_service`) if they are configured
- Selects `ContainerRuntime` when `container_runtime` setting is non-empty; falls back to `HostRuntime` otherwise

### Agent (`agent/llm/agent.py`)
- Manages the LLM conversation loop via `run(system_prompt, messages, orchestrator)`
- `compress()` — incremental conversation compression: serialises history into a structured summary
- **Does NOT** own tool registration, system prompt construction, or message sending (SRP)
- Clones the shared `ToolRegistry` per `Orchestrator` so concurrent workers don't interfere

### Orchestrator (`agent/llm/agent.py`)
- ABC with two implementations:
  - `HumanInputOrchestrator` — sends partial content before tool calls; calls `channel.register_tools(registry)` at construction time to expose backend-specific capabilities as agent tools
  - `BackgroundOrchestrator` — silently executes tools; only sends final response if content is non-empty and does not end with `NO_REPORT`
- `process(message, finish_reason)` — dispatches tool calls or handles final response
- `_handle_tool_call()` — resolves handler from registry, validates args against JSON schema, executes

### SystemPromptBuilder (`agent/llm/prompt.py`)
- Builds the system prompt from the current OS, skills, and workspace bootstrap files
- Bootstrap files loaded from CWD with mtime-based caching: `IDENTITY.md`, `USER.md`, `MEMORY.md`, `CONTEXT.md`
- `build()` — base prompt
- `build_with_conversation_summary(summary)` — base prompt + prior compression summary section
- `build_for_heartbeat()` — base prompt + `HEARTBEAT.md`
- `build_for_cron()` — base prompt + `CRON.md`

### ToolRegistry (`agent/tools/registry.py`)
- Holds tool name → (handler, schema) mappings
- Wraps handlers with timeout and error handling
- `clone()` — returns a shallow copy of the registry used per-orchestrator
- Adding new tools requires only a `register()` call (OCP)

### LLM Client (`agent/llm/`)
- `CompletionClient` Protocol used by `Agent` — only requires `do_completion()`
- `OpenAIProvider` — OpenAI-compatible API with retry logic

### Runtime (`agent/core/runtime.py`)
- `Runtime` ABC — Strategy pattern for command execution
- `ContainerRuntime` — executes commands via `podman exec` inside the workspace container; transfers files via base64
- `HostRuntime` — executes commands directly on the host machine
- Both implement: `execute()`, `read_file()`, `write_file()`, `read_raw_bytes()`
- `edit_file()` — default implementation on `Runtime` base: fuzzy-matches search blocks using `difflib.SequenceMatcher` (ratio ≥ 0.6) and reports the closest match on failure

### Scheduler (`agent/engine/scheduler.py`)
- `Scheduler` consumes from the shared event queue and dispatches to per-chat `ConversationWorker` tasks
- `SchedulerContext` Protocol decouples the scheduler from `App` — no upward import needed
- Slash commands are parsed inside `Scheduler._dispatch_text` and translated to typed events

### Workers (`agent/engine/worker.py`)
- `ConversationWorker` — owns a private asyncio.Queue, processes events sequentially; constructed with explicit deps (`Settings`, `Agent`, `SystemPromptBuilder`, `OrchestratorFactory`) — no dependency on `SchedulerContext`
- `CronWorker` — manages aiocron lifecycle for one chat session; constructed with `chat_id`, an `asyncio.Queue`, and `CronLoader` — no dependency on `ConversationWorker`

### ApiService (`agent/api/server.py`)
- `ApiService` ABC with `UvicornApiService` implementation
- `UvicornApiService` wraps FastAPI with uvicorn server
- `WS /api/bot` — WebSocket endpoint; each connection gets its own `chat_id` session
- `GET /api/health` — health check
- Queues `TextInputEvent` / `ImageInputEvent` on message; queues `DropSessionEvent` on disconnect

### Channel and Gateway (`agent/core/messaging.py`)
- **Channel** — ABC for a bound reply handle for a single conversation turn
  - Methods: `send(text)`, `start_thinking()`, `end_thinking()`, `register_tools(registry)`
  - Concrete backends (Feishu, WebSocket) implement `register_tools` to selectively add features like reactions or file sending as standard agent tools.
- **Gateway** — ABC for a background task that receives inbound messages and queues events
- Decouples core logic from messaging backends: core depends on `Channel`/`Gateway` abstractions, not concrete providers.

## 4. Tools

### Default tools (always registered)

| Tool | Description |
|------|-------------|
| `run_command` | Execute shell commands in the workspace (container or host) |
| `read_file` | Read file content with pagination (max 500 lines, `start_line` for pagination) |
| `write_file` | Write content to a file (parent dirs created automatically) |
| `edit_file` | Surgically replace exact text blocks; fuzzy-suggests closest match on failure |
| `grep` | Regex search across files; supports context lines, glob include, case flag |
| `glob` | List files matching a glob pattern (supports `**`) |
| `web_search` | Search the web via DuckDuckGo |
| `fetch` | Fetch and extract main content from a web page via trafilatura |
| `use_skill` | Load detailed instructions for a named skill |
| `read_image` | Read image file as vision content block (only when `vision_support=true`) |

### Channel-specific tools (registered via `Channel.register_tools`)

| Tool | Description |
|------|-------------|
| `add_reaction` | React to the current message with an emoji |
| `send_image` | Send an image file to the user |
| `send_file` | Send a file to the user |

## 5. Configuration

Settings are managed via `pydantic-settings` and loaded from `.env`:

| Setting | Default | Description |
|---------|---------|-------------|
| `openai_base_url` | `https://api.openai.com/v1` | LLM API endpoint |
| `openai_model` | `gpt-4o` | Model to use |
| `openai_api_key` | `""` | API key |
| `container_name` | `sys-agent-workspace` | Workspace container name |
| `container_runtime` | `""` | Container runtime (`podman`/`docker`); empty = use `HostRuntime` |
| `tool_timeout` | `60` | Default tool execution timeout (seconds) |
| `max_output_chars` | `10000` | Max characters returned from command output |
| `web_search_proxy` | `""` | HTTP proxy for web search |
| `cwd` | `./workspace` | Working directory the agent changes into on startup |
| `project_dir` | *(project root)* | Absolute path to the project root (auto-resolved) |
| `skills_dir` | `./.skills` | Directory containing skill definitions |
| `crons_dir` | `./.cron` | Directory containing cron job definitions |
| `wake_interval_seconds` | `1800` | Heartbeat interval when none specified |
| `context_auto_compression_enabled` | `false` | Enable automatic conversation compression |
| `context_max_tokens` | `100000` | Token threshold triggering auto-compression |
| `context_num_keep_last` | `9` | Number of recent messages kept verbatim during compression |
| `feishu_app_id` | `""` | Feishu app ID |
| `feishu_app_secret` | `""` | Feishu app secret |
| `feishu_encrypt_key` | `""` | Feishu event encrypt key |
| `feishu_verification_token` | `""` | Feishu verification token |
| `vision_support` | `false` | Enable vision (image input) support |
| `max_image_size_bytes` | `5242880` | Max image size (5 MB) |
| `webui_enabled` | `true` | Enable the WebSocket API server |
| `webui_host` | `localhost` | API server bind address |
| `webui_port` | `8017` | API server port |

## 6. Skills

Skills are Markdown files in `.skills/<name>/SKILL.md` with YAML frontmatter:

```yaml
---
name: skill-name
description: What this skill does
---

# Skill Instructions
...
```

Skills are discovered at startup and listed in the system prompt. The `use_skill` tool returns the full instructions on demand.

## 7. Cron Jobs

Cron job definitions live in `.cron/<job-name>/*.md`. Each Markdown file represents one task:

```yaml
---
name: optional-task-name   # defaults to filename stem
cron: "0 9 * * 1-5"        # standard cron expression (required)
---

Task prompt / instructions sent to the agent when this task fires.
```

- Files without a `cron` frontmatter key are silently skipped
- Files within a job group are loaded in lexicographic order
- Slash commands manage cron jobs at runtime (see Section 9)

## 8. Context Compression

When `context_auto_compression_enabled` is true and `total_tokens` reaches `context_max_tokens`:

1. `ConversationWorker._compress_conversation()` is triggered before the next LLM call
2. All messages except the last `context_num_keep_last` are sent to `Agent.compress()`
3. `Agent.compress()` pairs tool calls with their results and generates a structured Markdown summary (sections: Active Tasks, Completed Tasks, Established Facts, Key Files, Pending Issues)
4. The summary is stored in `Conversation.previous_summary` and prepended to future prompts via `build_with_conversation_summary()`
5. The compressed messages are discarded; the retained tail replaces the full history

## 9. Event System

### Event types (`agent/core/events.py`)

| Event | Trigger |
|---|---|
| `TextInputEvent` | Inbound text message |
| `ImageInputEvent` | Inbound image message |
| `HeartbeatEvent` | `/heartbeat [seconds]` command or recurring wake timer |
| `CronEvent` | Scheduled cron task fired by `CronWorker` |
| `NewSessionEvent` | `/new` command — resets conversation history |
| `DropSessionEvent` | WebSocket/session disconnect — tears down the worker |

### Slash commands (intercepted by `Scheduler._dispatch_text`)

| Command | Description |
|---|---|
| `/heartbeat [seconds]` | Start a recurring autonomous wake cycle |
| `/cron load <job>` | Load and schedule cron tasks from `.cron/<job>/` |
| `/cron unload <job>` | Stop a loaded cron job group |
| `/cron ls` | List available and loaded cron jobs |
| `/new` | Reset conversation history |
| `/drop` | Tear down the current session worker |

`Scheduler` routes each event to a `ConversationWorker` keyed by `chat_id`. Workers process events sequentially; different chats run concurrently.

`SchedulerContext` is a `Protocol` capturing only what the scheduler needs (`settings`, `agent`, `prompt_builder`, `orchestrator_factory`, `event_queue`) — `App` satisfies it structurally, avoiding upward imports.

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

## 11. Project Structure

```
agent/
├── main.py                      # Entry point only
├── engine/
│   ├── app.py                   # Composition root (App)
│   ├── scheduler.py             # Scheduler, SchedulerContext
│   └── worker.py                # ConversationWorker, CronWorker, Conversation
├── api/
│   └── server.py                # ApiService ABC + FastAPI + WebSocket
├── core/
│   ├── messaging.py             # Channel ABC + Gateway ABC
│   ├── events.py                # Event types
│   ├── runtime.py               # Runtime ABC, ContainerRuntime, HostRuntime
│   └── settings.py              # Configuration (Pydantic)
├── llm/
│   ├── agent.py                 # Agent + Orchestrator ABC + BackgroundOrchestrator + HumanInputOrchestrator + OrchestratorFactory + DefaultOrchestratorFactory
│   ├── openai.py                # OpenAI implementation
│   ├── prompt.py                # SystemPromptBuilder
│   └── types.py                 # Shared LLM type views
├── messaging/
│   ├── feishu.py                # FeishuGateway + FeishuChannel
│   ├── gateway.py               # create_gateway factory
│   └── websocket.py             # WebSocketChannel
└── tools/
    ├── registry.py              # ToolRegistry (OCP)
    ├── toolbox.py               # Tool implementations (default)
    ├── skill.py                 # SkillLoader
    ├── cron.py                  # CronLoader + CronJobDef
    └── markdown.py              # YAML frontmatter parser
```

## 12. Dependency Graph

One-way, no cycles:

```
core/           channel, events, settings, runtime   (no agent imports)
  ↑
tools/          → core/
llm/            → core/, tools/
messaging/      → core/, gateway
api/            → core/, messaging/websocket
  ↑
engine/         → core/, llm/, tools/, messaging/, api/
  ↑
main.py         → engine/, core/settings
```

## 13. Design Principles

- **Single Responsibility**: Each class has one clear purpose (Agent ≠ ToolRegistry ≠ PromptBuilder ≠ Orchestrator)
- **Open/Closed**: New tools added via `ToolRegistry.register()` — no existing code changes
- **Liskov Substitution**: `Runtime` implementations are interchangeable; `Gateway` implementations are interchangeable
- **Interface Segregation**: `Runtime` ABC is focused; `Channel` ABC is minimal; `SchedulerContext` Protocol exposes only what the scheduler needs
- **Dependency Inversion**: Core depends on abstractions, not concretions. No module-level singletons — everything wired via composition root (`App`). Factories like `create_gateway` and `create_api_service` return `Optional` types instead of null objects when features are disabled.
