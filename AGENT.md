# Agent Developer Reference

This document is a comprehensive guide for developers working on the autonomous agent codebase. It covers architecture, design patterns, extension points, and coding standards.

## Architecture Overview

The agent follows **SOLID principles** with emphasis on:
- **Dependency Injection** via composition root
- **Abstract interfaces** for pluggable components
- **Event-driven scheduling** for autonomous and interactive modes
- **Container isolation** for safe command execution

### Core Components

```
engine/App (engine/app.py)
  ├── Settings (configuration)
  ├── Runtime (ContainerRuntime or HostRuntime)
  ├── ToolRegistry (tool management)
  ├── OpenAIProvider (LLM client)
  ├── Agent (conversation loop)
  ├── SystemPromptBuilder (prompt construction)
  ├── Gateway (FeishuGateway/None)
  └── ApiService (FastAPI)

Scheduler (engine/scheduler.py)
  ├── ConversationWorker  — per-chat, sequential processing (engine/worker.py)
  ├── CronWorker          — aiocron lifecycle per session (engine/worker.py)
  ├── HeartbeatEvent      → autonomous wake cycles
  ├── CronEvent           → scheduled cron task execution
  ├── TextInputEvent      → human chat messages
  ├── ImageInputEvent     → image messages
  ├── NewSessionEvent     → reset conversation
  └── DropSessionEvent    → tear down worker (e.g. WS disconnect)
```

## Composition Root Pattern

`App` is the **single place** where all dependencies are wired together. This eliminates scattered singletons and makes the dependency graph explicit and testable.

**Location**: `agent/engine/app.py`

```python
class App:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.event_queue: asyncio.Queue[AgentEvent] = asyncio.Queue()
        self.runtime = ContainerRuntime(...) if settings.container_runtime else HostRuntime(...)
        self.skill = SkillLoader(settings.skills_dir)
        self.tool_registry = ToolRegistry()
        register_default_tools(self.tool_registry, self.runtime, self.skill, settings)
        self.gateway = create_gateway(settings, self.event_queue, self.runtime)
        self.api_service = create_api_service(settings, self.event_queue)
        self.prompt_builder = SystemPromptBuilder(self.skill)
        self.llm_client = OpenAIProvider(url=settings.openai_base_url, api_key=settings.openai_api_key)
        self.model_name = settings.openai_model
        self.agent = Agent(self.llm_client, self.model_name, self.tool_registry)

    async def run(self) -> None:
        os.chdir(self.settings.cwd)
        self.background_tasks = [asyncio.create_task(self.api_service.run())]
        if self.gateway is not None:
            self.background_tasks.append(asyncio.create_task(self.gateway.run()))
```

### Benefits
- **Testability**: Inject mock dependencies via constructor
- **Clarity**: See entire object graph in one place
- **Flexibility**: Swap implementations without touching client code

## Extension Points

### 1. Adding New Tools

Tools are registered in `agent/tools/toolbox.py`. The `ToolRegistry` follows the **Open/Closed Principle** — add new tools without modifying existing code.

**Step 1**: Implement the tool handler inside `register_default_tools()`. The function's docstring becomes the tool description; the function name becomes the tool name.
```python
async def my_tool(param1: str, param2: int) -> ToolContent:
    """
    What this tool does — used directly as the description sent to the LLM.
    """
    result = await do_something(param1, param2)
    return ToolContent.from_dict("success", {"result": result})
```

**Step 2**: Register by passing the function and its **parameters** schema (not a full `{"type": "function", ...}` wrapper):
```python
registry.register(
    my_tool,
    {
        "type": "object",
        "properties": {
            "param1": {"type": "string", "description": "..."},
            "param2": {"type": "integer", "description": "..."},
        },
        "required": ["param1", "param2"],
    },
)
```

Use the optional `name=` and `description=` keyword arguments to override the function name or docstring:
```python
registry.register(my_tool, schema, name="my_tool", description="Override description")
```

Tools only available during human interaction (e.g. `add_reaction`, `send_image`) should be registered by overriding `Channel.register_tools()` — they are cloned into a per-turn `HumanInputOrchestrator`.

### 2. Adding New Messaging Backends

The `Gateway` / `Channel` pair abstracts inbound and outbound messaging.

**Location**: `agent/messaging/gateway.py` (factory), `agent/core/messaging.py` (ABCs)

**Step 1**: Implement `Gateway` (background task that enqueues events) and a paired `Channel` (reply handle for a single turn). Override `register_tools` to expose backend capabilities as agent tools:
```python
class MyChannel(Channel):
    async def send(self, text: str) -> None: ...
    async def start_thinking(self) -> None: ...
    async def end_thinking(self) -> None: ...

    def register_tools(self, registry: ToolRegistry) -> None:
        # Register only what this backend supports
        async def add_reaction(emoji: str) -> ToolContent: ...
        registry.register(add_reaction, schema)

class MyGateway(Gateway):
    async def run(self) -> None:
        async for msg in my_stream():
            channel = MyChannel(msg)
            await self.event_queue.put(
                TextInputEvent(chat_id=msg.chat_id, message=msg.text,
                               message_id=msg.id, sender=channel)
            )
```

**Step 2**: Update `create_gateway()` factory in `agent/messaging/gateway.py`:
```python
def create_gateway(settings, event_queue, runtime) -> Gateway | None:
    if settings.my_backend_enabled:
        return MyGateway(settings, event_queue)
    return None
```

Tools registered via `register_tools` are added to the orchestrator's cloned registry at the start of each human-input turn, so they appear in the LLM's tool list *only* when that backend is active.

### 3. Adding New API Service Implementations

The `ApiService` abstraction allows different HTTP frameworks or disabling the API entirely.

**Location**: `agent/api/server.py`

**Current implementations**:
- `UvicornApiService`: FastAPI with Uvicorn serving WebSocket + health check

**To add a new implementation**:
```python
class MyApiService(ApiService):
    async def run(self) -> None:
        # Start your HTTP server
        ...
```

## Design Patterns

### 1. Strategy Pattern
**Runtime**: Swap execution strategies (`ContainerRuntime` vs. `HostRuntime`)
```python
runtime = ContainerRuntime(container_name=...) if settings.container_runtime else HostRuntime()
result = await runtime.execute("ls /workspace")
```

### 2. Template Method / ABC
**Orchestrator**: `BackgroundOrchestrator` and `HumanInputOrchestrator` share the tool-dispatch loop inherited from `Orchestrator` but differ in how they handle streaming content and final responses.

### 3. Observer Pattern
**Event Queue**: Scheduler observes events pushed by API and messaging sources
```python
await event_queue.put(TextInputEvent(chat_id=..., message="hello", ...))
# Scheduler picks up and routes to the right ConversationWorker
```

### 4. Dependency Inversion
The `Gateway` / `Channel` pair abstracts inbound and outbound messaging.
- `Gateway`: background task that receives inbound messages and queues events
- `Channel`: reply handle for a single conversation turn; advertises backend capabilities as tools via `register_tools`

## Event-Driven Scheduler

**Location**: `agent/engine/scheduler.py`, `agent/engine/worker.py`

Event types in `agent/core/events.py`:

| Event | Trigger |
|---|---|
| `TextInputEvent` | Inbound text message |
| `ImageInputEvent` | Inbound image message |
| `HeartbeatEvent` | `/heartbeat [seconds]` command or recurring timer |
| `CronEvent` | Scheduled cron task fired from a loaded job group |
| `NewSessionEvent` | `/new` command — resets conversation history |
| `DropSessionEvent` | WebSocket disconnect — cancels the worker |

`Scheduler` routes each event to a `ConversationWorker` keyed by `chat_id`. Workers process events sequentially; different chats run concurrently.

`SchedulerContext` is a `Protocol` that captures only what `Scheduler` needs from `App` — `App` satisfies it structurally, avoiding any upward import.

`ConversationWorker` and `CronWorker` live in `engine/worker.py` and take **explicit constructor deps** — neither depends on `SchedulerContext` nor on each other.

```python
while self.running:
    event = await self.app.event_queue.get()
    await self._dispatch(event)
    self.app.event_queue.task_done()
```

### Slash commands (parsed by `Scheduler._dispatch_text`)

| Command | Effect |
|---|---|
| `/heartbeat [seconds]` | Start recurring wake cycle |
| `/cron load <job>` | Load and schedule `.cron/<job>/` tasks |
| `/cron unload <job>` | Stop a loaded job group |
| `/cron ls` | List available / loaded jobs |
| `/new` | Reset conversation history |
| `/drop` | Tear down session worker |

## Testing Strategy

### Unit Tests
- Test individual components with mocked dependencies
- `App` accepts a `Settings` instance for dependency injection

```python
def test_registry():
    registry = ToolRegistry()

    async def handler() -> ToolContent:
        """test tool"""
        return ToolContent.from_dict("success", {})

    registry.register(handler, {"type": "object", "properties": {}, "required": []})
    assert registry.get_handler("handler") is not None
```

### Integration Tests
- Test API endpoints with WebSocket TestClient
- Test runtime with real container or `HostRuntime`

```python
async def test_ws_endpoint():
    queue: asyncio.Queue[AgentEvent] = asyncio.Queue()
    app = create_fastapi_app(queue)
    client = TestClient(app)
    with client.websocket_connect("/api/bot") as ws:
        ws.send_json({"type": "text", "message": "hi", "message_id": "1"})
```

## Coding Standards

### Python Version
- **3.12+** managed via `uv`

### Style
- **Formatting**: `ruff format` (no manual formatting)
- **Linting**: `ruff check` (no errors allowed in PRs)
- **Type Checking**: `basedpyright` in standard mode
- **Async**: Use `asyncio` for all I/O operations

### Naming Conventions
- Variables/Functions: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Private helper methods: `_leading_underscore` (e.g., `_compress_conversation`)
- Instance variables: **no `_` prefix** — use plain names (`self.sender`, not `self._sender`)
- Module-level private helpers: `_leading_underscore` (e.g., `_truncate`)

### Code Organization
- **Single Responsibility**: Each class/function has one clear purpose
- **Dependency Injection**: Avoid module-level singletons
- **Type Hints**: All functions must have complete type annotations
- **Docstrings**: Public APIs must have docstrings
- **No class-scope member declarations**: Do not pre-declare instance variables as class-level annotations. Assign instance variables directly in `__init__` (e.g., `self.name = name`, not a separate `name: str` in the class body).

## Development Workflow

### Pre-Commit Checklist

```bash
# 1. Format code
uv run ruff format .

# 2. Lint and auto-fix
uv run ruff check --fix .

# 3. Type check
uv run basedpyright

# 4. Run tests
uv run pytest tests/ -v

# Or run all at once:
uv run ruff format . && uv run ruff check . && uv run basedpyright && uv run pytest
```

### Tools

**Ruff**: Fast Python linter and formatter
- Config: `[tool.ruff]` in `pyproject.toml`
- Replaces: black, isort, flake8

**Basedpyright**: Strict type checker (Pyright fork)
- Config: `[tool.basedpyright]` in `pyproject.toml`
- Mode: standard (strict)

**Pytest**: Testing framework
- Config: `[tool.pytest]` in `pyproject.toml`
- Fixtures: `tests/conftest.py`

## Project Structure

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
│   ├── messaging.py             # Channel/Gateway ABCs
│   ├── events.py                # Event types + WorkerEvent union
│   ├── runtime.py               # Runtime ABC, ContainerRuntime, HostRuntime
│   └── settings.py              # Configuration (Pydantic)
├── llm/
│   ├── agent.py                 # Agent + Orchestrator ABC + BackgroundOrchestrator + HumanInputOrchestrator
│   ├── factory.py               # LLM client factory
│   ├── openai.py                # OpenAI implementation
│   ├── prompt.py                # SystemPromptBuilder
│   └── types.py                 # Shared LLM type definitions
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

### Dependency graph (one-way, no cycles)

```
core/      channel, events, settings, runtime   (no agent imports)
  ↑
tools/     → core/
llm/       → core/, tools/
messaging/ → core/, gateway
api/       → core/, messaging/websocket
  ↑
engine/    → core/, llm/, tools/, messaging/, api/
  ↑
main.py    → engine/, core/settings
```

## References

- [Specification](docs/specification.md): Detailed system specification
- [README](README.md): User guide and setup instructions
- [pyproject.toml](pyproject.toml): Dependencies and tool configuration
