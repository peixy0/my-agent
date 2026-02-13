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
AppWithDependencies (app.py)
  ├── Settings (configuration)
  ├── EventLogger (remote logging)
  ├── ContainerCommandExecutor (command execution)
  ├── ToolRegistry (tool management)
  ├── OpenAIProvider (OpenAI-compatible API)
  ├── Agent (conversation loop)
  ├── Messaging (WeChat/Feishu/Null)
  └── ApiService (FastAPI/Null)

Scheduler (main.py)
  ├── HeartbeatEvent → autonomous wake cycles
  └── HumanInputEvent → API-triggered interactions
```

## Composition Root Pattern

`AppWithDependencies` is the **single place** where all dependencies are wired together. This eliminates scattered singletons and makes the dependency graph explicit and testable.

**Location**: `agent/app.py`

```python
class AppWithDependencies:
    def __init__(self, settings: Settings | None = None):
        # All dependencies created and wired here
        self.settings = settings or get_settings()
        self.event_queue = asyncio.Queue()
        self.event_logger = EventLogger(...)
        self.executor = ContainerCommandExecutor(...)
        self.tool_registry = ToolRegistry(...)
        self.llm_client = LLMFactory.create(...)
        self.agent = Agent(self.llm_client)
        self.messaging = create_messaging(...)
        self.api_service = create_api_service(...)

    async def run(self) -> None:
        """Start all background tasks."""
        self._background_tasks = [
            asyncio.create_task(self.event_logger.run()),
            asyncio.create_task(self.messaging.run()),
            asyncio.create_task(self.api_service.run()),
        ]
```

### Benefits
- **Testability**: Inject mock dependencies via constructor
- **Clarity**: See entire object graph in one place
- **Flexibility**: Swap implementations without touching client code

## Extension Points

### 1. Adding New Tools

Tools are registered in `agent/tools/toolbox.py`. The `ToolRegistry` follows the **Open/Closed Principle**—add new tools without modifying existing code.

**Step 1**: Implement the tool handler
```python
async def my_tool_handler(param1: str, param2: int) -> dict:
    """Your tool implementation."""
    result = await do_something(param1, param2)
    return {"result": result}
```

**Step 2**: Define the JSON schema
```python
MY_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "my_tool",
        "description": "What this tool does",
        "parameters": {
            "type": "object",
            "properties": {
                "param1": {"type": "string", "description": "..."},
                "param2": {"type": "integer", "description": "..."},
            },
            "required": ["param1", "param2"],
        },
    },
}
```

**Step 3**: Register in `register_default_tools()`
```python
def register_default_tools(registry: ToolRegistry, executor, skill_loader):
    # ... existing tools ...
    registry.register("my_tool", my_tool_handler, MY_TOOL_SCHEMA)
```

### 2. Adding New Messaging Backends

The `Messaging` abstraction allows pluggable notification systems.

**Location**: `agent/core/messaging.py`

**Step 1**: Implement the `Messaging` ABC
```python
class MyMessaging(Messaging):
    def __init__(self, config: MyMessagingConfig):
        self._config = config
        self.event_queue = asyncio.Queue()

    async def run(self) -> None:
        """Background task processing messages."""
        while True:
            event = await self.event_queue.get()
            if isinstance(event, SendMessageRequest):
                await self._send(event.content)
            self.event_queue.task_done()

    async def send_message(self, message: str) -> None:
        """Queue a message for sending."""
        await self.event_queue.put(SendMessageRequest(content=message))

    async def _send(self, message: str) -> None:
        # Actual API call to messaging service
        ...
```

**Step 2**: Update `create_messaging()` factory
```python
def create_messaging(settings: Settings, event_queue: asyncio.Queue) -> Messaging:
    if settings.my_messaging_enabled:
        config = MyMessagingConfig(...)
        return MyMessaging(config)
    # ... existing backends ...
    return NullMessaging()
```

### 3. Adding New API Service Implementations

The `ApiService` abstraction allows different HTTP frameworks or disabling the API entirely.

**Location**: `agent/api/server.py`

**Current implementations**:
- `NullApiService`: No-op when API is disabled
- `UvicornApiService`: FastAPI with Uvicorn

**To add a new implementation**:
```python
class MyApiService(ApiService):
    async def run(self) -> None:
        # Start your HTTP server
        ...
```

## Design Patterns

### 1. Strategy Pattern
**CommandExecutor**: Swap execution strategies (container vs. local)
```python
class ContainerCommandExecutor:
    async def execute(self, command: str) -> CommandResult:
        # Execute in container via podman/docker exec
```

### 2. Factory Pattern
**LLMFactory**, **create_messaging()**, **create_api_service()**
- Encapsulate object creation logic
- Return appropriate implementation based on configuration

### 3. Observer Pattern
**Event Queue**: Scheduler observes events from API and timer
```python
await event_queue.put(HumanInputEvent(content=message))
# Scheduler picks up and processes
```

### 4. Dependency Inversion Principle
Core logic depends on **abstractions**, not concretions:
- `Messaging` (not `FeishuMessaging`)
- `ApiService` (not `UvicornApiService`)
- `CommandExecutor` protocol (not specific runtime)

## Event-Driven Scheduler

**Location**: `agent/main.py`

The `Scheduler` processes two event types:

1. **HeartbeatEvent**: Periodic autonomous wake-ups
   - Triggered by `_schedule_heartbeat()` after each cycle
   - Agent works on tasks, decides whether to report to user

2. **HumanInputEvent**: API-triggered interactions
   - Queued by `POST /api/bot` endpoint
   - Agent processes immediately, responds via messaging

```python
while self.running:
    event = await self.app.event_queue.get()
    if isinstance(event, HeartbeatEvent):
        await self._process_heartbeat()
    elif isinstance(event, HumanInputEvent):
        await self._process_human_input(event)
    self.app.event_queue.task_done()
```

## Testing Strategy

### Unit Tests
- Test individual components with mocked dependencies
- `AppWithDependencies` accepts `Settings` for testing

```python
def test_tool_registry():
    registry = ToolRegistry(tool_timeout=10)
    registry.register("test_tool", handler, schema)
    assert "test_tool" in registry.list_tools()
```

### Integration Tests
- Test API endpoints with TestClient
- Test command executor with real container

```python
async def test_api_endpoint():
    app = create_api(asyncio.Queue())
    client = TestClient(app)
    response = client.post("/api/bot", json={"message": "test"})
    assert response.status_code == 200
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
- Private members: `_leading_underscore`

### Code Organization
- **Single Responsibility**: Each class/function has one clear purpose
- **Dependency Injection**: Avoid module-level singletons
- **Type Hints**: All functions must have complete type annotations
- **Docstrings**: Public APIs must have docstrings

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
├── app.py                   # Composition root (AppWithDependencies)
├── main.py                  # Entry point, Scheduler
├── api/
│   └── server.py           # ApiService abstraction + FastAPI
├── core/
│   ├── events.py           # Event types (HeartbeatEvent, HumanInputEvent)
│   ├── event_logger.py     # Remote event streaming
│   ├── messaging.py        # Messaging abstraction (WeChat, Feishu, Null)
│   └── settings.py         # Configuration (Pydantic)
├── llm/
│   ├── agent.py            # Conversation loop
│   ├── factory.py          # LLM client factory
│   ├── openai.py           # OpenAI implementation
│   └── prompt_builder.py   # System prompt construction
└── tools/
    ├── command_executor.py  # Container command execution
    ├── skill_loader.py      # Skill discovery
    ├── tool_registry.py     # Tool registration (OCP)
    └── toolbox.py           # Tool implementations
```

## References

- [Specification](docs/specification.md): Detailed system specification
- [README](README.md): User guide and setup instructions
- [pyproject.toml](pyproject.toml): Dependencies and tool configuration

