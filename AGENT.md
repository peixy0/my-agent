# Agent Development Guide

This document outlines the coding standards, tools, and workflows for developing the `sys-agent` project.

## Coding Standards

- **Python Version**: 3.12+ (managed via `uv`)
- **Formatting**: Strictly follow `ruff format`. No manual formatting.
- **Linting**: No linting errors allowed in PRs. Use `ruff check`.
- **Type Checking**: All code must pass `basedpyright` in `standard` mode.
- **Async**: Use `asyncio` for non-blocking I/O.
- **Naming**: 
    - Variables/Functions: `snake_case`
    - Classes: `PascalCase`
    - Constants: `UPPER_SNAKE_CASE`

## Tools

### 1. Ruff
Fast Python linter and formatter. Replaces `black`, `isort`, and `flake8`.
- **Config**: Defined in `[tool.ruff]` in `pyproject.toml`.
- **Run**: `uv run ruff check .` or `uv run ruff format .`

### 2. Basedpyright
A more strict and feature-rich fork of Pyright.
- **Config**: Defined in `[tool.basedpyright]` in `pyproject.toml`.
- **Run**: `uv run basedpyright`

## Workflow (Step-by-Step)

For every change, follow these steps before committing:

### Step 1: Format Code
Automatically fix formatting and import sorting.
```bash
uv run ruff format .
```

### Step 2: Lint and Fix
Run linter and apply automatic fixes.
```bash
uv run ruff check --fix .
```

### Step 3: Type Check
Ensure all types are correct.
```bash
uv run basedpyright
```

### Step 4: Run Tests
Ensure no regressions.
```bash
uv run pytest
```

## Summary Command
To run all checks at once:
```bash
uv run ruff format . && uv run ruff check . && uv run basedpyright && uv run pytest
```

