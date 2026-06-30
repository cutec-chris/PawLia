# PawLia — Agent Guide

## Key Documents

- **[vision.md](vision.md)** — long-term direction, active priorities, design decisions.
  Read this first when making trade-off decisions or planning larger changes.
- **[agents.md](agents.md)** — architecture deep-dive: agent system, scheduler, skills, tools,
  memory, LLM config, development guidelines, release process.

## Quick Start

Run tests via the venv (never bare `pytest` or system Python):

```bash
.venv/bin/python -m pytest tests/ -q
```

## Most Important Rules

- **Never write to a git remote from within the assistant.** The workspace-git feature was
  removed after a security incident — see `vision.md` for context. Do not re-introduce it.
- **Skills over core changes.** New capabilities belong in `skills/`, not in the scheduler
  or agent core, unless they are cross-cutting (security, correctness).
- **Obsidian-compatible workspace.** Workspace files must stay readable by plain Obsidian.

## Development Guidelines

See [agents.md § Development Guidelines](agents.md#development-guidelines) for the full list.
