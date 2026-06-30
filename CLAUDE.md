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

- **Never push to a git remote on your own initiative.** Push only when the user
  explicitly instructs it in the current turn. The `github` and `codeberg` remotes
  are off-limits — they are updated exclusively via `scripts/release_public.sh` after
  a squash-merge to `main`. The `origin` remote (local ssh) may be pushed to on
  explicit user instruction.
- **Skills over core changes.** New capabilities belong in `skills/`, not in the scheduler
  or agent core, unless they are cross-cutting (security, correctness).
- **Obsidian-compatible workspace.** Workspace files must stay readable by plain Obsidian.

## Development Guidelines

See [agents.md § Development Guidelines](agents.md#development-guidelines) for the full list.
