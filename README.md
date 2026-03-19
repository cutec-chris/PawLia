```
     █   ░█░░
     ██▒░░████   ▓
     ▓██▓ ▓▓██  ░██     ___       ___       ___       ___       ___       ___
     ░▓██      ▓██▒    /\  \     /\  \     /\__\     /\__\     /\  \     /\  \
 ░█▒░ ░░░░ ░░  ▓▓█▓   /::\  \   /::\  \   /:/\__\   /:/  /    _\:\  \   /::\  \
 ░▓██▓█  █▓▓▒▒░      /::\:\__\ /::\:\__\ /:/:/\__\ /:/__/    /\/::\__\ /::\:\__\
  ░▓███ ░████████░    \/\::/  / \/\::/  / \::/:/  / \:\  \    \::/\/__/ \/\::/  /
       ░▓▓████████        \/__/    /:/  /   \::/  /   \:\__\    \:\__\     /:/  /
       ▓▓▓▓██████                  \/__/     \/__/     \/__/     \/__/     \/__/
        ░▓▓█
```

**A lightweight, open-source AI assistant built for local hardware.**

PawLia runs small language models (e.g. Qwen, Llama) with persistent memory, multi-user sessions, and extensible skills — no cloud required.

## Why PawLia?

- **Runs locally** — any OpenAI-compatible backend: Ollama, vLLM, Groq, OpenRouter
- **Meets you where you are** — Telegram, Matrix, CLI, or HTTP webhook, all at once
- **Remembers** — per-user memory and conversation history persisted as Markdown
- **Extensible** — drop a `SKILL.md` in `skills/user/` and the agent picks it up automatically
- **Proactive** — built-in scheduler delivers reminders and calendar alerts through your active interface

## Quick Start

```bash
# 1. Configure
cp config.sample.yaml config.yaml
# edit config.yaml — add your Ollama/Groq/etc. credentials

# 2. Run
python -m pawlia            # interactive CLI
python -m pawlia --mode server  # Telegram + Matrix + Webhook

# Or with Docker
docker compose up -d
```

## Interfaces

| Interface | How to run | Notes |
|-----------|-----------|-------|
| **CLI** | `python -m pawlia` | Interactive terminal |
| **Telegram** | `--mode server` | Voice, images, threads |
| **Matrix** | `--mode server` | Element-compatible, threads |
| **Webhook** | `--mode server` | `POST /chat`, `GET /notifications` |

All interfaces can run simultaneously in server mode.

## Skills

Skills are self-contained directories with a `SKILL.md` — same format as [AgentSkills](https://agentskills.io). Each skill runs as a sub-agent with its own LLM session.

**Bundled:** searxng · perplexica · browser · files · organizer

**Custom:** place your skill in `skills/user/` (gitignored) — it loads automatically.

## Documentation

- [Command reference](docs/commands.md) — `/model`, `/private`, and platform prefixes
- `config.sample.yaml` — full configuration reference with comments

## Project Structure

```
pawlia/
├── pawlia/          # Python package
│   ├── agents/      # ChatAgent (dispatcher), SkillRunnerAgent
│   ├── interfaces/  # CLI, Telegram, Matrix, Webhook
│   ├── tools/       # Built-in tools (bash, reminders)
│   └── memory.py    # Session & memory management
├── skills/          # Skill packages (user/ is gitignored)
├── session/         # Per-user session data
└── config.yaml      # Your configuration
```
