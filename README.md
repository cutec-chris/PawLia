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
cp config.sample.yaml config.yaml
# edit config.yaml — add your provider URL and bot tokens

docker compose up -d
```

See [docs/installation.md](docs/installation.md) for full setup instructions, including manual installation for development.

## Interfaces

CLI · Telegram · Matrix · Webhook — all run simultaneously in server mode. Telegram and Matrix support voice messages, images, and threads. Matrix additionally supports VoIP calls.

→ [docs/interfaces.md](docs/interfaces.md)

## Skills

Skills are self-contained sub-agents — drop a `SKILL.md` in `skills/user/` and it loads automatically. Bundled: searxng · perplexica · browser · files · organizer.

→ [docs/skills.md](docs/skills.md)

## Documentation

- [Installation](docs/installation.md) — Docker setup, first steps
- [Interfaces](docs/interfaces.md) — CLI, Telegram, Matrix, Webhook, sessions, scheduler
- [Configuration](docs/config.md) — providers, models, agents, fallback chain
- [Skills](docs/skills.md) — bundled skills, custom skills, SKILL.md format
- [Commands](docs/commands.md) — `/thread`, `/model`, `/private`

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
