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
- **Talk to it** — voice messages are transcribed automatically; Matrix VoIP calls let you speak to PawLia directly
- **Threads** — isolate side conversations in Telegram forum topics or Matrix threads, each with its own context and model
- **Multi-user** — every user gets their own session with separate memory, history, and model settings
- **Remembers** — conversation history and user facts persisted as Markdown, with automatic summarization
- **Switch models on the fly** — `/model qwen3:4b` swaps the LLM at runtime, per-thread or session-wide
- **Extensible** — drop a `SKILL.md` in `skills/user/` and the agent picks it up automatically
- **Proactive** — built-in scheduler delivers reminders and calendar alerts through your active interface
- **Private mode** — `/private` prevents messages from being written to disk

## Quick Start

```bash
cp config.sample.yaml config.yaml
# edit config.yaml — add your provider URL and bot tokens

docker compose up -d
```

See [docs/installation.md](docs/installation.md) for full setup instructions, including manual installation for development.

## Interfaces

| Interface | Voice | Images | Threads | VoIP |
|-----------|:-----:|:------:|:-------:|:----:|
| **Telegram** | transcription | vision agent | forum topics | — |
| **Matrix** | transcription | vision agent | thread replies | full duplex |
| **CLI** | — | — | `/thread` | — |
| **Webhook** | — | base64 | via `thread_id` | — |

All interfaces run simultaneously in server mode.

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
