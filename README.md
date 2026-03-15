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

A lightweight, open-source AI assistant built for small language models (e.g., Qwen 3.5 9b). PawLia brings persistent memory, multi-user sessions, and extensible skills to local hardware — no cloud API required.

## Key Features

### 🔌 Multiple Interfaces
All interfaces can run simultaneously in server mode:
- **CLI** — interactive terminal session
- **Telegram** — bot with voice transcription and image support
- **Matrix** — Element-compatible bot with voice transcription and image support
- **Webhook** — HTTP endpoint for custom integrations (`POST /chat`, `GET /notifications`)

### 👥 Multi-User Sessions
Each user gets their own isolated session with per-user memory and conversation history. Sessions are persisted to disk as Markdown files and expire from RAM after inactivity.

### ⏰ Scheduler
PawLia can act proactively — a background scheduler checks every 60 seconds for:
- **Due reminders** from the built-in reminder tool (with daily/weekly/monthly recurrence)
- **Upcoming calendar events** from the organizer skill (15 min before start)

Notifications are delivered directly through the active interface (CLI, Telegram, Matrix) or buffered for polling (Webhook).

### 🛠️ AgentSkills
PawLia supports the [AgentSkills](https://agentskills.io) specification. Skills are self-contained directories with a `SKILL.md` and optional scripts.

Each skill runs as a **sub-agent** with its own LLM session and access to tools (Bash, etc.). The dispatcher (ChatAgent) decides which skill to call based on the user's request.

**User skills:** Place custom skills in `skills/user/`. They are loaded exactly like built-in skills — same format, same `SKILL.md` frontmatter. The `user/` directory is gitignored so personal skills don't pollute the repo.

### ⚙️ Provider-Agnostic LLM
Any OpenAI-compatible API works as a backend — Groq, OpenRouter, Ollama, vLLM, or any local server.

Every agent type can use a **different model and provider**. The lookup follows a fallback chain so you only configure what you want to override:

| Agent type | Fallback chain |
|------------|----------------|
| `chat` | `agents.chat` → `defaults` |
| `skill_runner` | `agents.skill_runner` → `defaults` |
| `vision` | `agents.vision` → `agents.chat` → `defaults` |
| `skill.<name>` | `agents.skills.<name>` → `agents.skill_runner` → `defaults` |

Minimal example — one model for everything:
```yaml
agents:
  defaults:
    model: qwen3.5:latest
    provider: ollama
```

With a dedicated vision model and a fast/cheap model for search:
```yaml
agents:
  defaults:
    model: qwen3.5:latest
    provider: ollama

  vision:
    model: qwen2.5vl:latest

  skills:
    searxng:
      model: qwen3:4b
      provider: groq
```

LLMs with identical configuration are reused across agent types — no redundant connections.

## Quick Start

1. Copy `config.sample.yaml` to `config.yaml` and fill in your provider credentials.
2. Run in CLI mode:
   ```bash
   python -m pawlia
   ```
3. Run in server mode (Telegram, Matrix, Webhook):
   ```bash
   python -m pawlia --mode server
   ```
4. With Docker:
   ```bash
   docker compose up -d
   ```

## Project Structure

```
pawlia/
├── pawlia/              # Python package
│   ├── agents/          # ChatAgent (dispatcher), SkillRunnerAgent
│   ├── interfaces/      # CLI, Telegram, Matrix, Webhook
│   ├── tools/           # Built-in tools (bash, reminders)
│   ├── skills/          # Skill loader
│   ├── scheduler.py     # Proactive reminders & event notifications
│   └── memory.py        # Session & memory management
├── skills/              # Installed skill packages
│   └── user/            # Custom user skills (gitignored)
├── session/             # Per-user session data
├── config.yaml          # Your configuration
└── config.sample.yaml   # Configuration reference
```

## Configuration

See `config.sample.yaml` for all available options. Both `.yaml` and `.json` are supported.

| Key | Purpose |
|-----|---------|
| `providers` | LLM backend(s) and API keys |
| `agents.defaults` | Default model, provider, temperature |
| `agents.chat` | Override for the chat agent |
| `agents.skill_runner` | Default LLM for all skill sub-agents |
| `agents.vision` | LLM used when the user sends an image |
| `agents.skills.<name>` | Per-skill LLM override |
| `interfaces` | Enable Telegram / Matrix / Webhook |
| `skill-config` | Per-skill configuration (URLs, API keys, …) |
