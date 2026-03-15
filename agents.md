# PawLia Architecture

Guide for AI agents and developers working on PawLia.

## Two-Tier Agent System

```
User message (any interface)
       │
       ▼
   ChatAgent (dispatcher)
   ├─ Has NO tools, only skill descriptions
   ├─ Decides: answer directly or call a skill?
   │
   ├─ Direct answer → return to user
   │
   └─ Skill needed → spawn SkillRunnerAgent
       ├─ Has real tools (bash, reminders)
       ├─ Mode 1: Tool-call (LLM calls bash directly)
       ├─ Mode 2: Command fallback (LLM outputs shell command as text)
       └─ Returns raw result → ChatAgent formulates final answer
```

### ChatAgent (`pawlia/agents/chat.py`)

Dispatcher with a two-turn pattern:

1. **Turn 1:** Send user message + skill specs (as OpenAI tools) to LLM. The LLM either answers directly or requests a skill call.
2. **Turn 2:** If skills were called, feed their results back to the LLM (without tool bindings) for a final answer.

The ChatAgent also handles:
- Building the system prompt from identity files + memory + summary
- Replaying recent exchanges as structured message pairs
- Persisting exchanges to the daily log
- Triggering conversation summarization (as background asyncio task)

### SkillRunnerAgent (`pawlia/agents/skill_runner.py`)

Executes a single skill. Dual-mode with `command_fallback` parameter:

- **Tool-call mode:** LLM calls bash/tools via `bind_tools`. Multi-turn loop (max 5 turns).
- **Command mode** (fallback for small models): LLM outputs a shell command in a ` ```bash ` block. Command is extracted via regex and executed. Returns raw output — no LLM interpretation to avoid hallucination.

Retry: Up to `MAX_RETRIES=2` if both modes produce no output.

The SkillRunner receives **no conversation history** — it's isolated to prevent hallucination propagation from chat context.

### BaseAgent (`pawlia/agents/base.py`)

- `_invoke(messages, llm)` — async LLM call via `asyncio.to_thread()`
- `strip_thinking(text)` — removes `<think>`/`<thinking>` blocks (for reasoning models)
- `extract_text(response)` — extracts clean text from AIMessage

## App (`pawlia/app.py`)

Central state holder. Creates and wires everything:

- Two LLMs via `LLMFactory` — `chat_llm` and `runner_llm` (allows different models)
- `ToolRegistry` with BashTool + ReminderTool
- Skills via `SkillLoader.discover()`
- `MemoryManager` for session persistence
- `Scheduler` for proactive reminders and event notifications
- `make_agent(user_id)` — factory that creates a ChatAgent with a bound SkillRunner factory

## Interfaces

All interfaces follow the same pattern: get an agent via `app.make_agent(user_id)`, call `agent.run(text)`, return the response.

| Interface | File | Transport | Agent per |
|-----------|------|-----------|-----------|
| CLI | `interfaces/cli.py` | stdin/stdout (async reader) | fixed `cli_user` |
| Telegram | `interfaces/telegram.py` | python-telegram-bot polling | Telegram user ID |
| Matrix | `interfaces/matrix.py` | matrix-nio sync loop | Matrix sender |
| Webhook | `interfaces/webhook.py` | aiohttp `POST /chat` | `user_id` from JSON body |

Server mode (`--mode server`) starts all configured interfaces in parallel via `asyncio.gather`.

Each interface registers a notification callback with the Scheduler for proactive messages:
- **CLI**: Overwrites current prompt line, prints notification, reprints `You: `
- **Telegram**: Sends via `bot.send_message()` to tracked `chat_id`
- **Matrix**: Sends via `client.room_send()` to tracked `room_id`
- **Webhook**: Buffers notifications, polled via `GET /notifications?user_id=...`

## Skills (`pawlia/skills/loader.py`)

Skills are self-contained directories with a `SKILL.md` (YAML frontmatter + instructions).

```
skills/
├── bahn/
│   ├── SKILL.md          # name, description, instructions
│   └── scripts/bahn.mjs  # actual tool
├── browser/
│   ├── SKILL.md
│   └── scripts/browser.py
└── user/                  # custom user skills (gitignored)
    └── bike-routing/
        ├── SKILL.md
        └── scripts/route.py
```

`SkillLoader.discover()` scans direct children of `skills/` plus `skills/user/` subdirectories. Skills with `requires_config` in metadata are skipped if config is missing.

The ChatAgent sees skills as OpenAI function specs (name + description + query param). The SkillRunnerAgent gets the full instructions and runs in the skill's directory (`cwd = skill.skill_path`).

SKILL.md supports variable substitution: `<user_id>`, `<session_dir>`, `<scripts_dir>`.

## Tools (`pawlia/tools/`)

| Tool | Name | Purpose |
|------|------|---------|
| `BashTool` | `bash` | Execute shell commands. Respects `context["cwd"]` and `context["timeout"]` (default 120s). |
| `ReminderTool` | `schedule_reminder` | CRUD for reminders in `session/{user_id}/reminders.json`. Delivery via Scheduler. |

Tools extend `Tool(ABC)` and register in `ToolRegistry`. Each tool provides `as_openai_spec()` for LLM binding and `execute(args, context)` for actual execution.

## Memory & Sessions (`pawlia/memory.py`)

```
session/{user_id}/workspace/
├── memory/
│   ├── 2026-03-15.md        # daily chat log (append-only)
│   ├── memory.md             # persistent user facts
│   └── context_summary.md    # LLM-generated conversation summary
├── soul.md                   # agent personality (from template)
├── IDENTITY.md               # agent identity (from template)
├── USER.md                   # user context (from template)
└── bootstrap.md              # onboarding (removed once identity files are filled)
```

**System prompt** is built from all `.md` files in workspace root + summary + memory + skill instructions.

**Summarization triggers:**

| Trigger | Threshold | Constant |
|---------|-----------|----------|
| Exchange count | 20 | `MAX_EXCHANGES_BEFORE_SUMMARY` |
| Bot response repetition | 0.6 similarity | `SIMILARITY_THRESHOLD` (window of 4) |
| Idle timeout | 300s | `IDLE_TIMEOUT_SECONDS` |

Summarization runs as a background `asyncio.create_task()`. The summary **replaces** (not appends to) the previous summary — the LLM receives the prior summary as context and merges everything into max 4 bullet points.

## Scheduler (`pawlia/scheduler.py`)

Background asyncio task that runs every 60 seconds and scans all user sessions for:

- **Due reminders** (`session/{user_id}/reminders.json`): Fires when `fire_at <= now`. Handles recurrence (daily/weekly/monthly) by advancing `fire_at`. One-time reminders are marked `fired: true`.
- **Upcoming events** (`session/{user_id}/calendar/events.json`): Notifies 15 minutes before `start`. Marks events with `_notified` flag to avoid duplicates.

Interfaces register async callbacks via `scheduler.register(callback)`. The scheduler calls all registered callbacks when a notification is due.

## LLM Configuration (`pawlia/llm.py`)

```json
{
  "providers": {
    "ollama": {"apiBase": "http://192.168.177.120:11434/v1", "apiKey": "ollama"},
    "groq": {"apiBase": "https://api.groq.com/openai/v1", "apiKey": "gsk_..."}
  },
  "agents": {
    "defaults": {"model": "qwen3:4b", "provider": "ollama", "temperature": 0.7},
    "chat": {"model": "qwen3:4b", "think": false},
    "skill_runner": {"model": "qwen3.5:latest", "think": true, "temperature": 0.3}
  }
}
```

`LLMFactory.create(agent_type)` merges agent-specific config over defaults, resolves the provider, and returns a `ChatOpenAI` instance. Any OpenAI-compatible API works.

## Development Guidelines

- Run tests: `.venv/bin/python -m pytest tests/ -x -q`
- Keep it simple — PawLia targets small models on local hardware
- Skills are isolated: own directory, own config, no shared state
- Tools are pluggable: extend `Tool`, register in `App.__init__`
- Don't add conversation history to SkillRunnerAgent — isolation prevents hallucination
- Command mode returns raw script output — no LLM interpretation phase
