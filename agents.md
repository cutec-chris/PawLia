# PawLia Architecture

Guide for AI agents and developers working on PawLia.

## Local Test Environment

Run Python tests from the repository virtual environment, not the system Python.
Use `.venv/bin/python -m pytest ...` (or `.venv/bin/pytest ...`) for test runs.

## Three-Tier Agent System

```
User message (any interface)
       │
       ▼
   RouterAgent (backend dispatcher, pawlia/agents/router.py)
   ├─ Resolves the active backend per turn (per thread)
   ├─ backend: pawlia → ChatAgent
   └─ backend: hermes → HermesBackend (pawlia/backends/hermes.py)
       │
       ▼
   ChatAgent (local stack, pawlia/agents/chat.py)
   ├─ Has NO tools, only skill descriptions
   ├─ Decides: answer directly or call a skill?
   │
   ├─ Direct answer → return to user
   │
   └─ Skill needed → spawn SkillRunnerAgent
       ├─ Has real tools (bash)
       ├─ Mode 0: Workflow mode (if compiled workflow.yaml)
       ├─ Mode 1: Tool-call (LLM calls bash directly)
       ├─ Mode 2: Command fallback (LLM outputs shell command as text)
       └─ Returns raw result → ChatAgent formulates final answer
```

### RouterAgent (`pawlia/agents/router.py`)

The outer dispatcher returned by `App.make_agent(user_id)`. On every `run()` it resolves the backend for the active thread (or main conversation) and forwards the turn:

- `backend: pawlia` (default) → wraps a `ChatAgent` and routes the turn through the full local stack with skills.
- `backend: hermes` → wraps a `HermesBackend` (`pawlia/backends/hermes.py`) and forwards the turn to a remote Hermes-Agent server. PawLia still owns memory, summarization, identity/system-prompt assembly, and notifications; only the chat-completion call is delegated.

Backend selection follows the same per-thread / session / global override chain as model selection (`/model <path> <name>`).

### ChatAgent (`pawlia/agents/chat.py`)

Dispatcher with a two-turn pattern:

1. **Turn 1:** Send user message + skill specs (as OpenAI tools) to LLM. The LLM either answers directly or requests a skill call.
2. **Turn 2:** If skills were called, feed their results back to the LLM (without tool bindings) for a final answer.

The ChatAgent also handles:
- Building the system prompt from identity files + memory + summary
- Replaying recent exchanges as structured message pairs
- Replaying earlier skill-backed exchanges in a compact text form instead of
  restoring full historical tool payloads into context
- Persisting exchanges to the daily log
- Triggering conversation summarization (as background asyncio task)

### SkillRunnerAgent (`pawlia/agents/skill_runner.py`)

Executes a single skill. Dual-mode with `command_fallback` parameter:

- **Mode 0: Workflow mode:** If a compiled `workflow.yaml` exists and is valid,
  PawLia prefers it first. Building blocks become tools, and execution is
  verified programmatically where possible.
- **Tool-call mode:** LLM calls bash/tools via `bind_tools`. Multi-turn loop.
  The default budget is `MAX_TOOL_TURNS = 30`, overridable per-skill via
  `metadata.max_tool_turns` in `SKILL.md` and per-model via `max_tool_turns`
  in `models:` (config.yaml). When neither is set, `estimate_max_tool_turns`
  in `pawlia/llm.py` derives a budget from the model identifier (frontier
  APIs get 40, smaller local models scale down).
- **Command mode** (fallback for small models): LLM outputs a shell command in a ` ```bash ` block. Command is extracted via regex and executed. Returns raw output — no LLM interpretation to avoid hallucination.

Retry: Up to `MAX_RETRIES=2` if both modes produce no output.

The SkillRunner receives **no conversation history** — it's isolated to prevent hallucination propagation from chat context.

### BaseAgent (`pawlia/agents/base.py`)

- `_invoke(messages, llm)` — async LLM call via `asyncio.to_thread()`
- `strip_thinking(text)` — removes `<think>`/`<thinking>` blocks, handles unclosed tags, and strips chat-template tokens (e.g. `<|...|>`)
- `extract_text(response)` — extracts clean text from AIMessage

## App (`pawlia/app.py`)

Central state holder. Creates and wires everything:

- LLMs via `LLMFactory` — `chat_llm`, `vision_llm`, and per-skill LLMs (allows different models)
- `ToolRegistry` with BashTool
- Skills via `SkillLoader.discover()` (built-in + user + workspace)
- `MemoryManager` for session persistence
- `Scheduler` for proactive reminders, events, checklists, jobs, and background tasks
- `make_agent(user_id)` — factory that returns a `RouterAgent` wrapping either `ChatAgent` (pawlia backend) or `HermesBackend` (hermes backend), with `make_runner` and `make_local_agent` factories bound for per-turn dispatch

If `config["models"]` is empty when the process starts, `__main__.py` automatically launches the Web UI for first-run setup instead of dropping into the CLI prompt.

## Interfaces

All interfaces follow the same pattern: get a router via `app.make_agent(user_id)`, call `agent.run(text)`, return the response.

| Interface | File | Transport | Agent per |
|-----------|------|-----------|-----------|
| CLI | `interfaces/cli.py` | stdin/stdout (async reader) | fixed `cli_user` |
| Telegram | `interfaces/telegram.py` | python-telegram-bot polling | Telegram user ID |
| Matrix | `interfaces/matrix.py` | matrix-nio sync loop | Matrix sender |
| Web | `interfaces/web.py` | aiohttp (HTML + JSON API) | Cookie-authenticated user |
| Webhook | `interfaces/webhook.py` | aiohttp `POST /chat` | `user_id` from JSON body |
| OpenAI-compatible | `interfaces/openai_compat.py` | aiohttp `POST /v1/chat/completions` (+ Ollama-style `/api/chat`) | `user_id` from `X-User-Id` header or default |

Server mode (`--mode server`) starts all configured interfaces in parallel via `asyncio.gather`.

**End-to-end API evaluation workflow:** For full-system tests, run PawLia in `--mode server` with `interfaces.openai` enabled and send normal chat turns to `POST /v1/chat/completions`. This path exercises the real `RouterAgent`/`ChatAgent`, memory, workspace search, and skill stack exactly as external clients would use it. For stable multi-turn sessions, keep the same `X-User-Id` header (or `user` field) across requests; do not test individual skill scripts directly if the goal is to validate the integrated agent behavior.

Each interface registers a notification callback with the Scheduler for proactive messages:
- **CLI**: Overwrites current prompt line, prints notification, reprints `You: `
- **Telegram**: Sends via `bot.send_message()` to tracked `chat_id`
- **Matrix**: Sends via `client.room_send()` to tracked `room_id`
- **Web**: Server-sent events stream pushes notifications to the open browser tab
- **Webhook**: Buffers notifications, polled via `GET /notifications?user_id=...`
- **OpenAI-compatible**: No notification channel (stateless API)

**Thread Support**: ChatAgent supports `thread_id` parameter for isolated conversation contexts. Threads have their own exchange logs, model overrides, and can be marked private (no disk persistence).

**Private Mode**: CLI supports `/private` command to enable session-level private mode. Exchanges are kept in RAM only and not written to disk. Per-thread private mode is also supported.

## Skills (`pawlia/skills/loader.py`)

Skills are self-contained directories with a `SKILL.md` (YAML frontmatter + instructions).

```
skills/
├── automation/           # scheduled jobs (formerly part of organizer)
├── browser/              # headless browsing via Playwright
├── config/               # read/write workspace config files
├── files/                # workspace file CRUD + grep + outline
├── memory/               # long-term RAG memory + Dream Wiki triggers
├── organizer/            # events, tasks, reminders (calendar/tasks.md)
├── perplexica/           # web answer engine
├── researcher/           # per-project document collections (workspace-local)
├── searxng/              # meta-search
├── skill-creator/        # build new skills interactively
├── workspace-git/        # commit/squash/sync the workspace repo
└── user/                 # custom user skills (gitignored)
    └── bike-routing/
        ├── SKILL.md
        └── scripts/route.py
```

`SkillLoader.discover()` scans direct children of `skills/` plus `skills/user/` subdirectories. With `skill-install.allow_workspace: true`, also discovers skills from `session/{user_id}/workspace/skills/`. Skills with `requires_config` in metadata are skipped if config is missing.

The ChatAgent sees skills as OpenAI function specs (name + description + query param). The SkillRunnerAgent gets the full instructions and runs in the skill's directory (`cwd = skill.skill_path`) by default. This can be overridden via `openclaw.cwd` metadata ("skill" or "workspace").

SKILL.md supports variable substitution: `<user_id>`, `<session_dir>`, `<scripts_dir>`.

## Tools (`pawlia/tools/`)

| Tool | Name | Purpose |
|------|------|---------|
| `BashTool` | `bash` | Execute shell commands. Respects `context["cwd"]` and `context["timeout"]` (default 120s). Injects `PAWLIA_USER_ID` and `PAWLIA_SESSION_DIR` as env vars. |

**Note**: `ReminderTool` exists in code but is not registered in the App. Reminders are managed directly by the Scheduler.

Tools extend `Tool(ABC)` and register in `ToolRegistry`. Each tool provides `as_openai_spec()` for LLM binding and `execute(args, context)` for actual execution.

## Memory & Sessions (`pawlia/memory.py`)

```
session/{user_id}/
├── session_version.txt             # on-disk log/session format version
├── workspace/                      # Obsidian vault
│   ├── memory/
│   │   ├── 2026-03-15.md           # daily chat log (main + embedded thread sections)
│   │   ├── memory.md                # persistent user facts
│   │   ├── context_summary.md       # LLM-generated conversation summary
│   ├── config.yaml                  # session-level overrides (agents, tts, disabled_skills, user)
│   ├── calendar/                   # Full Calendar plugin events
│   │   └── 2026-04-10 Meeting.md   # one .md per event (frontmatter)
│   ├── tasks.md                    # Obsidian Tasks plugin format
│   ├── wiki/                       # Dream Wiki (structured knowledge base)
│   │   ├── index.md                # catalog of all pages
│   │   ├── log.md                  # chronological audit log
│   │   └── topics/                 # one .md page per topic/entity
│   ├── research/                   # researcher skill projects
│   │   └── {project}/
│   │       ├── README.md           # project metadata (name, description)
│   │       └── {sha1}.md           # scraped document (SHA1 of URL as filename)
│   ├── soul.md                     # agent personality (from template)
│   ├── identity.md                 # agent identity (from template)
│   ├── user.md                     # user context (from template)
│   ├── bootstrap.md                # onboarding (removed once identity files are filled)
│   ├── skills/                     # workspace skills (with allow_workspace: true)
│   └── .git/                       # optional: auto-managed by workspace git sync
├── scheduler_state.json            # internal scheduler flags (notified, fired, etc.)
├── automations/
│   └── jobs.json                   # scheduled automation jobs
└── memory_index/                   # RAG backend index (not in vault)
    └── dreamed_files.json          # Dream Wiki tracking
```

**System prompt** is built from the workspace identity files actually used by the
code (`bootstrap.md`, `identity.md`, `user.md`, `soul.md`, `memory.md`) plus
the conversation summary, current time block, mode-specific instructions, skill
instructions, and — on the first turn of each session — a **Workspace-Referenzen**
block with relevant workspace content (see Workspace Search below).

Identity files (`soul.md`, `identity.md`, `user.md`) are copied as templates from `pawlia/prompts/` on first use. `bootstrap.md` is removed once all three identity files are filled and differ from their templates.

**Workspace Search** (`pawlia/workspace_search.py`): On each **substantive user turn**, `ChatAgent` runs a BM25 keyword search across the workspace's navigable markdown knowledge sources and stores the current hits as `session.workspace_refs`. Results are injected into the user message as a `## Workspace Notes Available` block so the model knows what's available and can navigate further via the files skill.

- **Scope:** `workspace/wiki/topics/`, `workspace/research/`, loose `.md` files at workspace root. Excludes `workspace/memory/` (raw chat logs — the Dream Wiki is the distilled version) and all identity files. This search stays available even when a different memory backend is configured.
- **Algorithm:** BM25 via `rank_bm25`; falls back to simple term-frequency if the library is absent. Retrieval units are Markdown sections (ATX headings, or paragraph chunks for long headingless files) rather than whole files.
- **Ranking boosts:** heading matches, page-title/path matches, `## Related` sections, and lightly linked wiki pages receive small additive boosts. No embeddings are used.
- **Injection format:** Each hit renders as an Obsidian-compatible section ref such as `[[topic/slug#Heading]]`, `[[person/max#Heading]]`, or `[[path/to/file#Heading]]` plus a file-level `files read` suggestion. The files skill resolves path-style wikilinks to workspace paths.
- **Config** (`workspace-search:` in `config.yaml`): `enabled`, `top_k` (default 5), `min_score` (0–1, fraction of best hit), `snippet_chars`, `exclude_dirs`, `include_root_files`.

**Summarization triggers:**

| Trigger | Threshold | Constant |
|---------|-----------|----------|
| Token budget reached | `summary_threshold_tokens(model)` | resolved per turn from `LLMFactory` |
| Token budget exceeded by 50 % | bypasses the idle gate | `should_summarize` returns `"tokens_force"` |
| Exchange count (safety net) | 20 | `MAX_EXCHANGES_BEFORE_SUMMARY` |
| Bot response repetition | 0.6 similarity | `SIMILARITY_THRESHOLD` (window of 4) |
| Idle timeout | 300s | `IDLE_TIMEOUT_SECONDS` |
| Force threshold (exchange count) | 30 | `FORCE_SUMMARY_EXCHANGES` |

The token threshold for the active chat model resolves to `summarize_at_tokens` (absolute) if set in `models:`, otherwise `summarize_at_fraction × context_size` (default fraction `0.6`). Token usage is approximated cheaply via `estimate_session_tokens()` (chars / 4) — tiktoken would dominate the scheduler tick.

Summarization runs as a background `asyncio.create_task()`. The summary **replaces** (not appends to) the previous summary — the LLM receives the prior summary as context and merges everything into max 4 bullet points.

## Scheduler (`pawlia/scheduler.py`)

Background asyncio task that runs every 60 seconds and scans all user sessions for:

**High priority (every tick, no idle requirement):**
- **Due reminders** (`workspace/tasks.md`, lines with 🔔 and ⏳): Fires when the scheduled time has passed. State (notified flag, last-fired) lives in `scheduler_state.json`.
- **Upcoming events** (`workspace/calendar/*.md`, per-event files with Full Calendar frontmatter): Notifies 15 minutes before `start`. Marked in `scheduler_state.json` to avoid duplicates.
- **Checklists**: Event-frontmatter checklists processed via `ChecklistProcessor`.
- **Task reminders**: Tasks-file reminders processed via `TaskReminderProcessor`.
- **Automation jobs**: Runs scheduled automation jobs via `JobRunner` (`automations/jobs.json`).
- **Token-forced summarization**: When `should_summarize` returns `"tokens_force"`, summarize immediately even if the user is active.
- **Exchange-count-forced summarization**: When `exchange_count ≥ FORCE_SUMMARY_EXCHANGES` (30), summarize immediately.

**Low priority (idle-gated, LLM must be free):**
- **Conversation summarization**: After 5 min idle (`IDLE_SUMMARIZE_MIN`). Soft `"tokens"` and legacy `"exchange_limit"`/`"repetition"` triggers fire here.
- **Background tasks** (`/background`): After 10 min idle (`IDLE_BACKGROUND_MIN`).
- **Memory indexing** (RAG backend): After configurable idle time, default 20 min (`IDLE_MEMORY_MIN`, override via `skill-config.memory.idle_minutes`).
- **Workspace git auto-commit** (when `workspace-git.enabled`): throttled to max 1 commit per 5 minutes; daily and weekly squashes run at the configured time.

The scheduler provides LLM priority gating via `acquire_llm()` / `release_llm()` to block background tasks during active chat sessions. Interfaces register async callbacks via `scheduler.register(callback)` for proactive message delivery.

## LLM Configuration (`pawlia/llm.py`)

**New format (recommended):** Models are defined separately, agents reference them by key.

```yaml
providers:
  ollama:
    apiBase: http://localhost:11434/v1
    apiKey: ollama
    timeout: 120
  groq:
    apiBase: https://api.groq.com/openai/v1
    apiKey: gsk_...

models:
  fast:
    model: qwen3:4b
    provider: ollama
    temperature: 0.7
  smart:
    model: qwen3.5:latest
    provider: ollama
    temperature: 0.9
    think: true
  vision:
    model: qwen2.5vl:latest
    provider: ollama

agents:
  default: smart       # fallback for any unspecified agent type
  chat: smart
  skill_runner: fast
  vision: vision
  skills:              # per-skill overrides
    searxng: fast
    browser: smart
```

**Fallback chains:**
- `get("chat")` → `agents.chat` → `agents.default`
- `get("vision")` → `agents.vision` → `agents.chat` → `agents.default`
- `get("skill_runner")` → `agents.skill_runner` → `agents.chat` → `agents.default`
- `get("skill.<name>")` → `agents.skills.<name>` → `agents.skill_runner` → `agents.chat` → `agents.default`
- `get("compiler")` → `agents.compiler` → `agents.skill_runner` → `agents.chat` → `agents.default` (used by the skill workflow compiler)

**Valid agent paths for `/model <path> <name>`:** `default`, `chat`, `skill_runner`, `vision`, `compiler`, `skills.<name>` (see `_VALID_AGENT_PATHS` in `pawlia/interfaces/common.py`).

**Legacy format** (inline model config) is also supported for backwards compatibility.

`LLMFactory.get(agent_type)` merges agent-specific config over defaults, resolves the provider, and returns a `ChatOpenAI` or `ChatOllama` instance. Any OpenAI-compatible API works.

**Per-model heuristics** (`pawlia/llm.py`):
- `estimate_max_tool_turns(name)` — derives the SkillRunner tool-call budget from the model identifier; override per-model via `max_tool_turns` in `models:` or per-skill via `metadata.max_tool_turns` in `SKILL.md`.
- `estimate_context_size(name)` — derives `num_ctx` for Ollama-backed models so prompts don't get silently truncated to Ollama's 2048 default; override via `context_size` (alias `num_ctx`).
- `summary_threshold_tokens(name)` — derives the token threshold for auto-summarization; override via `summarize_at_tokens` (absolute) or `summarize_at_fraction` (of context_size).
- `audio_input: true` on a model marks it as natively audio-capable; PawLia bypasses Whisper transcription for those models in the VoIP and voice-message paths.

## Development Guidelines

**Working directory:** Always run commands from the project root (the directory containing `pawlia/` and `requirements.txt`).

**Virtualenv + PYTHONPATH:** `pawlia` is not installed as a package (no `pyproject.toml`), so `PYTHONPATH=.` is required for any command that imports it.

```bash
# Run tests
PYTHONPATH=. .venv/bin/pytest tests/ -x -q

# Start CLI
PYTHONPATH=. .venv/bin/python -m pawlia

# Start CLI with piped input (non-interactive, for scripted testing)
printf 'Hello\nexit\n' | PYTHONPATH=. .venv/bin/python -m pawlia 2>/dev/null
```

- Keep it simple — PawLia targets small models on local hardware
- Skills are isolated: own directory, own config, no shared state
- Tools are pluggable: extend `Tool`, register in `App.__init__`
- Don't add conversation history to SkillRunnerAgent — isolation prevents hallucination
- Prefer compiled workflows for reliability when a skill provides one; free-form
  tool calling remains the fallback path inside the SkillRunner
- Command mode returns raw script output — no LLM interpretation phase
