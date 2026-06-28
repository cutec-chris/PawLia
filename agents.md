# PawLia Architecture

Guide for AI agents and developers working on PawLia.

## Test Commands

Always run tests through the repository virtual environment. Do **not** use
system Python, `python -m pytest`, or bare `pytest`; those may miss project
dependencies.

Use this pattern:

```bash
.venv/bin/python -m pytest <test-paths> -q
```

Examples:

```bash
.venv/bin/python -m pytest tests/test_scheduler.py -q
.venv/bin/python -m pytest tests/test_organizer_recurring.py tests/test_radicale_recurring.py -q
```

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

Supporting modules in `pawlia/agents/`:

- **`error_classifier.py`** — classifies LLM exceptions into an `ErrorCategory`
  (context_overflow, rate_limit, auth_error, timeout, server_error,
  format_error, unknown). `BaseAgent` uses `is_retryable()` /
  `should_compact()` to decide between retry, context compaction, or raising.
- **`iteration_budget.py`** — `IterationBudget`, a thread-safe per-agent
  iteration counter with one grace call after exhaustion and `refund()`
  support. Guards against skill-runner runaway loops; the maximum is derived
  from the model size in `app.py`.

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
| Discord | `interfaces/discord.py` | discord.py gateway | Discord user ID (threads via channel threads) |
| Webhook | `interfaces/webhook.py` | aiohttp `POST /chat` | `user_id` from JSON body |
| OpenAI-compatible | `interfaces/openai_compat.py` | aiohttp `POST /v1/chat/completions` (+ Ollama-style `/api/chat`) | `user_id` from `X-User-Id` header or default |

**Voice:** Matrix supports incoming WebRTC VoIP calls (`interfaces/matrix_call.py`,
requires `aiortc` — included in the `Dockerfile.voip` image, an Alpine-based
variant that adds aiortc, audio dependencies, and Node.js 20). Each call gets its own
isolated thread context; LLM responses are streamed sentence-by-sentence into TTS, and
a hold-audio loop (`tts.hold_audio`, default `assets/keyboard.m4a`) plays while the
agent thinks. `//stop` (or `//stop all`) cancels running skill turns in Matrix.
Discord has a voice counterpart in `interfaces/discord_voice.py`. Shared audio
plumbing (AGC, VAD, recording) lives in `pawlia/audio/`; STT in
`pawlia/transcription.py` (groq / openai / local Whisper), TTS in `pawlia/tts.py`
(edge / piper).

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
└── user/                 # custom user skills (gitignored, created on demand)
```

`SkillLoader.discover()` scans direct children of `skills/` plus `skills/user/` subdirectories. With `skill-install.allow_workspace: true`, also discovers skills from `session/{user_id}/workspace/skills/`. Skills with `requires_config` in metadata are skipped if config is missing.

**Skill config**: Global per-skill settings live under `skill-config:` in
`config.yaml`. The session-level `workspace/config.yaml` can override them per
skill; `app.py` merges session over global when building the skill context.

**Write access**: Skills may only write under `session/<user_id>/` and `/tmp`;
skill directories are out of bounds. When bubblewrap can create a nested user
namespace this is *enforced* (read-only root); under rootless podman it cannot,
so it degrades to a post-hoc detective scan of `/app/skills` + `/app/pawlia`
that reports stray writes but does not prevent them (see Tools › Write sandbox).
Skills that generate files — notably skill-creator — write into the user's
workspace instead.

**Credentials** (`pawlia/credentials.py`): Per-user credential store at
`session/.credentials/{user_id}.json` — deliberately **outside** the
sandbox-writable `session/{user_id}/` so skill code cannot read or tamper with
it via bash (legacy `session/{user_id}/.credentials.json` is migrated
automatically). Credentials are injected into skill processes as environment
variables (`build_env_extra()`, names derived via `env_key_for()`).

The ChatAgent sees skills as OpenAI function specs (name + description + query param). The SkillRunnerAgent gets the full instructions and runs in the skill's directory (`cwd = skill.skill_path`) by default. This can be overridden via `openclaw.cwd` metadata ("skill" or "workspace").

SKILL.md supports variable substitution: `<user_id>`, `<session_dir>`, `<scripts_dir>`.

**Coding backend** (`pawlia/coding.py`): `skill-creator`'s `implement`/`fix`
delegate script writing/debugging to a coding backend selected by `coding.backend`
(`opencode` | `aider` | `llm` | `auto`) or the per-skill override
`skill-config.skill-creator.coding_backend`. `auto` detects in order
**opencode → aider → llm**. opencode and aider run as subprocesses (own agentic
loop / turn budget); `llm` is a single in-process call via the `coder` model.
Each CLI uses its own model and authentication — PawLia does not pass
`--model` or forward provider API keys, so there is no `coder`-chain coupling
to manage. Users configure opencode via `opencode auth login` / project
`opencode.json`; aider has its own model setting. Both CLIs are bundled in
the images; the config skill switches the backend and installs a missing CLI
at runtime via `config.py coding --backend <name>`.

## Tools (`pawlia/tools/`)

| Tool | Name | Purpose |
|------|------|---------|
| `BashTool` | `bash` | Execute shell commands. Respects `context["cwd"]` and `context["timeout"]` (default 120s). Injects `PAWLIA_USER_ID` and `PAWLIA_SESSION_DIR` as env vars. Runs inside the write sandbox (see below) unless `context["sandbox"]` is falsy. |
| `AttachFileTool` | `attach_file` | Send a file to the user. Not in the ToolRegistry — passed to ChatAgent as `direct_tools` (`app.py`) and executed inline; queues bytes on `agent.pending_attachments`, which each interface delivers (Matrix: into the active thread). Paths resolve relative to the workspace; absolute paths are symlink-resolved and validated against allowed roots (workspace, Downloads, /tmp, `attachments.extra_allowed_roots`). Size limit `attachments.max_outgoing_bytes` (default 25 MB). |

**Write sandbox** (`pawlia/sandbox.py`): Bash commands run under a bubblewrap
write-sandbox when bubblewrap can create a nested user namespace — read-only
root with only `session/<user_id>/` and `/tmp` bind-mounted writable. This is
the only mode that actually *prevents* out-of-bounds writes.

**Caveat — rootless podman:** under rootless podman (the production deployment
on thalia) the container already runs inside a mapped user namespace, and
bubblewrap cannot create a nested one — `bwrap` fails with `Operation not
permitted` and `bwrap_available()` returns `False`. The sandbox then degrades
to a **detective fallback**: an mtime snapshot taken before execution, compared
afterwards to report stray writes. Note its limits — it is *post-hoc* (the
write has already happened, it is only flagged, not reverted) and it only scans
`/app/skills` and `/app/pawlia` (`bash.py`), so writes elsewhere (e.g. other
users' session dirs, `/etc`) are invisible to it. **Do not rely on it as an
isolation boundary in prod.** To get real enforcement under rootless podman,
run the pawlia container with `security_opt: ["seccomp=unconfined"]` (or a
custom profile that permits `clone3`/`unshare` with `CLONE_NEWUSER`), which
unblocks the nested-userns `clone3` that the default seccomp profile rejects.

**Note**: `ReminderTool` and the tools in `files_tools.py` (ReadFile/ListFiles/GrepFiles, delegating to the files skill script) exist in code but are not registered in the App. Reminders are managed directly by the Scheduler.

Tools extend `Tool(ABC)` and register in `ToolRegistry`. Each tool provides `as_openai_spec()` for LLM binding and `execute(args, context)` for actual execution.

## Attachments (`pawlia/attachments.py`)

- **Incoming:** files received via any interface are saved to
  `workspace/Downloads/` with a markdown sidecar (YAML frontmatter +
  description), making them discoverable through workspace search. Images are
  additionally bridged into live VoIP calls.
- **Outgoing:** the `attach_file` direct tool (see Tools) lets the agent send
  workspace files back to the user.
- **Config** (`attachments:` in `config.yaml`): `max_outgoing_bytes`
  (default 25 MB), `extra_allowed_roots`.

## Memory & Sessions (`pawlia/memory.py`)

```
session/.credentials/{user_id}.json # per-user credential store (outside the sandbox)
session/{user_id}/
├── session_version.txt             # on-disk log/session format version
├── workspace/                      # Obsidian vault
│   ├── Downloads/                  # incoming attachments + markdown sidecars
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

When context compaction happens mid-turn (e.g. after a context-overflow error),
a **compression marker** is persisted in `session.exchanges` so replay stays
consistent, and the compaction is surfaced to the user via the `on_interim`
callback. Memory search additionally recalls recently active threads
index-free (without requiring the RAG index).

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
- **Workspace git auto-commit** (when `workspace-git.enabled`): throttled to max 1 commit per 5 minutes; invalid-character paths are auto-renamed (commit skipped if still unrepresentable); daily and weekly squashes run at the configured time, plus a monthly GC pass (`reflog expire` + `gc --aggressive` + `repack -ad` + `--force-with-lease`) that is the only thing that actually shrinks the remote on disk. When `push` is on, the scheduler pushes then pulls (`--ff-only`); on divergence the remote is authoritative (`reset --hard origin/HEAD`, discarding local commits). See `pawlia/workspace_git.py`.

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

**Session override inheritance:** When you set a session override with `/model chat <models>`, that selection is automatically inherited by `skill_runner` and all `skill.*` roles **unless** you explicitly set them too. This lets you control the fallback chain for chat and skills from a single override without touching the global config.

**Valid agent paths for `/model <path> <name>`:** `default`, `chat`, `skill_runner`, `vision`, `compiler`, `skills.<name>` (see `_VALID_AGENT_PATHS` in `pawlia/interfaces/common.py`).

**Legacy format** (inline model config) is also supported for backwards compatibility.

`LLMFactory.get(agent_type)` merges agent-specific config over defaults, resolves the provider, and returns a `ChatOpenAI` or `ChatOllama` instance. Any OpenAI-compatible API works.

**Per-model heuristics** (`pawlia/llm.py`):
- `estimate_max_tool_turns(name)` — derives the SkillRunner tool-call budget from the model identifier; override per-model via `max_tool_turns` in `models:` or per-skill via `metadata.max_tool_turns` in `SKILL.md`.
- `estimate_context_size(name)` — derives `num_ctx` for Ollama-backed models so prompts don't get silently truncated to Ollama's 2048 default; override via `context_size` (alias `num_ctx`). Before the heuristic applies, `pawlia/context_probe.py` tries to read the **real** context window from the provider API (Ollama `/api/show`, OpenAI-style `/models`; 4 s timeout, falls back silently).
- `summary_threshold_tokens(name)` — derives the token threshold for auto-summarization; override via `summarize_at_tokens` (absolute) or `summarize_at_fraction` (of context_size).
- `audio_input: true` on a model marks it as natively audio-capable; PawLia bypasses Whisper transcription for those models in the VoIP and voice-message paths.

**Vision capability detection** (`pawlia/vision_probe.py`): whether a model
accepts images is resolved at runtime — explicit config flag →
`model_capabilities.json` cache → live probe (sends a colored-band PNG, 30 s
timeout; inconclusive results are not cached) → name heuristic. If the chat
model can't take images, PawLia falls back to describing the image with the
vision model and feeding the description as text.

**Model blacklist**: models that error are blacklisted *per failure reason* —
e.g. a vision failure doesn't block the model for plain chat.

## Logs

Container logs are stored under `log/*/container.log`. Each subfolder is named with the pattern `<user>_<ip>-<container>-<timestamp>`.

Example: `log/chris_192.168.177.105-thalia_pawlia_1-20260603-064728/container.log`

## Development Guidelines

**Working directory:** Always run commands from the project root (the directory containing `pawlia/` and `requirements.txt`).

**Virtualenv + PYTHONPATH:** `pawlia` is not installed as a package (no `pyproject.toml`), so `PYTHONPATH=.` is required for any command that imports it. The project version lives in `pawlia/__init__.py` (`__version__`), not in packaging metadata — see "Versioning & Releases" below.

```bash
# Run tests
PYTHONPATH=. .venv/bin/pytest tests/ -x -q

# Start CLI
PYTHONPATH=. .venv/bin/python -m pawlia

# Start CLI with piped input (non-interactive, for scripted testing)
printf 'Hello\nexit\n' | PYTHONPATH=. .venv/bin/python -m pawlia 2>/dev/null
```

**Filename hygiene** (`pawlia/utils.py`): `sanitize_filename()` strips
characters that break cross-platform sync (`<>:"|?*`, control chars), applies
NFC normalization, and trims leading/trailing dots. `sanitize_workspace()`
renames offending files bottom-up and runs over all workspaces at app startup.
`find_similar_slug()` (SequenceMatcher, threshold 0.7) powers did-you-mean
suggestions for mistyped wiki/file slugs.

- Keep it simple — PawLia targets small models on local hardware
- Skills are isolated: own directory, own config, no shared state
- Tools are pluggable: extend `Tool`, register in `App.__init__`
- Don't add conversation history to SkillRunnerAgent — isolation prevents hallucination
- Prefer compiled workflows for reliability when a skill provides one; free-form
  tool calling remains the fallback path inside the SkillRunner
- Command mode returns raw script output — no LLM interpretation phase

## Versioning & Releases (git-flow)

**These rules are binding for all contributors and AI models.**

**Single source of truth.** The project version is `pawlia.__version__` in
`pawlia/__init__.py` — a SemVer string `MAJOR.MINOR.PATCH`. The User-Agent
(`pawlia.utils.PAWLIA_USER_AGENT`) is **derived** from it as `f"PawLia/{__version__}"`.
**Never hardcode the version anywhere else** (no literal `"PawLia/1.2.3"`, no second
copy). `python -m pawlia --version` prints it. (The browser-emulating UA from
`web_user_agent()` is intentionally a Chrome string and is unrelated.)

**SemVer bump rules:**
- **PATCH** (`0.1.0 → 0.1.1`): backward-compatible bug fixes only (typical hotfix).
- **MINOR** (`0.1.0 → 0.2.0`): new, backward-compatible features.
- **MAJOR** (`1.2.3 → 2.0.0`): breaking changes. While `0.y.z`, the API is considered
  unstable — breaking changes bump MINOR, and `0.x → 1.0.0` declares first stable API.

**Branch model (classic git-flow, plain git — no git-flow CLI needed):**

| Branch | Purpose | Branches from | Merges into |
|--------|---------|---------------|-------------|
| `main` | stable, tagged releases only | — | — |
| `develop` | integration branch (base for daily work) | `main` | — |
| `feature/<name>` | features / fixes | `develop` | `develop` |
| `release/<x.y.z>` | release stabilization + version bump | `develop` | `main` **and** `develop` |
| `hotfix/<x.y.z>` | urgent production fixes | `main` | `main` **and** `develop` |

**Version bumps happen ONLY on `release/*` or `hotfix/*` branches. Tags (`vX.Y.Z`,
annotated) are created ONLY on `main`.** Commits follow Conventional Commits
(`feat(...)`, `fix(...)`, `chore(release): x.y.z`). Record changes in `CHANGELOG.md`
under `[Unreleased]`, then rename to the version + date on the release branch.

**Cut a release:**
```bash
git switch develop && git switch -c release/0.2.0
# bump __version__ in pawlia/__init__.py; finalize CHANGELOG.md
git commit -am "chore(release): 0.2.0"
git switch main && git merge --no-ff release/0.2.0
git tag -a v0.2.0 -m "PawLia 0.2.0"
git switch develop && git merge --no-ff release/0.2.0
git branch -d release/0.2.0
git push origin main develop --tags
```
**Hotfix:** same shape, branch from `main`, PATCH bump, merge back into both `main`
and `develop`.

**Never commit, tag, or push without an explicit instruction from the user.**
