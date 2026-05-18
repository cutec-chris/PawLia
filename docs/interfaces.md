# Interfaces

All interfaces share the same agent, memory, and skills. In server mode they all run simultaneously.

## CLI

```bash
PYTHONPATH=. python -m pawlia
```

Interactive terminal session. One session per run, identified as `cli_user`. Supports the full command set (`/thread`, `/model`, `/status`, `/background`, `/private`, `/reload`) and receives proactive notifications from the scheduler inline.

Interrupt a running response with `Ctrl+C` — the current generation is cancelled and the prompt returns immediately.

## Telegram

Requires a bot token in `config.yaml`:

```yaml
interfaces:
  telegram:
    token: YOUR_BOT_TOKEN
```

### Supported input types

| Type | Notes |
|------|-------|
| Text | Plain messages and commands |
| Photos | Sent to the vision agent; caption is used as the prompt |
| Voice messages | Transcribed via the configured STT provider, then sent to the agent. If the active model has `audio_input: true` (e.g. `gemma4:e4b`), audio is consumed natively without transcription. |

### Threads (forum topics)

Each forum topic gets its own isolated context window — a clean slate, no seeding from the main conversation. Thread history is logged separately and does not appear in the main conversation log. Model and agent overrides apply per-thread independently.

### Commands

See [commands.md](commands.md) for the full reference. Quick overview:

| Command | Effect |
|---------|--------|
| `/thread <msg>` | Run message in a new isolated thread context, reply in-thread |
| `/model [name]` | Show or switch the active session chat model |
| `/model <path> <name>` | Override a specific session agent role (`chat`, `skill_runner`, `vision`, `compiler`, `skills.<name>`, `default`) |
| `/status` | Show session status (active model, idle, exchange count, threads) |
| `/private` | Toggle private mode (threads only) |
| `/reload` | Reload config, models, bundled skills, and scheduler settings |
| `/background <msg>` | Queue a message for deferred background processing |

### Skill status messages

When a skill is running, the bot sends a live status message that is edited in-place as the skill progresses (step counter, current action). Replaced with a ✓ summary on completion.

## Matrix

Requires credentials in `config.yaml`:

```yaml
interfaces:
  matrix:
    homeserver: https://matrix.org
    user_id: "@yourbot:matrix.org"
    password: YOUR_PASSWORD
    # access_token: OR_USE_THIS_INSTEAD_OF_PASSWORD
    # allowed_users: ["@you:matrix.org"]   # optional allow-list
    # always_thread: false                  # see below
```

### Supported input types

| Type | Notes |
|------|-------|
| Text | Plain messages and `//`-prefixed commands |
| Images | Sent to the vision agent; message body is used as caption |
| Voice messages | Transcribed and forwarded to the agent (or consumed natively if the model has `audio_input: true`) |
| VoIP calls | Full duplex voice calls (requires `aiortc`; see VoIP section) |

### Threads

Matrix thread replies (messages with `m.thread` relation) get their own isolated context window. Use `//thread <msg>` to start a new thread from the main room. Each thread starts with a clean context — only the initial question is visible to the model.

#### Always-thread mode

When `always_thread: true` is set in the Matrix interface config, every message automatically creates a new thread. The user's message becomes the thread root and the bot replies inside it. Messages that are already part of an existing thread stay there. This keeps the main room clean and gives each conversation its own isolated context.

```yaml
interfaces:
  matrix:
    always_thread: true
```

### Commands

Commands use `//` as prefix instead of `/`:

| Command | Effect |
|---------|--------|
| `//thread <msg>` | Respond as a Matrix thread reply (proper `m.thread` relation) |
| `//model [name]` | Show or switch the active room/session chat model |
| `//model <path> <name>` | Override a specific session agent role for the room |
| `//status` | Show session status |
| `//private` | Toggle private mode (thread replies only) |
| `//reload` | Reload config, models, bundled skills, and scheduler settings |
| `//background <msg>` | Queue a message for deferred background processing |
| `//clear` | Clear the in-memory conversation context (daily log on disk is kept) |

### VoIP (optional)

PawLia can accept Matrix voice calls using WebRTC. Requires `aiortc` to be installed (included in the Docker image). Configure a STUN server and a TTS provider:

```yaml
interfaces:
  matrix:
    stun_servers:
      - stun:stun.l.google.com:19302

voip:
  call_inactivity_seconds: 180
  silence_seconds: 2.2
  silence_threshold: 0.018
  bargein_rms_threshold: 0.05
  agc_window_seconds: 15.0
  agc_target_rms: 0.10
  agc_max_gain: 12.0
  agc_smoothing: 0.15

tts:
  provider: piper
  piper:
    executable: piper
    model: /app/piper/de_DE-kerstin-low.onnx
    config: /app/piper/de_DE-kerstin-low.onnx.json
    sample_rate: 16000
  # hold_audio: /app/assets/keyboard.m4a   # background sound while waiting (default: assets/keyboard.m4a)
```

Each call gets its own isolated thread context (like `//thread`) — all transcriptions and responses appear in a dedicated Matrix thread rooted at a "📞 Eingehender Anruf" message. Conversation history from different calls does not leak into each other.

#### Streamed TTS

LLM responses are streamed token-by-token. As soon as a complete sentence is detected in the stream, it is synthesised and enqueued for playback immediately. This reduces the delay before the caller hears the first word of the response.

#### Hold audio

While the agent is processing (thinking / skill execution), a hold audio loop is played to the caller so they don't sit in silence. The default is `assets/keyboard.m4a`. Override it via `tts.hold_audio` in the config, or remove the file to disable it. A Matrix typing indicator is also kept alive during processing.

## Web Interface

A browser-based UI with chat, provider/model management, skill administration, and a memory graph viewer. Always launched when no models are configured (first-run setup wizard); otherwise opt-in via `interfaces.web`.

```yaml
interfaces:
  web:
    host: 0.0.0.0
    port: 8888
    # token: OPTIONAL_FIXED_TOKEN
```

On startup a random access token is printed to the console unless `token:` is set. Enter it in the browser to authenticate. Sessions are cookie-based (7 days).

### Features

| Feature | Description |
|---------|-------------|
| Chat | Full chat with the agent, supports `/` commands |
| Setup wizard | First-run: pick provider, pull model, configure interfaces |
| Providers | View and edit provider config (API base, key, timeout) |
| Models | View and edit model definitions (incl. `context_size`, `max_tool_turns`, `summarize_at_tokens`) |
| Skills | List all skills, upload new ones (ZIP), configure skill settings, delete user skills |
| Settings | Edit interface, TTS, transcription, VoIP, workspace-git config |
| Memory | Browse per-user memory state and the Dream Wiki graph |

### Commands

Commands use `/` as prefix (same as CLI/Telegram):

| Command | Effect |
|---------|--------|
| `/status` | Show session status |
| `/model [name]` | Show or switch the active model |
| `/model <path> <name>` | Override a specific agent role |
| `/private` | Toggle private mode |
| `/reload` | Reload config, models, bundled skills, and scheduler settings |
| `/thread <msg>` | Start a new isolated thread context |
| `/background <msg>` | Queue a message for deferred background processing |

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Web UI (single-page app) |
| `GET` | `/health` | Health check |
| `POST` | `/api/auth` | Authenticate with token, set session cookie |
| `POST` | `/api/logout` | Drop the session cookie |
| `POST` | `/api/chat` | Send a message |
| `GET` | `/api/notifications` | Poll for scheduler notifications |
| `GET` / `POST` | `/api/providers` | Read/write provider config |
| `GET` / `POST` | `/api/models` | Read/write model config |
| `GET` | `/api/skills` | List all skills |
| `POST` | `/api/skills/upload` | Upload a skill ZIP |
| `DELETE` | `/api/skills/{name}` | Delete a user skill |
| `GET` / `POST` | `/api/skill-config` | Read/write skill configuration |
| `GET` / `POST` | `/api/settings` | Read/write interface / TTS / transcription / workspace-git config |
| `GET` | `/api/setup-status` | Whether the bootstrap wizard still has work to do |
| `POST` | `/api/setup/auto` | Auto-detect provider, pull a default model, write config |
| `GET` | `/api/memory/users` | List users with persisted memory |
| `GET` | `/api/memory/graph` | Memory / Dream Wiki graph data for the viewer |

### Skill Upload

Upload a ZIP file containing a skill directory with a `SKILL.md`. The ZIP can have the skill files at the root or nested one level deep. After upload, dependencies declared in `requirements.txt` inside the skill are installed automatically. A restart is required for the skill to become active.

## OpenAI-compatible API

Exposes PawLia under the OpenAI Chat Completions schema (`POST /v1/chat/completions`, `GET /v1/models`) and the Ollama schema (`POST /api/chat`, `GET /api/tags`, `GET /api/version`). External tools like Continue.dev, OpenWebUI, and Cursor can connect to PawLia as if it were a generic LLM provider — every turn still goes through the local agent stack, so skills, memory, and the scheduler all stay active.

```yaml
interfaces:
  openai:
    host: 127.0.0.1
    port: 11435
    api_key: optional-bearer-token    # if set, clients must send Authorization: Bearer <key>
```

User identity is taken from the `X-User-Id` header. If absent, a single shared default user is used — usually fine for solo desktops, less appropriate for shared multi-tenant setups. Streaming is supported.

This interface has no notification channel (the protocol is stateless), so scheduler-triggered messages are silently dropped while a client is connected only via the OpenAI API.

## Webhook

A minimal HTTP API for custom integrations:

```yaml
interfaces:
  webhook:
    port: 8080
    # token: OPTIONAL_BEARER_TOKEN
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/chat` | Send a message, get a response |
| `GET` | `/notifications` | Poll for pending scheduler notifications |
| `GET` | `/health` | Health check |

### POST /chat

Request:
```json
{
  "user_id": "alice",
  "message": "What time is it?"
}
```

Response:
```json
{
  "response": "It's 14:32."
}
```

Optionally include `"thread_id"` to route the message into a thread context.

## Multi-User Sessions

Each user gets an isolated session:

- **Telegram**: one session per Telegram user ID (`tg_<id>`)
- **Matrix**: one session per Matrix sender (`mx_<sanitized_mxid>`), not per room — the same `@you:matrix.org` keeps one session across rooms
- **Web**: one session per cookie-authenticated user
- **CLI**: single session (`cli_user`)
- **Webhook**: one session per `user_id` in the request body
- **OpenAI-compatible**: one session per `X-User-Id` header value (or a single default user if absent)

Sessions are persisted to disk as Markdown files under `session/<user_id>/` and expire from RAM after inactivity. Memory, identity files, skills, and workspace are shared across threads within a session.

## Scheduler

A background task runs every 60 seconds and checks for due items. Work is split into two priority tiers:

### High priority (every tick)

- **Due reminders** from `workspace/tasks.md` (lines tagged 🔔 with ⏳ scheduled times; supports daily / weekly / monthly recurrence)
- **Upcoming calendar events** from `workspace/calendar/*.md` (one file per event with Full Calendar frontmatter; notified 15 minutes before start)
- **Event checklists** — script-based automation tied to events
- **Task reminders** — reminders attached to tasks with due dates
- **Scheduled jobs** — cron-like recurring automation scripts in `automations/jobs.json`
- **Token-forced summarization** — when the conversation history exceeds 1.5× the per-model token threshold, summarize immediately (bypasses the idle gate)
- **Exchange-count-forced summarization** — when `exchange_count ≥ 30`, summarize immediately

### Low priority (idle-based)

Low-priority tasks use per-user idle time as their priority. Each task type has a minimum idle threshold (in minutes). Tasks only run when the LLM is free (no active chat request).

| Idle (min) | Task | Description |
|------------|------|-------------|
| 5 | **Summarization** | Soft trigger when token threshold reached, exchange limit hit, or repetition detected |
| 10 | **Background tasks** | Deferred `agent.run()` calls queued via `/background` |
| 20 | **Memory indexing** | RAG backend (markdown / lightrag / simple / mem0) indexing of conversation logs |

The token threshold per model resolves to `summarize_at_tokens` (absolute) or `summarize_at_fraction × context_size` (default fraction 0.6). See [config.md](config.md#auto-summarization-threshold).

Tasks are processed per-user: if Alice is idle for 10 minutes but Bob just sent a message, Alice's background tasks will still run (as long as the LLM is free).

### LLM priority gate

Chat requests have priority over all background work. Each interface calls `acquire_llm()` / `release_llm()` around `agent.run()` calls. While any chat request is active, all low-priority tasks are deferred. Between each low-priority task, the scheduler re-checks `llm_busy` before proceeding.

Notifications are delivered through the active interface. For Webhook, they are buffered and returned on the next `GET /notifications` poll. The OpenAI-compatible interface has no notification channel.
