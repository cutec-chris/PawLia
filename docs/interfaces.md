# Interfaces

All interfaces share the same agent, memory, and skills. In server mode they all run simultaneously.

## CLI

```bash
python -m pawlia
```

Interactive terminal session. One session per run, identified as `cli_user`. Supports the full command set (`/thread`, `/model`, `/private`) and receives proactive notifications from the scheduler inline.

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
| Voice messages | Transcribed via the configured STT provider, then sent to the agent |

### Threads (forum topics)

Each forum topic gets its own isolated context window. The first message in a topic is seeded with the last 5 exchanges from the main conversation. Thread history is logged separately and does not appear in the main conversation log. Model overrides can be set per-thread independently.

### Commands

See [commands.md](commands.md) for the full reference. Quick overview:

| Command | Effect |
|---------|--------|
| `/thread <msg>` | Run message in a new isolated thread context, reply in-thread |
| `/model [name]` | Show or switch the active model for this context |
| `/private` | Toggle private mode (threads only) |
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
```

### Supported input types

| Type | Notes |
|------|-------|
| Text | Plain messages and `//`-prefixed commands |
| Images | Sent to the vision agent; message body is used as caption |
| Voice messages | Transcribed and forwarded to the agent |
| VoIP calls | Full duplex voice calls (requires `aiortc`; see VoIP section) |

### Threads

Matrix thread replies (messages with `m.thread` relation) get their own isolated context window — same behaviour as Telegram forum topics. Use `//thread <msg>` to start a new thread from the main room.

### Commands

Commands use `//` as prefix instead of `/`:

| Command | Effect |
|---------|--------|
| `//thread <msg>` | Respond as a Matrix thread reply (proper `m.thread` relation) |
| `//model [name]` | Show or switch the active model |
| `//private` | Toggle private mode (thread replies only) |
| `//background <msg>` | Queue a message for deferred background processing |

### VoIP (optional)

PawLia can accept Matrix voice calls using WebRTC. Requires `aiortc` to be installed (included in the Docker image). Configure a STUN server and a TTS provider:

```yaml
interfaces:
  matrix:
    stun_servers:
      - stun:stun.l.google.com:19302

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

A browser-based UI with chat, provider/model management, and skill administration. Always active in server mode.

```yaml
interfaces:
  web:                    # optional — works without config
    host: 0.0.0.0
    port: 8888
    # token: OPTIONAL_FIXED_TOKEN
```

On startup a random access token is printed to the console. Enter it in the browser to authenticate. Sessions are cookie-based (7 days).

### Features

| Feature | Description |
|---------|-------------|
| Chat | Full chat with the agent, supports `/` commands |
| Providers | View and edit provider config (API base, key, timeout) |
| Models | View and edit model definitions |
| Skills | List all skills, upload new ones (ZIP from ClawHub), configure skill settings, delete user skills |

### Commands

Commands use `/` as prefix (same as CLI/Telegram):

| Command | Effect |
|---------|--------|
| `/status` | Show session status |
| `/model [name]` | Show or switch the active model |
| `/private` | Toggle private mode |
| `/thread <msg>` | Start a new isolated thread context |
| `/background <msg>` | Queue a message for deferred background processing |

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Web UI (single-page app) |
| `POST` | `/api/auth` | Authenticate with token |
| `POST` | `/api/chat` | Send a message |
| `GET` | `/api/notifications` | Poll for scheduler notifications |
| `GET/POST` | `/api/providers` | Read/write provider config |
| `GET/POST` | `/api/models` | Read/write model config |
| `GET` | `/api/skills` | List all skills |
| `POST` | `/api/skills/upload` | Upload a skill ZIP |
| `DELETE` | `/api/skills/{name}` | Delete a user skill |
| `GET/POST` | `/api/skill-config` | Read/write skill configuration |

### Skill Upload

Upload a ZIP file containing a skill directory with a `SKILL.md`. The ZIP can have the skill files at the root or nested one level deep. After upload, dependencies declared in `requirements.txt` inside the skill are installed automatically. A restart is required for the skill to become active.

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
- **Matrix**: one session per room (`mx_<room_id>`)
- **CLI**: single session (`cli_user`)
- **Webhook**: one session per `user_id` in the request body

Sessions are persisted to disk as Markdown files under `session/<user_id>/` and expire from RAM after inactivity. Memory, identity files, skills, and workspace are shared across threads within a session.

## Scheduler

A background task runs every 60 seconds and checks for due items. Work is split into two priority tiers:

### High priority (every tick)

- **Due reminders** set via the built-in reminder tool (supports daily / weekly / monthly recurrence)
- **Upcoming calendar events** from the organizer skill (notified 15 minutes before start)
- **Event checklists** — script-based automation tied to events
- **Task reminders** — reminders attached to tasks with due dates
- **Scheduled jobs** — cron-like recurring automation scripts

### Low priority (idle-based)

Low-priority tasks use per-user idle time as their priority. Each task type has a minimum idle threshold (in minutes). Tasks only run when the LLM is free (no active chat request).

| Idle (min) | Task | Description |
|------------|------|-------------|
| 5 | **Summarization** | Summarize the conversation when exchange limit, repetition, or idle trigger fires |
| 10 | **Background tasks** | Deferred `agent.run()` calls queued via `/background` |
| 20 | **Memory indexing** | LightRAG knowledge graph indexing of conversation logs |

Tasks are processed per-user: if Alice is idle for 10 minutes but Bob just sent a message, Alice's background tasks will still run (as long as the LLM is free).

### LLM priority gate

Chat requests have priority over all background work. Each interface calls `acquire_llm()` / `release_llm()` around `agent.run()` calls. While any chat request is active, all low-priority tasks are deferred. Between each low-priority task, the scheduler re-checks `llm_busy` before proceeding.

Notifications are delivered through the active interface. For Webhook, they are buffered and returned on the next `GET /notifications` poll.
