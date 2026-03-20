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
| Text | Plain messages and `!`-prefixed commands |
| Images | Sent to the vision agent; message body is used as caption |
| Voice messages | Transcribed and forwarded to the agent |
| VoIP calls | Full duplex voice calls (requires `aiortc`; see VoIP section) |

### Threads

Matrix thread replies (messages with `m.thread` relation) get their own isolated context window — same behaviour as Telegram forum topics. Use `!thread <msg>` to start a new thread from the main room.

### Commands

Commands use `!` as prefix instead of `/`:

| Command | Effect |
|---------|--------|
| `!thread <msg>` | Respond as a Matrix thread reply (proper `m.thread` relation) |
| `!model [name]` | Show or switch the active model |
| `!private` | Toggle private mode (thread replies only) |

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
```

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

A background task runs every 60 seconds and checks for:

- **Due reminders** set via the built-in reminder tool (supports daily / weekly / monthly recurrence)
- **Upcoming calendar events** from the organizer skill (notified 15 minutes before start)

Notifications are delivered through the active interface. For Webhook, they are buffered and returned on the next `GET /notifications` poll.
