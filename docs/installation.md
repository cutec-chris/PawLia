# Installation

## Recommended: Docker

Docker is the recommended way to run PawLia in production. It provides process isolation and sandboxing — important because PawLia can execute shell commands via the built-in `bash` tool on behalf of the AI agent. Running inside a container limits the blast radius of any unintended command execution.

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) with Compose plugin (v2)
- An LLM backend reachable from the container (e.g. Ollama running on the host)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/your-org/pawlia.git
cd pawlia

# 2. Create your config
cp config.sample.yaml config.yaml
# Edit config.yaml — at minimum set your provider and model
```

Key things to configure before starting:

| Section | What to fill in |
|---------|----------------|
| `providers.ollama.apiBase` | URL of your Ollama/vLLM/etc. instance |
| `interfaces.telegram.token` | Telegram bot token (if using Telegram) |
| `interfaces.matrix.*` | Matrix homeserver + credentials (if using Matrix) |
| `agents.default` | The model key to use as fallback |

If Ollama runs on the Docker host, use `http://host.docker.internal:11434/v1` as `apiBase` (Linux: add `extra_hosts: ["host.docker.internal:host-gateway"]` to `compose.yml`).

```bash
# 3. Start
docker compose up -d

# View logs
docker compose logs -f
```

Session data (memory, conversation history) is persisted to `./session/` on the host via the volume mount in `compose.yml`.

---

## Manual Installation (development)

Use this if you want to develop PawLia or run it without Docker.

### Prerequisites

- Python 3.11+
- An LLM backend (Ollama, Groq, etc.)

### Steps

PawLia is not packaged for `pip` — there is no `pyproject.toml`. Install the deps into a virtualenv and run via `PYTHONPATH=.`:

```bash
# 1. Clone and install
git clone https://github.com/your-org/pawlia.git
cd pawlia
python -m venv .venv
.venv/bin/pip install -r requirements.txt -c constraints.txt
# Optional extras (only what you need):
#   - VoIP / voice input:    pip install -r requirements-voip.txt
#   - Test suite:            pip install -r requirements-test.txt
#   - End-to-end tests:      pip install -r requirements-e2e.txt

# 2. Configure
cp config.sample.yaml config.yaml
# Edit config.yaml (or skip — when no models are configured PawLia auto-launches
# the Web UI setup wizard at http://localhost:8080)

# 3. Run
PYTHONPATH=. .venv/bin/python -m pawlia                  # interactive CLI
PYTHONPATH=. .venv/bin/python -m pawlia --mode server    # all configured interfaces
PYTHONPATH=. .venv/bin/python -m pawlia --debug          # verbose logging
```

> **Note:** In manual mode the agent can execute shell commands with the permissions of the running user. For production deployments, prefer Docker.

---

## Configuration Reference

See `config.sample.yaml` for all available options with inline comments.

### Providers and backends

Providers define both the transport details and the backend type.

- `backend: pawlia` or omitted: normal PawLia stack with `llm.py`, skills, local routing
- `backend: hermes`: PawLia forwards turns to a Hermes API server and keeps local logs in sync

For normal PawLia providers, any OpenAI-compatible API works:

```yaml
providers:
  ollama:
    backend: pawlia   # optional; default if omitted
    apiBase: http://localhost:11434/v1
  groq:
    backend: pawlia
    apiBase: https://api.groq.com/openai/v1
    apiKey: gsk_...
```

Hermes example:

```yaml
providers:
  hermes_local:
    backend: hermes
    apiBase: http://127.0.0.1:8642/v1
    apiKey: change-me
    conversation_namespace: pawlia
```

### Models and agents

```yaml
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

agents:
  default: smart,fast   # runtime fallback order
  chat: smart,fast      # main chat agent
  skill_runner: fast    # model for skill sub-agents
```

Agent model values can be comma-separated. PawLia tries them in order and falls back automatically on invocation errors.

When a selected model points at a Hermes provider, PawLia switches from its own skill stack to Hermes for that conversation turn. Daily logs and thread logs are still written by PawLia so Dream Wiki and local follow-up sessions continue to work on the same visible chat history.

### Enabling interfaces

Uncomment the relevant section in `config.yaml`:

```yaml
interfaces:
  telegram:
    token: YOUR_BOT_TOKEN
  matrix:
    homeserver: https://matrix.org
    user_id: "@yourbot:matrix.org"
    password: YOUR_PASSWORD
    always_thread: false            # if true, every reply lives in a thread
    allowed_users: ["@you:matrix.org"]   # optional allow-list
  web:
    host: 0.0.0.0
    port: 8080
    api_key: optional-bearer-token
  webhook:
    port: 8081
  openai:
    host: 127.0.0.1
    port: 11435
    api_key: optional-bearer-token  # exposes /v1/chat/completions and /api/chat
```

All enabled interfaces run simultaneously in server mode. The OpenAI-compatible interface mirrors both the OpenAI Chat Completions API and the Ollama `/api/chat` endpoint so tools like Continue.dev, OpenWebUI, and Cursor can connect to PawLia as if it were a generic LLM provider.

### Other optional sections

`config.sample.yaml` also documents these sections that are not required for a minimal install:

| Section | Purpose |
|---------|---------|
| `transcription` | Whisper STT for voice messages and VoIP |
| `tts` | Text-to-speech (Piper or edge-tts) |
| `voip` | Matrix VoIP call settings (silence detection, barge-in) |
| `caldav` | Sync workspace `calendar/*.md` events to a Radicale/Nextcloud server |
| `workspace` / `workspace-git` | Per-user workspace template + auto-commit/squash |
| `workspace-search` | BM25 search across workspace files for system-prompt injection |
| `skill-config` | Per-skill runtime config (e.g. `memory.rag_backend`, `perplexica.url`) |
| `skill-install` | Allow workspace-local skills (`allow_workspace: true`) |
