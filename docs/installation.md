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

```bash
# 1. Clone and install
git clone https://github.com/your-org/pawlia.git
cd pawlia
pip install -e ".[all]"

# 2. Configure
cp config.sample.yaml config.yaml
# Edit config.yaml

# 3. Run
python -m pawlia                  # interactive CLI
python -m pawlia --mode server    # all server interfaces (Telegram, Matrix, Webhook)
python -m pawlia --debug          # verbose logging
```

> **Note:** In manual mode the agent can execute shell commands with the permissions of the running user. For production deployments, prefer Docker.

---

## Configuration Reference

See `config.sample.yaml` for all available options with inline comments.

### LLM providers

Any OpenAI-compatible API works. Define one or more providers and reference them from model definitions:

```yaml
providers:
  ollama:
    apiBase: http://localhost:11434/v1
  groq:
    apiBase: https://api.groq.com/openai/v1
    apiKey: gsk_...
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
  default: smart        # global fallback
  skill_runner: fast    # model for skill sub-agents
```

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
  webhook:
    port: 8080
```

All enabled interfaces run simultaneously in server mode.
