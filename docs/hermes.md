# Hermes Backend

PawLia can run in front of a Hermes agent and act as a thin interface layer.
In this mode, PawLia keeps its own interfaces, logging, and session storage,
while Hermes handles the live agent/tool execution.

## How It Works

- Hermes keeps the active runtime context via its own API conversation state
- PawLia still writes daily logs and thread logs
- Dream Wiki, summaries, and later local PawLia sessions still see the visible conversation history
- `RouterAgent` dispatches each turn to the appropriate backend per thread (`backend: pawlia` → `ChatAgent`, `backend: hermes` → `HermesBackend`), so the main room and a thread can run on different backends at the same time
- `/model <name>` swaps the active chat model; `/model <path> <name>` overrides any agent role for the current session/thread (`default`, `chat`, `skill_runner`, `vision`, `compiler`, `skills.<name>`)

The backend is selected on the provider, not on the model.

## Provider Configuration

`backend: pawlia` is the default if omitted.

```yaml
providers:
  ollama:
    backend: pawlia
    apiBase: http://localhost:11434/v1
    apiKey: ollama

  hermes_local:
    backend: hermes
    apiBase: http://127.0.0.1:8642/v1
    apiKey: change-me
    timeout: 600
    conversation_namespace: pawlia
    store: true

models:
  fast:
    model: qwen3.5:4b
    provider: ollama

  hermes:
    model: hermes-agent
    provider: hermes_local

agents:
  default: fast
```

Then switch at runtime with `/model hermes` (shorthand for `chat`) or back with `/model fast`.
For role-specific overrides use `/model chat hermes` or `/model default fast,hermes` — comma-separated values become a runtime failover chain.

## Conversation State

Hermes-backed turns use Hermes' stateful Responses API. PawLia derives stable
conversation identifiers from its own session and thread context, so Hermes can
keep server-side context aligned with PawLia threads.

At the same time, PawLia still stores the visible conversation locally:

- main chat goes into the normal daily log
- thread replies go into the matching thread log
- Dream Wiki and local follow-up sessions continue to work on the same transcript

This means PawLia logs are the shared visible journal, while Hermes owns the
live tool/runtime state for Hermes-backed turns.

## Current Behavior

- PawLia-backed models use the normal `ChatAgent` + `SkillRunnerAgent` stack
- Hermes-backed models bypass PawLia skills for that turn
- status views show model, backend, and provider
- VoIP / streamed calls work, but Hermes currently uses the simple full-response path rather than Hermes-native progress streaming

## Hermes API Server

PawLia expects a running Hermes API server that exposes an OpenAI-compatible
endpoint plus Hermes conversation handling. A typical local setup looks like:

```bash
hermes gateway
```

With environment like:

```bash
API_SERVER_ENABLED=true
API_SERVER_KEY=change-me
```

By default Hermes commonly listens on:

```text
http://127.0.0.1:8642/v1
```

## Notes

- If you omit `backend` on a provider, PawLia treats it as `pawlia`
- The model name itself does not select the backend; the referenced provider does
- Daily logs and thread logs are intentionally still written in Hermes mode
- That local logging is what keeps Dream Wiki and local-mode context usable across backend switches
