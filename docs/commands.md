# Command Reference

PawLia supports a set of slash commands across all interactive interfaces. The prefix differs by platform.

| Interface | Prefix | Example |
|-----------|--------|---------|
| Telegram  | `/`    | `/model qwen3:4b` |
| Matrix    | `//`   | `//model qwen3:4b` |
| CLI       | `/`    | `/model qwen3:4b` |
| Web       | `/`    | `/model qwen3:4b` |

---

## `/private` — Toggle private mode

Prevents messages from being written to the conversation history on disk.

```
/private          # Telegram & CLI
//private          # Matrix
```

- **Telegram / Matrix:** thread-only — private mode is scoped to the current thread/room-thread. Sending the command outside a thread returns an error.
- **CLI:** session-level — affects the entire CLI session (no thread concept).

Toggling the command again disables private mode. The state is held in memory and resets when the bot restarts.

---

## `/thread` — Start a new thread and reply there

Runs a message in its own isolated thread context and delivers the response as a thread reply — keeps side conversations out of the main chat.

```
/thread <message>      # Telegram & CLI
//thread <message>      # Matrix
```

- **Telegram**: the bot replies to the `/thread` command message (visual reply chain), and the conversation is tracked in an isolated context keyed to that message.
- **Matrix**: the bot responds as a proper Matrix thread reply (`m.thread` relation), rooted at the `//thread` event. Element and other clients display this as a collapsible thread.
- **CLI**: the response is printed with `[Thread]` label; the context uses a time-based thread ID.

Subsequent messages to that thread work exactly like any other thread: reply inside the thread in Telegram/Matrix, or use `/thread` again in the CLI (which creates a new isolated context each time).

---

## `/model` — Show or switch the active model

```
/model                  # show current model
/model <name>           # switch to a different model
```

Matrix prefix: `//model` / `//model <name>`

`<name>` accepts either a **model key** defined in `config.yaml` (e.g. `fast`, `smart`) or a **raw model name** (e.g. `qwen3:4b`, `llama3.1:8b`).

### Scope

| Interface | Scope of `/model <name>` |
|-----------|--------------------------|
| Telegram  | Thread-local when sent inside a thread; session-wide otherwise |
| Matrix    | Thread-local when sent as a thread reply; room-wide otherwise |
| CLI       | Session-wide |

Thread-local overrides do not affect the main conversation or other threads. All overrides are persisted to disk and survive restarts.

### Examples

```
/model                  # → "Aktives Modell: smart"
/model fast             # switch to the "fast" model key from config
/model qwen3:4b         # switch to a raw Ollama model name
/model groq-fast        # switch to the "groq-fast" model key
```

To reset to the default model, restart the session or (CLI) set the override to the default key.

---

## `/status` — Show session status

Displays information about the current session or thread: active model, context size, private mode, loaded skills, and more.

```
/status          # Telegram & CLI
//status          # Matrix
```

When sent inside a thread, the output reflects the thread's context (exchanges, model override). Otherwise it shows the main session.

### Output fields

| Field | Description |
|-------|-------------|
| **Model** | Active model name (marked with "override" if set via `/model`) |
| **Temp** | Sampling temperature |
| **Provider** | API base URL of the LLM provider |
| **Context** | Number of exchanges and estimated token count |
| **Summary** | Size of the auto-generated conversation summary (if any) |
| **Private** | Whether private mode is active |
| **Threads** | Number of active thread contexts in this session |
| **Skills** | Loaded skill names |
| **Idle** | Time since last exchange |
