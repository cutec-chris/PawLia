# Command Reference

PawLia supports a set of slash commands across all interactive interfaces. The prefix differs by platform.

| Interface | Prefix | Example |
|-----------|--------|---------|
| Telegram  | `/`    | `/model qwen3.5:4b` |
| Matrix    | `//`   | `//model qwen3.5:4b` |
| CLI       | `/`    | `/model qwen3.5:4b` |
| Web       | `/`    | `/model qwen3.5:4b` |

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

## `/model` — Show or switch a model (per agent role)

```
/model                       # show all active model overrides for this context
/model <name>                # shorthand for /model chat <name> (session-wide)
/model <path> <name>         # override one agent role
/model <path> off            # clear that override
```

Matrix prefix: `//model …`.

`<name>` accepts either a **model key** defined in `config.yaml` (e.g. `fast`, `smart`) or a **raw model name** (e.g. `qwen3.5:4b`). Comma-separated lists (e.g. `smart,fast`) are accepted and become a runtime failover chain. `off` clears the override.

`<path>` selects which agent role to override:

| Path | Used by |
|------|---------|
| `default` | Global fallback for every agent role |
| `chat` | Main conversation agent |
| `skill_runner` | Default model for all skill sub-agents |
| `vision` | Vision agent (image messages) |
| `compiler` | Skill workflow compiler |
| `skills.<name>` | Per-skill override (e.g. `skills.browser`) |

### Scope

| Interface | Scope of `/model …` |
|-----------|---------------------|
| Telegram  | Session-wide |
| Matrix    | Room/session-wide |
| CLI       | Session-wide |

All overrides are persisted in `session/<user>/config.yaml` and survive restarts. Threads inherit the same session-level agent selection; only `/private` remains thread-local.

### Examples

```
/model                            # → "chat=smart  skills.browser=fast"
/model qwen3.5:4b                 # shorthand: agents.chat = qwen3.5:4b
/model chat smart,fast            # explicit session chat override with failover chain
/model default smart,fast         # change the global fallback
/model skills.browser fast        # override one skill's model
/model skills.browser off         # clear that override
/model chat off                   # back to the config default
```

---

## `/status` — Show session status

Displays information about the current session or thread: active model, context size, private mode, loaded skills, and more.

```
/status          # Telegram & CLI
//status          # Matrix
```

When sent inside a thread, the output reflects the thread's context (exchanges, agent overrides). Otherwise it shows the main session.

### Output fields

| Field | Description |
|-------|-------------|
| **Model** | Active chat-model selector (marked with "override" if session/thread agent overrides are active) |
| **Agent Overrides** | Active flattened override paths such as `default=smart,fast` or `skills.browser=fast` |
| **Temp** | Sampling temperature |
| **Provider** | API base URL of the LLM provider |
| **Context** | Number of exchanges and estimated token count |
| **Summary** | Size of the auto-generated conversation summary (if any) |
| **Private** | Whether private mode is active |
| **Threads** | Number of active thread contexts in this session |
| **Skills** | Loaded skill names |
| **Idle** | Time since last exchange |

---

## `/reload` — Reload config-driven runtime state

Reloads the active configuration from disk and rebuilds the app's runtime state without stopping the process.

```
/reload          # Telegram, CLI & Web
//reload         # Matrix
```

Reload currently refreshes:

- config values from `config.yaml`
- model/provider definitions and the LLM factory
- bundled skill discovery
- scheduler config

Still requires a full process restart:

- interface listener settings such as ports, bot tokens, Matrix login/session details
- `session_dir` changes

Use this after editing providers, models, workflow settings, or bundled skills when you want PawLia to pick them up immediately.

---

## `/background` — Run a message in the background

Queues a message for deferred processing. The agent processes it when the active user has been idle for at least `IDLE_BACKGROUND_MIN` (default 10 minutes) and no chat request is currently using the LLM.

```
/background <message>      # Telegram, CLI & Web
//background <message>      # Matrix
```

The task is stored under `session/<user>/automations/` and processed through the normal agent pipeline (including skills). Once complete, the result is delivered as a scheduler notification through whichever interface the user is on.

Use this for long-running or low-priority tasks that don't need an immediate response — e.g. research, bulk operations, or anything that would block the LLM for other users.

---

## `//clear` — Clear the in-memory context (Matrix only)

```
//clear
```

Drops the in-memory conversation context (exchanges, recent bot responses) for the current room or thread without touching the daily log on disk. Useful to restart a stuck conversation without losing the persistent record.
