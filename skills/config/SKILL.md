---
name: config
description: Read and write pawlia configuration settings (interfaces, TTS, transcription, agents, skill-config), override session-local agent selection at runtime, and toggle private mode. Use this to enable/disable features, change providers, adjust interface settings, configure skill parameters, change active agent models, or enable/disable private mode.
license: MIT
metadata:
  author: Christian Ulrich
  version: "1.0"
---

# Config Skill

Reads and writes `config.yaml` using dot-notation paths.

## IMPORTANT

Always use the **bash tool** to run the commands below.
`--user-id` and `--session-dir` are NOT needed for this skill.

---

## Settable sections

Only these top-level sections can be modified:

| Section | Examples |
|---------|---------|
| `interfaces` | `interfaces.matrix.always_thread`, `interfaces.web.port` |
| `tts` | `tts.provider`, `tts.hold_audio`, `tts.edge.voice` |
| `transcription` | `transcription.provider`, `transcription.groq.language` |
| `agents` | `agents.default`, `agents.chat`, `agents.skill_runner` |
| `skill-config` | `skill-config.searxng.url`, `skill-config.memory.idle_minutes` |

`providers` and `models` are managed via the web UI and cannot be changed here.

---

## Show current settings

Show all settable config sections:

```
python <scripts_dir>/config.py show
```

Show a single section:

```
python <scripts_dir>/config.py show --section interfaces
```

## Get a specific value

```
python <scripts_dir>/config.py get --path interfaces.matrix.always_thread
```

## Set a value

Values are parsed as YAML scalars: `true`/`false` become booleans, numbers become integers/floats, everything else is a string.

```
python <scripts_dir>/config.py set --path interfaces.matrix.always_thread --value true
python <scripts_dir>/config.py set --path tts.provider --value edge
python <scripts_dir>/config.py set --path tts.edge.voice --value de-DE-KatjaNeural
python <scripts_dir>/config.py set --path transcription.groq.language --value de
python <scripts_dir>/config.py set --path agents.default --value fast
python <scripts_dir>/config.py set --path skill-config.memory.idle_minutes --value 10
```

**TTS validation:** writes to `tts.provider`, `tts.piper.model`, and `tts.edge.voice` are validated — an unknown value is rejected with a list of valid options instead of being written. A wrong voice silently breaks TTS in the VoIP call, so always pick from the returned `available_voices`. To force a value not in the list (e.g. an English Edge voice), append `--force`.

## Switch the active chat model (runtime shorthand)

Show the current model:

```
python <scripts_dir>/config.py model
```

Switch to a different model (takes effect immediately, no restart needed):

```
python <scripts_dir>/config.py model --name qwen3.5:latest
```

This is shorthand for overriding `agents.chat`.
The model name must match a key in the `models` section of config.yaml.

## Override runtime `agents:` selection

Show all session overrides:

```
python <scripts_dir>/config.py agent
```

Show one override path:

```
python <scripts_dir>/config.py agent --path chat
python <scripts_dir>/config.py agent --path skills.browser
```

Set an override path:

```
python <scripts_dir>/config.py agent --path chat --value smart,fast
python <scripts_dir>/config.py agent --path default --value smart,fast
python <scripts_dir>/config.py agent --path skills.browser --value fast
```

Thread-local override:

```
python <scripts_dir>/config.py agent --thread abc123 --path chat --value hermes
```

Clear one override path:

```
python <scripts_dir>/config.py agent --path skills.browser --off
```

Supported runtime paths:
- `default`
- `chat`
- `skill_runner`
- `vision`
- `compiler`
- `skills.<name>`

## Switch the TTS voice (per-user)

The voice command is **provider-aware** — it reads `tts.provider` from `config.yaml` and lists/validates voices for that provider. Always run `voice` (without `--name`) first to see the current `available_voices` for the active provider, then pick one from that list. A name that doesn't match the provider is rejected to avoid silently breaking TTS.

Show current voice + available voices for the active provider:

```
python <scripts_dir>/config.py voice
```

Set a voice (persists in `workspace/memory/voice_override.txt`):

```
# When tts.provider = piper:
python <scripts_dir>/config.py voice --name de_DE-thorsten-low

# When tts.provider = edge:
python <scripts_dir>/config.py voice --name de-DE-KatjaNeural
```

Clear the override (falls back to global `tts.piper.model` / `tts.edge.voice`):

```
python <scripts_dir>/config.py voice --off
```

**How `available_voices` is discovered:**
- Piper: glob `/app/piper/*.onnx` (the voice files baked into the VoIP image). **This list is authoritative** — a voice not listed does not exist on disk. Setting a non-listed Piper voice is always rejected, even with `--force`, because Piper would immediately crash with "Model file doesn't exist" and break TTS. If a user asks for a Piper voice that isn't available, tell them so and offer one from the list; never try to force it.
- Edge: `edge_tts.list_voices()` (full Microsoft catalog). If the dynamic listing is unreachable (`edge_tts` not installed / no internet) the list may be empty — only then may `--force` be used to write a voice name the user explicitly asked for:

  ```
  python <scripts_dir>/config.py voice --name en-US-AriaNeural --force
  ```

## Private mode

Enable private mode for the current session (messages won't be saved):

```
python <scripts_dir>/config.py private
```

Enable private mode for a specific thread:

```
python <scripts_dir>/config.py private --thread <thread_id>
```

Disable private mode:

```
python <scripts_dir>/config.py private --off
python <scripts_dir>/config.py private --thread <thread_id> --off
```

## Output

All commands return JSON. On success: `{"success": true, ...}`. On error: `{"success": false, "error": "..."}`.

After `set`, the response includes `"value_read_back"` — the value actually written to disk. Always compare it against what you intended to set and report any discrepancy to the user.

**Note:** Changes to config.yaml (via `set`) take effect after the next restart. The `model` and `agent` commands change the active runtime selection immediately without restart.
