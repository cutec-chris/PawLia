---
name: config
description: Read and write pawlia configuration settings (interfaces, TTS, transcription, agents, skill-config). Use this to enable/disable features, change providers, adjust interface settings like always_thread, or configure skill parameters.
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

## Output

All commands return JSON. On success: `{"success": true, ...}`. On error: `{"success": false, "error": "..."}`.

After `set`, the response includes `"value_read_back"` — the value actually written to disk. Always compare it against what you intended to set and report any discrepancy to the user.

**Note:** Changes to config.yaml take effect after the next restart. Inform the user that a restart is required for the changes to become active.
