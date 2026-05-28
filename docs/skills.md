# Skills

Skills are self-contained agent extensions that run as sub-agents with their own LLM session and tool access. PawLia follows the [AgentSkills](https://agentskills.io) specification.

## How skills work

When the user sends a message, the dispatcher (ChatAgent) decides whether to call a skill or respond directly. If a skill is selected, it receives the relevant query and runs in its own LLM session with access to tools like Bash. The result is returned to the main agent, which incorporates it into the response.

## Bundled skills

| Skill | Description | Requires |
|-------|-------------|---------|
| `memory` | Long-term conversation memory — search past chat logs and trigger the Dream Wiki | `skill-config.memory.rag_model` + `embedding_host`; other settings depend on `rag_backend` |
| `researcher` | Per-project document collections in `workspace/research/<project>/`. Scrapes URLs and runs keyword / embedding search over the saved Markdown. The Dream Wiki is *not* fed by research projects — only by conversations. | `skill-config.researcher` (embedding settings optional — falls back to keyword search) |
| `searxng` | Web search via a SearXNG instance | `skill-config.searxng.url` |
| `perplexica` | AI-powered search via Perplexica or Vane | `skill-config.perplexica.url` (+ model config for Vane) |
| `browser` | Browse and extract content from web pages | — |
| `files` | Read, write, edit, grep, outline, and delete files in the workspace | — |
| `organizer` | Calendar events (Full Calendar), tasks (Obsidian Tasks), reminders — Obsidian vault native | — |
| `automation` | Scheduled jobs (cron-like recurring tasks) and event-bound checklists | — |
| `config` | Read/write config files in the workspace | — |
| `skill-creator` | Create, scaffold, validate, and package new skills; manage centralized credentials | — |
| `workspace-git` | Commit, squash, push, and pull the workspace git repo on demand | — |

## Custom skills

With `skill-install.allow_workspace: true` in config.yaml, skills placed in a user's workspace are loaded automatically:

```
session/<user>/workspace/skills/
└── my-skill/
    ├── SKILL.md         # required
    ├── scripts/         # optional helper scripts
    ├── references/      # optional docs loaded into context as needed
    └── assets/          # optional files used in output (templates, etc.)
```

Use the `skill-creator` skill to scaffold new skills — it handles directory structure, templates, and validation.

## SKILL.md format

Each skill needs a `SKILL.md` with a YAML frontmatter header followed by the instructions for the sub-agent:

```markdown
---
name: my-skill
description: One-line description used by the dispatcher to decide when to call this skill.
license: MIT
metadata:
  author: Your Name
  version: "1.0"
  requires_config:         # optional: keys under skill-config.<name>.*
    - url
requires_credentials:      # optional: credential keys the skill needs
  - api_key
---

# My Skill

## Instructions

Describe step by step what the sub-agent should do...
```

### Frontmatter fields

| Field | Required | Description |
|-------|----------|-------------|
| `name` | yes | Skill identifier (matches directory name) |
| `description` | yes | Used by the dispatcher to decide when to invoke the skill |
| `license` | no | License identifier |
| `metadata.author` | no | Free-form author label, surfaced in the Web UI skill list |
| `metadata.version` | no | Free-form version label |
| `metadata.trust` | no | `internal` for bundled skills; affects how skill output is framed in the prompt |
| `metadata.requires_config` | no | List of `skill-config.<name>.*` keys that must exist; the skill is skipped if any are missing |
| `metadata.optional_config` | no | List of recognised but optional `skill-config.<name>.*` keys (documentation only) |
| `metadata.max_tool_turns` | no | Override the SkillRunner tool-call budget for this skill (wins over the model heuristic and the `models.<name>.max_tool_turns` override) |
| `metadata.openclaw.cwd` | no | `"skill"` (default) or `"workspace"` — what directory `BashTool` runs in |
| `requires_credentials` | no | List of credential key names (see [Credentials](#credentials)) |

### Per-skill model assignment

Assign a specific model to a skill in `config.yaml`:

```yaml
agents:
  skills:
    searxng: groq-fast,fast    # use groq-fast first, then fast as fallback
    browser: smart,fast
```

Falls back to `agents.skill_runner` → `agents.default` if not set. See [config.md](config.md#agents) for the full fallback chain.

### Model recommendations

Most skills work fine with small/fast models because the SkillRunner guides execution
via system prompts, nudging, and command-mode fallback. Some skills however are
significantly more complex and benefit from a larger model:

| Skill | Recommended | Why |
|-------|-------------|-----|
| `skill-creator` | `smart` (or larger) | Generates SKILL.md files, writes scripts, makes design decisions — requires strong reasoning and code generation |
| `browser` | `smart` | Multi-step navigation with context-dependent decisions |
| Most others | `fast` is fine | Follow clear script → parse output → return result |

Example config for mixed model sizes:

```yaml
agents:
  default: fast
  skill_runner: fast          # default for all skills
  skills:
    skill-creator: smart      # override: needs a capable model
    browser: smart
    searxng: fast
```

## Skill configuration

Skills that need deployment settings such as URLs, hosts, timeouts, or model
names read from `skill-config` in `config.yaml`:

```yaml
skill-config:
  searxng:
    url: http://localhost:8888
    timeout: 10
```

The full per-skill config is injected into skill scripts as JSON in `PAWLIA_SKILL_CONFIG`.
Scripts should read deployment config from that env var rather than requiring the
LLM to pass URLs, hosts, timeouts, embedding settings, or model names as CLI
arguments:

```python
import json
import os

skill_config = json.loads(os.environ.get("PAWLIA_SKILL_CONFIG", "{}"))
url = skill_config.get("url")
timeout = int(skill_config.get("timeout", 30))
```

Compiled workflows also receive config automatically: placeholders such as
`{url}` or `{timeout}` are filled from `skill-config.<skill>` when present and
are not exposed as model-provided parameters.

## Credentials

Skills that need API keys or tokens declare them via `requires_credentials` in their SKILL.md frontmatter. Credentials are stored centrally per user at `session/<user>/.credentials.json` — outside the workspace so skills can't read the file directly.

When the SkillRunner starts, it injects matching credentials as environment variables (`CRED_<KEY_NAME>`) into the skill's execution context. Skill scripts read them via `os.environ`:

```python
import os
api_key = os.environ.get("CRED_MY_API_KEY")
```

### Credential management

The `skill-creator` skill provides credential management commands. The main model asks the user for credentials when a skill needs them, then stores them:

```
python <scripts_dir>/credentials.py set --key "my_api_key" --value "sk-..."
python <scripts_dir>/credentials.py list
python <scripts_dir>/credentials.py check --keys "my_api_key,other_key"
```

### Flow

1. Skill declares `requires_credentials: ["my_api_key"]` in SKILL.md
2. Main model sees the credential requirement in the skill's description
3. Main model asks the user: "I need your API key for this skill"
4. Main model stores it via `credentials.py set`
5. SkillRunner loads matching credentials and injects as `CRED_*` env vars
6. Skill script reads the env var

## Coding Backend

The `skill-creator` can delegate script generation and debugging to external
coding tools. This is useful when a skill has been scaffolded (`init`) but the
actual scripts still need to be written, or when a script fails and needs
fixing.

### Backends

| Backend | How | Requirements |
|---------|-----|-------------|
| **aider** | Runs `aider --message ... --yes` in the skill directory | `aider` CLI in PATH, configured LLM |
| **opencode** | Runs `opencode run ...` in the skill directory | `opencode` CLI in PATH |
| **llm** | Direct LLM call via the `coder` model from config | Always available (fallback) |

Auto-detection order: aider > opencode > llm. Override globally or per-skill.

### Commands

**Implement** — generate or rewrite skill scripts:

```
python <scripts_dir>/creator.py implement --name "my-skill" --task "what to implement"
```

**Fix** — debug a failing script:

```
python <scripts_dir>/creator.py fix --name "my-skill" --error "SyntaxError..." --failed-cmd "python scripts/main.py"
```

### Configuration

```yaml
agents:
  coder: coder              # model for LLM fallback (aider/opencode use their own)

coding:
  backend: auto             # auto | aider | opencode | llm

# Per-skill override:
skill-config:
  skill-creator:
    coding_backend: aider
```

When `backend: auto` (default), PawLia picks the first available backend from
the detection order. Set explicitly to skip detection.

The `coder` model is only used by the LLM fallback backend. Aider and opencode
use their own configured models.
