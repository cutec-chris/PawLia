# Skills

Skills are self-contained agent extensions that run as sub-agents with their own LLM session and tool access. PawLia follows the [AgentSkills](https://agentskills.io) specification.

## How skills work

When the user sends a message, the dispatcher (ChatAgent) decides whether to call a skill or respond directly. If a skill is selected, it receives the relevant query and runs in its own LLM session with access to tools like Bash. The result is returned to the main agent, which incorporates it into the response.

## Bundled skills

| Skill | Description | Requires |
|-------|-------------|---------|
| `memory` | Long-term conversation memory — indexes daily chat logs by topic | `skill-config.memory` (embedding settings) |
| `researcher` | Per-project research knowledge bases (index URLs, PDFs, query) | `skill-config.researcher` (embedding settings) |
| `searxng` | Web search via a SearXNG instance | `skill-config.searxng.url` |
| `perplexica` | AI-powered search via Perplexica | `skill-config.perplexica.url` |
| `browser` | Browse and extract content from web pages | — |
| `files` | Read, write, and manage files in the workspace | — |
| `organizer` | Calendar events (Full Calendar), tasks (Obsidian Tasks), reminders, scheduled jobs — Obsidian vault native | — |
| `skill-creator` | Create, scaffold, validate, and package new skills; manage centralized credentials | — |

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
requires_config:           # optional: config keys that must be present
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
| `metadata.requires_config` | no | List of `skill-config.<name>.*` keys that must exist |
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

Skills that need external URLs or API keys read from `skill-config` in `config.yaml`:

```yaml
skill-config:
  searxng:
    url: http://localhost:8888
    timeout: 10
```

The values are passed to the skill's scripts via environment variables or arguments — see each skill's `SKILL.md` for specifics.

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
