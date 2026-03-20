# Skills

Skills are self-contained agent extensions that run as sub-agents with their own LLM session and tool access. PawLia follows the [AgentSkills](https://agentskills.io) specification.

## How skills work

When the user sends a message, the dispatcher (ChatAgent) decides whether to call a skill or respond directly. If a skill is selected, it receives the relevant query and runs in its own LLM session with access to tools like Bash. The result is returned to the main agent, which incorporates it into the response.

## Bundled skills

| Skill | Description | Requires |
|-------|-------------|---------|
| `searxng` | Web search via a SearXNG instance | `skill-config.searxng.url` |
| `perplexica` | AI-powered search via Perplexica | `skill-config.perplexica.url` |
| `browser` | Browse and extract content from web pages | — |
| `files` | Read, write, and manage files in the workspace | — |
| `organizer` | Calendar and task management | — |

## Custom skills

Place your skill directory in `skills/user/` — it is gitignored and loaded automatically alongside bundled skills.

```
skills/
└── user/
    └── my-skill/
        ├── SKILL.md         # required
        └── scripts/         # optional helper scripts
```

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
  requires_config:       # optional: config keys that must be present
    - url
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

## Per-skill model assignment

Assign a specific model to a skill in `config.yaml`:

```yaml
agents:
  skills:
    searxng: groq-fast    # use the groq-fast model for the searxng skill
    browser: smart
```

Falls back to `agents.skill_runner` → `agents.default` if not set. See [config.md](config.md#agents) for the full fallback chain.

## Skill configuration

Skills that need external URLs or API keys read from `skill-config` in `config.yaml`:

```yaml
skill-config:
  searxng:
    url: http://localhost:8888
    timeout: 10
```

The values are passed to the skill's scripts via environment variables or arguments — see each skill's `SKILL.md` for specifics.
