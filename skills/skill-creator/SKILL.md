---
name: skill-creator
description: >
  Create new PawLia skills from scratch, improve or audit existing ones.
  Also manages centralized credentials for skills — store, retrieve, check
  API keys and tokens that other skills need at runtime.
  Use when the user wants to: create a new skill, scaffold a skill directory,
  manage skill credentials, improve or review an existing skill, validate
  a SKILL.md against the spec, package a skill for distribution.
  Triggers on phrases like "create a skill", "new skill", "store api key",
  "add credentials", "improve this skill", "validate skill".
license: MIT
metadata:
  author: Christian Ulrich
  version: "2.0"
---

# Skill Creator

Create, edit, improve, or audit PawLia AgentSkills. Manage credentials.

## Core Principles

### Concise is Key

The context window is a shared resource. Skills share it with the system prompt,
conversation history, and other skills' metadata. The default assumption is that
the model is already smart — only add what it doesn't already know.

### Set Appropriate Degrees of Freedom

Match specificity to the task's fragility:
- **High freedom** (text instructions): Multiple valid approaches, context-dependent
- **Medium freedom** (pseudocode / parameterized scripts): Preferred pattern, some variation OK
- **Low freedom** (specific scripts, few params): Fragile operations, consistency critical

### Progressive Disclosure

Skills use three loading levels:
1. **Metadata** (name + description) — always in context (~100 tokens)
2. **SKILL.md body** — loaded when skill triggers (<500 lines ideal)
3. **Bundled resources** — loaded as needed (scripts execute without context loading)

---

## Credential Management

Skills that need API keys or tokens declare them via `requires_credentials` in
their SKILL.md frontmatter. Credentials live at `session/{user_id}/.credentials.json`
— outside the workspace. The SkillRunner injects matching credentials as env vars
(`CRED_<KEY_NAME>`) when running a skill. Skills never read the credential file.

### When a skill needs credentials

The main model (ChatAgent) should:

1. Ask the user for the required credential (e.g. "I need your OpenMeteo API key")
2. Store it:
   ```
   python <scripts_dir>/credentials.py set --key "<key_name>" --value "<value>"
   ```
3. Invoke the target skill — credentials are injected automatically

### Check if credentials exist

```
python <scripts_dir>/credentials.py check --keys "api_key,other_key"
```

Returns `{"success": true, "available": [...], "missing": [...]}`.

### Other commands

```
python <scripts_dir>/credentials.py list                              # list key names
python <scripts_dir>/credentials.py get --key "<key_name>"           # retrieve value
python <scripts_dir>/credentials.py delete --key "<key_name>"        # remove
```

---

## Skill Creation Process

### Step 1: Understand Intent

Gather concrete usage examples from the user. Ask:
1. What should this skill enable the agent to do?
2. When should it trigger? (what user phrases/contexts)
3. What's the expected input/output format?
4. Does it need credentials (API keys, tokens)?

Avoid overwhelming the user — start with the most important questions, follow up as needed.

### Step 2: Plan Reusable Contents

Analyze examples to determine what the skill needs:

| Resource | When to include | Example |
|----------|----------------|---------|
| `scripts/` | Repeated logic or deterministic reliability | `scripts/weather.py` for API calls |
| `references/` | Docs the agent should reference | `references/api_docs.md` for schemas |
| `assets/` | Files used in output (templates, boilerplate) | `assets/template.html` |
| `requires_credentials` | External APIs, services | `["openmeteo_api_key"]` |

Keep SKILL.md under 500 lines. Split detailed content into references/ when approaching the limit.

### Step 3: Initialize

```
python <scripts_dir>/creator.py init --name "<skill-name>" --description "<desc>" [--resources scripts,references,assets]
```

Skills are created in the user's workspace (`$PAWLIA_SESSION_DIR/$PAWLIA_USER_ID/workspace/skills/<name>/`).

### Step 4: Implement

#### Write SKILL.md

**Frontmatter fields:**
- `name`: Skill identifier, matches directory name. Lowercase + hyphens.
- `description`: Primary triggering mechanism. Include what the skill does AND when to use it.
- `requires_credentials`: List of credential key names the skill needs. Example:
  ```yaml
  requires_credentials:
    - openmeteo_api_key
  ```

**Body** instructions:
- Use imperative form ("Run the script", "Parse the output")
- Include step-by-step workflow
- Reference bundled resources with guidance on when to read them
- Include error handling / self-repair table
- Define output format clearly

#### Write Scripts

Scripts access credentials via env vars — PawLia injects them as `CRED_<KEY_NAME>`
(uppercased, non-alphanumeric replaced with `_`):

```python
import os
api_key = os.environ.get("CRED_OPENMETEO_API_KEY")
```

Other conventions:
- Accept arguments via `argparse`
- Return JSON: `{"success": true, ...}` or `{"success": false, "error": "..."}`
- Use `<scripts_dir>` placeholder (substituted at runtime)

### Step 5: Validate

```
python <scripts_dir>/creator.py validate --name "<skill-name>"
```

### Step 6: Package (optional)

```
python <scripts_dir>/creator.py package --name "<skill-name>"
```

### Step 7: Iterate

Use the skill on real tasks, notice struggles, update. Repeat.

---

## Skill Anatomy

```
skill-name/
├── SKILL.md            # required: frontmatter + instructions
├── scripts/            # optional: executable code
├── references/         # optional: docs loaded into context as needed
└── assets/             # optional: files used in output
```

---

## Commands Quick Reference

| Command | Script | What it does |
|---------|--------|-------------|
| `init` | creator.py | Create a new skill in the user's workspace |
| `validate` | creator.py | Check SKILL.md for common errors |
| `list` | creator.py | Show all skills (workspace + bundled) |
| `package` | creator.py | Create a `.skill` zip for distribution |
| `set` | credentials.py | Store a credential |
| `get` | credentials.py | Retrieve a credential |
| `list` | credentials.py | List credential key names |
| `delete` | credentials.py | Remove a credential |
| `check` | credentials.py | Check if keys exist |
