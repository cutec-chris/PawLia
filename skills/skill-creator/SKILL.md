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

For design principles and writing patterns, read `references/design-principles.md`
and `references/patterns.md` when writing or reviewing a SKILL.md.

## Credential Management

Skills declare required credentials via `requires_credentials` in SKILL.md frontmatter.
Credentials live at `session/{user_id}/.credentials.json` — outside the workspace.
The SkillRunner injects them as env vars (`CRED_<KEY_NAME>`) at runtime.

### Store credentials

Ask the user for the value, then:
```
python <scripts_dir>/credentials.py set --key "<key_name>" --value "<value>"
```

### Check / list / delete

```
python <scripts_dir>/credentials.py check --keys "api_key,other_key"
python <scripts_dir>/credentials.py list
python <scripts_dir>/credentials.py delete --key "<key_name>"
```

---

## Creating a Skill

### 1. Understand Intent

Ask the user:
1. What should this skill do?
2. When should it trigger? (phrases, contexts)
3. Expected input/output?
4. Does it need credentials?

### 2. Plan Resources

| Resource | When needed | Example |
|----------|-------------|---------|
| `scripts/` | Repeated logic, deterministic reliability | `scripts/weather.py` |
| `references/` | Docs the agent should reference | `references/api_docs.md` |
| `assets/` | Templates, boilerplate | `assets/template.html` |
| `requires_credentials` | External APIs | `["my_api_key"]` |

### 3. Scaffold

```
python <scripts_dir>/creator.py init --name "<skill-name>" --description "<desc>" [--resources scripts,references,assets] [--credentials "key1,key2"]
```

Creates the skill in `$PAWLIA_SESSION_DIR/$PAWLIA_USER_ID/workspace/skills/<name>/`.

### 4. Implement

**SKILL.md frontmatter** — required fields:
- `name`: lowercase + hyphens, matches directory name
- `description`: what the skill does AND when to trigger it. Be specific — models undertrigger vague descriptions.

Optional:
- `requires_credentials`: list of credential key names

**SKILL.md body** — instructions for the sub-agent:
- Imperative form ("Run the script", "Parse the output")
- Step-by-step workflow
- Error handling table (so the sub-agent self-repairs)
- Clear output format
- Keep under 500 lines; split into `references/` if longer

**Scripts**:
- Credentials via env vars: `os.environ.get("CRED_MY_API_KEY")`
- Arguments via `argparse`
- Output JSON: `{"success": true, ...}` or `{"success": false, "error": "..."}`
- Use `<scripts_dir>` placeholder in SKILL.md (substituted at runtime)

### 5. Validate

```
python <scripts_dir>/creator.py validate --name "<skill-name>"
```

### 6. Package (optional)

```
python <scripts_dir>/creator.py package --name "<skill-name>"
```

### 7. Iterate

Use the skill on real tasks, notice struggles, update.

---

## Skill Directory Structure

```
skill-name/
├── SKILL.md            # required
├── scripts/            # optional
├── references/         # optional
└── assets/             # optional
```

---

## Commands

| Command | Script | What it does |
|---------|--------|-------------|
| `init` | creator.py | Scaffold a new skill |
| `validate` | creator.py | Check SKILL.md for errors |
| `list` | creator.py | Show all skills |
| `package` | creator.py | Create .skill zip |
| `set` | credentials.py | Store a credential |
| `list` | credentials.py | List credential keys |
| `delete` | credentials.py | Remove a credential |
| `check` | credentials.py | Check if keys exist |
