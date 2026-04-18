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
  "add credentials", "improve this skill", "validate skill", "audit skill",
  "scaffold a skill".
license: MIT
metadata:
  author: Christian Ulrich
  version: "3.1"
  max_tool_turns: 50
---

# Skill Creator

Create, edit, improve, or audit PawLia AgentSkills. Manage credentials.

## About Skills

Skills are modular packages that extend PawLia's capabilities — specialized
workflows, tool integrations, domain expertise, and bundled scripts/references/
assets. They give the agent procedural knowledge no model can possess on its own.

A skill runs as a sub-agent with its own LLM session: the dispatcher reads the
skill's `description` to decide whether to invoke it, then the SkillRunner loads
the full SKILL.md body, injects credentials as env vars, and hands control to
the sub-agent with access to bash and other tools.

## Anatomy of a Skill

```
skill-name/
├── SKILL.md            # required — frontmatter + instructions
├── scripts/            # optional — executable code (python/bash/node)
├── references/         # optional — docs loaded into context as needed
├── assets/             # optional — templates/boilerplate used in OUTPUT (not read into context)
└── harness.sh          # optional — smoke-test, runs via `creator.py test` (may also be .py/.mjs)
```

You may see `workflow.yaml` in existing skills. **Never hand-write it** — it is
LLM-compiled from SKILL.md by the workflow compiler. Compilation is NOT
automatic on load or on file change — trigger it explicitly via
`creator.py compile --name <name>` after you finish implementing or edit
SKILL.md substantively. Without a fresh `workflow.yaml`, the skill still runs
(falls back to tool-call mode or command mode) — just without the optimised
building-block pipeline.

### Three loading levels (progressive disclosure)

1. **Frontmatter** (`name` + `description`) — always in context; triggers the skill
2. **SKILL.md body** — loaded only when the skill triggers; keep under 500 lines
3. **Bundled resources** — loaded as needed; scripts execute without being read into context

`references/` = docs the agent reads WHILE working (schemas, API docs, policies).
`assets/` = files used IN the output (templates, images, boilerplate) — not read into context.

### What NOT to include

No README.md, CHANGELOG.md, INSTALLATION_GUIDE.md, test files, or auxiliary
docs. Only files the agent needs to do its job.

For design principles and writing patterns, read `references/design-principles.md`
and `references/patterns.md` before writing or reviewing a SKILL.md.

---

## Frontmatter Reference

The complete frontmatter for a PawLia skill. Note the exact nesting — the loader
is strict about where each field lives:

```yaml
---
name: my-skill                    # required — lowercase + hyphens, matches folder name
description: >                    # required — THE dispatch trigger. Include what + when
  What the skill does. Use when [trigger phrases and contexts]. Also triggers
  on phrases like "X", "Y", "Z".
license: MIT                      # convention
metadata:
  author: Your Name
  version: "1.0"
  compatibility: Requires X       # optional — human-readable deployment note
  max_tool_turns: 30              # optional — overrides the default tool-call budget (30); bump for skills that need deep iterative work
  requires_config:                # optional — NESTED under metadata
    - url                         # keys that must exist under skill-config.<name>.* in config.yaml
    - timeout
requires_credentials:             # optional — TOP-LEVEL (sibling to metadata, NOT nested)
  - my_api_key                    # each becomes env var CRED_MY_API_KEY at runtime
---
```

**Placement rules** (these are enforced by the loader — getting them wrong breaks the skill):

| Field | Location |
|-------|----------|
| `name`, `description`, `license` | top-level |
| `author`, `version`, `compatibility`, `max_tool_turns` | under `metadata:` |
| `requires_config` | under `metadata:` (NESTED) |
| `requires_credentials` | top-level (SIBLING to `metadata`) |

### Credentials vs. Configuration

Two distinct mechanisms — pick the right one:

- **`requires_credentials`** — per-user secrets (API keys, tokens). Stored in
  `session/<user_id>/.credentials.json` via `credentials.py`. Injected as env
  vars `CRED_<KEY>` where `<KEY>` is uppercased and non-alphanumeric chars become
  underscores. Example: key `api-key` → env var `CRED_API_KEY`.
- **`metadata.requires_config`** — deployment-level settings in `config.yaml`
  under `skill-config.<skill-name>.*`. The web UI prompts the user to fill
  missing keys. Skills without these values configured are not loaded at all.
  Use for URLs, model names, hostnames — shared, non-secret settings.

### Description writing (CRITICAL — this is what triggers the skill)

The `description` is the ONLY field the dispatcher reads to decide whether to
invoke your skill. Put ALL "when to use" information here. Body content is
invisible to the dispatcher.

Bad:
```yaml
description: Search the web.
```

Good:
```yaml
description: >
  Perform web searches. Use when the user asks for current information, news,
  or wants to find online resources. Also triggers on "what is...", "look up...",
  "search for..." — even without the explicit word "search".
```

Rules:
- Include **what** + **when** (both required)
- List specific trigger phrases
- Be slightly "pushy" — models tend to undertrigger
- Cover edge cases where the skill SHOULD activate

---

## Runtime Environment (what scripts can read)

When a skill's script runs, the SkillRunner injects these env vars:

| Env var | What it is | Set when |
|---------|------------|----------|
| `PAWLIA_SESSION_DIR` | Absolute path to the session root | always |
| `PAWLIA_USER_ID` | Current user ID | always |
| `CRED_<KEY>` | Each credential from `requires_credentials` | credential is stored |

Config values (`metadata.requires_config`) are **not** auto-injected — they are
passed explicitly on the command line, the sub-agent fills them in when calling
the script. Example: `python <scripts_dir>/search.py --url "<url>"`.

**Placeholders in SKILL.md text** (substituted by the runner before the sub-agent
sees the prompt):

| Placeholder | Becomes |
|-------------|---------|
| `<scripts_dir>` | Absolute path to the skill's `scripts/` directory |
| `<user_id>` | Current user ID |
| `<session_dir>` | Absolute session directory |

Always reference scripts with `<scripts_dir>/<script_name>`, never a relative path.

---

## Credential Management

### Store a credential

Ask the user for the value, then:
```
python <scripts_dir>/credentials.py set --key "<key_name>" --value "<value>"
```

The response includes `value_read_back` — compare it to what you intended to
set and report any discrepancy to the user.

### Check / list / delete

```
python <scripts_dir>/credentials.py check --keys "api_key,other_key"
python <scripts_dir>/credentials.py list
python <scripts_dir>/credentials.py delete --key "<key_name>"
```

---

## Creating a Skill

### 1. Understand Intent

To build an effective skill, gather concrete examples. Without these, the skill
will be too generic to trigger reliably and too vague to guide the sub-agent.

Ask:
- "What should this skill do? Can you give me a concrete example?"
- "What would a user say to trigger this skill?"
- "What's the expected input and output?"
- "Does it need API keys (credentials) or instance settings (config.yaml)?"

Don't ask all questions at once — start with the most important, follow up.
Conclude when you have a clear picture of functionality and triggers.

### 2. Plan Resources

For each concrete example, ask:
> "What would need to be rewritten or re-discovered from scratch each time?"

That becomes a script, reference, or asset.

Examples:
- "Rotate this PDF" → same code every time → `scripts/rotate_pdf.py`
- "Build a todo app" → same HTML boilerplate every time → `assets/todo-template/`
- "How many users logged in today?" → re-discovering schemas → `references/schema.md`

| Resource | When needed |
|----------|-------------|
| `scripts/` | Repeated logic; deterministic reliability |
| `references/` | Docs the agent should read while working |
| `assets/` | Templates/boilerplate used IN the output |
| `requires_credentials` | User-specific API keys |
| `metadata.requires_config` | Deployment settings from config.yaml |

### 3. Scaffold

**Naming:** lowercase + hyphens only, max 63 chars, matches directory name
(e.g., `pdf-editor`, `gh-address-comments`). Prefer short, verb-led phrases.
Namespace by tool when it clarifies triggering.

```
python <scripts_dir>/creator.py init \
  --name "<skill-name>" \
  --description "<desc with triggers>" \
  [--resources scripts,references,assets] \
  [--credentials "key1,key2"] \
  [--config "url,timeout"] \
  [--script python|node|bash]
```

Creates the skill in `$PAWLIA_SESSION_DIR/$PAWLIA_USER_ID/workspace/skills/<name>/`.

### 4. Implement

**SKILL.md body** — write in imperative form ("Run the script", "Parse the output"):

1. Clear step-by-step workflow
2. Output format section (explicit structure — show the exact shape)
3. Error handling table for self-repair (agent recovers, doesn't escalate)
4. Reference any `references/` files with a note on WHEN to read them
5. Keep under 500 lines; split overflow into `references/`

**Scripts** — every script must:
- Accept arguments via `argparse`
- Read credentials via `os.environ.get("CRED_<KEY>")` (remember the uppercase/underscore normalization)
- Read PawLia env: `os.environ.get("PAWLIA_USER_ID")`, `os.environ.get("PAWLIA_SESSION_DIR")`
- Output valid JSON: `{"success": true, ...}` or `{"success": false, "error": "..."}`
- Exit 0 on success, non-zero on failure

**Test the script** by running it manually before declaring the skill done.

See `references/patterns.md` for writing patterns (workflows, error handling,
output formats, description writing).

### 5. Validate

```
python <scripts_dir>/creator.py validate --name "<skill-name>"
```

Fix all `issues` (fatal); review `warnings` and fix anything that matters.

### 6. Harness & Test (REQUIRED before declaring done)

Every non-trivial skill gets a **harness** — a smoke-test script at the skill
root that exercises the primary workflow end-to-end. The harness is what lets
you (and future edits to this skill) prove it still works without hand-rolling
credentials and env vars each time.

**File:** `harness.sh` (POSIX `sh`, preferred), or `harness.py`, or `harness.mjs`.
Placed at the skill root — sibling to `SKILL.md`, not inside `scripts/`.

**Contract:**
- Reads the same `CRED_*` env vars the skill uses at runtime
- Runs one or two read-only probes (e.g. `status` + one simple GET)
- For write-capable skills: do a write-then-delete roundtrip, or skip writes unless `--write` is passed
- Prints one final JSON line: `{"success": true, "checks": [{"name": "...", "ok": true, "detail": "..."}]}`
- Exits 0 on all-green, non-zero on any failure
- **Leaves no persistent side effects** (clean up test data)

**Run it:**
```
python <scripts_dir>/creator.py test --name "<skill-name>"
```

This loads credentials from `.credentials.json`, injects them as `CRED_*`
env vars using the same normalization the runtime uses, sets
`PAWLIA_SESSION_DIR` / `PAWLIA_USER_ID`, and runs the harness. Output
includes full `stdout`, `stderr`, `exit_code`, and any `missing_credentials`
— nothing is truncated or wrapped, so errors are readable.

**Do not declare a skill "done" until `test` exits 0.**

### 7. Compile (recommended)

Turn the SKILL.md instructions into a structured `workflow.yaml` so the
SkillRunner can execute it as building blocks:

```
python <scripts_dir>/creator.py compile --name "<skill-name>"
```

This invokes the workflow compiler LLM — it reads SKILL.md + the `scripts/`
listing and produces `workflow.yaml` with verified commands, parameter
extraction, and error-recovery blocks. Re-run after every substantive SKILL.md
edit. If the skill's metadata `version` is unchanged and `workflow.yaml`
already exists, it's skipped — pass `--force` to override.

The skill works even without `workflow.yaml` (tool-call / command fallback),
but compilation is the difference between guided execution and the model
reasoning about every step from scratch.

### 8. Package (optional)

```
python <scripts_dir>/creator.py package --name "<skill-name>"
```

Produces a `.skill` file (zip archive) for distribution.

### 9. Iterate

Use the skill on real tasks. Notice struggles (wrong triggers, missing params,
unclear instructions, brittle error handling). Update SKILL.md or scripts.
Validate + `test`, then **bump `metadata.version`** and re-compile — the
compiler skips up-to-date skills unless the version changed (or you pass
`--force`).

---

## Fixing / Auditing an Existing Skill

When a user reports "skill X is broken" or asks for an audit, follow this
loop — do NOT try to guess the fix from SKILL.md alone.

### 1. Reproduce with `test`

```
python <scripts_dir>/creator.py test --name "<skill-name>"
```

Read the full `stdout` / `stderr`. If there's no harness yet, write one
first — you cannot fix what you cannot reproduce.

### 2. Get the real error — never trust a wrapper

Skill scripts often wrap upstream errors into generic messages like
`"HTTP 500 - server error"`. That strips the information you need. When
the harness output is too generic:

- Open the script and check whether it swallows response bodies in error
  branches. Fix the script to **always include status code + full response
  body** in failure output (see `references/patterns.md` § Error Handling).
- Re-run the harness. The real error (validation detail, missing field,
  auth message) should now be visible.

### 3. Compare against a working reference

If the skill integrates an external API, find the closest-working sibling
(e.g. if `sparkyfitness` is broken, read `fittrackee`'s script + SKILL.md).
Compare payload shape, auth headers, endpoint paths. Most skill bugs are
payload-schema mismatches that another skill already got right.

### 4. Fix, re-test, repeat

Edit, run `test` again. Only declare "fixed" when the harness is green.

### 5. If you hit 2–3 failed fix attempts → stop and report

Don't loop indefinitely. Report what you found (full error from step 2,
what you tried, what you suspect) and ask the user. Looping burns context
without progress.

---

## Commands

| Command | Script | What it does |
|---------|--------|-------------|
| `init` | creator.py | Scaffold a new skill |
| `validate` | creator.py | Check SKILL.md for errors |
| `list` | creator.py | Show all skills (workspace + bundled) |
| `test` | creator.py | Run the skill's harness with real credentials/env |
| `compile` | creator.py | LLM-compile SKILL.md → workflow.yaml |
| `package` | creator.py | Create `.skill` zip |
| `set` | credentials.py | Store a credential |
| `list` | credentials.py | List credential keys |
| `delete` | credentials.py | Remove a credential |
| `check` | credentials.py | Check if keys exist |
