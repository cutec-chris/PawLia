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
  version: "3.2"
  max_tool_turns: 50
---

# Skill Creator

Create, edit, improve, or audit PawLia AgentSkills. Manage credentials.

A skill is a sub-agent with its own LLM session. The dispatcher reads the skill's
`description` to decide whether to invoke it; the SkillRunner then loads the full
SKILL.md body, injects credentials as env vars, and hands control to the
sub-agent with bash + other tools.

## Anatomy

```
skill-name/
├── SKILL.md        # required — frontmatter + instructions
├── scripts/        # optional — executable code (python/bash/node)
├── references/     # optional — docs the agent reads while working
├── assets/         # optional — templates/boilerplate used IN the output
└── harness.sh      # optional — smoke-test (also .py / .mjs); run via `creator.py test`
```

Three loading levels: **frontmatter** always in context (~100 tokens, triggers
the skill) → **SKILL.md body** loaded when triggered (keep <500 lines) →
**bundled resources** loaded on demand (scripts execute without entering
context).

No README / CHANGELOG / test-suites / setup guides. Only files the agent needs.

`workflow.yaml`, if present, was LLM-compiled from SKILL.md — never hand-write it.
Compile with `creator.py compile --name <name>` after substantive SKILL.md edits.

For design patterns, read [references/patterns.md](references/patterns.md) and
[references/design-principles.md](references/design-principles.md).

---

## Frontmatter

```yaml
---
name: my-skill                    # required — lowercase+hyphens, matches folder
description: >                    # required — the dispatch trigger; include what AND when
  What the skill does. Use when [trigger phrases and contexts]. Triggers on
  phrases like "X", "Y", "Z".
license: MIT
metadata:
  author: Your Name
  version: "1.0"
  max_tool_turns: 30              # optional — overrides the default budget (30)
  requires_config:                # optional — NESTED under metadata
    - url                         # keys under skill-config.<name>.* in config.yaml
requires_credentials:             # optional — TOP-LEVEL (sibling to metadata)
  - my_api_key                    # each becomes CRED_MY_API_KEY at runtime
---
```

Placement matters — the loader reads `requires_config` from `metadata`,
`requires_credentials` from top-level. Getting it wrong silently breaks the
skill.

**Description writing** decides whether your skill triggers at all. Include
both *what* and *when*, list trigger phrases, cover edge cases. Be slightly
pushy — models tend to undertrigger. Put all "when to use" info here, never
in the body (the body is invisible to the dispatcher).

## Credentials vs. Config

- `requires_credentials` — per-user secrets (API keys, tokens). Stored in
  `session/<user_id>/.credentials.json` via `credentials.py`. Injected as
  `CRED_<NORMALIZED>` where `<NORMALIZED>` is the key uppercased with
  non-alphanumerics → `_`. Example: `api-key` → `CRED_API_KEY`.
- `metadata.requires_config` — deployment-level settings in `config.yaml` under
  `skill-config.<name>.*`. Not auto-injected; pass on the CLI. Skills with
  missing config are not loaded.

## Runtime Environment

Scripts receive:

| Env var | Value |
|---------|-------|
| `PAWLIA_SESSION_DIR` | Absolute path to the session root |
| `PAWLIA_USER_ID` | Current user ID |
| `CRED_<KEY>` | Each credential declared in `requires_credentials` |

Placeholders in the SKILL.md body — substituted by the runner before the
sub-agent sees them: `<scripts_dir>`, `<user_id>`, `<session_dir>`. Always
reference scripts as `<scripts_dir>/<name>`, never relative paths.

---

## Credential Management

```
python <scripts_dir>/credentials.py set --key "<name>" --value "<val>"
python <scripts_dir>/credentials.py check --keys "a,b,c"
python <scripts_dir>/credentials.py list
python <scripts_dir>/credentials.py delete --key "<name>"
```

After `set`, the response includes `value_read_back` — compare it to what you
intended and flag any mismatch.

---

## Creating a Skill

1. **Understand intent.** Ask for concrete examples: what should trigger it,
   what's the input/output, does it need credentials or config? One question
   at a time.

2. **Plan resources.** For each example, ask *"what would be rewritten or
   re-discovered every time?"* — that becomes `scripts/`, `references/`, or
   `assets/`.

3. **Scaffold.**
   ```
   python <scripts_dir>/creator.py init \
     --name "<name>" --description "<desc>" \
     [--resources scripts,references,assets] \
     [--credentials "k1,k2"] [--config "url,timeout"] \
     [--script python|node|bash]
   ```

4. **Implement.** Write scripts first, test each one by running it directly
   with the right env vars, then write the SKILL.md body that guides the
   sub-agent. SKILL.md is imperative ("Run the script", "Parse the output"),
   shows the exact output shape, lists error-recovery steps in a table, and
   references any `references/` files with a note on *when* to read them.

   Scripts must: parse args via `argparse` (or equivalent), read credentials
   from `CRED_*`, output `{"success": bool, ...}` as JSON, exit 0 on success
   and non-zero on failure.

5. **Validate.** `creator.py validate --name "<name>"` — fix all `issues`,
   review `warnings`.

6. **Harness (recommended).** Add `harness.sh` at the skill root (also `.py`
   or `.mjs`) that runs 1–3 read-only probes and prints one JSON line
   `{"success": true, "checks": [...]}`. Write-capable skills do a
   write-then-delete roundtrip or gate writes behind `--write`. Harness
   leaves no side effects.

   Run via `creator.py test --name "<name>"` — loads real credentials and
   env, prints full stdout/stderr (no truncation). See
   [references/patterns.md](references/patterns.md) § Harness for the
   skeleton.

7. **Compile.** `creator.py compile --name "<name>"` — LLM-compiles SKILL.md
   into `workflow.yaml`. Skipped if version matches; pass `--force` to
   override. The skill runs without it (fallback mode), but compiled is
   better.

8. **Package (optional).** `creator.py package --name "<name>"` produces a
   `.skill` zip.

9. **Iterate.** Use it on real tasks, notice struggles, update SKILL.md or
   scripts, bump `metadata.version`, re-compile.

---

## Fixing / Auditing an Existing Skill

Three phases with hard stop-gates. Do not blur them.

### Phase 1 — Diagnose (≤5 tool calls)

Read the skill files **once**. Run the harness or reproduce the failing
command. Capture the full error (status code + response body).

Skill scripts often wrap upstream errors into generic "HTTP 500 - server
error" strings. If the output is too generic, **the first fix is to the
script's error branch** — make it print the real status + body — before any
further investigation.

**Stop-gate:** as soon as you have a concrete, actionable root cause
(specific missing field, wrong endpoint, validation message), **stop
diagnosing**. Do not fuzz parameter names or endpoint variants once you have
a working signal.

### Phase 2 — Implement (≤5 tool calls)

Edit the script. Update SKILL.md only if the external contract changed
(endpoint path, payload shape, auth flow).

**Rule:** in Phase 2, no new probes. Every tool call must be `write_file`,
`edit_file`, or a single targeted re-read of a file you are editing. If you
feel the urge to probe again, you ended Phase 1 too early — go back and
capture what you missed, then resume Phase 2 fresh.

If an external reference skill exists (e.g. `fittrackee` vs. `sparkyfitness`),
read it for payload / auth patterns. That's allowed in Phase 2 — it's
referencing, not probing.

### Phase 3 — Verify (≤3 tool calls)

Run the harness (or reproduce the original failing command). Green → done,
report a short summary to the user. Red → one loop back to Phase 2, same
budget. Never go back to Phase 1 from here.

### Failure exit

After 2–3 failed fix attempts → stop and report. Include: full error from
Phase 1, what you changed in Phase 2, what still fails. Do not keep looping
— it burns context without progress.

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
| `set` / `list` / `delete` / `check` | credentials.py | Manage credentials |
