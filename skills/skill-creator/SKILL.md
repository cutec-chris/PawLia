---
name: skill-creator
description: >
  Create new PawLia skills from scratch, improve or audit existing ones.
  Also manages credentials for skills — store, check, list and delete
  API keys and tokens that other skills need at runtime (skills
  themselves only see the runtime `CRED_*` env vars, never the store).
  When a skill has bugs or needs changes, delegate the full task here —
  describe the problem and let the skill-creator autonomously diagnose
  and fix it. Do not pre-read the skill files yourself.
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
  max_tool_turns: 60
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
  `session/.credentials/<user_id>.json` (sandboxed, **outside** the
  per-user session dir) via `credentials.py`. Injected at runtime as
  `CRED_<NORMALIZED>` where `<NORMALIZED>` is the key uppercased with
  non-alphanumerics → `_`. Example: `api-key` → `CRED_API_KEY`. Skill
  scripts read them from env — they must never `cat` the credential
  store.
- `metadata.requires_config` — deployment-level settings in `config.yaml` under
  `skill-config.<name>.*`. Skills with missing required config are not loaded.
  The runtime injects the full per-skill config as JSON in
  `PAWLIA_SKILL_CONFIG`. Scripts must read config from that env var instead of
  requiring the LLM to pass URLs, timeouts, hosts, or model names as CLI args.
  Compiled workflow placeholders such as `{url}` or `{timeout}` are also filled
  automatically from `skill-config.<name>` when present.

## Runtime Environment

Scripts receive:

| Env var | Value |
|---------|-------|
| `PAWLIA_SESSION_DIR` | Absolute path to the session root |
| `PAWLIA_USER_ID` | Current user ID |
| `PAWLIA_SKILL_CONFIG` | JSON object from `skill-config.<skill-name>` |
| `CRED_<KEY>` | Each credential declared in `requires_credentials` |

Placeholders in the SKILL.md body — substituted by the runner before the
sub-agent sees them: `<scripts_dir>`, `<user_id>`, `<session_dir>`. Always
reference scripts as `<scripts_dir>/<name>`, never relative paths.

Python scripts should use:

```python
skill_config = json.loads(os.environ.get("PAWLIA_SKILL_CONFIG", "{}"))
url = skill_config.get("url")
```

Do not teach skills to pass config values around as ordinary model-generated
arguments. The model should provide user intent (`query`, `limit`, `project`),
while the system supplies deployment config.

## Filesystem rules — where a skill may write

**Hard rule: a skill may only create or modify files under two roots.**

| Write to | When |
|----------|------|
| `$PAWLIA_SESSION_DIR/$PAWLIA_USER_ID/...` (the workspace, `Downloads/`, etc.) | Files the **user keeps** — documents, results worth re-reading later. Reachable via the `files` skill. |
| `/tmp/...` | **Throwaway, generated artefacts** — a rendered chart, a rain-radar PNG, an intermediate download. The default for anything ephemeral. Prefer a unique name (`/tmp/<skill>_<something>.png`). |

Everything else is **forbidden and blocked**: the `session/` root (e.g.
`$PAWLIA_SESSION_DIR/radar` — a common mistake), `/app`, `$HOME`, the skill's
own bundled directory, or any other absolute path. At runtime the bash tool
runs commands inside a sandbox with a read-only root, so such writes fail with
a permission/read-only error. **`creator.py test` enforces the same rule** and
fails the harness if the skill writes outside these roots — so a violation is
caught at the latest during testing.

**Delivering a file to the user** (image, PDF, GIF): write it to `/tmp` (or the
workspace if it should be kept) and return its **path** in the JSON payload.
Do **not** embed the bytes as a base64 `data:` URI in the response text —
chat surfaces like Matrix render that as raw text, not an image. The dispatcher
attaches the file via the `attach_file` tool, which accepts workspace and
`/tmp` paths.

---

## Credential Management

```
python <scripts_dir>/credentials.py set --key "<name>" --value "<val>"
python <scripts_dir>/credentials.py check --keys "a,b,c"
python <scripts_dir>/credentials.py list
python <scripts_dir>/credentials.py delete --key "<name>"
```

The store is located outside the bash-sandboxed per-user dir, so
ordinary skill scripts cannot reach it. The CLI is the only legitimate
write path; reads happen implicitly at runtime via `CRED_*` env vars.

After `set`, the response confirms `{"success": true, "key": "<name>"}` —
no value is echoed back. Verify success by checking the returned `key`
matches what you intended.

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

   Scripts must: parse user-provided args via `argparse` (or equivalent), read
   deployment config from `PAWLIA_SKILL_CONFIG`, read credentials from `CRED_*`,
   output `{"success": bool, ...}` as JSON, exit 0 on success and non-zero on
   failure.

   **Data vs. presentation — hard rule:** Scripts output raw structured data
   (facts, numbers, lists, timestamps) in the JSON payload. They do NOT
   pre-format the final answer as a user-facing string. The LLM sub-agent is
   responsible for turning the data into a response: choosing what to highlight,
   applying Pawlia's tone, trimming noise, and structuring the text. A script
   that returns a pre-built wall of text locks out the LLM and makes the skill
   impossible to adjust conversationally.

   **Exception:** skills whose output is explicitly required to be verbatim
   (e.g. a pre-formatted report) MUST say so in the SKILL.md with a clear
   "Return verbatim" rule AND provide a `## Example output` that shows the
   exact expected format including any links or special elements. Without the
   example, the sub-agent will helpfully reformat — and destroy links and structure.

   **`## Example output` — MANDATORY in every SKILL.md body.** It must:
   - Show one realistic sample of what the final user-facing text looks like
   - Explicitly mark elements that must not be changed: links, special
     formatting, exact phrases — annotate them with a `← keep` comment or bold
   - Be 5–20 lines; representative but not exhaustive
   - Be updated first whenever the user requests a change to output format —
     agree on the new example, then change the script/instructions to match it

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

## Installing a Skill from a Zip

When the user sends a `.skill` zip or any zip containing a skill:

1. **Extract to `/tmp/`** — never extract directly into any skill directory.
2. **Read the SKILL.md** from the extracted zip to determine `name`.
3. **Create the target directory** in the user's workspace:
   ```
   python <scripts_dir>/creator.py init --name "<name>" --resources scripts
   ```
   This creates the skill under `workspace/skills/<name>/`.
4. **Copy files** from `/tmp/<extracted>/` into `workspace/skills/<name>/`,
   overwriting the scaffold files from `init` with the real ones from the zip.
5. **Adapt the SKILL.md** for PawLia (ensure `<scripts_dir>` placeholders,
   `## Example output` section, PawLia-compatible frontmatter).
6. **Remove the `/tmp/` extraction** when done.
7. **Validate:** `creator.py validate --name "<name>"`.

If `creator.py init` fails with a "refusing to overwrite" error, pass `--force`
(the existing directory is from a previous failed install, not user work).

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

Edit the script. Update SKILL.md when the external contract changed
(endpoint path, payload shape, auth flow) **or when the user requested
a change to the output format** — in that case update `## Example output`
first to reflect the desired result, then adjust scripts and instructions
to match it.

**Rule:** in Phase 2, no new probes. Every tool call must be `write_file`,
`edit_file`, or a single targeted re-read of a file you are editing. If you
feel the urge to probe again, you ended Phase 1 too early — go back and
capture what you missed, then resume Phase 2 fresh.

If an external reference skill exists (e.g. `fittrackee` vs. `sparkyfitness`),
read it for payload / auth patterns. That's allowed in Phase 2 — it's
referencing, not probing.

### Phase 3 — Verify (≤3 tool calls)

Run the harness (`test`) — or reproduce the original failing command. **If the
skill has discrete, scriptable commands (a "simple" skill — lookups, CRUD,
status checks), you must also re-compile its workflow and re-test:** run
`compile --name "<name>"`, then `test --name "<name>"`. A workflow-backed skill
is only fixed once both the script *and* the compiled `workflow.yaml` are green.
Invoke every script in the SKILL.md as `<scripts_dir>/<script>` so the compiler
emits a runnable `{scripts_dir}/...` command — never a hand-written or invented
path placeholder.

**What "green" means — hard rule.** A passing test is NOT "the command exited 0".
For any command you added or changed, capture its `--json` output and confirm
**both**:
1. It contains a top-level `"success": true`.
2. Its payload fields carry **real data** for the command's purpose — not all
   `null`/`[]`/`{}`.

An exit code 0 with an all-null envelope (every result field `null`) or with no
`success` field is **RED**, period. Do not declare "alle Tests laufen
einwandfrei" after eyeballing null output — that ships a broken command. Re-read
the command's wiring (is its result actually written into the output envelope?
is `success` set?) and loop back to Phase 2.

**Commit atomically when green.** A scheduled workspace-sync (`workspace-git`)
can `git add -A && commit` at any moment — including mid-edit. To stop it
capturing a half-finished fix, commit your own work the moment Phase 3 is green:
```
git -C "$PAWLIA_SESSION_DIR/$PAWLIA_USER_ID/workspace" \
  add skills/<name> && git commit -m "fix(<name>): <what>"
```
Leave the workspace either fully green+committed or rolled back — never broken
and uncommitted across a pause.

Green → done, report a short summary to the user. Red → one loop back to
Phase 2, same budget. Never go back to Phase 1 from here.

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
| `implement` | creator.py | Generate scripts via coding backend (aider/opencode/llm) |
| `fix` | creator.py | Debug and fix a broken script via coding backend |
| `set` / `list` / `delete` / `check` | credentials.py | Manage credentials |

---

## Implement (coding backend)

Use when a skill has been scaffolded (`init`) and the SKILL.md describes what
the scripts should do, but the actual code still needs to be written — or when
existing scripts need a substantial rewrite.

```
python <scripts_dir>/creator.py implement --name "<name>" --task "<what to implement>"
```

If `--task` is omitted, the backend implements all scripts described in SKILL.md.

The command auto-detects the best available coding backend:

| Backend | How | When |
|---------|-----|------|
| **aider** | `aider --message ... --yes` CLI | `aider` in PATH |
| **opencode** | `opencode run ...` CLI | `opencode` in PATH |
| **llm** | Direct LLM call via config `agents.coder` | Always available (fallback) |

Override per-skill via `skill-config.skill-creator.coding_backend` in config.yaml,
or globally via `coding.backend`.

After `implement`, run `validate` and `compile` separately.

## Fix (coding backend)

Use when a skill's script fails with a specific error. The backend receives the
failing command, error output, and the full skill context, then edits the script
in-place.

```
python <scripts_dir>/creator.py fix --name "<name>" --error "<error message>" --command "<failing command>"
```

After `fix`, re-compile the workflow (`compile`) when the skill is
workflow-suitable, then run the harness (`test`) to verify the fix worked.
