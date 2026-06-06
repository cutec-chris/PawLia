# Skill Writing Patterns

Proven patterns for writing effective PawLia skills.

## Table of Contents

- [Multi-Step Workflow](#multi-step-workflow)
- [Command Reference Table](#command-reference-table)
- [Output Format Templates](#output-format-templates)
- [Error Handling / Self-Repair](#error-handling--self-repair)
- [Config Injection](#config-injection)
- [Never Swallow Upstream Errors](#never-swallow-upstream-errors)
- [Harness](#harness)
- [Conditional Logic](#conditional-logic)
- [Description Writing](#description-writing)

---

## Multi-Step Workflow

For skills with sequential operations, number the steps clearly:

```markdown
## Step-by-step instructions

1. Parse the query to extract the required parameters.
2. Run the search:
   ```
   python <scripts_dir>/search.py --query "<query>" --limit 10
   ```
3. Parse the JSON output (array of objects with `title`, `url`, `content`).
4. Return results as a structured list.
```

For multi-step interactions across multiple calls (e.g., browser), show the
sequence explicitly:

```markdown
## Multi-step example

To fill and submit a form across multiple calls:
- Call 1: `open https://example.com/login`
- Call 2: `fill I1 myusername`
- Call 3: `fill I2 mysecretpassword`
- Call 4: `submit F1`
```

---

## Command Reference Table

When a skill has multiple commands, use a table:

```markdown
## Commands

| Command | Syntax | Description |
|---------|--------|-------------|
| `open` | `open <url>` | Navigate to a URL |
| `click` | `click <ID>` | Click an element |
| `fill` | `fill <ID> <value>` | Fill a form field |
| `submit` | `submit <FORM_ID>` | Submit a form |
```

---

## Output Format Templates

Define the expected output format explicitly:

```markdown
## Output format

Return results like this:
```
1. **<title>**
   <url>
   <content>
```

For JSON scripts:

```markdown
## Output format

The script returns JSON:
```json
{"success": true, "result": "..."}
```

On error:
```json
{"success": false, "error": "error message"}
```
```

After a `set` command, include a read-back check:

```markdown
After `set`, the response includes `"value_read_back"` — compare it against
what you intended to set and report any discrepancy to the user.
```

---

## Error Handling / Self-Repair

Include a self-repair table so the sub-agent can recover without reporting
errors to the user:

```markdown
## Error handling — SELF-REPAIR

When a command fails, DO NOT report the error to the user. Instead, recover:

| Error | Recovery action |
|-------|-----------------|
| `No element [X]` | Run `show` to see available elements, retry |
| `Connection error` | Retry once, then try alternative URL |
| `Invalid input` | Check format, correct and retry |

General recovery strategy:
1. After ANY error, run `show` to see current state.
2. Compare expected vs actual.
3. Adjust approach.
4. Give up after 2-3 recovery attempts.
```

Key principle: the sub-agent should self-recover rather than bubble errors up.

---

## Config Injection

Deployment settings such as URLs, hosts, timeouts, model names, embedding
dimensions, and API base paths belong in `skill-config.<skill-name>` and are
injected by PawLia. Do not make the sub-agent invent or pass them as ordinary
CLI arguments.

In `SKILL.md`, declare required keys under `metadata.requires_config`:

```yaml
metadata:
  requires_config:
    - url
    - timeout
```

In scripts, read the per-skill config from `PAWLIA_SKILL_CONFIG`:

```python
import json
import os

skill_config = json.loads(os.environ.get("PAWLIA_SKILL_CONFIG", "{}"))
url = skill_config.get("url")
timeout = int(skill_config.get("timeout", 30))
```

Good command example:

```markdown
python <scripts_dir>/search.py --query "<query>" --limit 5
```

Bad command example:

```markdown
python <scripts_dir>/search.py --query "<query>" --url "<url>" --timeout "<timeout>"
```

Compiled workflow commands may still contain placeholders like `{url}` or
`{timeout}` when needed; if those keys exist in `skill-config.<skill-name>`,
the workflow executor fills them from config and does not ask the model for
them.

---

## Never Swallow Upstream Errors

Scripts that talk to external APIs must propagate **the real upstream
response** on failure — status code AND body. A generic
`"HTTP 500 - server error"` strips away the exact field that makes the
failure fixable (validation message, missing-column detail, auth reason).

Bad:
```sh
5*) die "HTTP $http_code - server error. Try again later." ;;
```

Good:
```sh
5*) printf '{"success": false, "status": %s, "error": "HTTP %s", "body": %s}\n' \
       "$http_code" "$http_code" "$(printf '%s' "$rbod" | json_escape)"
    exit 1 ;;
```

Apply the same rule to 4xx branches. If an endpoint returns structured
error JSON, pass it through untouched under a `body` or `details` field.

This is what lets the harness (and a human debugger) see the actual
constraint violation or schema mismatch — not just "something broke".

---

## Harness

Every skill should ship a **harness** at the skill root
(`harness.sh` / `harness.py` / `harness.mjs`) that smoke-tests the primary
workflow end-to-end. It's what `creator.py test --name <skill>` runs.

**Contract:**
- Reads the same `CRED_*` env vars the skill uses at runtime
- Runs 1–3 read-only probes (e.g. `status` + one simple GET)
- Write-capable skills: do a write-then-delete roundtrip, or gate writes
  behind `--write`
- Prints exactly one final JSON line — final line must parse as JSON
- Exits 0 on all-green, non-zero on any failure
- Leaves no persistent side effects
- Writes only under the workspace or `/tmp` (see SKILL.md § Filesystem rules).
  `creator.py test` runs the harness in a write-sandbox and **fails** it if the
  skill touches anything outside those roots — scratch files go to `/tmp`.

**Output format:**

```json
{
  "success": true,
  "checks": [
    {"name": "status",    "ok": true,  "detail": "{\"status\":\"UP\"}"},
    {"name": "list_recent", "ok": true, "detail": "5 entries"}
  ]
}
```

On failure, `success: false` and at least one check has `ok: false` with
the **full upstream error** in `detail` (status code + body). This is
what makes debugging possible — see § Never Swallow Upstream Errors.

**Minimal `harness.sh` skeleton:**

```sh
#!/bin/sh
# Harness for <skill-name>
set -e
SCRIPT="$(dirname "$0")/scripts/<script>.sh"

run_check() {
  _name="$1"; shift
  _out=$("$@" 2>&1) && _ok=true || _ok=false
  printf '{"name":"%s","ok":%s,"detail":%s}' \
    "$_name" "$_ok" "$(printf '%s' "$_out" | node -e 'process.stdout.write(JSON.stringify(require("fs").readFileSync(0,"utf8")))')"
}

c1=$(run_check "status" sh "$SCRIPT" status)
c2=$(run_check "list"   sh "$SCRIPT" list 1)

if printf '%s\n%s' "$c1" "$c2" | grep -q '"ok":false'; then
  success=false; code=1
else
  success=true; code=0
fi

printf '{"success":%s,"checks":[%s,%s]}\n' "$success" "$c1" "$c2"
exit $code
```

Keep it small. The harness is a smoke test, not an integration-test suite.

---

## Conditional Logic

For skills with different modes or branches:

```markdown
## Step-by-step instructions

1. Parse the query:
   - If the query looks like a URL → prepend `open`: use `open <query>`
   - If the query starts with a known command → use as-is
   - Otherwise → treat as a search query
2. Run the appropriate command.
3. Return the output.
```

---

## Description Writing

The `description` field in frontmatter is the primary triggering mechanism.
It determines whether the dispatcher calls this skill at all.

Guidelines:
- Include both what the skill does AND when to use it
- Be slightly "pushy" — models tend to undertrigger
- List specific trigger phrases
- Include edge cases where the skill should activate
- All "when to use" info goes here, NOT in the body

Bad: "Search the web."

Good: "Perform web searches using a SearXNG instance. Use when the user asks
for web search results, current information, news, or wants to find online
resources."

Better: "Perform web searches using a SearXNG instance. Use when the user asks
for web search results, current information, news, or wants to find online
resources. Also use when the user asks 'what is...', 'how does...', 'look up...',
or 'search for...' — even if they don't explicitly say 'search'."
