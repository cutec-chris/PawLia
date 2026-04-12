# Skill Writing Patterns

Proven patterns for writing effective PawLia skills.

## Table of Contents

- [Multi-Step Workflow](#multi-step-workflow)
- [Command Reference Table](#command-reference-table)
- [Output Format Templates](#output-format-templates)
- [Error Handling / Self-Repair](#error-handling--self-repair)
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
