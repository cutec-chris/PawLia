---
name: skill-creator
description: >
  Create new PawLia skills from scratch, improve or audit existing ones.
  Use when the user wants to: create a new skill, scaffold a skill directory,
  improve or review an existing skill, validate a SKILL.md against the spec,
  package a skill for distribution, or tidy up a skill directory.
  Triggers on phrases like "create a skill", "build a skill", "new skill",
  "improve this skill", "review the skill", "fix the skill", "validate skill".
license: MIT
metadata:
  author: Christian Ulrich
  version: "2.0"
---

# Skill Creator

Create, edit, improve, or audit PawLia AgentSkills.

## Core Principles

### Concise is Key

The context window is a shared resource. Skills share it with the system prompt,
conversation history, and other skills' metadata. The default assumption is that
the model is already smart — only add what it doesn't already know. Challenge
each piece: "Does this justify its token cost?"

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

## Skill Creation Process

### Step 1: Understand Intent

Gather concrete usage examples from the user. Ask:
1. What should this skill enable the agent to do?
2. When should it trigger? (what user phrases/contexts)
3. What's the expected input/output format?
4. Does it need external APIs, tools, or config?

Avoid overwhelming the user — start with the most important questions, follow up as needed.

### Step 2: Plan Reusable Contents

Analyze examples to determine what the skill needs:

| Resource | When to include | Example |
|----------|----------------|---------|
| `scripts/` | Same code rewritten repeatedly, or deterministic reliability needed | `scripts/weather.py` for API calls |
| `references/` | Documentation the agent should reference while working | `references/api_docs.md` for API schemas |
| `assets/` | Files used in output (templates, boilerplate) | `assets/template.html` for HTML generation |

Keep SKILL.md under 500 lines. Split detailed content into references/ when approaching the limit.

### Step 3: Initialize

Run the init script to scaffold the skill:

```
python <scripts_dir>/creator.py init --name "<skill-name>" --path "<target-path>" [--resources scripts,references,assets]
```

This creates the directory, SKILL.md template, and resource directories.

### Step 4: Implement

#### Write SKILL.md

**Frontmatter** (required fields):
- `name`: Skill identifier, matches directory name. Lowercase + hyphens.
- `description`: Primary triggering mechanism. Include what the skill does AND when to use it. All "when to use" info goes here, not in the body. Be slightly "pushy" — models tend to undertrigger skills.

**Body** instructions:
- Use imperative form ("Run the script", "Parse the output")
- Include step-by-step workflow
- Reference bundled resources with guidance on when to read them
- Include error handling / self-repair table
- Define output format clearly

#### Write Scripts

Scripts follow these conventions:
- Accept arguments via `argparse`
- Return JSON on stdout: `{"success": true, ...}` or `{"success": false, "error": "..."}`
- Use `<scripts_dir>` placeholder in SKILL.md (PawLia substitutes at runtime)
- Keep self-contained (stdlib + requests preferred)
- Test scripts by actually running them

### Step 5: Validate

```
python <scripts_dir>/creator.py validate --name "<skill-name>"
```

Checks: frontmatter fields, name match, instruction body, `<scripts_dir>` usage, script existence.

### Step 6: Package (optional)

```
python <scripts_dir>/creator.py package --name "<skill-name>"
```

Creates a `.skill` zip file for distribution.

### Step 7: Iterate

Use the skill on real tasks, notice struggles, update. Repeat until satisfied.

---

## Skill Anatomy

```
skill-name/
├── SKILL.md            # required: frontmatter + instructions
├── scripts/            # optional: executable code (Python, Bash, etc.)
├── references/         # optional: docs loaded into context as needed
└── assets/             # optional: files used in output (templates, etc.)
```

### What NOT to include

- No README.md, CHANGELOG.md, or other auxiliary docs
- No test files (testing happens externally)
- Only files the agent needs to do its job

---

## Reference Files

For detailed skill-writing patterns and examples, read `references/patterns.md`:

- Multi-step workflow patterns
- Output format templates
- Error handling patterns
- Command reference table patterns

---

## Commands Quick Reference

| Command | What it does |
|---------|-------------|
| `init` | Scaffold a new skill directory with template files |
| `validate` | Check SKILL.md for common errors |
| `list` | Show all loaded skills and their status |
| `package` | Create a `.skill` zip for distribution |
