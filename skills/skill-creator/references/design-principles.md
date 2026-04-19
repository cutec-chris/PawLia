# Skill Design Principles

Apply these when writing or reviewing a SKILL.md.

## Concise is Key

The context window is a shared resource. Skills share it with the system prompt,
conversation history, and other skills' metadata. The default assumption is that
the model is already smart — only add what it doesn't already know. Challenge
each piece: "Does this justify its token cost?"

Prefer concise examples over verbose explanations.

## Set Appropriate Degrees of Freedom

Match specificity to the task's fragility:

- **High freedom** (text instructions): Multiple valid approaches, context-dependent heuristics
- **Medium freedom** (pseudocode / parameterized scripts): Preferred pattern, some variation OK
- **Low freedom** (specific scripts, few params): Fragile operations, consistency critical

Think of the agent exploring a path: a narrow bridge with cliffs needs guardrails (low
freedom), while an open field allows many routes (high freedom).

## Progressive Disclosure

Skills use three loading levels:
1. **Frontmatter** (name + description) — always in context (~100 tokens); triggers the skill
2. **SKILL.md body** — loaded when skill triggers; keep under 500 lines
3. **Bundled resources** — loaded as needed; scripts execute without context loading

Keep SKILL.md body to the essentials. Split content into `references/` files when
approaching the limit. When splitting, always reference those files in SKILL.md and
describe **when** to read them — otherwise the agent won't know they exist.

**Key principle:** When a skill supports multiple variants, frameworks, or options, keep
only the core workflow and selection guidance in SKILL.md. Move variant-specific details
to reference files.

### Pattern 1: High-level guide with links to detail

```markdown
# PDF Processing

## Quick start

Extract text with pdfplumber:
[code example]

## Advanced features

- **Form filling**: See [references/forms.md](references/forms.md) for the complete guide
- **API reference**: See [references/reference.md](references/reference.md) for all methods
```

The agent reads `forms.md` or `reference.md` only when needed.

### Pattern 2: Domain-specific organization

For skills covering multiple domains, organize by domain to avoid loading irrelevant context:

```
bigquery-skill/
├── SKILL.md (overview + which file to read for which domain)
└── references/
    ├── finance.md    (revenue, billing metrics)
    ├── sales.md      (opportunities, pipeline)
    └── product.md    (API usage, feature flags)
```

When a user asks about sales metrics, the agent only reads `sales.md`.

Similarly for multi-provider or multi-framework skills:

```
cloud-deploy/
├── SKILL.md (workflow + provider selection guide)
└── references/
    ├── aws.md
    ├── gcp.md
    └── azure.md
```

### Pattern 3: Conditional details

Show basic content inline, link to advanced content:

```markdown
## Creating documents

Use docx-js for new documents. See [references/docx-js.md](references/docx-js.md).

## Editing documents

For simple edits, modify the XML directly.

**For tracked changes**: See [references/redlining.md](references/redlining.md)
**For OOXML details**: See [references/ooxml.md](references/ooxml.md)
```

**Important:** Keep references one level deep from SKILL.md. For reference files longer
than 100 lines, include a table of contents at the top.

## What NOT to Include in a Skill

- No README.md, CHANGELOG.md, or other auxiliary docs
- No test files (testing happens externally)
- Only files the agent needs to do its job
- Information should live in SKILL.md **or** in a reference file, not both
