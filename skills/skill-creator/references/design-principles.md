# Skill Design Principles

Apply these when writing or reviewing SKILL.md instructions.

## Concise is Key

The context window is a shared resource. Skills share it with the system prompt,
conversation history, and other skills' metadata. The default assumption is that
the model is already smart — only add what it doesn't already know. Challenge
each piece: "Does this justify its token cost?"

## Set Appropriate Degrees of Freedom

Match specificity to the task's fragility:
- **High freedom** (text instructions): Multiple valid approaches, context-dependent
- **Medium freedom** (pseudocode / parameterized scripts): Preferred pattern, some variation OK
- **Low freedom** (specific scripts, few params): Fragile operations, consistency critical

## Progressive Disclosure

Skills use three loading levels:
1. **Metadata** (name + description) — always in context (~100 tokens)
2. **SKILL.md body** — loaded when skill triggers (<500 lines ideal)
3. **Bundled resources** — loaded as needed (scripts execute without context loading)

## What NOT to Include in a Skill

- No README.md, CHANGELOG.md, or other auxiliary docs
- No test files (testing happens externally)
- Only files the agent needs to do its job
