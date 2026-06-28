# PawLia — Vision & Priorities

This document captures the long-term direction and active priorities for PawLia.
It is the authoritative reference for trade-off decisions during development.
Reference it in planning and reviews when direction is unclear.

---

## Core Philosophy

PawLia is a **personal AI assistant infrastructure** — not a product, not a platform.
It runs at home, handles sensitive data, and must be trustworthy above all else.

**The hierarchy:**
1. **Security & privacy** — the assistant must never be a liability for the user's data
2. **Reliability** — the assistant should work quietly and correctly, without babysitting
3. **Extensibility** — new capabilities via skills, not core bloat
4. **User experience** — the assistant should feel natural across all interfaces

---

## Design Decisions (with rationale)

### No in-app workspace git sync
The workspace-git feature was removed after an incident where a hallucinated remote URL
caused a user's vault to be force-pushed to a public GitHub repo. Out-of-band sync
(syncthing) is the prescribed alternative — PawLia never touches git in the workspace.
**Do not re-introduce any feature that writes to a git remote from within the assistant.**

### Obsidian vault as canonical workspace format
The workspace (`session/<user>/workspace/`) is a native Obsidian vault.
- Events: Full Calendar frontmatter
- Tasks: Obsidian Tasks emoji format
- Dream Wiki: interlinked topic pages

New workspace features should stay compatible with plain Obsidian — no proprietary formats.

### Coding backend: opencode first, aider on demand
`opencode-ai` is baked into the VoIP image. `aider-chat` is intentionally excluded from
the image (aiohttp conflict + Python 3.14 wheel issues) and installed at runtime via
`config.py coding --backend aider` when needed. Keep it this way until the conflict resolves.

### Skills over core changes
New functionality belongs in a skill (`skills/user/` or bundled skills), not in core
scheduler/agent code. Core changes require stronger justification (security, correctness,
cross-cutting concerns).

---

## Wishes & Future Directions

<!-- Free-form section — add ideas here, no commitment implied -->

- **Skill-level identity and continuity.** Persistent per-skill agent
  identity that survives across days — opinions, named relationships,
  remembered preferences — backed by a per-skill vector store. This
  is the difference between "a tool" and "a colleague."

- **Linguistic dispatch.** The regex-plus-table is brittle. A small classifier
  (or a single prompted LLM call) that recognises "I want code written" in any
  language would replace imperative-table maintenance. Until then, extend the
  table on demand and add a test per language.

- **Local-model-first as a first-class goal.** PawLia is built for Qwen 3.5 4b and
  similar small models, but most optimisations still happen because a good model
  is on the other end. The dream: same UX, fully local, no API key, no egress.

- **Backup story for config and credentials.** Workspace sync is syncthing's job,
  but `config.yaml`, the credentials vault, and `session/<user>/memory/` are not.
  Manual copies are fragile. A documented restore path and a `pawlia backup`
  command would close this.

- **Skill sharing.** `skills/user/` is per-user. A curated bundle (e.g.
  "home-lab-bundle", "writer-bundle") would lower the bar for non-technical users
  without giving up the per-user sandbox.

- **Onboarding for non-technical users.** The web setup wizard exists, but creating
  a skill is still LLM-assisted and intimidating. A guided "create your first skill"
  flow in the web UI would help.

---

## Non-Goals

- PawLia is **not** a hosted SaaS product — multi-tenancy, billing, rate limiting are out of scope
- PawLia is **not** a general agent framework — opinionated, single-purpose, no pluggable everything
- PawLia is **not** a model trainer or finetuner
- PawLia does **not** manage git history in the workspace — that's syncthing's job
- PawLia does **not** expose the workspace to the internet directly

---

*Last updated: 2026-06-28. Keep this document current — a stale vision is worse than none.*
