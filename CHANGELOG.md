# Changelog

All notable changes to PawLia are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

Process: add changes under **[Unreleased]** as you go; on a `release/*` branch,
rename it to the new version with a date and bump `pawlia.__version__`.
See `agents.md` › "Versioning & Releases (git-flow)".

## [Unreleased]
### Added
- Filesystem write-sandbox for skill execution (`pawlia/sandbox.py`). A skill
  may only write under the per-user session dir (`session/<user_id>/` — the
  workspace, `Downloads/`, credentials, memory) or `/tmp` for throwaway
  scratch. The bash tool wraps commands in `bubblewrap` (read-only root, only
  the writable roots bind-mounted rw) so out-of-bounds writes are rejected by
  the kernel; falls back to a logged warning when bubblewrap / user namespaces
  are unavailable. `skill-creator`'s `creator.py test` runs the harness the
  same way and additionally scans for stray writes, so a violation fails the
  smoke test at the latest. `bubblewrap` added to both Dockerfiles.
- `attach_file` now accepts `/tmp` paths so generated throwaway artefacts
  (e.g. a rain-radar PNG) can be attached without being written into the
  workspace git tree.
- File / image attachments over Matrix, Telegram, and Discord.
- Incoming files and images are saved to `<workspace>/Downloads/` and tracked
  in `downloads_index.json` (kept outside the workspace so it does not
  pollute the workspace git repo).
- `attach_file` direct tool — the LLM can re-attach a previously received or
  generated file (e.g. a rain-radar GIF) to its next reply. Path is validated
  against the workspace + `attachments.extra_allowed_roots`; symlinks and
  path traversal are rejected.
- Per-interface attachment senders:
  - Matrix: `client.upload()` + `room_send(msgtype=m.image|video|audio|file)`
  - Telegram: `send_photo()` for images, `send_document()` for everything else
  - Discord: `channel.send(file=discord.File(...))`
- New `on_document` handler on the Telegram interface (PDFs, Office docs,
  archives, etc.) — previously only photos and voice were handled.
- Config section `attachments:` with `max_incoming_bytes` (25 MB),
  `max_outgoing_bytes` (25 MB) and `extra_allowed_roots` defaults.
- Runtime vision-capability detection: when an image arrives, PawLia checks
  whether the image-handling model can actually see. Detection order is the
  explicit `supports_images` model-config flag → a cached probe result →
  a one-time verified probe (a tiny generated image is sent and the answer
  checked) → a name heuristic. Results are cached in
  `<session_dir>/model_capabilities.json`.
- Vision-blind fallback: if the image model can't see, a vision-capable model
  from the fallback chain describes the image and the description is injected
  into context as text — invisible to the user — so a text-only chat model can
  still reason about images and keep driving the conversation.
- Searchable sidecar markdown for incoming attachments, plus bridging of
  received images into live (voice) calls.
- Index-free recall of recent threads in memory search — recent conversation
  context is surfaced even before the memory index has been built.

### Changed
- Incoming attachments are unified into a single link + description note
  instead of separate handling per type.

### Fixed
- Reminders fire in the user's timezone and @-mention the user on notify;
  `tzdata` added so `zoneinfo` resolves IANA zones on Alpine.
- Matrix: the conversation turn is persisted even when `agent.run()` raises,
  so an errored turn is no longer lost.
- Skill runner: oversized tool outputs are truncated to 4 kB to keep them from
  blowing up the context window.

## [0.1.0] - 2026-06-04
### Added
- Single source of truth for the project version: `pawlia.__version__` (SemVer).
- `python -m pawlia --version`.
- `PAWLIA_USER_AGENT` is now derived from `pawlia.__version__` (was hardcoded
  `PawLia/1.0`); an explicit User-Agent is sent on all outbound HTTP requests
  (LLM/embedding/provider, search-API, transcription, internal services) so
  providers behind Cloudflare (e.g. Groq) no longer 403 the default urllib UA.
- Configurable browser-emulating User-Agent for web fetches (browser/researcher)
  via the top-level `web_user_agent` config key / `$PAWLIA_WEB_USER_AGENT`.

### Changed
- dream-wiki: `analyze` requests a JSON *array* and `consolidate` a JSON *object*
  via the provider-appropriate hint, fixing array-collapse on strict
  OpenAI-compatible providers.

[Unreleased]: about:blank
[0.1.0]: about:blank
