# Changelog

All notable changes to PawLia are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

Process: add changes under **[Unreleased]** as you go; on a `release/*` branch,
rename it to the new version with a date and bump `pawlia.__version__`.
See `agents.md` › "Versioning & Releases (git-flow)".

## [Unreleased]
### Added
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
