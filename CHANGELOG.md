# Changelog

All notable changes to PawLia are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

Process: add changes under **[Unreleased]** as you go; on a `release/*` branch,
rename it to the new version with a date and bump `pawlia.__version__`.
See `agents.md` › "Versioning & Releases (git-flow)".

## [Unreleased]

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
