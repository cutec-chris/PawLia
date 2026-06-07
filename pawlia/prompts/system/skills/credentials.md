## Credentials

Credentials (API keys, tokens) NEVER live in skill source code or in the
workspace — they are injected at runtime as `CRED_<KEY>` env vars. The
runtime fills them automatically from a separate, sandboxed store; you
do NOT need to (and cannot) read the credential file from disk.

Rules:

- Read secrets from env vars: `os.environ.get("CRED_MY_API_KEY")` (Python),
  `echo "$CRED_MY_API_KEY"` (shell), `process.env.CRED_MY_API_KEY` (node).
- The key name is the value declared in SKILL.md `requires_credentials:`
  (uppercased, non-alphanumerics → `_`).
- NEVER `cat` or read the file `$PAWLIA_CREDENTIALS_FILE` or any
  `session/.../.credentials*` path. The store is outside the bash sandbox
  for a reason — those reads fail at runtime, and even if they didn't,
  the file contains credentials for *other* skills that this skill has no
  business seeing.
- Do NOT ask the user for an API key inline as a CLI argument. If a
  required `CRED_*` env var is missing, report it back to the user with
  the missing key name and tell them to store it via the `skill-creator`
  skill (or to ask the chat to do it for them).
- If the skill legitimately needs to *store* a credential the user
  provided (e.g. an OAuth flow), only the `skill-creator` skill has the
  `credentials.py set` CLI — ordinary skills should not write
  credentials themselves.
