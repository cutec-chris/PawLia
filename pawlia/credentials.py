"""Credential storage for PawLia skills.

Credentials live at ``<session_dir>/.credentials/<user_id>.json`` — a
sibling of the per-user session dir, deliberately **outside** the
``session/<user_id>/`` root that the bash sandbox bind-mounts read-write.
That way a skill running shell commands cannot read another skill's (or
the same skill's) stored secrets by ``cat``-ing the file directly; the
only legitimate read path is via the ``CRED_*`` env vars the
``SkillRunner`` injects per ``requires_credentials`` declaration.

The legacy location ``session/<user_id>/.credentials.json`` is migrated
on first access (the move happens outside the sandbox, from PawLia's own
process).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Optional


def path_for(session_dir: str, user_id: str) -> Path:
    """Return the credential file path for ``user_id``.

    Always returns the new-style path — callers that need to detect the
    legacy file should use :func:`legacy_path` separately.
    """
    return Path(session_dir) / ".credentials" / f"{user_id}.json"


def legacy_path(session_dir: str, user_id: str) -> Path:
    """Pre-isolation location — kept only for one-shot migration."""
    return Path(session_dir) / user_id / ".credentials.json"


def migrate_if_needed(session_dir: str, user_id: str) -> None:
    """Move the legacy credential file to the new location if it exists.

    Idempotent: a no-op when the new file is already there or the legacy
    file is absent. Runs from PawLia's process (outside the bash sandbox)
    so the rename can reach the new sibling directory.
    """
    new_path = path_for(session_dir, user_id)
    if new_path.is_file():
        return
    old_path = legacy_path(session_dir, user_id)
    if not old_path.is_file():
        return
    new_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.replace(old_path, new_path)
        try:
            os.chmod(new_path, 0o600)
        except OSError:
            pass
    except OSError:
        # If migration fails (e.g. race with another process) leave the
        # legacy file in place — callers will fall back to it.
        pass


def load(session_dir: str, user_id: str) -> dict:
    """Load and return the user's credential dict.

    Migrates the legacy file on the fly when encountered. Returns an
    empty dict when nothing is stored yet or the file is unreadable.
    """
    migrate_if_needed(session_dir, user_id)
    path = path_for(session_dir, user_id)
    if not path.is_file():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def save(session_dir: str, user_id: str, data: dict) -> Path:
    """Write the credential dict and return the file path."""
    path = path_for(session_dir, user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return path


def env_key_for(credential_key: str) -> str:
    """Normalize a credential key into its ``CRED_*`` env var name.

    Non-alphanumerics become ``_``, the result is uppercased. Matches
    the historical behaviour in ``SkillRunner._load_credentials`` and
    ``creator._build_cred_env``.
    """
    return "CRED_" + re.sub(r"[^A-Za-z0-9]", "_", credential_key).upper()


def build_env_extra(
    session_dir: Optional[str],
    user_id: Optional[str],
    required_keys: list,
) -> dict:
    """Build the ``CRED_*`` env-var overlay for the declared keys.

    Returns only the env vars for keys that actually exist in the store;
    undeclared credentials are never exposed. Also includes
    ``PAWLIA_CREDENTIALS_FILE`` so skill-management scripts (e.g.
    ``credentials.py``) can locate the store without re-implementing the
    path logic.
    """
    env: dict = {}
    if session_dir and user_id:
        creds = load(session_dir, user_id)
        for key in required_keys or []:
            if key in creds:
                env[env_key_for(key)] = str(creds[key])
        env["PAWLIA_CREDENTIALS_FILE"] = str(path_for(session_dir, user_id))
    return env
