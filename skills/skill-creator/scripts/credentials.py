#!/usr/bin/env python3
"""Central credential management for PawLia skills.

Manages credentials at session/{user_id}/.credentials.json —
outside the workspace so skills can't read the file directly.
Credentials are injected as env vars (CRED_*) by the SkillRunner.

Uses PAWLIA_SESSION_DIR and PAWLIA_USER_ID env vars (set by PawLia).
"""

import argparse
import json
import os
import sys
from pathlib import Path


def _cred_path() -> Path:
    """Return the credential file path for the current user."""
    session_dir = os.environ.get("PAWLIA_SESSION_DIR")
    user_id = os.environ.get("PAWLIA_USER_ID")
    if not session_dir or not user_id:
        return None
    return Path(session_dir) / user_id / ".credentials.json"


def _load() -> dict:
    """Load credentials from disk."""
    path = _cred_path()
    if not path or not path.is_file():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save(data: dict) -> None:
    """Save credentials to disk."""
    path = _cred_path()
    if not path:
        print(json.dumps({"success": False, "error": "PAWLIA_SESSION_DIR / PAWLIA_USER_ID not set"}))
        sys.exit(1)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def cmd_set(args):
    """Store a credential."""
    creds = _load()
    creds[args.key] = args.value
    _save(creds)
    print(json.dumps({"success": True, "key": args.key}))


def cmd_get(args):
    """Retrieve a credential value."""
    creds = _load()
    value = creds.get(args.key)
    if value is None:
        print(json.dumps({"success": False, "error": f"Credential '{args.key}' not found"}))
        sys.exit(1)
    print(json.dumps({"success": True, "key": args.key, "value": value}))


def cmd_list(args):
    """List credential key names (no values)."""
    creds = _load()
    print(json.dumps({"success": True, "keys": sorted(creds.keys())}))


def cmd_delete(args):
    """Delete a credential."""
    creds = _load()
    if args.key not in creds:
        print(json.dumps({"success": False, "error": f"Credential '{args.key}' not found"}))
        sys.exit(1)
    del creds[args.key]
    _save(creds)
    print(json.dumps({"success": True, "deleted": args.key}))


def cmd_check(args):
    """Check if credential keys exist. Returns missing keys."""
    creds = _load()
    keys = [k.strip() for k in args.keys.split(",")]
    missing = [k for k in keys if k not in creds]
    available = [k for k in keys if k in creds]
    print(json.dumps({
        "success": len(missing) == 0,
        "available": available,
        "missing": missing,
    }))


def main():
    parser = argparse.ArgumentParser(description="PawLia credential management")
    sub = parser.add_subparsers(dest="command")

    p_set = sub.add_parser("set", help="Store a credential")
    p_set.add_argument("--key", required=True, help="Credential key name")
    p_set.add_argument("--value", required=True, help="Credential value")

    p_get = sub.add_parser("get", help="Retrieve a credential")
    p_get.add_argument("--key", required=True, help="Credential key name")

    sub.add_parser("list", help="List credential key names")

    p_del = sub.add_parser("delete", help="Delete a credential")
    p_del.add_argument("--key", required=True, help="Credential key name")

    p_check = sub.add_parser("check", help="Check if keys exist")
    p_check.add_argument("--keys", required=True, help="Comma-separated key names")

    args = parser.parse_args()

    if args.command == "set":
        cmd_set(args)
    elif args.command == "get":
        cmd_get(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "delete":
        cmd_delete(args)
    elif args.command == "check":
        cmd_check(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
