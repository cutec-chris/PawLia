"""Config skill script — read and write pawlia config.yaml via dot-notation paths.

Usage:
  python config.py show [--section <section>]
  python config.py get --path <dot.notation.path>
  python config.py set --path <dot.notation.path> --value <value>
"""

import argparse
import json
import os
import sys
from typing import Any, Optional

import yaml




# ---------------------------------------------------------------------------
# Config file helpers
# ---------------------------------------------------------------------------

SETTABLE_SECTIONS = {"interfaces", "tts", "transcription", "skill-config", "agents"}

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _find_config() -> Optional[str]:
    path = os.environ.get("PAWLIA_CONFIG_PATH")
    if path and os.path.isfile(path):
        return path
    for base in (os.getcwd(), _PKG_ROOT):
        for name in ("config.yaml", "config.yml"):
            candidate = os.path.join(base, name)
            if os.path.isfile(candidate):
                return candidate
    return None


def _read(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _write(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def _out(data: Any) -> None:
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(json.dumps(data, ensure_ascii=False, default=str))


def _get_path(data: dict, path: str) -> Any:
    current = data
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _set_path(data: dict, path: str, value: Any) -> None:
    keys = path.split(".")
    current = data
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


def _coerce(value_str: str) -> Any:
    """Parse YAML scalar so 'true'→True, '42'→42, 'null'→None, etc."""
    try:
        return yaml.safe_load(value_str)
    except Exception:
        return value_str


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_show(args) -> None:
    path = _find_config()
    if not path:
        _out({"success": False, "error": "config.yaml not found"})
        return
    data = _read(path)
    if args.section:
        _out({"success": True, "section": args.section, "value": data.get(args.section, {}), "config_path": path})
    else:
        result = {s: data[s] for s in SETTABLE_SECTIONS if s in data}
        _out({"success": True, "config": result, "config_path": path})


def cmd_get(args) -> None:
    path = _find_config()
    if not path:
        _out({"success": False, "error": "config.yaml not found"})
        return
    data = _read(path)
    value = _get_path(data, args.path)
    _out({"success": True, "path": args.path, "value": value})




def cmd_model(args) -> None:
    config_path = _find_config()
    models: dict = {}
    if config_path:
        data = _read(config_path)
        models = data.get("models", {})

    if not args.name:
        # show: current override + available model keys
        user_id = args.user_id or os.environ.get("PAWLIA_USER_ID")
        session_dir = args.session_dir or os.environ.get("PAWLIA_SESSION_DIR")
        if not user_id or not session_dir:
            _out({"success": False, "error": "user-id and session-dir required"})
            return
        override_path = os.path.join(session_dir, user_id, "workspace", "memory", "model_override.txt")
        current = ""
        if os.path.isfile(override_path):
            with open(override_path, encoding="utf-8") as f:
                current = f.read().strip()
        available = {key: cfg.get("model", key) for key, cfg in models.items() if isinstance(cfg, dict)}
        _out({"success": True, "model": current or "(default)", "available_models": available})
        return

    # Name must be a known config key
    if args.name not in models:
        available = {key: cfg.get("model", key) for key, cfg in models.items() if isinstance(cfg, dict)}
        _out({"success": False, "error": f"Unknown model '{args.name}'", "available_models": available})
        return

    resolved = models[args.name]["model"]

    # set model — write file for consistent readback, directive triggers in-memory update
    user_id = args.user_id or os.environ.get("PAWLIA_USER_ID")
    session_dir = args.session_dir or os.environ.get("PAWLIA_SESSION_DIR")
    if user_id and session_dir:
        override_path = os.path.join(session_dir, user_id, "workspace", "memory", "model_override.txt")
        os.makedirs(os.path.dirname(override_path), exist_ok=True)
        with open(override_path, "w", encoding="utf-8") as f:
            f.write(resolved)
    _out({"__directive__": "set_model", "model": resolved})
    _out({"success": True, "model": resolved, "message": f"Model auf '{resolved}' gesetzt."})


def cmd_private(args) -> None:
    user_id = args.user_id or os.environ.get("PAWLIA_USER_ID")
    session_dir = args.session_dir or os.environ.get("PAWLIA_SESSION_DIR")
    if not user_id or not session_dir:
        _out({"success": False, "error": "user-id and session-dir required"})
        return
    memory_dir = os.path.join(session_dir, user_id, "memory")
    os.makedirs(memory_dir, exist_ok=True)

    if args.thread:
        path = os.path.join(memory_dir, f"private_thread_{args.thread}")
    else:
        path = os.path.join(memory_dir, "private_session")

    scope = f"Thread {args.thread}" if args.thread else "Session"

    if args.off:
        if os.path.isfile(path):
            os.remove(path)
        _out({"success": True, "private": False, "scope": scope})
    else:
        with open(path, "w") as f:
            f.write("")
        _out({"success": True, "private": True, "scope": scope})


def cmd_set(args) -> None:
    config_path = _find_config()
    if not config_path:
        _out({"success": False, "error": "config.yaml not found"})
        return
    top_key = args.path.split(".")[0]
    if top_key not in SETTABLE_SECTIONS:
        _out({
            "success": False,
            "error": f"Section '{top_key}' is read-only via this skill. "
                     f"Settable sections: {', '.join(sorted(SETTABLE_SECTIONS))}",
        })
        return
    data = _read(config_path)
    value = _coerce(args.value)
    _set_path(data, args.path, value)
    _write(config_path, data)
    written = _get_path(_read(config_path), args.path)
    _out({"success": True, "path": args.path, "value_set": value, "value_read_back": written})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    p = sub.add_parser("show")
    p.add_argument("--section", default=None, help="Limit to one config section")

    p = sub.add_parser("get")
    p.add_argument("--path", required=True, help="Dot-notation path, e.g. interfaces.matrix.always_thread")

    p = sub.add_parser("set")
    p.add_argument("--path", required=True)
    p.add_argument("--value", required=True, help="Value (YAML scalar: true/false/number/string)")

    p = sub.add_parser("model")
    p.add_argument("--name", default=None, help="Model name to switch to (omit to show current)")
    p.add_argument("--user-id", default=None)
    p.add_argument("--session-dir", default=None)

    p = sub.add_parser("private")
    p.add_argument("--thread", default=None, help="Thread ID (omit for session-level)")
    p.add_argument("--off", action="store_true", help="Disable private mode")
    p.add_argument("--user-id", default=None)
    p.add_argument("--session-dir", default=None)

    args = parser.parse_args()
    if not args.cmd:
        parser.print_help()
        sys.exit(1)

    dispatch = {"show": cmd_show, "get": cmd_get, "set": cmd_set, "model": cmd_model, "private": cmd_private}
    try:
        dispatch[args.cmd](args)
    except Exception as e:
        _out({"success": False, "error": str(e)})
        sys.exit(1)


if __name__ == "__main__":
    main()
