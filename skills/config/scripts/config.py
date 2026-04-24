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
VALID_AGENT_PATHS = {"default", "defaults", "chat", "skill_runner", "vision", "compiler"}

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


def _flatten_overrides(data: dict, prefix: str = "") -> dict:
    flat = {}
    for key, value in (data or {}).items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten_overrides(value, path))
        elif isinstance(value, str) and value.strip():
            flat[path] = value.strip()
    return flat


def _valid_agent_path(path: str) -> bool:
    return path in VALID_AGENT_PATHS or (path.startswith("skills.") and len(path.split(".")) == 2)


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
        current = ""
        agents_path = os.path.join(session_dir, user_id, "workspace", "memory", "agent_overrides.yaml")
        if os.path.isfile(agents_path):
            overrides = _read(agents_path)
            current = str((overrides or {}).get("chat", "") or "").strip()
        if not current:
            override_path = os.path.join(session_dir, user_id, "workspace", "memory", "model_override.txt")
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

    _out({"__directive__": "set_agent_override", "path": "chat", "value": args.name})
    _out({"success": True, "model": args.name, "message": f"Model auf '{args.name}' gesetzt."})


def cmd_agent(args) -> None:
    config_path = _find_config()
    models: dict = {}
    if config_path:
        data = _read(config_path)
        models = data.get("models", {})

    user_id = args.user_id or os.environ.get("PAWLIA_USER_ID")
    session_dir = args.session_dir or os.environ.get("PAWLIA_SESSION_DIR")
    if not user_id or not session_dir:
        _out({"success": False, "error": "user-id and session-dir required"})
        return

    memory_dir = os.path.join(session_dir, user_id, "workspace", "memory")
    os.makedirs(memory_dir, exist_ok=True)
    override_path = os.path.join(
        memory_dir,
        f"thread_{args.thread}_agents.yaml" if args.thread else "agent_overrides.yaml",
    )
    overrides = _read(override_path) if os.path.isfile(override_path) else {}
    flat = _flatten_overrides(overrides)

    if not args.path:
        _out({
            "success": True,
            "scope": f"thread:{args.thread}" if args.thread else "session",
            "overrides": overrides,
            "flat_overrides": flat,
            "available_models": {key: cfg.get("model", key) for key, cfg in models.items() if isinstance(cfg, dict)},
        })
        return

    if not _valid_agent_path(args.path):
        _out({
            "success": False,
            "error": "Invalid agent path",
            "valid_examples": ["default", "chat", "skill_runner", "vision", "skills.browser"],
        })
        return

    if args.off:
        _out({"__directive__": "set_agent_override", "path": args.path, "value": None, "thread": args.thread})
        _out({"success": True, "path": args.path, "value": "(default)", "scope": f"thread:{args.thread}" if args.thread else "session"})
        return

    if not args.value:
        _out({
            "success": True,
            "path": args.path,
            "value": flat.get(args.path, "(default)"),
            "scope": f"thread:{args.thread}" if args.thread else "session",
            "available_models": {key: cfg.get("model", key) for key, cfg in models.items() if isinstance(cfg, dict)},
        })
        return

    _out({"__directive__": "set_agent_override", "path": args.path, "value": args.value, "thread": args.thread})
    _out({"success": True, "path": args.path, "value": args.value, "scope": f"thread:{args.thread}" if args.thread else "session"})


_PIPER_DIR = "/app/piper"


def _current_tts_provider() -> str:
    """Read tts.provider from config.yaml (defaults to 'piper')."""
    path = _find_config()
    if not path:
        return "piper"
    data = _read(path)
    return (data.get("tts") or {}).get("provider") or "piper"


def _list_piper_voices() -> list:
    """Return Piper voice names by globbing the model dir."""
    import glob
    if not os.path.isdir(_PIPER_DIR):
        return []
    return sorted(
        os.path.basename(p)[:-len(".onnx")]
        for p in glob.glob(os.path.join(_PIPER_DIR, "*.onnx"))
    )


def _list_edge_voices() -> list:
    """Return all Edge voices via edge_tts.list_voices(); empty if unavailable."""
    try:
        import asyncio as _asyncio
        import edge_tts  # type: ignore
        voices = _asyncio.run(edge_tts.list_voices())
        return sorted(v["ShortName"] for v in voices if "ShortName" in v)
    except Exception:
        return []


def _voices_for_provider(provider: str) -> list:
    if provider == "edge":
        return _list_edge_voices()
    return _list_piper_voices()


def cmd_voice(args) -> None:
    user_id = args.user_id or os.environ.get("PAWLIA_USER_ID")
    session_dir = args.session_dir or os.environ.get("PAWLIA_SESSION_DIR")
    if not user_id or not session_dir:
        _out({"success": False, "error": "user-id and session-dir required"})
        return

    override_path = os.path.join(
        session_dir, user_id, "workspace", "memory", "voice_override.txt",
    )

    provider = _current_tts_provider()
    available = _voices_for_provider(provider)

    if args.off:
        if os.path.isfile(override_path):
            os.remove(override_path)
        _out({"__directive__": "set_voice", "voice": None})
        _out({"success": True, "voice": "(default)", "message": "Voice-Override entfernt."})
        return

    if not args.name:
        current = ""
        if os.path.isfile(override_path):
            with open(override_path, encoding="utf-8") as f:
                current = f.read().strip()
        _out({
            "success": True,
            "voice": current or "(default)",
            "provider": provider,
            "available_voices": available,
        })
        return

    if args.name not in available:
        # For Piper the list is authoritative — it globs the on-disk .onnx files,
        # so a voice not in the list does not exist and --force cannot conjure it.
        # For Edge the dynamic list may be incomplete, so --force is allowed there.
        if provider == "piper" or not args.force:
            _out({
                "success": False,
                "error": (
                    f"Unknown voice '{args.name}' for provider '{provider}'. "
                    + ("Pick from available_voices — the Piper list is the on-disk model files."
                       if provider == "piper"
                       else "Pick from available_voices or use --force for Edge voices not in the dynamic list.")
                ),
                "provider": provider,
                "available_voices": available,
            })
            return

    os.makedirs(os.path.dirname(override_path), exist_ok=True)
    with open(override_path, "w", encoding="utf-8") as f:
        f.write(args.name)
    _out({"__directive__": "set_voice", "voice": args.name})
    _out({"success": True, "voice": args.name, "provider": provider,
          "message": f"Voice auf '{args.name}' gesetzt."})


def cmd_private(args) -> None:
    scope = f"Thread {args.thread}" if args.thread else "Session"
    private = not args.off

    _out({"__directive__": "set_private", "private": private, "thread": args.thread})
    _out({"success": True, "private": private, "scope": scope})


_TTS_VOICE_PATHS = {
    "tts.piper.model": "piper",
    "tts.edge.voice": "edge",
}


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
    value = _coerce(args.value)

    # Validate TTS voice/model writes — wrong values silently break TTS.
    # Piper: the list is authoritative (on-disk .onnx files), --force can't
    # conjure a missing file. Edge: --force allowed because the dynamic list
    # may be incomplete when edge_tts isn't installed.
    if args.path in _TTS_VOICE_PATHS:
        provider = _TTS_VOICE_PATHS[args.path]
        available = _voices_for_provider(provider)
        if value not in available:
            if provider == "piper" or not args.force:
                _out({
                    "success": False,
                    "error": (
                        f"'{value}' is not a known {provider} voice. "
                        + ("Pick from available_voices — the Piper list is the on-disk model files."
                           if provider == "piper"
                           else "Pick from available_voices or use --force for Edge voices not in the dynamic list.")
                    ),
                    "provider": provider,
                    "available_voices": available,
                })
                return

    # Validate provider switch — only piper/edge are wired up.
    if args.path == "tts.provider" and not args.force and value not in ("piper", "edge"):
        _out({
            "success": False,
            "error": f"Unknown tts.provider '{value}'. Supported: piper, edge. "
                     f"Use --force to override.",
        })
        return

    data = _read(config_path)
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
    p.add_argument("--force", action="store_true",
                   help="Skip TTS voice/provider validation (use a voice not in the curated list)")

    p = sub.add_parser("model")
    p.add_argument("--name", default=None, help="Model name to switch to (omit to show current)")
    p.add_argument("--user-id", default=None)
    p.add_argument("--session-dir", default=None)

    p = sub.add_parser("agent")
    p.add_argument("--path", default=None, help="Relative agents path, e.g. chat or skills.browser")
    p.add_argument("--value", default=None, help="Selector value, e.g. smart,fast")
    p.add_argument("--off", action="store_true", help="Clear the override at --path")
    p.add_argument("--thread", default=None, help="Thread ID (omit for session-level)")
    p.add_argument("--user-id", default=None)
    p.add_argument("--session-dir", default=None)

    p = sub.add_parser("voice")
    p.add_argument("--name", default=None,
                   help="Voice name — Piper (e.g. de_DE-thorsten-low) or Edge (e.g. de-DE-KatjaNeural)")
    p.add_argument("--off", action="store_true", help="Clear voice override")
    p.add_argument("--force", action="store_true",
                   help="Skip validation against the available_voices list for the active provider")
    p.add_argument("--user-id", default=None)
    p.add_argument("--session-dir", default=None)

    p = sub.add_parser("private")
    p.add_argument("--thread", default=None, help="Thread ID (omit for session-level)")
    p.add_argument("--off", action="store_true", help="Disable private mode")

    args = parser.parse_args()
    if not args.cmd:
        parser.print_help()
        sys.exit(1)

    dispatch = {
        "show": cmd_show, "get": cmd_get, "set": cmd_set,
        "model": cmd_model, "agent": cmd_agent, "voice": cmd_voice, "private": cmd_private,
    }
    try:
        dispatch[args.cmd](args)
    except Exception as e:
        _out({"success": False, "error": str(e)})
        sys.exit(1)


if __name__ == "__main__":
    main()
