"""
File workspace script – sandboxed read/write/list within session/{user_id}/workspace/.

Usage:
  python files.py list   --user-id <id> --session-dir <dir>
  python files.py read   --user-id <id> --session-dir <dir> --filename <name>
  python files.py write  --user-id <id> --session-dir <dir> --filename <name>
                         (file content is read from stdin)
"""

import argparse
import json
import os
import sys


# ---------------------------------------------------------------------------
# Sandbox helpers
# ---------------------------------------------------------------------------

def _workdir(user_id: str, session_dir: str) -> str:
    path = os.path.join(session_dir, user_id, "workspace")
    os.makedirs(path, exist_ok=True)
    return path


def _safe_path(workdir: str, filename: str) -> str:
    resolved = os.path.realpath(os.path.join(workdir, filename))
    root = os.path.realpath(workdir)
    if not resolved.startswith(root + os.sep) and resolved != root:
        raise ValueError(f"Access denied: '{filename}' is outside the workspace.")
    return resolved


def _out(data) -> None:
    print(json.dumps(data, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_list(args) -> None:
    workdir = _workdir(args.user_id, args.session_dir)
    entries = []
    for name in sorted(os.listdir(workdir)):
        path = os.path.join(workdir, name)
        entry = {"name": name, "type": "directory" if os.path.isdir(path) else "file"}
        if entry["type"] == "file":
            entry["size"] = os.path.getsize(path)
        entries.append(entry)
    _out({"success": True, "files": entries, "count": len(entries)})


def cmd_read(args) -> None:
    workdir = _workdir(args.user_id, args.session_dir)
    try:
        filepath = _safe_path(workdir, args.filename)
    except ValueError as e:
        _out({"success": False, "error": str(e)})
        return
    if not os.path.exists(filepath):
        _out({"success": False, "error": f"File '{args.filename}' not found."})
        return
    if os.path.isdir(filepath):
        _out({"success": False, "error": f"'{args.filename}' is a directory."})
        return
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    _out({"success": True, "filename": args.filename, "content": content, "size": len(content)})


def cmd_delete(args) -> None:
    workdir = _workdir(args.user_id, args.session_dir)
    try:
        filepath = _safe_path(workdir, args.filename)
    except ValueError as e:
        _out({"success": False, "error": str(e)})
        return
    if not os.path.exists(filepath):
        _out({"success": False, "error": f"File '{args.filename}' not found."})
        return
    if os.path.isdir(filepath):
        _out({"success": False, "error": f"'{args.filename}' is a directory."})
        return
    os.remove(filepath)
    _out({"success": True, "message": f"File '{args.filename}' deleted."})


def cmd_write(args) -> None:
    workdir = _workdir(args.user_id, args.session_dir)
    try:
        filepath = _safe_path(workdir, args.filename)
    except ValueError as e:
        _out({"success": False, "error": str(e)})
        return
    # Accept content from --content flag, CONTENT env var, or stdin
    if args.content is not None:
        content = args.content.replace("\\n", "\n").replace("\\t", "\t").replace("\\r", "\r")
    elif "CONTENT" in os.environ:
        content = os.environ["CONTENT"]
    else:
        content = sys.stdin.read()
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    with open(filepath, "r", encoding="utf-8") as f:
        written = f.read()
    _out({"success": True, "message": f"File '{args.filename}' written.", "bytes_written": len(written), "content_written": written})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    def _base(p):
        p.add_argument("--user-id", default=os.environ.get("PAWLIA_USER_ID"))
        p.add_argument("--session-dir", default=os.environ.get("PAWLIA_SESSION_DIR"))

    p = sub.add_parser("list")
    _base(p)

    p = sub.add_parser("read")
    _base(p)
    p.add_argument("--filename", required=True)

    p = sub.add_parser("write")
    _base(p)
    p.add_argument("--filename", required=True)
    p.add_argument("--content", default=None, help="File content (alternative to stdin)")

    p = sub.add_parser("delete")
    _base(p)
    p.add_argument("--filename", required=True)

    args = parser.parse_args()

    if not args.user_id or not args.session_dir:
        _out({"success": False, "error": "user-id and session-dir are required (via args or PAWLIA_USER_ID / PAWLIA_SESSION_DIR env vars)."})
        sys.exit(1)

    dispatch = {"list": cmd_list, "read": cmd_read, "write": cmd_write, "delete": cmd_delete}
    fn = dispatch.get(args.cmd)
    if not fn:
        _out({"success": False, "error": f"Unknown subcommand: {args.cmd}"})
        sys.exit(1)

    try:
        fn(args)
    except Exception as e:
        _out({"success": False, "error": str(e)})
        sys.exit(1)


if __name__ == "__main__":
    main()
