"""
File workspace script – sandboxed read/write/list/edit/grep within
session/{user_id}/workspace/.

Usage:
  python files.py list   --user-id <id> --session-dir <dir>
  python files.py read   --user-id <id> --session-dir <dir> --filename <name>
                         [--offset N] [--limit M]
  python files.py write  --user-id <id> --session-dir <dir> --filename <name>
                         (file content via CONTENT env var or stdin)
  python files.py edit   --user-id <id> --session-dir <dir> --filename <name>
                         [--replace-all]
                         (old/new strings via OLD_STRING / NEW_STRING env vars)
  python files.py grep   --user-id <id> --session-dir <dir> --pattern <regex>
                         [--filename <name>]
  python files.py delete --user-id <id> --session-dir <dir> --filename <name>
"""

import argparse
import json
import os
import re
import sys

# Add the project root to the Python path so we can import pawlia modules
# __file__ = skills/files/scripts/files.py -> up 4 levels to project root
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

from pawlia.utils import ensure_dir


# ---------------------------------------------------------------------------
# Sandbox helpers
# ---------------------------------------------------------------------------

def _workdir(user_id: str, session_dir: str) -> str:
    path = os.path.join(session_dir, user_id, "workspace")
    return ensure_dir(path)


def _ci_resolve(base: str, rel: str) -> str:
    """Resolve rel within base using case-insensitive matching at each path component.
    If a component exists with different casing, the on-disk name wins.
    Components that don't exist yet keep their original casing.
    """
    parts = rel.replace("\\", "/").split("/")
    current = base
    for i, part in enumerate(parts):
        if not part:
            continue
        exact = os.path.join(current, part)
        if os.path.exists(exact):
            current = exact
        else:
            try:
                lower = part.lower()
                match = next((e for e in os.listdir(current) if e.lower() == lower), None)
            except OSError:
                match = None
            if match:
                current = os.path.join(current, match)
            else:
                # Not found — keep original casing for this and remaining parts
                current = os.path.join(current, *parts[i:])
                break
    return current


def _safe_path(workdir: str, filename: str) -> str:
    resolved = os.path.realpath(_ci_resolve(workdir, filename))
    root = os.path.realpath(workdir)
    if not resolved.startswith(root + os.sep) and resolved != root:
        raise ValueError(f"Access denied: '{filename}' is outside the workspace.")
    return resolved


def _out(data) -> None:
    # Force UTF-8 output on Windows to avoid charmap codec errors with emoji/unicode
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(json.dumps(data, ensure_ascii=False))


def _walk_files(workdir: str):
    """Yield (abs_path, rel_path) for every file under workdir, recursively."""
    for root, dirs, files in os.walk(workdir):
        dirs.sort()
        for name in sorted(files):
            abs_path = os.path.join(root, name)
            rel_path = os.path.relpath(abs_path, workdir)
            # Normalize to forward slashes so the LLM gets consistent paths
            yield abs_path, rel_path.replace(os.sep, "/")


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_WIKILINK_INPUT_RE = re.compile(r"^\[\[([^\]|]+)(?:\|[^\]]+)?\]\]$")


def _resolve_wikilink(workdir: str, filename: str) -> str:
    """If *filename* is a [[wikilink]], resolve it to a workspace-relative path.

    Resolution order:
    1. Contains a slash → treated as path (`.md` appended if missing).
    2. Slug only → `wiki/topics/{slug}.md` if it exists.
    3. Fallback: first `{slug}.md` found anywhere in the workspace.
    4. If nothing found, return `wiki/topics/{slug}.md` as best guess.
    """
    m = _WIKILINK_INPUT_RE.match(filename.strip())
    if not m:
        return filename

    ref = m.group(1).strip()

    # Path-style reference (contains slash)
    if "/" in ref:
        return ref if ref.endswith(".md") else ref + ".md"

    # Slug-style: wiki/topics first
    wiki_candidate = f"wiki/topics/{ref}.md"
    if os.path.isfile(os.path.join(workdir, wiki_candidate)):
        return wiki_candidate

    # Search entire workspace for {slug}.md
    target = f"{ref}.md"
    for dirpath, _dirs, files in os.walk(workdir):
        if target in files:
            rel = os.path.relpath(os.path.join(dirpath, target), workdir)
            return rel.replace(os.sep, "/")

    # Nothing found — return wiki/topics guess (will produce "not found" naturally)
    return wiki_candidate


def _parse_headings(lines: list[str]) -> list[dict]:
    """Return ATX markdown headings as [{level, title, line}], skipping fenced code."""
    out: list[dict] = []
    in_fence = False
    fence_marker = ""
    for i, raw in enumerate(lines, 1):
        line = raw.rstrip("\n")
        fence_match = _FENCE_RE.match(line)
        if fence_match:
            marker = fence_match.group(1)
            if not in_fence:
                in_fence = True
                fence_marker = marker
            elif marker == fence_marker:
                in_fence = False
                fence_marker = ""
            continue
        if in_fence:
            continue
        # ATX headings must not be indented by 4+ spaces (would be code block)
        if line.startswith("    "):
            continue
        m = _HEADING_RE.match(line.lstrip(" "))
        if m:
            out.append({"level": len(m.group(1)), "title": m.group(2).strip(), "line": i})
    return out


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_list(args) -> None:
    workdir = _workdir(args.user_id, args.session_dir)
    entries = []
    for abs_path, rel_path in _walk_files(workdir):
        entries.append({
            "name": rel_path,
            "type": "file",
            "size": os.path.getsize(abs_path),
        })
    _out({"success": True, "files": entries, "count": len(entries)})


def cmd_read(args) -> None:
    workdir = _workdir(args.user_id, args.session_dir)
    args.filename = _resolve_wikilink(workdir, args.filename)
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

    if args.offset is None and args.limit is None:
        _out({"success": True, "filename": args.filename, "content": content, "size": len(content)})
        return

    lines = content.splitlines(keepends=True)
    total = len(lines)
    offset = max(0, args.offset or 0)
    limit = args.limit if args.limit is not None else total
    sliced = "".join(lines[offset:offset + limit])
    _out({
        "success": True,
        "filename": args.filename,
        "content": sliced,
        "offset": offset,
        "limit": limit,
        "lines_returned": max(0, min(total - offset, limit)),
        "total_lines": total,
    })


def cmd_outline(args) -> None:
    workdir = _workdir(args.user_id, args.session_dir)
    args.filename = _resolve_wikilink(workdir, args.filename)
    try:
        filepath = _safe_path(workdir, args.filename)
    except ValueError as e:
        _out({"success": False, "error": str(e)})
        return
    if not os.path.isfile(filepath):
        _out({"success": False, "error": f"File '{args.filename}' not found."})
        return
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    headings = _parse_headings(lines)
    _out({
        "success": True,
        "filename": args.filename,
        "headings": headings,
        "count": len(headings),
        "total_lines": len(lines),
    })


def cmd_read_section(args) -> None:
    workdir = _workdir(args.user_id, args.session_dir)
    args.filename = _resolve_wikilink(workdir, args.filename)
    try:
        filepath = _safe_path(workdir, args.filename)
    except ValueError as e:
        _out({"success": False, "error": str(e)})
        return
    if not os.path.isfile(filepath):
        _out({"success": False, "error": f"File '{args.filename}' not found."})
        return
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    headings = _parse_headings(lines)
    target = args.section.lstrip("#").strip()
    matches = [h for h in headings if h["title"] == target]
    if not matches:
        # Fall back to case-insensitive comparison so the LLM doesn't have to nail the case
        ci_matches = [h for h in headings if h["title"].lower() == target.lower()]
        if not ci_matches:
            available = [h["title"] for h in headings]
            _out({
                "success": False,
                "error": f"Section '{args.section}' not found in '{args.filename}'.",
                "available_sections": available,
            })
            return
        matches = ci_matches
    if len(matches) > 1:
        _out({
            "success": False,
            "error": (
                f"Section '{args.section}' is ambiguous ({len(matches)} matches at lines "
                f"{[m['line'] for m in matches]}). Heading titles must be unique to use read_section."
            ),
        })
        return
    head = matches[0]
    start = head["line"] - 1  # 0-based, include the heading line itself
    # End at the next heading with level <= head['level']
    end = len(lines)
    for h in headings:
        if h["line"] > head["line"] and h["level"] <= head["level"]:
            end = h["line"] - 1
            break
    section_text = "".join(lines[start:end])
    _out({
        "success": True,
        "filename": args.filename,
        "section": head["title"],
        "level": head["level"],
        "start_line": start + 1,
        "end_line": end,
        "content": section_text,
    })


def cmd_delete(args) -> None:
    workdir = _workdir(args.user_id, args.session_dir)
    args.filename = _resolve_wikilink(workdir, args.filename)
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


def cmd_edit(args) -> None:
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

    old_string = os.environ.get("OLD_STRING")
    new_string = os.environ.get("NEW_STRING")
    if old_string is None:
        old_string = args.old_string
    if new_string is None:
        new_string = args.new_string
    if old_string is None or new_string is None:
        _out({"success": False, "error": "OLD_STRING and NEW_STRING (env vars or --old-string/--new-string) are required."})
        return
    if old_string == "":
        _out({"success": False, "error": "OLD_STRING must not be empty."})
        return
    if old_string == new_string:
        _out({"success": False, "error": "OLD_STRING and NEW_STRING are identical — nothing to do."})
        return

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    count = content.count(old_string)
    if count == 0:
        _out({
            "success": False,
            "error": f"OLD_STRING not found in '{args.filename}'. Read the file again and copy the exact text to replace."
        })
        return
    if count > 1 and not args.replace_all:
        _out({
            "success": False,
            "error": (
                f"OLD_STRING occurs {count} times in '{args.filename}'. "
                "Either include more surrounding context to make it unique, or pass --replace-all."
            ),
        })
        return

    if args.replace_all:
        new_content = content.replace(old_string, new_string)
        replacements = count
    else:
        new_content = content.replace(old_string, new_string, 1)
        replacements = 1

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(new_content)
    with open(filepath, "r", encoding="utf-8") as f:
        written = f.read()

    _out({
        "success": True,
        "message": f"File '{args.filename}' edited ({replacements} replacement{'s' if replacements != 1 else ''}).",
        "replacements": replacements,
        "bytes_written": len(written),
        "content_after": written,
    })


def cmd_grep(args) -> None:
    workdir = _workdir(args.user_id, args.session_dir)
    try:
        regex = re.compile(args.pattern)
    except re.error as e:
        _out({"success": False, "error": f"Invalid regex: {e}"})
        return

    if args.filename:
        try:
            filepath = _safe_path(workdir, args.filename)
        except ValueError as e:
            _out({"success": False, "error": str(e)})
            return
        if not os.path.isfile(filepath):
            _out({"success": False, "error": f"File '{args.filename}' not found."})
            return
        targets = [(filepath, os.path.relpath(filepath, workdir).replace(os.sep, "/"))]
    else:
        targets = list(_walk_files(workdir))

    matches = []
    skipped: list[str] = []
    for abs_path, rel_path in targets:
        try:
            with open(abs_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f, 1):
                    if regex.search(line):
                        matches.append({
                            "filename": rel_path,
                            "line": i,
                            "text": line.rstrip("\n"),
                        })
        except (UnicodeDecodeError, OSError):
            skipped.append(rel_path)
            continue

    out = {"success": True, "matches": matches, "count": len(matches)}
    if skipped:
        out["skipped_binary_or_unreadable"] = skipped
    _out(out)


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
    p.add_argument("--offset", type=int, default=None, help="0-based line offset")
    p.add_argument("--limit", type=int, default=None, help="max lines to return")

    p = sub.add_parser("write")
    _base(p)
    p.add_argument("--filename", required=True)
    p.add_argument("--content", default=None, help="File content (alternative to stdin)")

    p = sub.add_parser("edit")
    _base(p)
    p.add_argument("--filename", required=True)
    p.add_argument("--old-string", default=None, help="text to replace (or use OLD_STRING env var)")
    p.add_argument("--new-string", default=None, help="replacement text (or use NEW_STRING env var)")
    p.add_argument("--replace-all", action="store_true", help="replace every occurrence instead of requiring uniqueness")

    p = sub.add_parser("grep")
    _base(p)
    p.add_argument("--pattern", required=True, help="Python regex")
    p.add_argument("--filename", default=None, help="restrict to a single file (default: whole workspace)")

    p = sub.add_parser("outline")
    _base(p)
    p.add_argument("--filename", required=True, help="markdown file to outline")

    p = sub.add_parser("read-section")
    _base(p)
    p.add_argument("--filename", required=True)
    p.add_argument("--section", required=True, help="heading title (without leading #s)")

    p = sub.add_parser("delete")
    _base(p)
    p.add_argument("--filename", required=True)

    args = parser.parse_args()

    if not args.user_id or not args.session_dir:
        _out({"success": False, "error": "user-id and session-dir are required (via args or PAWLIA_USER_ID / PAWLIA_SESSION_DIR env vars)."})
        sys.exit(1)

    dispatch = {
        "list": cmd_list,
        "read": cmd_read,
        "write": cmd_write,
        "edit": cmd_edit,
        "grep": cmd_grep,
        "outline": cmd_outline,
        "read-section": cmd_read_section,
        "delete": cmd_delete,
    }
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
