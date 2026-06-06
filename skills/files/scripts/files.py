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


def _suggest_similar_files(workdir: str, filename: str, limit: int = 5) -> list[str]:
    """Return up to *limit* files that look similar to the requested filename."""
    base = filename.replace(".md", "").lower()
    suggestions = []
    for _, rel_path in _walk_files(workdir):
        if not rel_path.endswith(".md"):
            continue
        rel_lower = rel_path.lower()
        # Check if any part of the requested path appears in the file
        parts = [p.lower() for p in base.replace("/", " ").split()]
        score = sum(1 for p in parts if p in rel_lower)
        if score > 0:
            suggestions.append((score, rel_path))
    suggestions.sort(key=lambda x: -x[0])
    return [p for _, p in suggestions[:limit]]


def _file_not_found_response(workdir: str, filename: str) -> dict:
    """Build a helpful 'file not found' response with suggestions."""
    result = {"success": False, "error": f"File '{filename}' not found."}

    # Suggest similar files
    similar = _suggest_similar_files(workdir, filename)
    if similar:
        result["suggestions"] = similar
        result["hint"] = (
            "Did you mean one of these? Pass the exact path to retry. "
            "For wikilinks like [[person/name]], the file is usually under wiki/topics/."
        )

    # If it looks like a wikilink slug, remind about wiki/topics/
    if "/" not in filename and not filename.startswith("[["):
        wiki_path = f"wiki/topics/{filename}"
        if not filename.endswith(".md"):
            wiki_path += ".md"
        if os.path.isfile(os.path.join(workdir, wiki_path)):
            result["suggestions"] = [wiki_path]
            result["hint"] = f"File exists at: {wiki_path}"

    return result


def _safe_path(workdir: str, filename: str) -> str:
    resolved = os.path.realpath(_ci_resolve(workdir, filename))
    root = os.path.realpath(workdir)
    if not resolved.startswith(root + os.sep) and resolved != root:
        raise ValueError(f"Access denied: '{filename}' is outside the workspace.")
    return resolved


def _resolve_filepath(workdir: str, filename: str) -> str | None:
    """Resolve and validate a file path. Returns None (with error output) on failure."""
    try:
        return _safe_path(workdir, filename)
    except ValueError as e:
        _out({"success": False, "error": str(e)})
        return None


def _out(data) -> None:
    # Force UTF-8 output on Windows to avoid charmap codec errors with emoji/unicode
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(json.dumps(data, ensure_ascii=False))


def _walk_files(workdir: str):
    """Yield (abs_path, rel_path) for every file under workdir, recursively.

    Hidden entries (names starting with ``.``) are skipped entirely — this
    prunes ``.git`` (thousands of objects), ``.obsidian``, ``.trash`` etc. so
    listings/greps reach real content like ``Downloads/`` instead of drowning.
    """
    for root, dirs, files in os.walk(workdir):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for name in sorted(files):
            if name.startswith("."):
                continue
            abs_path = os.path.join(root, name)
            rel_path = os.path.relpath(abs_path, workdir)
            # Normalize to forward slashes so the LLM gets consistent paths
            yield abs_path, rel_path.replace(os.sep, "/")


def _walk_entries(workdir: str):
    """Yield (abs_path, rel_path) for every file and directory under workdir, recursively.

    Hidden entries (names starting with ``.``) are skipped — see ``_walk_files``.
    """
    for root, dirs, files in os.walk(workdir):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for d in dirs:
            abs_path = os.path.join(root, d)
            rel_path = os.path.relpath(abs_path, workdir)
            yield abs_path, rel_path.replace(os.sep, "/")
        for name in sorted(files):
            if name.startswith("."):
                continue
            abs_path = os.path.join(root, name)
            rel_path = os.path.relpath(abs_path, workdir)
            yield abs_path, rel_path.replace(os.sep, "/")


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_WIKILINK_INPUT_RE = re.compile(r"^\[\[([^\]|]+)(?:\|[^\]]+)?\]\]$")
_WIKI_TOPIC_DIRS = {"person", "place", "object", "project", "topic"}


def _split_filename_section(filename: str) -> tuple[str, str | None]:
    """Split a file reference into path-like part and optional heading anchor."""
    text = filename.strip()
    m = _WIKILINK_INPUT_RE.match(text)
    if m:
        ref = m.group(1).strip()
        if "#" in ref:
            base, section = ref.split("#", 1)
            return f"[[{base}]]", section.strip() or None
        return filename, None

    if "#" in text:
        base, section = text.split("#", 1)
        base = base.strip()
        if base.endswith(".md") or "/" in base:
            return base, section.strip() or None

    return filename, None


def _resolve_wikilink(workdir: str, filename: str) -> str:
    """If *filename* is a [[wikilink]], resolve it to a workspace-relative path.

    Resolution order:
    1. Exact path match (with/without .md).
    2. If first path component is a wiki topic dir → try wiki/topics/{path}.md.
    3. Slug only → try wiki/topics/{slug}.md.
    4. Fallback: search entire workspace for matching filename.
    5. If nothing found, return best guess.
    """
    m = _WIKILINK_INPUT_RE.match(filename.strip())
    if not m:
        return filename

    ref = m.group(1).strip()

    _md_cache: list[tuple[str, str]] | None = None

    def _md_files():
        """Lazily collect all .md files in the workspace as (rel_path, fname)."""
        nonlocal _md_cache
        if _md_cache is None:
            _md_cache = [
                (rel, os.path.basename(rel))
                for _abs, rel in _walk_files(workdir)
                if rel.endswith(".md")
            ]
        return _md_cache

    # --- Path-style reference (contains slash) ---
    if "/" in ref:
        candidate = ref if ref.endswith(".md") else ref + ".md"

        # 1. Direct path match
        if os.path.isfile(os.path.join(workdir, candidate)):
            return candidate

        # 2. If first component is a wiki topic dir, try wiki/topics/{path}.md
        first = ref.split("/", 1)[0]
        if first in _WIKI_TOPIC_DIRS and not ref.startswith("wiki/topics/"):
            typed_candidate = f"wiki/topics/{candidate}"
            if os.path.isfile(os.path.join(workdir, typed_candidate)):
                return typed_candidate

        # 3. Search workspace for basename match, then fuzzy
        basename = os.path.basename(candidate)
        slug_parts = ref.replace("/", " ").lower()
        for rel, fname in _md_files():
            if fname == basename:
                return rel
        for rel, _ in _md_files():
            if slug_parts in rel.lower().replace("_", " ").replace("-", " "):
                return rel

        return candidate

    # --- Slug-style: wiki/topics first ---
    wiki_candidate = f"wiki/topics/{ref}.md"
    if os.path.isfile(os.path.join(workdir, wiki_candidate)):
        return wiki_candidate

    # Search workspace for {slug}.md, then fuzzy
    target = f"{ref}.md"
    slug_lower = ref.lower()
    for rel, fname in _md_files():
        if fname == target:
            return rel
    for rel, _ in _md_files():
        if slug_lower in rel.lower().replace("_", " ").replace("-", " "):
            return rel

    # Nothing found — return wiki/topics guess (will produce "not found" naturally)
    return wiki_candidate


def _load_section(filepath: str, filename: str, section: str) -> dict:
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    headings = _parse_headings(lines)
    target = section.lstrip("#").strip()
    matches = [h for h in headings if h["title"] == target]
    if not matches:
        ci_matches = [h for h in headings if h["title"].lower() == target.lower()]
        if not ci_matches:
            return {
                "success": False,
                "error": f"Section '{section}' not found in '{filename}'.",
                "available_sections": [h["title"] for h in headings],
            }
        matches = ci_matches
    if len(matches) > 1:
        return {
            "success": False,
            "error": (
                f"Section '{section}' is ambiguous ({len(matches)} matches at lines "
                f"{[m['line'] for m in matches]}). Heading titles must be unique to use read_section."
            ),
        }

    head = matches[0]
    start = head["line"] - 1
    end = len(lines)
    for h in headings:
        if h["line"] > head["line"] and h["level"] <= head["level"]:
            end = h["line"] - 1
            break
    return {
        "success": True,
        "filename": filename,
        "section": head["title"],
        "level": head["level"],
        "start_line": start + 1,
        "end_line": end,
        "content": "".join(lines[start:end]),
    }


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
    offset = max(0, int(getattr(args, "offset", 0) or 0))
    limit = max(1, int(getattr(args, "limit", 200) or 200))
    entries = []
    all_entries = list(_walk_entries(workdir))
    total = len(all_entries)
    for abs_path, rel_path in all_entries[offset:offset + limit]:
        is_dir = os.path.isdir(abs_path)
        entry = {
            "name": rel_path,
            "type": "dir" if is_dir else "file",
        }
        if not is_dir:
            entry["size"] = os.path.getsize(abs_path)
        entries.append(entry)
    has_more = offset + limit < total
    payload = {
        "success": True,
        "files": entries,
        "count": len(entries),
        "total_count": total,
        "offset": offset,
        "limit": limit,
        "has_more": has_more,
    }
    if has_more:
        payload["next_offset"] = offset + limit
        payload["hint"] = (
            "More files available. Call list again with "
            f"--offset {offset + limit} --limit {limit} if needed."
        )
    _out(payload)


_DEFAULT_READ_LIMIT = 150
_QUERY_CONTEXT_LINES = 3


def _filter_by_query(lines: list, query: str) -> tuple:
    """Return (filtered_text, match_count) keeping only lines near query tokens.

    Non-matching stretches are replaced with ``[... N lines skipped ...]``.
    Returns (None, 0) if no matches found.
    """
    tokens = {t for t in re.findall(r"\w+", query.lower()) if len(t) >= 3}
    if not tokens:
        return None, 0

    keep: set = set()
    for i, line in enumerate(lines):
        if tokens & set(re.findall(r"\w+", line.lower())):
            for j in range(max(0, i - _QUERY_CONTEXT_LINES),
                           min(len(lines), i + _QUERY_CONTEXT_LINES + 1)):
                keep.add(j)

    if not keep:
        return None, 0

    parts: list = []
    skip_start: int | None = None
    for i, line in enumerate(lines):
        if i in keep:
            if skip_start is not None:
                parts.append(f"[... {i - skip_start} lines skipped, no matches ...]\n")
                skip_start = None
            parts.append(line)
        else:
            if skip_start is None:
                skip_start = i
    if skip_start is not None and skip_start < len(lines):
        parts.append(f"[... {len(lines) - skip_start} lines skipped, no matches ...]\n")

    return "".join(parts), len(keep)


def cmd_read(args) -> None:
    workdir = _workdir(args.user_id, args.session_dir)
    filename, section = _split_filename_section(args.filename)
    args.filename = _resolve_wikilink(workdir, filename)
    filepath = _resolve_filepath(workdir, args.filename)
    if filepath is None:
        return
    if not os.path.exists(filepath):
        _out(_file_not_found_response(workdir, args.filename))
        return
    if os.path.isdir(filepath):
        _out({"success": False, "error": f"'{args.filename}' is a directory."})
        return

    # Section refs like [[page#Heading]] resolve directly to the section content.
    if section and args.query is None and args.offset is None and args.limit is None:
        _out(_load_section(filepath, args.filename, section))
        return

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    lines = content.splitlines(keepends=True)
    total = len(lines)

    # Query-based filtering: return only relevant blocks, skip the rest
    if args.query:
        filtered, match_count = _filter_by_query(lines, args.query)
        if filtered is None:
            _out({
                "success": True,
                "filename": args.filename,
                "content": "",
                "total_lines": total,
                "matches": 0,
                "note": "No lines matched the query.",
            })
        else:
            _out({
                "success": True,
                "filename": args.filename,
                "content": filtered,
                "total_lines": total,
                "matches": match_count,
            })
        return

    offset = max(0, args.offset or 0)
    limit = args.limit if args.limit is not None else _DEFAULT_READ_LIMIT
    sliced = "".join(lines[offset:offset + limit])
    returned = max(0, min(total - offset, limit))
    result = {
        "success": True,
        "filename": args.filename,
        "content": sliced,
        "offset": offset,
        "limit": limit,
        "lines_returned": returned,
        "total_lines": total,
    }
    if offset + returned < total:
        result["has_more"] = True
        result["next_offset"] = offset + returned
    _out(result)


def cmd_search(args) -> None:
    """BM25 search across workspace markdown files.

    Returns a list of hits with filename, headings, and a short snippet
    so the model can gauge relevance without having to ``files read``
    every candidate first.
    """
    workdir = _workdir(args.user_id, args.session_dir)
    limit = getattr(args, "limit", 10) or 10
    try:
        from pawlia.workspace_search import WorkspaceSearch
    except ImportError:
        _out({"success": False, "error": "workspace_search module not available"})
        return

    searcher = WorkspaceSearch(workdir, config={"top_k": limit})
    try:
        hits = searcher.search(args.query)
    except Exception as e:
        _out({"success": False, "error": f"search failed: {e}"})
        return

    results = []
    for hit in hits:
        rel = os.path.relpath(hit.path, workdir)
        with open(hit.path, "r", encoding="utf-8") as f:
            headings = _parse_headings(f.readlines())
        results.append({
            "filename": rel,
            "heading": hit.heading,
            "headings": [h["title"] for h in headings],
            "snippet": hit.snippet,
            "score": round(hit.score, 4),
        })
    _out({"success": True, "query": args.query, "results": results, "count": len(results)})


def cmd_outline(args) -> None:
    workdir = _workdir(args.user_id, args.session_dir)
    filename, _section = _split_filename_section(args.filename)
    args.filename = _resolve_wikilink(workdir, filename)
    filepath = _resolve_filepath(workdir, args.filename)
    if filepath is None:
        return
    if not os.path.isfile(filepath):
        _out(_file_not_found_response(workdir, args.filename))
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
    filename, embedded_section = _split_filename_section(args.filename)
    args.filename = _resolve_wikilink(workdir, filename)
    filepath = _resolve_filepath(workdir, args.filename)
    if filepath is None:
        return
    if not os.path.isfile(filepath):
        _out(_file_not_found_response(workdir, args.filename))
        return
    _out(_load_section(filepath, args.filename, embedded_section or args.section))


def cmd_delete(args) -> None:
    workdir = _workdir(args.user_id, args.session_dir)
    filename, _section = _split_filename_section(args.filename)
    args.filename = _resolve_wikilink(workdir, filename)
    filepath = _resolve_filepath(workdir, args.filename)
    if filepath is None:
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
    filepath = _resolve_filepath(workdir, args.filename)
    if filepath is None:
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
    filepath = _resolve_filepath(workdir, args.filename)
    if filepath is None:
        return
    if not os.path.exists(filepath):
        _out(_file_not_found_response(workdir, args.filename))
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
        _out({
            "success": False,
            "error": f"Invalid regex: {e}",
            "hint": (
                "--pattern is a Python re pattern, not a shell glob. "
                "Use '.*' (not '*') to match any chars, escape literal "
                "dots with '\\.', and use '[ ]' for character classes."
            ),
        })
        return

    if args.filename:
        filename, _section = _split_filename_section(args.filename)
        args.filename = _resolve_wikilink(workdir, filename)
        filepath = _resolve_filepath(workdir, args.filename)
        if filepath is None:
            return
        if not os.path.isfile(filepath):
            _out(_file_not_found_response(workdir, args.filename))
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
    p.add_argument("--offset", type=int, default=0, help="0-based file offset")
    p.add_argument("--limit", type=int, default=200, help="max files to return (default: 200)")

    p = sub.add_parser("read")
    _base(p)
    p.add_argument("--filename", required=True)
    p.add_argument("--offset", type=int, default=None, help="0-based line offset")
    p.add_argument("--limit", type=int, default=None, help="max lines to return (default: 150)")
    p.add_argument("--query", default=None, help="return only lines matching this query (skips replace irrelevant blocks)")

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

    p = sub.add_parser("search")
    _base(p)
    p.add_argument("--query", required=True)
    p.add_argument("--limit", type=int, default=10)

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
        "search": cmd_search,
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
