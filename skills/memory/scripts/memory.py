#!/usr/bin/env python3
"""Memory skill — query long-term conversation memory via the configured RAG backend.

Indexing is handled automatically by the scheduler. This script is the
query interface and provides manual index/status/dream/lint commands.

Usage:
    memory.py <user_id> search <question>
    memory.py <user_id> index          # manual trigger (debug)
    memory.py <user_id> consolidate    # merge duplicate topic files
    memory.py <user_id> dream          # manual dream wiki trigger
    memory.py <user_id> lint           # wiki health check
    memory.py <user_id> status
"""

import asyncio
import json
import os
import pathlib
import re
import sys

import yaml

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------

_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_SKILL_DIR = _SCRIPT_DIR.parent
_PROJECT_ROOT = _SKILL_DIR.parent.parent  # thalia/
_SESSION_DIR = pathlib.Path(os.environ["PAWLIA_SESSION_DIR"]) if "PAWLIA_SESSION_DIR" in os.environ else _PROJECT_ROOT / "session"

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}\.md$")

sys.path.insert(0, str(_PROJECT_ROOT))


def _load_skill_config() -> dict:
    raw_config = os.environ.get("PAWLIA_SKILL_CONFIG")
    if raw_config:
        try:
            return json.loads(raw_config)
        except json.JSONDecodeError:
            pass

    for candidate in (_PROJECT_ROOT / "config.yaml", _PROJECT_ROOT / "config.yml"):
        if candidate.is_file():
            with open(candidate, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            skill_config_root = cfg.get("skill-config") or {}
            return skill_config_root.get("memory", {})
    return {}


CFG = _load_skill_config()

# ---------------------------------------------------------------------------
# RAG backend (read-only — same index the scheduler writes to)
# ---------------------------------------------------------------------------

_backend_instance = None


async def _get_backend(user_id: str):
    from pawlia.rag_backend import create_backend

    global _backend_instance
    if _backend_instance is not None:
        return _backend_instance

    index_path = _SESSION_DIR / user_id / "memory_index"
    if not index_path.exists():
        return None

    # naive mode needs no LLM — only embeddings for similarity search
    _backend_instance = create_backend(
        str(index_path),
        CFG,
        max_async_embedding=4,
    )
    return _backend_instance


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def _recent_threads_block(user_id: str, question: str, limit: int = 5, max_days: int = 2) -> str:
    """Search the recent (not-yet-indexed) threads straight from the daily logs.

    Covers the gap the RAG index lags behind on — threads from today and the
    last day are matched against *question* by keyword, so a call or file from
    minutes ago is findable before the background indexer has processed it.
    """
    try:
        from pawlia.memory import MemoryManager

        threads = MemoryManager(str(_SESSION_DIR)).recent_threads(
            user_id, limit=limit, max_days=max_days, query=question,
        )
    except Exception:
        return ""
    if not threads:
        return ""
    lines = ["## Treffer in den jüngsten Gesprächen (noch nicht indiziert, direkt aus den Chat-Logs)"]
    for t in threads:
        lines.append(f"\n### {t['date']} — {t['title']}")
        lines.append(t["body"])
    return "\n".join(lines)


async def cmd_search(user_id: str, question: str):
    # Search the recent, not-yet-indexed threads too. These go first so they
    # survive any downstream output truncation.
    recent = _recent_threads_block(user_id, question)

    backend = await _get_backend(user_id)
    if backend is None:
        print(json.dumps({
            "result": recent or "Noch kein Langzeitgedächtnis vorhanden. Chatlogs werden automatisch im Hintergrund indiziert.",
        }, ensure_ascii=False))
        return

    rag_result = await backend.query(question)
    parts = []
    if recent:
        parts.append(recent)
    if rag_result and rag_result.strip():
        parts.append("## Langzeit-Wissen (Wiki)\n" + rag_result.strip())
    print(json.dumps({"result": "\n\n".join(parts)}, ensure_ascii=False))


async def cmd_index(user_id: str):
    """Manual index trigger — imports and runs the scheduler's indexer."""
    full_cfg = {}
    for candidate in (_PROJECT_ROOT / "config.yaml", _PROJECT_ROOT / "config.yml"):
        if candidate.is_file():
            with open(candidate, encoding="utf-8") as f:
                full_cfg = yaml.safe_load(f) or {}
            break

    from pawlia.memory_indexer import MemoryIndexer
    indexer = MemoryIndexer(str(_SESSION_DIR), full_cfg)
    if not indexer.enabled:
        print(json.dumps({"error": "Memory indexer not configured"}))
        sys.exit(1)

    await indexer.process_user(user_id)
    print(json.dumps({"status": "ok", "message": "Indexing complete"}))


async def cmd_consolidate(user_id: str):
    """Manually trigger topic consolidation on the existing index."""
    index_path = _SESSION_DIR / user_id / "memory_index"
    if not index_path.exists():
        print(json.dumps({"error": "Kein Index vorhanden"}))
        sys.exit(1)

    from pawlia.dream_wiki import DreamWikiBackend
    wiki_dir = str(_SESSION_DIR / user_id / "workspace" / "wiki")
    backend = DreamWikiBackend(str(index_path), CFG, wiki_dir=wiki_dir)
    await backend.consolidate()
    print(json.dumps({"status": "ok", "message": "Konsolidierung abgeschlossen"}))


async def cmd_dream(user_id: str):
    """Manually trigger Dream Wiki: process unprocessed daily logs into wiki pages."""
    full_cfg = {}
    for candidate in (_PROJECT_ROOT / "config.yaml", _PROJECT_ROOT / "config.yml"):
        if candidate.is_file():
            with open(candidate, encoding="utf-8") as f:
                full_cfg = yaml.safe_load(f) or {}
            break

    from pawlia.memory_indexer import MemoryIndexer
    indexer = MemoryIndexer(str(_SESSION_DIR), full_cfg)
    if not indexer.enabled:
        print(json.dumps({"error": "Memory indexer not configured"}))
        sys.exit(1)

    await indexer.process_user(user_id)
    print(json.dumps({"status": "ok", "message": "Dream Wiki Verarbeitung abgeschlossen"}))


async def cmd_lint(user_id: str):
    """Run wiki health check: merge overlapping pages, fix links, detect orphans."""
    index_path = _SESSION_DIR / user_id / "memory_index"
    if not index_path.exists():
        print(json.dumps({"error": "Kein Wiki vorhanden"}))
        sys.exit(1)

    from pawlia.dream_wiki import DreamWikiBackend
    wiki_dir = str(_SESSION_DIR / user_id / "workspace" / "wiki")
    backend = DreamWikiBackend(str(index_path), CFG, wiki_dir=wiki_dir)
    await backend.consolidate()
    print(json.dumps({"status": "ok", "message": "Wiki Lint abgeschlossen"}))


async def cmd_status(user_id: str):
    backend = CFG.get("rag_backend", "markdown")
    tracker_path = _SESSION_DIR / user_id / "memory_index" / f".indexed_files_{backend}.json"
    indexed = {}
    if tracker_path.exists():
        indexed = json.loads(tracker_path.read_text(encoding="utf-8"))

    memory_dir = _SESSION_DIR / user_id / "workspace" / "memory"
    total_logs = 0
    pending = []
    if memory_dir.exists():
        for f in sorted(memory_dir.iterdir()):
            if _DATE_RE.match(f.name):
                total_logs += 1
                if f.name not in indexed:
                    pending.append(f.name)

    print(json.dumps({
        "indexed_days": len(indexed),
        "total_logs": total_logs,
        "pending": len(pending),
        "pending_files": pending[:10],
    }, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    env_user_id = os.environ.get("PAWLIA_USER_ID")

    if env_user_id and len(sys.argv) >= 2 and sys.argv[1] in (
        "search", "index", "status", "consolidate", "dream", "lint"
    ):
        user_id = env_user_id
        command = sys.argv[1]
        args = sys.argv[2:]
    elif len(sys.argv) >= 3:
        user_id = sys.argv[1]
        command = sys.argv[2]
        args = sys.argv[3:]
    else:
        print("Usage: memory.py [<user_id>] <command> [args...]", file=sys.stderr)
        print("       (user_id can be set via PAWLIA_USER_ID env var)", file=sys.stderr)
        sys.exit(1)

    if command == "search":
        if not args:
            print("Usage: memory.py <user_id> search <question>", file=sys.stderr)
            sys.exit(1)
        await cmd_search(user_id, " ".join(args))
    elif command == "index":
        await cmd_index(user_id)
    elif command == "consolidate":
        await cmd_consolidate(user_id)
    elif command == "dream":
        await cmd_dream(user_id)
    elif command == "lint":
        await cmd_lint(user_id)
    elif command == "status":
        await cmd_status(user_id)
    else:
        print(f"Unknown command: {command}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
