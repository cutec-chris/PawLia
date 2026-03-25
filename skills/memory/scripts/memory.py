#!/usr/bin/env python3
"""Memory skill — query long-term conversation memory via the configured RAG backend.

Indexing is handled automatically by the scheduler. This script is the
query interface and provides manual index/status commands for debugging.

Usage:
    memory.py <user_id> search <question>
    memory.py <user_id> index          # manual trigger (debug)
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
    for candidate in (_PROJECT_ROOT / "config.yaml", _PROJECT_ROOT / "config.yml"):
        if candidate.is_file():
            with open(candidate, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            return cfg.get("skill-config", {}).get("memory", {})
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

async def cmd_search(user_id: str, question: str):
    backend = await _get_backend(user_id)
    if backend is None:
        print(json.dumps({
            "result": "Noch kein Langzeitgedächtnis vorhanden. Chatlogs werden automatisch im Hintergrund indiziert.",
        }, ensure_ascii=False))
        return

    result = await backend.query(question)
    print(json.dumps({"result": result}, ensure_ascii=False))


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


async def cmd_status(user_id: str):
    backend = CFG.get("rag_backend", "lightrag")
    tracker_path = _SESSION_DIR / user_id / "memory_index" / f"indexed_files_{backend}.json"
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
        "search", "index", "status"
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
    elif command == "status":
        await cmd_status(user_id)
    else:
        print(f"Unknown command: {command}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
