#!/usr/bin/env python3
"""Memory skill — query long-term conversation memory via LightRAG.

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
_SESSION_DIR = _PROJECT_ROOT / "session"

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}\.md$")


def _load_skill_config() -> dict:
    for candidate in (_PROJECT_ROOT / "config.yaml", _PROJECT_ROOT / "config.yml"):
        if candidate.is_file():
            with open(candidate, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            return cfg.get("skill-config", {}).get("memory", {})
    return {}


CFG = _load_skill_config()

# ---------------------------------------------------------------------------
# LightRAG (read-only — same index the scheduler writes to)
# ---------------------------------------------------------------------------

_rag_instance = None


def _build_embedding_func(cfg: dict):
    import lightrag.utils
    import lightrag.llm.ollama
    import lightrag.llm.openai

    provider = cfg.get("embedding_provider", "ollama")
    model = cfg.get("embedding_model", "bge-m3:latest")
    dim = int(cfg.get("embedding_dim", 1024))
    host = cfg.get("embedding_host", "http://localhost:11434")

    if provider == "ollama":
        return lightrag.utils.EmbeddingFunc(
            embedding_dim=dim,
            func=lambda texts: lightrag.llm.ollama.ollama_embed(
                texts, host=host, embed_model=model, max_token_size=8192,
            ),
        )
    api_key = cfg.get("embedding_api_key")
    base_url = cfg.get("embedding_base_url")

    async def _embed(texts):
        return await lightrag.llm.openai.openai_embed(
            texts, embed_model=model, api_key=api_key, base_url=base_url,
        )

    return lightrag.utils.EmbeddingFunc(embedding_dim=dim, func=_embed)


def _build_llm_func(cfg: dict):
    import lightrag.llm.ollama
    import lightrag.llm.openai

    provider = cfg.get("rag_provider", cfg.get("embedding_provider", "ollama"))
    model = cfg.get("rag_model", "qwen3.5:latest")
    host = cfg.get("embedding_host", "http://localhost:11434")

    if provider == "ollama":
        async def _complete(prompt: str, system_prompt: str = "", **kw):
            return await lightrag.llm.ollama.ollama_model_complete(
                prompt, system_prompt=system_prompt,
                host=host,
                options={"num_ctx": int(cfg.get("rag_numctx", 4096))},
                **kw,
            )
        return _complete

    api_key = cfg.get("rag_api_key", cfg.get("embedding_api_key"))
    base_url = cfg.get("rag_base_url", cfg.get("embedding_base_url"))

    async def _complete(prompt, system_prompt=None, history_messages=[], **kw):
        return await lightrag.llm.openai.openai_complete_if_cache(
            model, prompt, system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=api_key, base_url=base_url, **kw,
        )
    return _complete


async def _get_rag(user_id: str):
    import lightrag
    import lightrag.kg.shared_storage

    global _rag_instance
    if _rag_instance is not None:
        return _rag_instance

    index_path = _SESSION_DIR / user_id / "memory_index"
    if not index_path.exists():
        return None

    _rag_instance = lightrag.LightRAG(
        str(index_path),
        llm_model_func=_build_llm_func(CFG),
        llm_model_name=CFG.get("rag_model", "qwen3.5:latest"),
        summary_max_tokens=8192,
        enable_llm_cache=False,
        llm_model_kwargs={},
        embedding_func=_build_embedding_func(CFG),
    )
    await _rag_instance.initialize_storages()
    await lightrag.kg.shared_storage.initialize_pipeline_status()
    return _rag_instance


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

async def cmd_search(user_id: str, question: str):
    import lightrag

    rag = await _get_rag(user_id)
    if rag is None:
        print(json.dumps({
            "result": "Noch kein Langzeitgedächtnis vorhanden. Chatlogs werden automatisch im Hintergrund indiziert.",
        }, ensure_ascii=False))
        return

    result = await rag.aquery(
        question,
        param=lightrag.QueryParam(mode="global", enable_rerank=False),
    )
    print(json.dumps({"result": result}, ensure_ascii=False))


async def cmd_index(user_id: str):
    """Manual index trigger — imports and runs the scheduler's indexer."""
    sys.path.insert(0, str(_PROJECT_ROOT))
    from pawlia.memory_indexer import MemoryIndexer

    full_cfg = {}
    for candidate in (_PROJECT_ROOT / "config.yaml", _PROJECT_ROOT / "config.yml"):
        if candidate.is_file():
            with open(candidate, encoding="utf-8") as f:
                full_cfg = yaml.safe_load(f) or {}
            break

    indexer = MemoryIndexer(str(_SESSION_DIR), full_cfg)
    if not indexer.enabled:
        print(json.dumps({"error": "Memory indexer not configured"}))
        sys.exit(1)

    await indexer.process_user(user_id)
    print(json.dumps({"status": "ok", "message": "Indexing complete"}))


async def cmd_status(user_id: str):
    tracker_path = _SESSION_DIR / user_id / "memory_index" / "indexed_files.json"
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
    if len(sys.argv) < 3:
        print("Usage: memory.py <user_id> <command> [args...]", file=sys.stderr)
        sys.exit(1)

    user_id = sys.argv[1]
    command = sys.argv[2]
    args = sys.argv[3:]

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
