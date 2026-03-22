"""Background memory indexer — indexes daily chat logs into LightRAG.

Called by the Scheduler once per tick. Tracks which files have been indexed
(by mtime) so only new or updated logs are processed.

Requires ``skill-config.memory`` to be configured with embedding settings.
"""

import asyncio
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger("pawlia.memory_indexer")

# Monkey-patch tiktoken to allow special tokens (LLMs like qwen emit
# <|endoftext|> etc. which LightRAG then tries to tokenize internally).
def _patch_tiktoken():
    try:
        import tiktoken.core
        _orig_encode = tiktoken.core.Encoding.encode
        def _permissive_encode(self, text, **kw):
            kw.setdefault("disallowed_special", ())
            return _orig_encode(self, text, **kw)
        tiktoken.core.Encoding.encode = _permissive_encode
    except Exception:
        pass
_patch_tiktoken()

# Pattern for daily log files: YYYY-MM-DD.md
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}\.md$")


RETRY_COOLDOWN = 3600  # seconds before retrying a failed file (1 hour)


class MemoryIndexer:
    """Indexes daily chat logs per user into a LightRAG instance."""

    def __init__(self, session_dir: str, config: dict, llm_busy_check=None):
        self._session_dir = session_dir
        self._llm_busy = llm_busy_check  # callable: () -> bool
        self._cfg = config.get("skill-config", {}).get("memory", {})
        self._enabled = bool(
            self._cfg.get("embedding_provider")
            and self._cfg.get("embedding_model")
            and self._cfg.get("embedding_dim")
            and self._cfg.get("embedding_host")
        )
        # Per-user RAG instances (lazy)
        self._rags: Dict[str, object] = {}
        # Track failed indexing attempts: {user_id: {fname: timestamp}}
        self._failures: Dict[str, Dict[str, float]] = {}

        if not self._enabled:
            logger.debug("Memory indexer disabled (skill-config.memory not configured)")

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ------------------------------------------------------------------
    # LightRAG setup
    # ------------------------------------------------------------------

    def _build_embedding_func(self):
        import lightrag.utils
        import lightrag.llm.ollama
        import lightrag.llm.openai

        cfg = self._cfg
        provider = cfg.get("embedding_provider", "ollama")
        model = cfg["embedding_model"]
        dim = int(cfg["embedding_dim"])
        host = cfg.get("embedding_host", "http://localhost:11434")

        if provider == "ollama":
            async def _ollama_embed(texts):
                import numpy as np
                try:
                    result = await lightrag.llm.ollama.ollama_embed(
                        texts, host=host, embed_model=model, max_token_size=8192,
                    )
                except Exception as e:
                    err_str = str(e)
                    if "NaN" in err_str:
                        # Ollama produced NaN embeddings server-side and failed
                        # to serialize them.  Return zero vectors so the caller
                        # can continue instead of crashing the whole pipeline.
                        logger.warning(
                            "Ollama NaN embedding error for %d texts, returning zero vectors: %s",
                            len(texts), err_str,
                        )
                        return np.zeros((len(texts), dim))
                    raise
                # Replace NaN values from buggy embedding responses
                arr = np.array(result)
                if np.isnan(arr).any():
                    logger.warning("NaN in embeddings detected, replacing with 0.0")
                    arr = np.nan_to_num(arr, nan=0.0)
                return arr
            return lightrag.utils.EmbeddingFunc(
                embedding_dim=dim,
                func=_ollama_embed,
            )
        else:
            api_key = cfg.get("embedding_api_key")
            base_url = cfg.get("embedding_base_url")

            async def _embed(texts):
                return await lightrag.llm.openai.openai_embed(
                    texts, embed_model=model, api_key=api_key, base_url=base_url,
                )

            return lightrag.utils.EmbeddingFunc(embedding_dim=dim, func=_embed)

    def _build_llm_func(self):
        import lightrag.llm.ollama
        import lightrag.llm.openai

        cfg = self._cfg
        provider = cfg.get("rag_provider", cfg.get("embedding_provider", "ollama"))
        model = cfg.get("rag_model", "qwen3.5:latest")
        host = cfg.get("embedding_host", "http://localhost:11434")

        if provider == "ollama":
            async def _complete(prompt: str, system_prompt: str = "", **kw):
                return await lightrag.llm.ollama.ollama_model_complete(
                    prompt, system_prompt=system_prompt,
                    host=host,
                    options={"num_ctx": int(cfg.get("rag_numctx", 4096)), "think": False},
                    **kw,
                )
            return _complete
        else:
            api_key = cfg.get("rag_api_key", cfg.get("embedding_api_key"))
            base_url = cfg.get("rag_base_url", cfg.get("embedding_base_url"))

            async def _complete(prompt, system_prompt=None, history_messages=[], **kw):
                return await lightrag.llm.openai.openai_complete_if_cache(
                    model, prompt, system_prompt=system_prompt,
                    history_messages=history_messages,
                    api_key=api_key, base_url=base_url, **kw,
                )
            return _complete

    async def _get_rag(self, user_id: str):
        import lightrag
        import lightrag.kg.shared_storage

        if user_id in self._rags:
            return self._rags[user_id]

        index_path = os.path.join(self._session_dir, user_id, "memory_index")
        os.makedirs(index_path, exist_ok=True)

        rag = lightrag.LightRAG(
            index_path,
            llm_model_func=self._build_llm_func(),
            llm_model_name=self._cfg.get("rag_model", "qwen3.5:latest"),
            summary_max_tokens=8192,
            enable_llm_cache=False,
            llm_model_kwargs={},
            embedding_func=self._build_embedding_func(),
            default_llm_timeout=int(self._cfg.get("rag_timeout", 600)),
            default_embedding_timeout=int(self._cfg.get("rag_embedding_timeout", 120)),
            llm_model_max_async=int(self._cfg.get("rag_max_async_llm", 2)),
            embedding_func_max_async=int(self._cfg.get("rag_max_async_embedding", 4)),
        )
        await rag.initialize_storages()
        await lightrag.kg.shared_storage.initialize_pipeline_status()
        self._rags[user_id] = rag
        return rag

    # ------------------------------------------------------------------
    # Tracking
    # ------------------------------------------------------------------

    def _tracker_path(self, user_id: str) -> str:
        d = os.path.join(self._session_dir, user_id, "memory_index")
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, "indexed_files.json")

    def _load_tracked(self, user_id: str) -> dict:
        p = self._tracker_path(user_id)
        if os.path.exists(p):
            try:
                with open(p, encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_tracked(self, user_id: str, data: dict):
        with open(self._tracker_path(user_id), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    # ------------------------------------------------------------------
    # Failure cooldown
    # ------------------------------------------------------------------

    def _is_on_cooldown(self, user_id: str, fname: str) -> bool:
        """Return True if a previous failure is still within the cooldown window."""
        user_fails = self._failures.get(user_id, {})
        fail_time = user_fails.get(fname)
        if fail_time is None:
            return False
        import time
        return (time.time() - fail_time) < RETRY_COOLDOWN

    def _mark_failed(self, user_id: str, fname: str):
        import time
        self._failures.setdefault(user_id, {})[fname] = time.time()

    def _clear_failure(self, user_id: str, fname: str):
        user_fails = self._failures.get(user_id, {})
        user_fails.pop(fname, None)

    # ------------------------------------------------------------------
    # Index
    # ------------------------------------------------------------------

    def _find_daily_logs(self, user_id: str) -> list:
        memory_dir = os.path.join(
            self._session_dir, user_id, "workspace", "memory",
        )
        if not os.path.isdir(memory_dir):
            return []
        return sorted(
            os.path.join(memory_dir, f)
            for f in os.listdir(memory_dir)
            if _DATE_RE.match(f)
        )

    async def process_user(self, user_id: str) -> None:
        """Index new/updated daily logs for a single user."""
        if not self._enabled:
            return

        logs = self._find_daily_logs(user_id)
        if not logs:
            return

        tracked = self._load_tracked(user_id)
        changed = False

        for log_path in logs:
            # Yield to high-priority LLM requests (chat)
            if self._llm_busy and self._llm_busy():
                logger.debug("LLM busy with chat, deferring memory indexing")
                break

            fname = os.path.basename(log_path)
            mtime = str(os.path.getmtime(log_path))

            if fname in tracked and tracked[fname] == mtime:
                continue

            if self._is_on_cooldown(user_id, fname):
                logger.debug("Skipping %s/%s (on cooldown after previous failure)", user_id, fname)
                continue

            try:
                with open(log_path, encoding="utf-8") as f:
                    content = f.read().strip()
            except Exception:
                continue

            if not content:
                continue

            date_str = fname.replace(".md", "")
            doc_id = f"chat_{user_id}_{date_str}"
            markdown = f"# Conversation log from {date_str}\n\n{content}"

            try:
                rag = await self._get_rag(user_id)
                await rag.ainsert(markdown, ids=doc_id)

                # Wait for processing (max 120s)
                for _ in range(120):
                    status = await rag.doc_status.get_by_id(doc_id)
                    if status and status.get("status") == "processed":
                        break
                    await asyncio.sleep(1)

                tracked[fname] = mtime
                changed = True
                self._clear_failure(user_id, fname)
                logger.info("Memory indexed: %s/%s", user_id, fname)
            except Exception as e:
                self._mark_failed(user_id, fname)
                logger.error("Memory indexing failed for %s/%s (retry in %dh): %s",
                             user_id, fname, RETRY_COOLDOWN // 3600, e)

        if changed:
            self._save_tracked(user_id, tracked)
