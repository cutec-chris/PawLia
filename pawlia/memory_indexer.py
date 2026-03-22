"""Background memory indexer — indexes daily chat logs into a RAG backend.

Called by the Scheduler once per tick. Tracks which files have been indexed
(by mtime) so only new or updated logs are processed.

Requires ``skill-config.memory`` to be configured with embedding settings.
The active RAG backend is selected via ``skill-config.memory.rag_backend``
(default: ``lightrag``; alternative: ``mem0``).
"""

import json
import logging
import os
import re
import time
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger("pawlia.memory_indexer")

# Pattern for daily log files: YYYY-MM-DD.md
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}\.md$")

RETRY_COOLDOWN = 3600  # seconds before retrying a failed file (1 hour)


class MemoryIndexer:
    """Indexes daily chat logs per user into a RAG backend instance."""

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
        # Per-user backend instances (lazy)
        self._backends: Dict[str, object] = {}
        # Track failed indexing attempts: {user_id: {fname: timestamp}}
        self._failures: Dict[str, Dict[str, float]] = {}

        if not self._enabled:
            logger.debug("Memory indexer disabled (skill-config.memory not configured)")

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ------------------------------------------------------------------
    # Backend setup
    # ------------------------------------------------------------------

    async def _get_backend(self, user_id: str):
        from pawlia.rag_backend import create_backend

        if user_id in self._backends:
            return self._backends[user_id]

        index_path = os.path.join(self._session_dir, user_id, "memory_index")
        os.makedirs(index_path, exist_ok=True)

        backend = create_backend(
            index_path,
            self._cfg,
            llm_busy_check=self._llm_busy,
            think=False,
        )
        self._backends[user_id] = backend
        return backend

    # ------------------------------------------------------------------
    # Tracking
    # ------------------------------------------------------------------

    def _tracker_path(self, user_id: str) -> str:
        d = os.path.join(self._session_dir, user_id, "memory_index")
        os.makedirs(d, exist_ok=True)
        backend = self._cfg.get("rag_backend", "lightrag")
        return os.path.join(d, f"indexed_files_{backend}.json")

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
        user_fails = self._failures.get(user_id, {})
        fail_time = user_fails.get(fname)
        if fail_time is None:
            return False
        return (time.time() - fail_time) < RETRY_COOLDOWN

    def _mark_failed(self, user_id: str, fname: str):
        self._failures.setdefault(user_id, {})[fname] = time.time()

    def _clear_failure(self, user_id: str, fname: str):
        self._failures.get(user_id, {}).pop(fname, None)

    # ------------------------------------------------------------------
    # Index
    # ------------------------------------------------------------------

    def _find_daily_logs(self, user_id: str) -> list:
        memory_dir = os.path.join(
            self._session_dir, user_id, "workspace", "memory",
        )
        if not os.path.isdir(memory_dir):
            return []
        today = datetime.now().strftime("%Y-%m-%d.md")
        return sorted(
            os.path.join(memory_dir, f)
            for f in os.listdir(memory_dir)
            if _DATE_RE.match(f) and f != today
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

            # Final guard: don't start insert if LLM just became busy
            if self._llm_busy and self._llm_busy():
                logger.debug("LLM busy before insert, deferring %s", fname)
                break

            try:
                backend = await self._get_backend(user_id)
                await backend.insert(markdown, doc_id)
                processed = await backend.wait_for_indexed(doc_id, timeout=120, poll_interval=5.0)

                if processed:
                    tracked[fname] = mtime
                    changed = True
                    self._clear_failure(user_id, fname)
                    logger.info("Memory indexed: %s/%s", user_id, fname)
                else:
                    self._mark_failed(user_id, fname)
                    logger.warning(
                        "Memory indexing incomplete for %s/%s (busy or failed), will retry",
                        user_id, fname,
                    )
            except Exception as e:
                self._mark_failed(user_id, fname)
                logger.error(
                    "Memory indexing failed for %s/%s (retry in %dh): %s",
                    user_id, fname, RETRY_COOLDOWN // 3600, e,
                )

        if changed:
            self._save_tracked(user_id, tracked)
