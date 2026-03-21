"""Simple background task queue — defers agent.run() calls to idle time.

A user can send ``/background <message>`` to queue a task that runs through
the normal agent/skill pipeline during the next idle window.  Results are
posted back via notification callbacks.
"""

import json
import logging
import os
import time
import uuid
from typing import Any, Callable, Coroutine, List, Optional

logger = logging.getLogger("pawlia.background_tasks")


class BackgroundTaskQueue:
    """Persist and process background agent tasks."""

    def __init__(self, session_dir: str):
        self._session_dir = session_dir

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _queue_dir(self, user_id: str) -> str:
        d = os.path.join(self._session_dir, user_id, "background_tasks")
        os.makedirs(d, exist_ok=True)
        return d

    def enqueue(self, user_id: str, message: str) -> dict:
        """Add a task.  Returns the task dict."""
        task_id = uuid.uuid4().hex[:10]
        thread_id = f"bg_{task_id}"
        task = {
            "id": task_id,
            "message": message,
            "thread_id": thread_id,
            "status": "pending",
            "created": time.time(),
        }
        path = os.path.join(self._queue_dir(user_id), f"{task_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(task, f, indent=2, ensure_ascii=False)
        logger.info("Background task queued: %s/%s — %s", user_id, task_id, message[:80])
        return task

    def pending(self) -> List[tuple]:
        """Return [(user_id, task), ...] for all pending tasks, oldest first."""
        result = []
        if not os.path.isdir(self._session_dir):
            return result
        for user_id in os.listdir(self._session_dir):
            d = os.path.join(self._session_dir, user_id, "background_tasks")
            if not os.path.isdir(d):
                continue
            for fname in sorted(os.listdir(d)):
                if not fname.endswith(".json"):
                    continue
                try:
                    with open(os.path.join(d, fname), encoding="utf-8") as f:
                        task = json.load(f)
                    if task.get("status") == "pending":
                        result.append((user_id, task))
                except Exception:
                    continue
        return result

    def list_tasks(self, user_id: str) -> List[dict]:
        """List all tasks for a user."""
        d = os.path.join(self._session_dir, user_id, "background_tasks")
        if not os.path.isdir(d):
            return []
        tasks = []
        for fname in sorted(os.listdir(d)):
            if not fname.endswith(".json"):
                continue
            try:
                with open(os.path.join(d, fname), encoding="utf-8") as f:
                    tasks.append(json.load(f))
            except Exception:
                continue
        return tasks

    def _update(self, user_id: str, task_id: str, **fields) -> None:
        path = os.path.join(self._queue_dir(user_id), f"{task_id}.json")
        if not os.path.isfile(path):
            return
        with open(path, encoding="utf-8") as f:
            task = json.load(f)
        task.update(fields)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(task, f, indent=2, ensure_ascii=False)

    def mark_running(self, user_id: str, task_id: str) -> None:
        self._update(user_id, task_id, status="running")

    def mark_done(self, user_id: str, task_id: str) -> None:
        self._update(user_id, task_id, status="done", finished=time.time())

    def mark_error(self, user_id: str, task_id: str, error: str) -> None:
        self._update(user_id, task_id, status="error", error=error, finished=time.time())
