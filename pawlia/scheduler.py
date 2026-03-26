"""Scheduler - periodic background task that fires due reminders, events,
checklist items, task reminders, scheduled automation jobs, and memory indexing.

Runs as an asyncio task alongside the interfaces. Every CHECK_INTERVAL seconds
it scans all user sessions for due items, then calls registered notification
callbacks to deliver messages proactively.
"""

import asyncio
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Coroutine, Dict, List, Optional

from pawlia.automation import ChecklistProcessor, JobRunner, TaskReminderProcessor
from pawlia.utils import load_json, save_json

CHECK_INTERVAL = 60  # seconds between checks
EVENT_REMINDER_MINUTES = 15  # notify this many minutes before an event

# ── Idle-based priority tiers (minutes) ──
# Lower = higher priority.  Each task only runs when the user has been idle
# for at least this many minutes AND no high-priority (chat) LLM request is
# in progress.
IDLE_SUMMARIZE_MIN = 5    # conversation summarization
IDLE_BACKGROUND_MIN = 10  # deferred /background tasks
IDLE_MEMORY_MIN = 20      # LightRAG memory indexing

# Type for notification callbacks: async def send(user_id, message) -> None
NotifyCallback = Callable[[str, str], Coroutine[Any, Any, None]]

# Type for LLM formatter: async def format(user_id, raw_message) -> str
LLMFormatter = Callable[[str, str], Coroutine[Any, Any, str]]

logger = logging.getLogger("pawlia.scheduler")



class Scheduler:
    """Periodically checks for due reminders, events, checklists, jobs, and memory indexing."""

    def __init__(self, session_dir: str, config: Optional[Dict] = None):
        self.session_dir = session_dir
        self._config = config or {}
        self._app: Optional[Any] = None  # set via set_app()
        self._callbacks: List[NotifyCallback] = []
        self._task: Optional[asyncio.Task] = None
        self._llm_formatter: Optional[LLMFormatter] = None

        # Automation processors (initialized lazily after callbacks are registered)
        self._checklist: Optional[ChecklistProcessor] = None
        self._jobs: Optional[JobRunner] = None
        self._task_reminders: Optional[TaskReminderProcessor] = None

        # Memory indexer (initialized lazily)
        self._memory_indexer: Optional[Any] = None
        # Background task queue (initialized lazily)
        self._bg_tasks: Optional[Any] = None
        # LLM priority gate: high-prio requests (chat) block low-prio (background)
        self._llm_active = 0
        # Track last user activity for idle-based background work
        # Start with current time so background work doesn't fire immediately on boot
        self._boot_time = time.monotonic()
        self._last_activity: Dict[str, float] = {}

    def set_app(self, app: Any) -> None:
        """Set reference to the App (needed for background task processing)."""
        self._app = app

    def register(self, callback: NotifyCallback) -> None:
        """Register a notification callback (one per interface)."""
        self._callbacks.append(callback)

    @property
    def bg_tasks(self):
        """Lazy-init and return the BackgroundTaskQueue."""
        if self._bg_tasks is None:
            from pawlia.background_tasks import BackgroundTaskQueue
            self._bg_tasks = BackgroundTaskQueue(self.session_dir)
        return self._bg_tasks

    def touch_activity(self, user_id: str) -> None:
        """Mark a user as active now (resets the idle timer for memory indexing)."""
        self._last_activity[user_id] = time.monotonic()

    def acquire_llm(self) -> None:
        """Signal that a high-priority LLM request (chat) is starting."""
        self._llm_active += 1

    def release_llm(self) -> None:
        """Signal that a high-priority LLM request has finished."""
        self._llm_active = max(0, self._llm_active - 1)

    @property
    def llm_busy(self) -> bool:
        """True if any high-priority LLM request is in progress."""
        return self._llm_active > 0

    def set_llm_formatter(self, formatter: LLMFormatter) -> None:
        """Set the LLM formatter for personalizing notifications.

        The formatter receives (user_id, raw_message) and returns a
        personalized message.  If it fails or times out, the raw message
        is used as fallback.
        """
        self._llm_formatter = formatter

    def start(self) -> None:
        """Start the scheduler as a background asyncio task."""
        if self._task and not self._task.done():
            return
        self._task = asyncio.create_task(self._loop())
        logger.info("Scheduler started (interval=%ds)", CHECK_INTERVAL)

    def stop(self) -> None:
        """Cancel the scheduler task."""
        if self._task and not self._task.done():
            self._task.cancel()
            logger.info("Scheduler stopped")

    async def _loop(self) -> None:
        """Main scheduler loop."""
        try:
            while True:
                await asyncio.sleep(CHECK_INTERVAL)
                try:
                    await self._check_all()
                except Exception as e:
                    logger.error("Scheduler check failed: %s", e)
        except asyncio.CancelledError:
            pass

    def _ensure_processors(self) -> None:
        """Lazily initialize automation processors on first use."""
        if self._checklist is None:
            self._checklist = ChecklistProcessor(self.session_dir, self._notify)
            self._jobs = JobRunner(self.session_dir, self._notify)
            self._task_reminders = TaskReminderProcessor(self.session_dir, self._notify)
        if self._memory_indexer is None:
            from pawlia.memory_indexer import MemoryIndexer
            self._memory_indexer = MemoryIndexer(
                self.session_dir, self._config,
                llm_busy_check=lambda: self.llm_busy,
            )

    def _user_idle_minutes(self, user_id: str) -> float:
        """Return how many minutes a user has been idle."""
        now = time.monotonic()
        last = self._last_activity.get(user_id, self._boot_time)
        return (now - last) / 60.0

    async def _check_all(self) -> None:
        """Scan all user sessions for due items.

        High-priority tasks (reminders, events, automation) run every tick.
        Low-priority tasks are gated by per-user idle time (in minutes):
          5 min  → conversation summarization
          10 min → background tasks (/background)
          20 min → memory indexing (LightRAG)
        All low-priority tasks require the LLM to be free (no active chat).
        """
        if not os.path.isdir(self.session_dir):
            return

        self._ensure_processors()

        user_ids = [
            uid for uid in os.listdir(self.session_dir)
            if os.path.isdir(os.path.join(self.session_dir, uid))
        ]

        # ── High priority (every tick, no idle requirement) ──
        for user_id in user_ids:
            user_dir = os.path.join(self.session_dir, user_id)

            reminders_path = os.path.join(user_dir, "reminders.json")
            await self._check_reminders(user_id, reminders_path)

            events_path = os.path.join(user_dir, "calendar", "events.json")
            await self._check_events(user_id, events_path)

            if self._checklist:
                try:
                    await self._checklist.process_user(user_id)
                except Exception as e:
                    logger.error("Checklist processing failed for %s: %s", user_id, e)

            if self._task_reminders:
                try:
                    await self._task_reminders.process_user(user_id)
                except Exception as e:
                    logger.error("Task reminder processing failed for %s: %s", user_id, e)

            if self._jobs:
                try:
                    await self._jobs.process_user(user_id)
                except Exception as e:
                    logger.error("Job processing failed for %s: %s", user_id, e)

        # ── Force-summarize when exchange count exceeds hard limit ──
        # This runs even when the user is active to prevent unbounded growth.
        if not self.llm_busy and self._app:
            from pawlia.memory import FORCE_SUMMARY_EXCHANGES
            for user_id in user_ids:
                session = self._app.memory.load_session(user_id)
                if session.exchange_count >= FORCE_SUMMARY_EXCHANGES:
                    try:
                        await self._summarize_user(user_id)
                    except Exception as e:
                        logger.error("Forced summarization failed for %s: %s", user_id, e)
                    if self.llm_busy:
                        break

        # ── Low priority (idle-based, ordered by priority) ──
        if self.llm_busy:
            return

        for user_id in user_ids:
            idle = self._user_idle_minutes(user_id)

            # Prio 1: Summarization (5 min idle)
            if idle >= IDLE_SUMMARIZE_MIN and self._app:
                try:
                    await self._summarize_user(user_id)
                except Exception as e:
                    logger.error("Summarization failed for %s: %s", user_id, e)
                if self.llm_busy:
                    return

            # Prio 2: Background tasks (10 min idle)
            if idle >= IDLE_BACKGROUND_MIN and self._app:
                try:
                    await self._process_background_tasks(user_id)
                except Exception as e:
                    logger.error("Background task failed for %s: %s", user_id, e)
                if self.llm_busy:
                    return

            # Prio 3: Memory indexing (configurable idle, default 20 min)
            idle_memory_min = int(
                self._config.get("skill-config", {}).get("memory", {}).get("idle_minutes", IDLE_MEMORY_MIN)
            )
            if idle >= idle_memory_min:
                if self._memory_indexer and self._memory_indexer.enabled:
                    try:
                        await self._memory_indexer.process_user(user_id)
                    except Exception as e:
                        logger.error("Memory indexing failed for %s: %s", user_id, e)
                    if self.llm_busy:
                        return

    async def _summarize_user(self, user_id: str) -> None:
        """Summarize a user's conversation if needed.

        Uses the MemoryManager to check whether summarization is due
        (exchange limit, repetition, or idle) and runs it through the
        chat LLM.
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        memory = self._app.memory
        session = memory.load_session(user_id)

        reason = memory.should_summarize(session)
        if not reason:
            return

        history = session.daily_history.strip()
        if not history:
            return

        logger.info("Summarizing conversation for %s (trigger: %s)", user_id, reason)

        prior = session.summary.strip()
        context = f"Previous summary:\n{prior}\n\n" if prior else ""

        messages = [
            SystemMessage(content=(
                "Summarize this conversation in 2-4 short bullet points.\n"
                "Keep ONLY:\n"
                "- User preferences and personal facts\n"
                "- Decisions made or tasks completed\n"
                "- Open/unanswered requests\n"
                "DISCARD:\n"
                "- Specific numbers, routes, or data (the user can ask again)\n"
                "- Failed attempts, errors, or debugging details\n"
                "- Greetings and small talk\n"
                "Write in the user's language. Maximum 4 lines."
            )),
            HumanMessage(content=(
                f"{context}Conversation to summarize:\n{history}"
            )),
        ]

        llm = self._app.llm.get("chat")

        try:
            response = await llm.ainvoke(messages)
        except Exception as e:
            logger.error("Summarization LLM call failed for %s: %s", user_id, e)
            return

        from pawlia.agents.base import BaseAgent
        summary = BaseAgent.strip_thinking(response.content or "").strip()
        if summary:
            memory.summarize(session, summary)
            logger.info("Conversation summarized for %s", user_id)

    async def _process_background_tasks(self, user_id: str) -> None:
        """Run one pending background task for a user."""
        tasks = self.bg_tasks.list_tasks(user_id)
        pending = [t for t in tasks if t.get("status") == "pending"]
        if not pending:
            return

        task = pending[0]  # oldest first
        task_id = task["id"]
        message = task["message"]
        thread_id = task["thread_id"]

        logger.info("Background task starting: %s/%s — %s", user_id, task_id, message[:80])
        self.bg_tasks.mark_running(user_id, task_id)

        try:
            agent = self._app.make_agent(user_id)
            response = await agent.run(message, thread_id=thread_id)
            self.bg_tasks.mark_done(user_id, task_id)
            await self._notify(user_id, f"**[Hintergrund {task_id[:8]}]**\n{response}")
            logger.info("Background task done: %s/%s", user_id, task_id)
        except Exception as e:
            self.bg_tasks.mark_error(user_id, task_id, str(e))
            await self._notify(user_id, f"**[Hintergrund {task_id[:8]}]** ❌ Fehler: {e}")
            logger.error("Background task failed: %s/%s: %s", user_id, task_id, e)

    async def _check_reminders(self, user_id: str, path: str) -> None:
        """Fire due reminders and handle recurrence."""
        reminders = load_json(path)
        if not reminders:
            return

        now = datetime.now()
        changed = False

        for reminder in reminders:
            if reminder.get("fired"):
                continue

            try:
                fire_at = datetime.fromisoformat(reminder["fire_at"])
                # Ensure naive comparison — strip timezone if present
                if fire_at.tzinfo is not None:
                    fire_at = fire_at.replace(tzinfo=None)
            except (ValueError, KeyError):
                continue

            if fire_at <= now:
                label = reminder.get("label", "Reminder")
                message = reminder.get("message", "")
                text = f"🔔 {label}: {message}"

                await self._notify(user_id, text)

                recurrence = reminder.get("recurrence", "none")
                if recurrence == "none":
                    reminder["fired"] = True
                else:
                    reminder["fire_at"] = _next_occurrence(fire_at, recurrence).isoformat()

                changed = True

        if changed:
            save_json(path, reminders)

    async def _check_events(self, user_id: str, path: str) -> None:
        """Notify about upcoming events within the reminder window."""
        events = load_json(path)
        if not events:
            return

        now = datetime.now()
        window = now + timedelta(minutes=EVENT_REMINDER_MINUTES)

        for event in events:
            if event.get("_notified"):
                continue

            try:
                start = datetime.fromisoformat(event["start"])
                if start.tzinfo is not None:
                    start = start.replace(tzinfo=None)
            except (ValueError, KeyError):
                continue

            if now <= start <= window:
                title = event.get("title", "Event")
                location = event.get("location", "")
                minutes_left = int((start - now).total_seconds() / 60)

                text = f"📅 In {minutes_left} Min: {title}"
                if location:
                    text += f" ({location})"

                await self._notify(user_id, text)
                event["_notified"] = True

        # Persist the _notified flags
        if any(e.get("_notified") for e in events):
            save_json(path, events)

    async def _notify(self, user_id: str, message: str) -> None:
        """Send a notification to all registered interfaces.

        If an LLM formatter is set, the raw message is first passed through
        the LLM for a personalized response.  On failure or timeout, the raw
        message is delivered as-is.
        """
        formatted = message
        if self._llm_formatter:
            try:
                formatted = await self._llm_formatter(user_id, message)
                if not formatted or not formatted.strip():
                    logger.warning("LLM returned empty response, using raw message")
                    formatted = message
            except Exception as e:
                logger.warning("LLM formatting failed for %s: %s, using raw message", user_id, e)
                formatted = message

        for callback in self._callbacks:
            try:
                await callback(user_id, formatted)
            except Exception as e:
                logger.error("Notify callback failed for %s: %s", user_id, e)


def _next_occurrence(fire_at: datetime, recurrence: str) -> datetime:
    """Calculate the next occurrence for a recurring reminder."""
    if recurrence == "daily":
        return fire_at + timedelta(days=1)
    elif recurrence == "weekly":
        return fire_at + timedelta(weeks=1)
    elif recurrence == "monthly":
        month = fire_at.month % 12 + 1
        year = fire_at.year + (1 if month == 1 else 0)
        try:
            return fire_at.replace(year=year, month=month)
        except ValueError:
            return fire_at.replace(year=year, month=month, day=28)
    return fire_at + timedelta(days=1)


