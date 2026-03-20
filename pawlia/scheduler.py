"""Scheduler - periodic background task that fires due reminders, events,
checklist items, task reminders, and scheduled automation jobs.

Runs as an asyncio task alongside the interfaces. Every CHECK_INTERVAL seconds
it scans all user sessions for due items, then calls registered notification
callbacks to deliver messages proactively.
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Callable, Coroutine, Dict, List, Optional

from pawlia.automation import ChecklistProcessor, JobRunner, TaskReminderProcessor

CHECK_INTERVAL = 60  # seconds between checks
EVENT_REMINDER_MINUTES = 15  # notify this many minutes before an event

# Type for notification callbacks: async def send(user_id, message) -> None
NotifyCallback = Callable[[str, str], Coroutine[Any, Any, None]]

# Type for LLM formatter: async def format(user_id, raw_message) -> str
LLMFormatter = Callable[[str, str], Coroutine[Any, Any, str]]

logger = logging.getLogger("pawlia.scheduler")



class Scheduler:
    """Periodically checks for due reminders, events, checklists, and jobs."""

    def __init__(self, session_dir: str):
        self.session_dir = session_dir
        self._callbacks: List[NotifyCallback] = []
        self._task: Optional[asyncio.Task] = None
        self._llm_formatter: Optional[LLMFormatter] = None

        # Automation processors (initialized lazily after callbacks are registered)
        self._checklist: Optional[ChecklistProcessor] = None
        self._jobs: Optional[JobRunner] = None
        self._task_reminders: Optional[TaskReminderProcessor] = None

    def register(self, callback: NotifyCallback) -> None:
        """Register a notification callback (one per interface)."""
        self._callbacks.append(callback)

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

        # Initialize automation processors — they route through self._notify
        # which handles LLM formatting + fallback before delivering to interfaces
        self._checklist = ChecklistProcessor(self.session_dir, self._notify)
        self._jobs = JobRunner(self.session_dir, self._notify)
        self._task_reminders = TaskReminderProcessor(self.session_dir, self._notify)

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

    async def _check_all(self) -> None:
        """Scan all user sessions for due items."""
        if not os.path.isdir(self.session_dir):
            return

        for user_id in os.listdir(self.session_dir):
            user_dir = os.path.join(self.session_dir, user_id)
            if not os.path.isdir(user_dir):
                continue

            # 1. Legacy reminders
            reminders_path = os.path.join(user_dir, "reminders.json")
            await self._check_reminders(user_id, reminders_path)

            # 2. Calendar events (basic 15-min notification)
            events_path = os.path.join(user_dir, "calendar", "events.json")
            await self._check_events(user_id, events_path)

            # 3. Event checklists (script-based automation)
            if self._checklist:
                try:
                    await self._checklist.process_user(user_id)
                except Exception as e:
                    logger.error("Checklist processing failed for %s: %s", user_id, e)

            # 4. Task reminders
            if self._task_reminders:
                try:
                    await self._task_reminders.process_user(user_id)
                except Exception as e:
                    logger.error("Task reminder processing failed for %s: %s", user_id, e)

            # 5. Scheduled jobs
            if self._jobs:
                try:
                    await self._jobs.process_user(user_id)
                except Exception as e:
                    logger.error("Job processing failed for %s: %s", user_id, e)

    async def _check_reminders(self, user_id: str, path: str) -> None:
        """Fire due reminders and handle recurrence."""
        reminders = _load_json(path)
        if not reminders:
            return

        now = datetime.now()
        changed = False

        for reminder in reminders:
            if reminder.get("fired"):
                continue

            try:
                fire_at = datetime.fromisoformat(reminder["fire_at"])
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
            _save_json(path, reminders)

    async def _check_events(self, user_id: str, path: str) -> None:
        """Notify about upcoming events within the reminder window."""
        events = _load_json(path)
        if not events:
            return

        now = datetime.now()
        window = now + timedelta(minutes=EVENT_REMINDER_MINUTES)

        for event in events:
            if event.get("_notified"):
                continue

            try:
                start = datetime.fromisoformat(event["start"])
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
            _save_json(path, events)

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


def _load_json(path: str) -> list:
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error("Failed to load %s: %s", path, e)
        return []


def _save_json(path: str, data: list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
