"""Scheduler - periodic background task that fires due reminders, events,
checklist items, task reminders, scheduled automation jobs, and memory indexing.

Runs as an asyncio task alongside the interfaces. Every CHECK_INTERVAL seconds
it scans all user sessions for due items, then calls registered notification
callbacks to deliver messages proactively.

Data sources:
  - Events:    workspace/calendar/*.md  (Full Calendar frontmatter)
  - Tasks:     workspace/tasks.md       (Obsidian Tasks emoji format)
  - Reminders: workspace/tasks.md       (scheduled tasks with 🔔 prefix)
  - Jobs:      automations/jobs.json    (scheduler-internal)
  - State:     scheduler_state.json     (internal flags: notified, fired, etc.)
"""

import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Coroutine, Dict, List, Optional

import yaml

from pawlia.automation import ChecklistProcessor, JobRunner, TaskReminderProcessor
from pawlia.prompt_utils import load_system_prompt
from pawlia.utils import load_json, save_json

CHECK_INTERVAL = 60  # seconds between checks
EVENT_REMINDER_MINUTES = 15  # notify this many minutes before an event

# ── Idle-based priority tiers (minutes) ──
IDLE_SUMMARIZE_MIN = 5
IDLE_BACKGROUND_MIN = 10
IDLE_MEMORY_MIN = 20

NotifyCallback = Callable[[str, str], Coroutine[Any, Any, None]]
LLMFormatter = Callable[[str, str], Coroutine[Any, Any, str]]

logger = logging.getLogger("pawlia.scheduler")


# ---------------------------------------------------------------------------
# Scheduler state I/O (separate from workspace to keep vault clean)
# ---------------------------------------------------------------------------

def _state_path(session_dir: str, user_id: str) -> str:
    return os.path.join(session_dir, user_id, "scheduler_state.json")


def _load_state(session_dir: str, user_id: str) -> dict:
    path = _state_path(session_dir, user_id)
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_state(session_dir: str, user_id: str, state: dict) -> None:
    path = _state_path(session_dir, user_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Markdown event parsing
# ---------------------------------------------------------------------------

def _read_event_frontmatter(filepath: str) -> dict | None:
    """Parse frontmatter from an event .md file."""
    try:
        with open(filepath, encoding="utf-8") as f:
            text = f.read()
    except Exception:
        return None
    if not text.lstrip().startswith("---"):
        return None
    parts = text.split("---", 2)
    if len(parts) < 3:
        return None
    try:
        fm = yaml.safe_load(parts[1]) or {}
    except Exception:
        return None
    fm["_filename"] = os.path.basename(filepath)
    return fm


def _list_workspace_events(session_dir: str, user_id: str) -> list[dict]:
    """List all events from workspace/calendar/*.md."""
    cal_dir = os.path.join(session_dir, user_id, "workspace", "calendar")
    if not os.path.isdir(cal_dir):
        return []
    events = []
    for f in os.listdir(cal_dir):
        if f.endswith(".md"):
            fm = _read_event_frontmatter(os.path.join(cal_dir, f))
            if fm:
                events.append(fm)
    return events


# ---------------------------------------------------------------------------
# Markdown task/reminder parsing
# ---------------------------------------------------------------------------

_TASK_RE = re.compile(r"^- \[([ xX\-])\] (.+)$")


def _parse_tasks_md(session_dir: str, user_id: str) -> list[dict]:
    """Parse workspace/tasks.md into task dicts."""
    path = os.path.join(session_dir, user_id, "workspace", "tasks.md")
    if not os.path.exists(path):
        return []
    tasks = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.rstrip()
            m = _TASK_RE.match(line)
            if not m:
                continue
            check = m.group(1)
            rest = m.group(2)
            task = {
                "_line": i,
                "_raw": line,
                "status": "completed" if check in ("x", "X") else ("cancelled" if check == "-" else "pending"),
                "is_reminder": "\U0001f514" in rest,  # 🔔
            }
            # Extract scheduled time (⏳)
            sm = re.search(r"\u23f3\s+([\d\-T:]+)", rest)
            if sm:
                task["scheduled"] = sm.group(1)
            # Extract due date (📅)
            dm = re.search(r"\U0001f4c5\s+([\d\-]+)", rest)
            if dm:
                task["due_date"] = dm.group(1)
            # Extract recurrence (🔁)
            rm = re.search(r"\U0001f501\s+(.+?)(?=\s[\u23f3\U0001f6eb\U0001f4c5\u2795\u2705\u274c\U0001f53a\u23eb\U0001f53c\U0001f53d\u23ec]|$)", rest)
            if rm:
                task["recurrence"] = rm.group(1).strip()
            # Extract title (everything before first emoji)
            title = rest
            for emoji in ("\U0001f53a", "\u23eb", "\U0001f53c", "\U0001f53d", "\u23ec",
                          "\U0001f501", "\u23f3", "\U0001f6eb", "\U0001f4c5", "\u2795", "\u2705", "\u274c"):
                if emoji in title:
                    title = title[:title.index(emoji)]
            task["title"] = title.strip()
            tasks.append(task)
    return tasks


def _mark_task_done(session_dir: str, user_id: str, line_idx: int) -> None:
    """Mark a task line as completed in tasks.md."""
    path = os.path.join(session_dir, user_id, "workspace", "tasks.md")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    if line_idx < len(lines):
        lines[line_idx] = lines[line_idx].replace("- [ ]", "- [x]", 1)
        # Add done date if not present
        done_date = datetime.now().strftime("%Y-%m-%d")
        if "\u2705" not in lines[line_idx]:
            lines[line_idx] = lines[line_idx].rstrip() + f" \u2705 {done_date}\n"
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def _reschedule_reminder(session_dir: str, user_id: str, line_idx: int, recurrence: str) -> None:
    """Reschedule a recurring reminder to the next occurrence."""
    path = os.path.join(session_dir, user_id, "workspace", "tasks.md")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    if line_idx >= len(lines):
        return

    line = lines[line_idx]
    # Find and replace the scheduled date
    sm = re.search(r"(\u23f3\s+)([\d\-T:]+)", line)
    if not sm:
        return
    try:
        old_dt = datetime.fromisoformat(sm.group(2))
    except ValueError:
        return
    new_dt = _next_occurrence(old_dt, recurrence)
    lines[line_idx] = line[:sm.start(2)] + new_dt.strftime("%Y-%m-%dT%H:%M") + line[sm.end(2):]

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


class Scheduler:
    """Periodically checks for due reminders, events, checklists, jobs, and memory indexing."""

    def __init__(self, session_dir: str, config: Optional[Dict] = None):
        self.session_dir = session_dir
        self._config = config or {}
        self._app: Optional[Any] = None
        self._callbacks: List[NotifyCallback] = []
        self._task: Optional[asyncio.Task] = None
        self._llm_formatter: Optional[LLMFormatter] = None

        self._checklist: Optional[ChecklistProcessor] = None
        self._jobs: Optional[JobRunner] = None
        self._task_reminders: Optional[TaskReminderProcessor] = None

        self._memory_indexer: Optional[Any] = None
        self._bg_tasks: Optional[Any] = None
        self._boot_time = time.monotonic()
        self._last_activity: Dict[str, float] = {}

        # Git config
        git_cfg = self._config.get("workspace", {}).get("git", {})
        self._git_enabled = git_cfg.get("enabled", False)
        self._git_daily_squash_time = git_cfg.get("daily_squash_time", "23:00")
        self._git_weekly_squash_day = int(git_cfg.get("weekly_squash_day", 6))  # 0=Mon, 6=Sun
        self._git_weekly_squash_time = git_cfg.get("weekly_squash_time", "23:30")
        self._git_push = git_cfg.get("push", False)
        self._git_daily_done: Dict[str, str] = {}   # user_id → date of last daily squash
        self._git_weekly_done: Dict[str, str] = {}   # user_id → week of last weekly squash

    @property
    def memory_indexer(self):
        if self._memory_indexer is None:
            from pawlia.memory_indexer import MemoryIndexer
            self._memory_indexer = MemoryIndexer(
                self.session_dir, self._config,
            )
        return self._memory_indexer

    def set_app(self, app: Any) -> None:
        self._app = app

    def register(self, callback: NotifyCallback) -> None:
        self._callbacks.append(callback)

    @property
    def bg_tasks(self):
        if self._bg_tasks is None:
            from pawlia.background_tasks import BackgroundTaskQueue
            self._bg_tasks = BackgroundTaskQueue(self.session_dir)
        return self._bg_tasks

    def touch_activity(self, user_id: str) -> None:
        self._last_activity[user_id] = time.monotonic()

    def set_llm_formatter(self, formatter: LLMFormatter) -> None:
        self._llm_formatter = formatter

    def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._task = asyncio.create_task(self._loop())
        logger.info("Scheduler started (interval=%ds)", CHECK_INTERVAL)

    def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            logger.info("Scheduler stopped")

    async def _loop(self) -> None:
        try:
            while True:
                try:
                    await self._check_all()
                except Exception as e:
                    logger.error("Scheduler check failed: %s", e)
                await asyncio.sleep(CHECK_INTERVAL)
        except asyncio.CancelledError:
            pass

    def _ensure_processors(self) -> None:
        if self._checklist is None:
            self._checklist = ChecklistProcessor(self.session_dir, self._notify)
            self._jobs = JobRunner(self.session_dir, self._notify)
            self._task_reminders = TaskReminderProcessor(self.session_dir, self._notify)
        _ = self.memory_indexer

    def _user_idle_minutes(self, user_id: str) -> float:
        now = time.monotonic()
        last = self._last_activity.get(user_id, self._boot_time)
        return (now - last) / 60.0

    async def _check_all(self) -> None:
        if not os.path.isdir(self.session_dir):
            return

        self._ensure_processors()

        user_ids = [
            uid for uid in os.listdir(self.session_dir)
            if os.path.isdir(os.path.join(self.session_dir, uid))
        ]

        # ── High priority (every tick) ──
        for user_id in user_ids:
            await self._check_reminders(user_id)
            await self._check_events(user_id)

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
        if self._app and self._app.memory:
            from pawlia.memory import FORCE_SUMMARY_EXCHANGES
            for user_id in user_ids:
                session = self._app.memory.load_session(user_id)
                if session.exchange_count >= FORCE_SUMMARY_EXCHANGES:
                    try:
                        await self._summarize_user(user_id)
                    except Exception as e:
                        logger.error("Forced summarization failed for %s: %s", user_id, e)

        # ── Low priority (idle-based) ──
        for user_id in user_ids:
            idle = self._user_idle_minutes(user_id)

            if idle >= IDLE_SUMMARIZE_MIN and self._app:
                try:
                    await self._summarize_user(user_id)
                except Exception as e:
                    logger.error("Summarization failed for %s: %s", user_id, e)

            if idle >= IDLE_BACKGROUND_MIN and self._app:
                try:
                    await self._process_background_tasks(user_id)
                except Exception as e:
                    logger.error("Background task failed for %s: %s", user_id, e)

            skill_config = self._config.get("skill-config") or {}
            memory_config = skill_config.get("memory") or {}
            idle_memory_min = int(memory_config.get("idle_minutes", IDLE_MEMORY_MIN))
            if idle >= idle_memory_min:
                if self._memory_indexer and self._memory_indexer.enabled:
                    try:
                        await self._memory_indexer.process_user(user_id)
                    except Exception as e:
                        logger.error("Memory indexing failed for %s: %s", user_id, e)

        # ── Workspace Git (auto-commit, daily/weekly squash) ──
        if self._git_enabled:
            for user_id in user_ids:
                try:
                    await self._git_sync(user_id)
                except Exception as e:
                    logger.error("Git sync failed for %s: %s", user_id, e)

    async def _git_sync(self, user_id: str) -> None:
        """Auto-commit workspace changes and run daily/weekly squash when due."""
        from pawlia.workspace_git import auto_commit, daily_squash, ensure_repo, push, weekly_squash

        workspace = os.path.join(self.session_dir, user_id, "workspace")
        if not os.path.isdir(workspace):
            return

        # Ensure git repo exists
        if not ensure_repo(workspace):
            return

        # Auto-commit (throttled to max 1 per 5 min inside auto_commit)
        auto_commit(workspace)

        now = datetime.now()
        today = now.strftime("%Y-%m-%d")
        _, week, _ = now.isocalendar()
        week_key = f"{now.year}-W{week:02d}"

        # Daily squash
        try:
            ds_hour, ds_min = (int(x) for x in self._git_daily_squash_time.split(":"))
        except ValueError:
            ds_hour, ds_min = 23, 0

        if now.hour == ds_hour and now.minute == ds_min and self._git_daily_done.get(user_id) != today:
            if daily_squash(workspace):
                self._git_daily_done[user_id] = today
                if self._git_push:
                    push(workspace)

        # Weekly squash
        try:
            ws_hour, ws_min = (int(x) for x in self._git_weekly_squash_time.split(":"))
        except ValueError:
            ws_hour, ws_min = 23, 30

        if (now.weekday() == self._git_weekly_squash_day
                and now.hour == ws_hour and now.minute == ws_min
                and self._git_weekly_done.get(user_id) != week_key):
            if weekly_squash(workspace):
                self._git_weekly_done[user_id] = week_key
                if self._git_push:
                    push(workspace)

    async def _summarize_user(self, user_id: str) -> None:
        if not self._app or not self._app.memory:
            return
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
            SystemMessage(content=load_system_prompt("scheduler/conversation_summary.md")),
            HumanMessage(content=(
                f"{context}Conversation to summarize:\n{history}"
            )),
        ]

        if not self._app or not self._app.llm:
            return
        llm = self._app.llm.get("chat")
        if not llm:
            return

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
        if not self._app:
            return
        tasks = self.bg_tasks.list_tasks(user_id)
        pending = [t for t in tasks if t and t.get("status") == "pending"]
        if not pending:
            return

        task = pending[0]
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
        except Exception as e:
            self.bg_tasks.mark_error(user_id, task_id, str(e))
            await self._notify(user_id, f"**[Hintergrund {task_id[:8]}]** Fehler: {e}")

    async def _check_reminders(self, user_id: str) -> None:
        """Fire due reminders from workspace/tasks.md (lines with 🔔 and ⏳)."""
        tasks = _parse_tasks_md(self.session_dir, user_id)
        reminders = [t for t in tasks if t.get("is_reminder") and t.get("status") == "pending"]
        if not reminders:
            return

        now = datetime.now()

        for rem in reminders:
            scheduled = rem.get("scheduled", "")
            if not scheduled:
                continue
            try:
                fire_at = datetime.fromisoformat(scheduled)
                if fire_at.tzinfo is not None:
                    fire_at = fire_at.replace(tzinfo=None)
            except ValueError:
                continue

            if fire_at <= now:
                title = rem.get("title", "Reminder")
                await self._notify(user_id, f"\U0001f514 {title}")

                recurrence = rem.get("recurrence", "")
                if recurrence:
                    _reschedule_reminder(self.session_dir, user_id, rem["_line"], recurrence)
                else:
                    _mark_task_done(self.session_dir, user_id, rem["_line"])

    async def _check_events(self, user_id: str) -> None:
        """Notify about upcoming events from workspace/calendar/*.md."""
        events = _list_workspace_events(self.session_dir, user_id)
        if not events:
            return

        state = _load_state(self.session_dir, user_id)
        notified = set(state.get("notified_events", []))
        now = datetime.now()
        window = now + timedelta(minutes=EVENT_REMINDER_MINUTES)
        changed = False

        for event in events:
            filename = event.get("_filename", "")
            if filename in notified:
                continue

            date_str = event.get("date", "")
            start_time = event.get("startTime", "")
            if not date_str:
                continue

            try:
                if start_time:
                    start = datetime.fromisoformat(f"{date_str}T{start_time}")
                else:
                    start = datetime.fromisoformat(date_str)
                if start.tzinfo is not None:
                    start = start.replace(tzinfo=None)
            except ValueError:
                continue

            if now <= start <= window:
                title = event.get("title", "Event")
                location = event.get("location", "")
                minutes_left = int((start - now).total_seconds() / 60)

                text = f"\U0001f4c5 In {minutes_left} Min: {title}"
                if location:
                    text += f" ({location})"

                await self._notify(user_id, text)
                notified.add(filename)
                changed = True

        if changed:
            state["notified_events"] = list(notified)
            _save_state(self.session_dir, user_id, state)

    async def _notify(self, user_id: str, message: str) -> None:
        formatted = message
        if self._llm_formatter:
            try:
                formatted = await self._llm_formatter(user_id, message)
                if not formatted or not formatted.strip():
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
    rec = recurrence.lower().strip()
    if "day" in rec:
        return fire_at + timedelta(days=1)
    elif "week" in rec:
        return fire_at + timedelta(weeks=1)
    elif "month" in rec:
        month = fire_at.month % 12 + 1
        year = fire_at.year + (1 if month == 1 else 0)
        try:
            return fire_at.replace(year=year, month=month)
        except ValueError:
            return fire_at.replace(year=year, month=month, day=28)
    return fire_at + timedelta(days=1)
