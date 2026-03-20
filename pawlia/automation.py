"""Automation engine — executes scripts for checklist items and scheduled jobs.

The LLM *plans* by creating checklist items and jobs with script references.
The automation engine *executes* them at the right time without LLM involvement.
Notification output is routed through the LLM for personalized delivery.

Two execution contexts:
- Checklist items: triggered relative to an event start time or on creation
- Scheduled jobs: triggered by cron expressions (daily, weekly, etc.)
"""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timedelta
from typing import Any, Callable, Coroutine, Dict, List, Optional

logger = logging.getLogger("pawlia.automation")

# async def notify(user_id, message) -> None  (already LLM-formatted by Scheduler)
NotifyFn = Callable[[str, str], Coroutine[Any, Any, None]]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

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
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _parse_offset(offset: str) -> timedelta:
    """Parse a relative offset string like '-90m', '-2h', '-1d', '+30m'."""
    s = offset.strip()
    sign = -1 if s.startswith("-") else 1
    s = s.lstrip("+-")

    if s.endswith("m"):
        return timedelta(minutes=sign * int(s[:-1]))
    elif s.endswith("h"):
        return timedelta(hours=sign * int(s[:-1]))
    elif s.endswith("d"):
        return timedelta(days=sign * int(s[:-1]))
    raise ValueError(f"Invalid offset format: {offset}")


# ---------------------------------------------------------------------------
# Script Executor
# ---------------------------------------------------------------------------

class ScriptExecutor:
    """Runs scripts in a subprocess and returns their output."""

    TIMEOUT = 120  # seconds

    @staticmethod
    async def run(script_path: str, params: Optional[Dict[str, Any]] = None,
                  cwd: Optional[str] = None) -> Dict[str, Any]:
        """Execute a script and return {success, output, error}.

        The script receives params as a JSON string via the AUTOMATION_PARAMS
        environment variable, and the working directory is set to cwd.
        """
        if not os.path.isfile(script_path):
            return {"success": False, "output": "", "error": f"Script not found: {script_path}"}

        env = os.environ.copy()
        if params:
            env["AUTOMATION_PARAMS"] = json.dumps(params, ensure_ascii=False)

        # Determine interpreter
        if script_path.endswith(".py"):
            cmd = ["python", script_path]
        elif script_path.endswith(".mjs") or script_path.endswith(".js"):
            cmd = ["node", script_path]
        elif script_path.endswith(".sh"):
            cmd = ["bash", script_path]
        else:
            cmd = ["python", script_path]  # default to python

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=cwd,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=ScriptExecutor.TIMEOUT,
            )
            output = stdout.decode("utf-8", errors="replace").strip()
            err = stderr.decode("utf-8", errors="replace").strip()

            if proc.returncode == 0:
                return {"success": True, "output": output, "error": ""}
            else:
                return {"success": False, "output": output, "error": err or f"Exit code {proc.returncode}"}
        except asyncio.TimeoutError:
            return {"success": False, "output": "", "error": f"Script timed out after {ScriptExecutor.TIMEOUT}s"}
        except Exception as e:
            return {"success": False, "output": "", "error": str(e)}


# ---------------------------------------------------------------------------
# Checklist Processor
# ---------------------------------------------------------------------------

class ChecklistProcessor:
    """Processes event checklists — fires script-based items at the right time."""

    def __init__(self, session_dir: str, notify: NotifyFn):
        self.session_dir = session_dir
        self._notify = notify

    async def process_user(self, user_id: str) -> None:
        """Check all events for this user and process due checklist items."""
        events_path = os.path.join(self.session_dir, user_id, "calendar", "events.json")
        events = _load_json(events_path)
        if not events:
            return

        now = datetime.now()
        changed = False

        for event in events:
            checklist = event.get("checklist", [])
            if not checklist:
                continue

            try:
                event_start = datetime.fromisoformat(event["start"])
            except (ValueError, KeyError):
                continue

            for item in checklist:
                if item.get("status") != "pending":
                    continue

                # Determine if this item should fire now
                trigger = item.get("trigger", "relative")
                should_fire = False

                if trigger == "on_create":
                    should_fire = True
                elif trigger == "relative":
                    offset_str = item.get("trigger_offset", "0m")
                    try:
                        offset = _parse_offset(offset_str)
                    except ValueError:
                        logger.error("Bad offset %r in event %s", offset_str, event.get("id"))
                        continue
                    fire_at = event_start + offset
                    should_fire = fire_at <= now
                elif trigger == "absolute":
                    try:
                        fire_at = datetime.fromisoformat(item.get("fire_at", ""))
                    except ValueError:
                        continue
                    should_fire = fire_at <= now

                if not should_fire:
                    continue

                # Execute the script
                script = item.get("script", "")
                if not script:
                    # Pure notification item (no script)
                    message = item.get("message", "")
                    if message:
                        message = self._interpolate(message, event)
                        await self._notify(user_id, f"📋 {event.get('title', 'Event')}: {message}")
                    item["status"] = "done"
                    changed = True
                    continue

                # Resolve script path
                script_path = self._resolve_script(user_id, script)
                params = item.get("params", {})
                params["event"] = {
                    "id": event.get("id"),
                    "title": event.get("title"),
                    "start": event.get("start"),
                    "location": event.get("location", ""),
                }
                params["previous_results"] = {
                    ci.get("id", ""): ci.get("result")
                    for ci in checklist if ci.get("result") is not None
                }

                result = await ScriptExecutor.run(script_path, params)
                item["result"] = result.get("output", "") if result["success"] else result.get("error", "")
                item["status"] = "done" if result["success"] else "failed"
                item["executed_at"] = now.isoformat()
                changed = True

                if item.get("notify", True):
                    if result["success"]:
                        output = result["output"][:500] if result["output"] else "erledigt"
                        await self._notify(user_id, f"📋 {event.get('title', '')}: {output}")
                    else:
                        await self._notify(user_id,
                            f"⚠️ {event.get('title', '')}: Script fehlgeschlagen — {result['error'][:200]}")

                logger.info("Checklist item %s for event %s: %s",
                           item.get("id"), event.get("id"), item["status"])

        if changed:
            _save_json(events_path, events)

    def _resolve_script(self, user_id: str, script: str) -> str:
        """Resolve a script path — check user automations dir first, then global."""
        user_path = os.path.join(self.session_dir, user_id, "automations", script)
        if os.path.isfile(user_path):
            return user_path

        pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        global_path = os.path.join(pkg_dir, "scripts", script)
        if os.path.isfile(global_path):
            return global_path

        skills_path = os.path.join(pkg_dir, "skills")
        for skill_dir in os.listdir(skills_path) if os.path.isdir(skills_path) else []:
            candidate = os.path.join(skills_path, skill_dir, "scripts", script)
            if os.path.isfile(candidate):
                return candidate

        return script

    @staticmethod
    def _interpolate(message: str, event: dict) -> str:
        """Replace {field} placeholders with event data."""
        for key in ("title", "start", "location", "description"):
            message = message.replace(f"{{{key}}}", event.get(key, ""))
        return message


# ---------------------------------------------------------------------------
# Job Runner (Cron-like scheduled scripts)
# ---------------------------------------------------------------------------

class JobRunner:
    """Executes scheduled jobs based on cron-like expressions."""

    def __init__(self, session_dir: str, notify: NotifyFn):
        self.session_dir = session_dir
        self._notify = notify

    async def process_user(self, user_id: str) -> None:
        """Check and execute due jobs for this user."""
        jobs_path = os.path.join(self.session_dir, user_id, "automations", "jobs.json")
        jobs = _load_json(jobs_path)
        if not jobs:
            return

        now = datetime.now()
        changed = False

        for job in jobs:
            if not job.get("enabled", True):
                continue

            if not self._is_due(job, now):
                continue

            script = job.get("script", "")
            if not script:
                continue

            script_path = self._resolve_script(user_id, script)
            params = job.get("params", {})
            params["job_name"] = job.get("name", "")
            params["user_id"] = user_id

            logger.info("Running job '%s' for %s", job.get("name"), user_id)
            result = await ScriptExecutor.run(script_path, params)

            job["last_run"] = now.isoformat()
            job["last_result"] = "success" if result["success"] else "failed"
            changed = True

            if job.get("notify", True):
                if result["success"]:
                    output = result["output"][:500] if result["output"] else "erledigt"
                    await self._notify(user_id, f"⚙️ {job.get('name', 'Job')}: {output}")
                else:
                    await self._notify(user_id,
                        f"⚠️ Job '{job.get('name', '')}' fehlgeschlagen: {result['error'][:200]}")

        if changed:
            _save_json(jobs_path, jobs)

    @staticmethod
    def _is_due(job: dict, now: datetime) -> bool:
        """Check if a job should run based on its schedule and last_run.

        Supports simple schedule formats:
        - 'HH:MM' — daily at that time
        - 'weekly:DOW:HH:MM' — weekly on day-of-week (0=Mon)
        - 'monthly:DD:HH:MM' — monthly on day DD
        - 'interval:Nm' / 'interval:Nh' — every N minutes/hours
        """
        schedule = job.get("schedule", "")
        if not schedule:
            return False

        last_run_str = job.get("last_run", "")
        last_run = None
        if last_run_str:
            try:
                last_run = datetime.fromisoformat(last_run_str)
            except ValueError:
                pass

        # interval:Nm or interval:Nh
        if schedule.startswith("interval:"):
            interval_str = schedule[len("interval:"):]
            try:
                delta = _parse_offset(f"+{interval_str}")
            except ValueError:
                return False
            if last_run is None:
                return True
            return now >= last_run + delta

        # weekly:DOW:HH:MM  (DOW: 0=Mon..6=Sun)
        if schedule.startswith("weekly:"):
            parts = schedule.split(":")
            if len(parts) != 4:
                return False
            try:
                dow = int(parts[1])
                hour = int(parts[2])
                minute = int(parts[3])
            except ValueError:
                return False
            if now.weekday() != dow:
                return False
            if now.hour == hour and now.minute == minute:
                if last_run is None:
                    return True
                return (now - last_run).total_seconds() > 120
            return False

        # monthly:DD:HH:MM
        if schedule.startswith("monthly:"):
            parts = schedule.split(":")
            if len(parts) != 4:
                return False
            try:
                day = int(parts[1])
                hour = int(parts[2])
                minute = int(parts[3])
            except ValueError:
                return False
            if now.day != day:
                return False
            if now.hour == hour and now.minute == minute:
                if last_run is None:
                    return True
                return (now - last_run).total_seconds() > 120
            return False

        # HH:MM — daily at that time
        try:
            parts = schedule.split(":")
            target_hour = int(parts[0])
            target_minute = int(parts[1]) if len(parts) > 1 else 0
        except (ValueError, IndexError):
            return False

        if now.hour == target_hour and now.minute == target_minute:
            if last_run is None:
                return True
            return (now - last_run).total_seconds() > 120
        return False

    def _resolve_script(self, user_id: str, script: str) -> str:
        """Resolve script path (same logic as ChecklistProcessor)."""
        user_path = os.path.join(self.session_dir, user_id, "automations", script)
        if os.path.isfile(user_path):
            return user_path
        pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        global_path = os.path.join(pkg_dir, "scripts", script)
        if os.path.isfile(global_path):
            return global_path
        return script


# ---------------------------------------------------------------------------
# Task Reminder Processor
# ---------------------------------------------------------------------------

class TaskReminderProcessor:
    """Fires task reminders based on due_date and reminder offsets."""

    def __init__(self, session_dir: str, notify: NotifyFn):
        self.session_dir = session_dir
        self._notify = notify

    async def process_user(self, user_id: str) -> None:
        """Check all tasks for due reminders."""
        tasks_path = os.path.join(self.session_dir, user_id, "tasks", "tasks.json")
        tasks = _load_json(tasks_path)
        if not tasks:
            return

        now = datetime.now()
        changed = False

        for task in tasks:
            if task.get("status") != "pending":
                continue

            due_str = task.get("due_date", "")
            if not due_str:
                continue

            try:
                if "T" in due_str:
                    due = datetime.fromisoformat(due_str)
                else:
                    due = datetime.fromisoformat(due_str + "T23:59:00")
            except ValueError:
                continue

            reminders = task.get("reminders", [])
            for reminder in reminders:
                if reminder.get("fired", False):
                    continue

                offset_str = reminder.get("offset", "")
                if not offset_str:
                    continue

                try:
                    offset = _parse_offset(offset_str)
                except ValueError:
                    continue

                fire_at = due + offset
                if fire_at <= now:
                    message = reminder.get("message", f"Aufgabe fällig: {task.get('title', '')}")
                    message = message.replace("{title}", task.get("title", ""))
                    message = message.replace("{due_date}", due_str)
                    await self._notify(user_id, f"📝 {message}")
                    reminder["fired"] = True
                    changed = True

        if changed:
            _save_json(tasks_path, tasks)


# ---------------------------------------------------------------------------
# Public helpers for creating checklist items and jobs
# ---------------------------------------------------------------------------

def create_checklist_item(
    script: str = "",
    trigger: str = "relative",
    trigger_offset: str = "0m",
    params: Optional[Dict[str, Any]] = None,
    message: str = "",
    notify: bool = True,
) -> Dict[str, Any]:
    """Create a checklist item dict for an event."""
    return {
        "id": f"chk-{uuid.uuid4().hex[:8]}",
        "script": script,
        "trigger": trigger,
        "trigger_offset": trigger_offset,
        "params": params or {},
        "message": message,
        "status": "pending",
        "result": None,
        "notify": notify,
    }


def create_job(
    name: str,
    script: str,
    schedule: str,
    params: Optional[Dict[str, Any]] = None,
    notify: bool = True,
) -> Dict[str, Any]:
    """Create a scheduled job dict."""
    return {
        "id": f"job-{uuid.uuid4().hex[:8]}",
        "name": name,
        "script": script,
        "schedule": schedule,
        "params": params or {},
        "notify": notify,
        "enabled": True,
        "created_at": datetime.now().isoformat(),
        "last_run": "",
        "last_result": "",
    }
