"""Automation engine — executes scheduled jobs via LLM.

Data sources (Obsidian-native):
  - Event checklists: workspace/calendar/*.md  (checklist in YAML frontmatter)
  - Task reminders:   scheduler_state.json     (reminder offsets per task title)
  - Jobs:             automations/jobs.json     (schedule + instruction)
  - State:            scheduler_state.json      (checklist status, fired flags)

Execution:
  - Checklist items: triggered relative to an event start time or on creation
  - Scheduled jobs: triggered by cron expressions, executed by LLM
  - Task reminders: triggered by due date offsets
"""

import asyncio
import json
import logging
import os
import re
import uuid
from datetime import datetime, timedelta
from typing import Any, Callable, Coroutine, Dict, List, Optional

import yaml

from pawlia.utils import load_json, resolve_script, save_json

logger = logging.getLogger("pawlia.automation")

NotifyFn = Callable[[str, str], Coroutine[Any, Any, None]]


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
# Scheduler state helpers
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
# Script Executor (used by ChecklistProcessor for script-based checklist items)
# ---------------------------------------------------------------------------

_INTERPRETERS: Dict[str, str] = {
    ".py": "python",
    ".mjs": "node",
    ".js": "node",
    ".sh": "bash",
}


class ScriptExecutor:
    """Runs scripts in a subprocess and returns their output."""

    TIMEOUT = 120

    @staticmethod
    def _is_allowed_path(script_path: str, user_id: Optional[str],
                         session_dir: Optional[str]) -> bool:
        real = os.path.realpath(script_path)
        pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        allowed = [
            os.path.realpath(os.path.join(pkg_dir, "skills")),
            os.path.realpath(os.path.join(pkg_dir, "scripts")),
        ]
        if session_dir and user_id:
            allowed.append(os.path.realpath(
                os.path.join(session_dir, user_id, "automations")
            ))
            allowed.append(os.path.realpath(
                os.path.join(session_dir, user_id, "workspace", ".scripts")
            ))
        return any(real.startswith(base + os.sep) for base in allowed)

    @staticmethod
    async def run(script_path: str, params: Optional[Dict[str, Any]] = None,
                  cwd: Optional[str] = None,
                  user_id: Optional[str] = None,
                  session_dir: Optional[str] = None) -> Dict[str, Any]:
        if not os.path.isfile(script_path):
            return {"success": False, "output": "", "error": f"Script not found: {script_path}"}

        if not ScriptExecutor._is_allowed_path(script_path, user_id, session_dir):
            logger.warning("Script path outside allowed dirs: %s", script_path)
            return {"success": False, "output": "",
                    "error": "Script liegt ausserhalb der erlaubten Verzeichnisse."}

        env = os.environ.copy()
        if user_id:
            env["PAWLIA_USER_ID"] = user_id
        if session_dir:
            env["PAWLIA_SESSION_DIR"] = session_dir
        if params:
            env["AUTOMATION_PARAMS"] = json.dumps(params, ensure_ascii=False)

        ext = os.path.splitext(script_path)[1]
        interpreter = _INTERPRETERS.get(ext, "python")
        cmd = [interpreter, script_path]

        proc: Optional[asyncio.subprocess.Process] = None
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
            return {"success": False, "output": output, "error": err or f"Exit code {proc.returncode}"}
        except asyncio.TimeoutError:
            if proc:
                proc.kill()
                await proc.wait()
            return {"success": False, "output": "", "error": f"Script timed out after {ScriptExecutor.TIMEOUT}s"}
        except Exception as e:
            return {"success": False, "output": "", "error": str(e)}


# ---------------------------------------------------------------------------
# Checklist Processor (reads event frontmatter, state in scheduler_state.json)
# ---------------------------------------------------------------------------

class ChecklistProcessor:
    """Processes event checklists from workspace/calendar/*.md frontmatter."""

    def __init__(self, session_dir: str, notify: NotifyFn):
        self.session_dir = session_dir
        self._notify = notify

    async def process_user(self, user_id: str) -> None:
        cal_dir = os.path.join(self.session_dir, user_id, "workspace", "calendar")
        if not os.path.isdir(cal_dir):
            return

        state = _load_state(self.session_dir, user_id)
        checklist_state = state.setdefault("checklist_state", {})
        now = datetime.now()
        changed = False

        for fname in os.listdir(cal_dir):
            if not fname.endswith(".md"):
                continue
            filepath = os.path.join(cal_dir, fname)
            fm = self._read_frontmatter(filepath)
            if not fm:
                continue

            checklist = fm.get("checklist", [])
            if not checklist:
                continue

            # Parse event start time
            date_str = fm.get("date", "")
            start_time = fm.get("startTime", "")
            if not date_str:
                continue
            try:
                if start_time:
                    event_start = datetime.fromisoformat(f"{date_str}T{start_time}")
                else:
                    event_start = datetime.fromisoformat(date_str)
            except ValueError:
                continue

            file_state = checklist_state.setdefault(fname, {})

            for item in checklist:
                item_id = item.get("id", "")
                if not item_id:
                    continue

                item_state = file_state.get(item_id, {})
                if item_state.get("status") in ("done", "failed"):
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

                # Execute
                script = item.get("script", "")
                event_info = {
                    "title": fm.get("title", ""),
                    "start": f"{date_str}T{start_time}" if start_time else date_str,
                    "location": fm.get("location", ""),
                }

                if not script:
                    message = item.get("message", "")
                    if message:
                        message = self._interpolate(message, event_info)
                        await self._notify(user_id, f"\U0001f4cb {event_info.get('title', 'Event')}: {message}")
                    file_state[item_id] = {"status": "done", "executed_at": now.isoformat()}
                    changed = True
                    continue

                script_path = resolve_script(self.session_dir, user_id, script)
                params = dict(item.get("params", {}))
                params["event"] = event_info
                # Collect previous results from state — only successful items
                params["previous_results"] = {
                    iid: ist.get("result")
                    for iid, ist in file_state.items()
                    if isinstance(ist, dict) and ist.get("status") == "done" and ist.get("result") is not None
                }

                result = await ScriptExecutor.run(
                    script_path, params,
                    user_id=user_id, session_dir=self.session_dir,
                )

                file_state[item_id] = {
                    "status": "done" if result["success"] else "failed",
                    "result": result.get("output", "") if result["success"] else result.get("error", ""),
                    "executed_at": now.isoformat(),
                }
                changed = True

                if item.get("notify", True):
                    if result["success"]:
                        output = result["output"][:500] if result["output"] else "erledigt"
                        await self._notify(user_id, f"\U0001f4cb {event_info.get('title', '')}: {output}")
                    else:
                        await self._notify(user_id,
                            f"\u26a0\ufe0f {event_info.get('title', '')}: Script fehlgeschlagen — {result['error'][:200]}")

                logger.info("Checklist item %s for %s: %s", item_id, fname, file_state[item_id]["status"])

        if changed:
            state["checklist_state"] = checklist_state
            _save_state(self.session_dir, user_id, state)

    @staticmethod
    def _read_frontmatter(filepath: str) -> dict | None:
        from pawlia.utils import parse_frontmatter
        return parse_frontmatter(filepath)

    @staticmethod
    def _interpolate(message: str, event: dict) -> str:
        for key in ("title", "start", "location", "description"):
            message = message.replace(f"{{{key}}}", event.get(key, ""))
        return message


# ---------------------------------------------------------------------------
# Job Runner — executes scheduled jobs via LLM
# ---------------------------------------------------------------------------

class JobRunner:
    """Executes scheduled jobs by running the instruction through the LLM."""

    def __init__(self, session_dir: str, notify: NotifyFn, app: Any = None):
        self.session_dir = session_dir
        self._notify = notify
        self._app = app

    async def process_user(self, user_id: str) -> None:
        jobs_path = os.path.join(self.session_dir, user_id, "automations", "jobs.json")
        jobs = load_json(jobs_path)
        if not jobs:
            return

        now = datetime.now()
        changed = False

        for job in jobs:
            if not job.get("enabled", True):
                continue
            force_run = job.get("force_run", False)
            if not force_run and not self._is_due(job, now):
                continue
            if force_run:
                job.pop("force_run", None)
                changed = True

            instruction = job.get("instruction", "")
            job_name = job.get("name", "Job")

            if not instruction:
                logger.warning("Job '%s' has no instruction, skipping", job_name)
                continue

            if not self._app:
                logger.error("Job '%s': no app reference, cannot execute", job_name)
                continue

            logger.info("Running job '%s' for %s via LLM", job_name, user_id)

            try:
                runner = self._app.run_instruction(instruction, user_id)
                result = await runner.run(instruction, thread_id=None)
                job["last_run"] = now.isoformat()
                job["last_result"] = "success"
                changed = True

                notify = job.get("notify", True)
                if notify is True:
                    output = result if result else "erledigt"
                    await self._notify(user_id, f"\u2699\ufe0f {job_name}:\n{output}")

            except Exception as e:
                logger.error("Job '%s' failed for %s: %s", job_name, user_id, e)
                job["last_run"] = now.isoformat()
                job["last_result"] = "failed"
                changed = True

                notify = job.get("notify", True)
                if notify is True or notify == "error":
                    await self._notify(user_id,
                        f"\u26a0\ufe0f Job '{job_name}' fehlgeschlagen: {str(e)[:200]}")

        if changed:
            save_json(jobs_path, jobs)

    @staticmethod
    def _is_due(job: dict, now: datetime) -> bool:
        schedule = job.get("schedule", "")
        if not schedule:
            return False

        last_run_str = job.get("last_run", "")
        last_run: Optional[datetime] = None
        if last_run_str:
            try:
                last_run = datetime.fromisoformat(last_run_str)
            except ValueError:
                pass

        def _not_run_recently() -> bool:
            return last_run is None or (now - last_run).total_seconds() > 120

        def _in_window(hour: int, minute: int) -> bool:
            target_today = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            window_start = last_run if last_run else (now - timedelta(minutes=2))
            return window_start < target_today <= now

        if schedule.startswith("interval:"):
            interval_str = schedule[len("interval:"):]
            try:
                delta = _parse_offset(f"+{interval_str}")
            except ValueError:
                return False
            if last_run is None:
                return True
            return now >= last_run + delta

        if schedule.startswith("weekly:"):
            parts = schedule.split(":")
            if len(parts) != 4:
                return False
            try:
                dow, hour, minute = int(parts[1]), int(parts[2]), int(parts[3])
            except ValueError:
                return False
            if now.weekday() != dow:
                return False
            return _in_window(hour, minute) and _not_run_recently()

        if schedule.startswith("monthly:"):
            parts = schedule.split(":")
            if len(parts) != 4:
                return False
            try:
                day, hour, minute = int(parts[1]), int(parts[2]), int(parts[3])
            except ValueError:
                return False
            if now.day != day:
                return False
            return _in_window(hour, minute) and _not_run_recently()

        try:
            parts = schedule.split(":")
            target_hour = int(parts[0])
            target_minute = int(parts[1]) if len(parts) > 1 else 0
        except (ValueError, IndexError):
            return False

        return _in_window(target_hour, target_minute) and _not_run_recently()


# ---------------------------------------------------------------------------
# Task Reminder Processor (reads tasks.md + scheduler_state.json)
# ---------------------------------------------------------------------------

_TASK_RE = re.compile(r"^- \[([ xX\-])\] (.+)$")


class TaskReminderProcessor:
    """Fires task reminders based on due_date and reminder config in scheduler_state."""

    def __init__(self, session_dir: str, notify: NotifyFn):
        self.session_dir = session_dir
        self._notify = notify

    async def process_user(self, user_id: str) -> None:
        """Check tasks.md for due tasks and fire reminders from scheduler_state."""
        tasks_path = os.path.join(self.session_dir, user_id, "workspace", "tasks.md")
        if not os.path.exists(tasks_path):
            return

        state = _load_state(self.session_dir, user_id)
        task_reminders = state.get("task_reminders", {})
        now = datetime.now()
        changed = False

        with open(tasks_path, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip()
                m = _TASK_RE.match(line)
                if not m:
                    continue
                check = m.group(1)
                if check != " ":
                    continue  # only pending tasks

                rest = m.group(2)
                # Skip reminders (🔔)
                if "\U0001f514" in rest:
                    continue

                # Extract due date (📅)
                dm = re.search(r"\U0001f4c5\s+([\d\-]+)", rest)
                if not dm:
                    continue
                due_str = dm.group(1)

                try:
                    if "T" in due_str:
                        due = datetime.fromisoformat(due_str)
                    else:
                        due = datetime.fromisoformat(due_str + "T23:59:00")
                except ValueError:
                    continue

                # Extract title
                title = rest
                for emoji in ("\U0001f53a", "\u23eb", "\U0001f53c", "\U0001f53d", "\u23ec",
                              "\U0001f501", "\u23f3", "\U0001f6eb", "\U0001f4c5", "\u2795", "\u2705", "\u274c"):
                    if emoji in title:
                        title = title[:title.index(emoji)]
                title = title.strip()

                # Check task-specific reminders from state
                task_key = title.lower()
                reminders = task_reminders.get(task_key, [])
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
                        message = reminder.get("message", f"Aufgabe f\u00e4llig: {title}")
                        message = message.replace("{title}", title).replace("{due_date}", due_str)
                        await self._notify(user_id, f"\U0001f4dd {message}")
                        reminder["fired"] = True
                        changed = True

        if changed:
            state["task_reminders"] = task_reminders
            _save_state(self.session_dir, user_id, state)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def create_checklist_item(
    script: str = "",
    trigger: str = "relative",
    trigger_offset: str = "0m",
    params: Optional[Dict[str, Any]] = None,
    message: str = "",
    notify: bool = True,
) -> Dict[str, Any]:
    return {
        "id": f"chk-{uuid.uuid4().hex[:8]}",
        "script": script,
        "trigger": trigger,
        "trigger_offset": trigger_offset,
        "params": params or {},
        "message": message,
        "notify": notify,
    }


def create_job(
    name: str,
    schedule: str,
    instruction: str,
    notify: bool | str = True,
) -> Dict[str, Any]:
    """Create a job definition.

    ``notify``: True = always deliver output, False = never, "error" = only on failure.
    """
    return {
        "id": f"job-{uuid.uuid4().hex[:8]}",
        "name": name,
        "schedule": schedule,
        "instruction": instruction,
        "notify": notify,
        "enabled": True,
        "created_at": datetime.now().isoformat(),
        "last_run": "",
        "last_result": "",
    }
