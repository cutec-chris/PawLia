"""Automation engine — executes scheduled jobs via LLM.

Data sources (Obsidian-native):
  - Event checklists: workspace/calendar/*.md  (checklist in YAML frontmatter)
  - Task reminders:   scheduler_state.json     (reminder offsets per task title)
  - Jobs:             automations/jobs.json     (schedule + instruction)
  - State:            scheduler_state.json      (checklist status, fired flags)

Execution:
  - Checklist items: triggered relative to an event start time or on creation
  - Scheduled jobs: triggered by cron expressions; run a deterministic script
    (preferred, silent unless it prints) or an LLM instruction (trivial cases)
  - Task reminders: triggered by due date offsets
"""

import asyncio
import json
import logging
import os
import re
import unicodedata
import uuid
from datetime import datetime, timedelta
from typing import Any, Callable, Coroutine, Dict, List, Optional

import yaml

from pawlia import sandbox
from pawlia.config import resolve_config_path
from pawlia.utils import load_json, resolve_script, save_json

logger = logging.getLogger("pawlia.automation")

NotifyFn = Callable[[str, str], Coroutine[Any, Any, None]]

# A script (or instruction) that prints exactly this marker is treated as
# "nothing to report" — same as empty output. Lets a job stay silent when there
# is nothing worth a notification (e.g. a thunderstorm monitor on a calm day).
SILENT_SENTINEL = "PAWLIA_SILENT"

# Markers that mean "say nothing this run". The canonical one is PAWLIA_SILENT
# (documented in the automation skill for scripts), but LLM instruction jobs
# naturally emit a bare "SILENT" instead. Accept both, case-insensitively, so a
# monitor stays quiet regardless of which form actually reaches us — otherwise
# the literal token gets posted every run (the recurring "SILENT alert" bug seen
# on the Gewitter and Bahn monitors).
_SILENT_MARKERS = frozenset({SILENT_SENTINEL, "SILENT"})

# Unicode categories with no visible glyph: format (Cf — e.g. U+200E
# LEFT-TO-RIGHT MARK, zero-width spaces, BOM), control (Cc), surrogates (Cs).
# A model that wants to "say nothing" can't emit a truly empty turn, so it
# emits one of these instead (mimo-v2.5 sends U+200E). We must treat such a
# reply as silence, not as a one-character message body.
_INVISIBLE_CATEGORIES = frozenset({"Cf", "Cc", "Cs"})


def _visible_text(output: "str | None") -> str:
    """*output* reduced to visible content: invisible/format/control characters
    removed, then surrounding whitespace stripped."""
    if not output:
        return ""
    return "".join(
        ch for ch in output if unicodedata.category(ch) not in _INVISIBLE_CATEGORIES
    ).strip()


def _is_silent_sentinel(output: "str | None") -> bool:
    """True when *output* carries no message worth sending.

    That is either a bare silent marker (PAWLIA_SILENT / SILENT, any case) or
    no visible content at all — whitespace, or only zero-width / format
    characters such as U+200E. A genuinely empty string is *not* silent here:
    it stays subject to the notify policy (notify=True → "erledigt").
    """
    if not output:
        return False
    visible = _visible_text(output)
    if not visible:
        return True
    return visible.upper() in _SILENT_MARKERS


def _strip_sentinel(output: str) -> str:
    """Return the output with the silent sentinel removed.

    Empty result means "say nothing". A bare marker (PAWLIA_SILENT or SILENT,
    optionally surrounded by whitespace, any case) collapses to empty so the job
    stays silent. A marker mixed with other text is left untouched.
    """
    if output is None:
        return ""
    if _is_silent_sentinel(output):
        return ""
    return output.strip()


def _success_notification(notify: "bool | str", output: str) -> Optional[str]:
    """Decide the notification body for a *successful* run, or ``None`` for silence.

    Failures are surfaced separately by the caller (always loud), so this only
    governs the success path.

    ``notify`` modes:
      - ``True``         → always send (empty output becomes "erledigt")
      - ``"output_only"``→ send only when there is real output (silent on empty)
      - ``"error"`` / ``False`` → nothing on success
    """
    if _is_silent_sentinel(output):
        # An explicit "stay silent" wins over the job's notify policy: a monitor
        # that reports "nothing to report" must never ping, even under
        # notify=True (the default for LLM instruction jobs).
        return None

    body = _strip_sentinel(output)

    if notify is True:
        return body if body else "erledigt"
    if notify == "output_only":
        return body if body else None
    return None


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
            user_root = os.path.realpath(
                os.path.join(session_dir, user_id)
            )
            allowed.append(os.path.join(user_root, "workspace", "skills", "scripts"))
            allowed.append(os.path.join(user_root, "automations"))
            allowed.append(os.path.join(user_root, "workspace", ".scripts"))
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

        # Make the automation harness importable regardless of cwd, and tell it
        # where the LLM config lives so ``llm_call`` can reach a provider.
        pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = pkg_root + (os.pathsep + existing_pp if existing_pp else "")
        config_path = resolve_config_path()
        if config_path:
            env["PAWLIA_CONFIG_PATH"] = config_path

        ext = os.path.splitext(script_path)[1]
        interpreter = _INTERPRETERS.get(ext, "python")
        cmd = [interpreter, script_path]

        # Sandbox the script the same way skill scripts run: read-only root,
        # only the user's session dir + /tmp writable. Network stays available
        # (weather/transit APIs). Degrade to an unsandboxed run when bubblewrap
        # is unavailable (some container runtimes disable user namespaces).
        if sandbox.bwrap_available():
            writable = sandbox.writable_roots(session_dir, user_id)
            cmd = sandbox.wrap_argv(cmd, writable)
        else:
            logger.warning(
                "bubblewrap unavailable — running script %s unsandboxed", script_path
            )

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

                # Reminders derived from the event's frontmatter ``reminders``
                # are fired recurrence-aware by the scheduler's _check_events,
                # so they must not also fire here (would double-notify and only
                # trigger on the original event date for recurring events).
                if item.get("source") == "event_reminder":
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
    """Executes scheduled jobs.

    A job runs either a deterministic ``script`` (preferred — it stays silent
    unless it prints something) or, for trivial cases, a natural-language
    ``instruction`` through the LLM.
    """

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

            script = job.get("script", "")
            instruction = job.get("instruction", "")
            job_name = job.get("name", "Job")

            if not script and not instruction:
                logger.warning("Job '%s' has neither script nor instruction, skipping", job_name)
                continue

            if not script and not self._app:
                logger.error("Job '%s': no app reference, cannot execute", job_name)
                continue

            # Committed to running — clear the flag and persist now so a crash
            # during execution doesn't cause the job to fire again on restart.
            if force_run:
                job.pop("force_run", None)
                save_json(jobs_path, jobs)

            # Script jobs stay silent unless they print something; instruction
            # jobs (the LLM always produces text) default to always delivering.
            notify = job.get("notify", "output_only" if script else True)

            try:
                if script:
                    logger.info("Running job '%s' for %s via script %s", job_name, user_id, script)
                    script_path = resolve_script(self.session_dir, user_id, script)
                    run_result = await ScriptExecutor.run(
                        script_path, job.get("params", {}),
                        user_id=user_id, session_dir=self.session_dir,
                    )
                    if not run_result["success"]:
                        # Surface script failures through the error path below.
                        raise RuntimeError(run_result.get("error") or "Script fehlgeschlagen")
                    result_text = run_result.get("output", "")
                else:
                    logger.info("Running job '%s' for %s via LLM", job_name, user_id)
                    runner = self._app.run_instruction(instruction, user_id)
                    result_text = await runner.run(instruction, thread_id=None)

                job["last_run"] = now.isoformat()
                job["last_result"] = "success"
                changed = True

                body = _success_notification(notify, result_text or "")
                if body is not None:
                    await self._notify(user_id, f"\u2699\ufe0f {job_name}:\n{body}")

            except Exception as e:
                logger.error("Job '%s' failed for %s: %s", job_name, user_id, e)
                job["last_run"] = now.isoformat()
                job["last_result"] = "failed"
                changed = True

                # Failures are always loud \u2014 a broken monitor must not fail silently.
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

        if _is_cron(schedule):
            return _cron_matches(schedule, now) and _not_run_recently()

        try:
            parts = schedule.split(":")
            target_hour = int(parts[0])
            target_minute = int(parts[1]) if len(parts) > 1 else 0
        except (ValueError, IndexError):
            return False

        return _in_window(target_hour, target_minute) and _not_run_recently()


def _is_cron(schedule: str) -> bool:
    parts = schedule.split()
    return len(parts) == 5 and all(
        p.replace("*", "").replace(",", "").replace("-", "").replace("/", "").isdigit()
        or p in ("*", "")
        for p in parts
    )


def _cron_field_matches(field: str, value: int, low: int, high: int) -> bool:
    if field == "*":
        return True

    if field.startswith("*/"):
        step = int(field[2:])
        if step <= 0:
            return False
        return value % step == 0

    for part in field.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_str, hi_str = part.split("-", 1)
            try:
                lo, hi = int(lo_str), int(hi_str)
                if lo <= value <= hi:
                    return True
            except ValueError:
                continue
        elif part == "*":
            return True
        else:
            try:
                if int(part) == value:
                    return True
            except ValueError:
                continue
    return False


def _cron_matches(schedule: str, now: datetime) -> bool:
    parts = schedule.split()
    if len(parts) != 5:
        return False
    try:
        return (
            _cron_field_matches(parts[0], now.minute, 0, 59)
            and _cron_field_matches(parts[1], now.hour, 0, 23)
            and _cron_field_matches(parts[2], now.day, 1, 31)
            and _cron_field_matches(parts[3], now.month, 1, 12)
            and _cron_field_matches(parts[4], (now.weekday() + 1) % 7, 0, 6)
        )
    except ValueError:
        return False


_CRON_FIELD_RE = re.compile(r"^(\*/\d+|[0-9*,/\-]+)$")


def _cron_field_values_in_range(field: str, low: int, high: int) -> bool:
    """Return True if every numeric value in `field` lies in [low, high]."""
    for part in field.split(","):
        part = part.strip()
        if not part or part == "*":
            continue
        if part.startswith("*/"):
            try:
                step = int(part[2:])
            except ValueError:
                return False
            if step <= 0 or step > (high - low + 1):
                return False
            continue
        if "-" in part:
            try:
                lo_s, hi_s = part.split("-", 1)
                lo, hi = int(lo_s), int(hi_s)
            except ValueError:
                return False
            if not (low <= lo <= high) or not (low <= hi <= high) or lo > hi:
                return False
            continue
        try:
            v = int(part)
        except ValueError:
            return False
        if not (low <= v <= high):
            return False
    return True


def _validate_cron_field(field: str, low: int, high: int, name: str) -> None:
    if not _CRON_FIELD_RE.match(field.strip()):
        raise ValueError(f"Cron {name} field '{field}' is invalid")
    if not _cron_field_values_in_range(field.strip(), low, high):
        raise ValueError(
            f"Cron {name} field '{field}' contains a value outside {low}-{high}"
        )


def validate_schedule(schedule: str) -> tuple[bool, str]:
    """Validate a schedule string. Returns (is_valid, error_message)."""
    if not schedule or not schedule.strip():
        return False, "Schedule ist leer."

    s = schedule.strip()

    if s.startswith("interval:"):
        try:
            _parse_offset(f"+{s[len('interval:'):]}")
            return True, ""
        except ValueError:
            return False, (
                f"Ungültiges Intervall '{s}'. "
                f"Erlaubte Formate: interval:30m, interval:2h, interval:1d"
            )

    if s.startswith("weekly:"):
        parts = s.split(":")
        if len(parts) != 4:
            return False, (
                f"Wöchentlicher Schedule muss weekly:DAY:HH:MM sein "
                f"(z.B. weekly:0:09:00, 0=Mo..6=So). Bekam: '{s}'"
            )
        try:
            dow, hour, minute = int(parts[1]), int(parts[2]), int(parts[3])
        except ValueError:
            return False, (
                f"Wöchentlicher Schedule muss weekly:DAY:HH:MM sein "
                f"(z.B. weekly:0:09:00, 0=Mo..6=So). Bekam: '{s}'"
            )
        if not (0 <= dow <= 6):
            return False, f"Wochentag muss 0-6 sein (0=Mo..6=So), nicht {dow}."
        if not (0 <= hour <= 23):
            return False, f"Stunde muss 0-23 sein, nicht {hour}."
        if not (0 <= minute <= 59):
            return False, f"Minute muss 0-59 sein, nicht {minute}."
        return True, ""

    if s.startswith("monthly:"):
        parts = s.split(":")
        if len(parts) != 4:
            return False, (
                f"Monatlicher Schedule muss monthly:DAY:HH:MM sein "
                f"(z.B. monthly:1:10:00). Bekam: '{s}'"
            )
        try:
            day, hour, minute = int(parts[1]), int(parts[2]), int(parts[3])
        except ValueError:
            return False, (
                f"Monatlicher Schedule muss monthly:DAY:HH:MM sein "
                f"(z.B. monthly:1:10:00). Bekam: '{s}'"
            )
        if not (1 <= day <= 31):
            return False, f"Tag muss 1-31 sein, nicht {day}."
        if not (0 <= hour <= 23):
            return False, f"Stunde muss 0-23 sein, nicht {hour}."
        if not (0 <= minute <= 59):
            return False, f"Minute muss 0-59 sein, nicht {minute}."
        return True, ""

    if _is_cron(s):
        parts = s.split()
        try:
            _validate_cron_field(parts[0], 0, 59, "Minute")
            _validate_cron_field(parts[1], 0, 23, "Stunde")
            _validate_cron_field(parts[2], 1, 31, "Tag")
            _validate_cron_field(parts[3], 1, 12, "Monat")
            _validate_cron_field(parts[4], 0, 6, "Wochentag")
            return True, ""
        except ValueError as e:
            return False, str(e)

    parts = s.split(":")
    if 2 <= len(parts) <= 3:
        try:
            hour, minute = int(parts[0]), int(parts[1])
        except ValueError:
            return False, (
                f"Unbekanntes Schedule-Format: '{s}'. "
                f"Erlaubt: HH:MM (täglich), interval:30m, weekly:0:09:00, "
                f"monthly:1:10:00, oder Cron-Syntax (5 Felder: '* * * * *')"
            )
        if not (0 <= hour <= 23):
            return False, f"Stunde muss 0-23 sein, nicht {hour}."
        if not (0 <= minute <= 59):
            return False, f"Minute muss 0-59 sein, nicht {minute}."
        return True, ""

    return False, (
        f"Unbekanntes Schedule-Format: '{s}'. "
        f"Erlaubte Formate:\n"
        f"  HH:MM          — täglich um diese Uhrzeit (z.B. 16:00)\n"
        f"  interval:30m   — alle 30 Minuten\n"
        f"  weekly:0:09:00 — wöchentlich Mo 09:00 (0=Mo..6=So)\n"
        f"  monthly:1:10:00 — monatlich am 1. um 10:00\n"
        f"  '* * * * *'    — Cron-Syntax (Minute Stunde Tag Monat Wochentag)"
    )


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
    instruction: str = "",
    notify: "bool | str | None" = None,
    script: str = "",
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a job definition.

    A job runs either a ``script`` (deterministic, preferred) or an
    ``instruction`` (LLM). ``notify``:
      - True          = always deliver output (empty becomes "erledigt")
      - "output_only" = deliver only when there is output (silent on empty)
      - "error"       = deliver only on failure
      - False         = never deliver on success (failures are always surfaced)

    When ``notify`` is left ``None`` the default depends on the kind: script
    jobs default to "output_only" (stay silent on a quiet day), instruction
    jobs default to True.
    """
    if notify is None:
        notify = "output_only" if script else True
    return {
        "id": f"job-{uuid.uuid4().hex[:8]}",
        "name": name,
        "schedule": schedule,
        "instruction": instruction,
        "script": script,
        "params": params or {},
        "notify": notify,
        "enabled": True,
        "created_at": datetime.now().isoformat(),
        "last_run": "",
        "last_result": "",
    }
