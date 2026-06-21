"""
Organizer script -- Obsidian-native storage for calendar events and tasks.

Events   -> workspace/calendar/<YYYY-MM-DD> <title>.md  (Full Calendar frontmatter)
Tasks    -> workspace/tasks.md                           (Obsidian Tasks emoji format)
Reminders-> workspace/tasks.md                           (scheduled tasks with clock emoji)
Jobs     -> automations/jobs.json                        (scheduler-internal, not in vault)

Usage:
  python organizer.py <subcommand> --user-id <id> --session-dir <dir> [options]

Subcommands:
  add-event, list-events, delete-event
  add-task, list-tasks, complete-task, delete-task
  add-reminder, list-reminders, delete-reminder
  add-job, list-jobs, delete-job, toggle-job, run-job
"""

import argparse
import hashlib
import json
import os
import re
import sys
import unicodedata
import uuid
from datetime import date, datetime, timedelta
from typing import Optional

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - py<3.9
    ZoneInfo = None  # type: ignore

import yaml

# Add the project root to the Python path so we can import pawlia modules
# __file__ = skills/organizer/scripts/organizer.py -> up 4 levels to project root
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

from pawlia.automation import validate_schedule as _validate_schedule


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _workspace_dir(user_id: str, session_dir: str) -> str:
    path = os.path.join(session_dir, user_id, "workspace")
    os.makedirs(path, exist_ok=True)
    return path


def _session_timezone(user_id: str, session_dir: str) -> Optional[str]:
    """Read the configured IANA timezone from the session config, if any."""
    if not user_id or not session_dir:
        return None
    cfg_path = os.path.join(session_dir, user_id, "config.yaml")
    try:
        with open(cfg_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        return None
    return (data.get("user") or {}).get("timezone") or None


def _local_now(user_id: str, session_dir: str) -> datetime:
    """Current wall-clock time in the user's timezone (naive).

    Reminder times must be stored in the same frame the scheduler compares
    against — the user's local wall clock — not the (often UTC) server clock.
    """
    tz_name = _session_timezone(user_id, session_dir)
    if tz_name and ZoneInfo is not None:
        try:
            return datetime.now(ZoneInfo(tz_name)).replace(tzinfo=None)
        except Exception:
            pass
    return datetime.now()


def _calendar_dir(user_id: str, session_dir: str) -> str:
    path = os.path.join(_workspace_dir(user_id, session_dir), "calendar")
    os.makedirs(path, exist_ok=True)
    return path


def _tasks_path(user_id: str, session_dir: str) -> str:
    return os.path.join(_workspace_dir(user_id, session_dir), "tasks.md")


def _jobs_path(user_id: str, session_dir: str) -> str:
    d = os.path.join(session_dir, user_id, "automations")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "jobs.json")


def _load_json(path: str):
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _save_json(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _out(data) -> None:
    print(json.dumps(data, ensure_ascii=False))
    if isinstance(data, dict) and data.get("success") is False:
        sys.exit(1)


def _strip_quotes(s: str) -> str:
    """Strip surrounding single quotes that cmd.exe leaves intact."""
    if s and len(s) >= 2 and s[0] == "'" and s[-1] == "'":
        return s[1:-1]
    return s


_WEEKDAY_TO_RRULE = {
    "mo": "MO",
    "mon": "MO",
    "monday": "MO",
    "montag": "MO",
    "tu": "TU",
    "tue": "TU",
    "tuesday": "TU",
    "di": "TU",
    "dienstag": "TU",
    "we": "WE",
    "wed": "WE",
    "wednesday": "WE",
    "mi": "WE",
    "mittwoch": "WE",
    "th": "TH",
    "thu": "TH",
    "thursday": "TH",
    "do": "TH",
    "donnerstag": "TH",
    "fr": "FR",
    "fri": "FR",
    "friday": "FR",
    "freitag": "FR",
    "sa": "SA",
    "sat": "SA",
    "saturday": "SA",
    "samstag": "SA",
    "su": "SU",
    "sun": "SU",
    "sunday": "SU",
    "so": "SU",
    "sonntag": "SU",
}

_RRULE_TO_FULLCALENDAR_DAY = {
    "SU": 0,
    "MO": 1,
    "TU": 2,
    "WE": 3,
    "TH": 4,
    "FR": 5,
    "SA": 6,
}


def _parse_date_or_datetime(value: str) -> date | datetime:
    if "T" in value:
        return datetime.fromisoformat(value)
    return date.fromisoformat(value[:10])


def _rrule_weekday_for_start(start: str) -> str:
    parsed = _parse_date_or_datetime(start)
    by_weekday = ["MO", "TU", "WE", "TH", "FR", "SA", "SU"]
    return by_weekday[parsed.weekday()]


def _normalize_recurrence_days(days: str) -> list[str]:
    result: list[str] = []
    for raw in re.split(r"[,\s]+", days or ""):
        token = raw.strip().lower()
        if not token:
            continue
        if token.isdigit():
            fullcalendar_day = int(token)
            reverse = {v: k for k, v in _RRULE_TO_FULLCALENDAR_DAY.items()}
            if fullcalendar_day in reverse:
                result.append(reverse[fullcalendar_day])
            continue
        day = _WEEKDAY_TO_RRULE.get(token)
        if day:
            result.append(day)
    return result


def _parse_weekdays_from_recurrence(recurrence: str) -> list[str] | None:
    """Extract specific weekday codes (MO, TU, ...) from a recurrence string.

    Returns None for plain cadences (daily/weekly/monthly) that carry no named
    weekdays — those remain tasks.md reminders. Weekday-specific recurrences
    (e.g. "mo,mi", "monday,wednesday", "FREQ=WEEKLY;BYDAY=MO,WE") return their
    RRULE day codes so the caller can redirect to a recurring calendar event.
    """
    rec = (recurrence or "").strip()
    if not rec:
        return None

    # RRULE form, e.g. FREQ=WEEKLY;BYDAY=MO,WE
    if rec.upper().startswith("FREQ="):
        byday = _rrule_part(rec, "BYDAY")
        if not byday:
            return None
        days = [
            d.strip().upper() for d in byday.split(",")
            if d.strip().upper() in _RRULE_TO_FULLCALENDAR_DAY
        ]
        return days or None

    # Plain text: scan tokens for weekday names in any supported locale.
    found: list[str] = []
    for token in re.split(r"[,\s/;&|+]+", rec.lower()):
        code = _WEEKDAY_TO_RRULE.get(token)
        if code and code not in found:
            found.append(code)
    return found or None


def _build_rrule(
    recurrence: str,
    start: str,
    recurrence_days: str = "",
    recurrence_until: str = "",
    recurrence_count: int | None = None,
) -> str:
    recurrence = (recurrence or "").strip()
    if not recurrence or recurrence.lower() == "none":
        return ""
    # If already an RRULE string, return as-is
    if recurrence.upper().startswith("FREQ="):
        return recurrence
    recurrence_lower = recurrence.lower()
    if recurrence_lower not in {"daily", "weekly", "monthly", "yearly"}:
        raise ValueError("recurrence must be one of: none, daily, weekly, monthly, yearly")

    parts = [f"FREQ={recurrence_lower.upper()}"]
    if recurrence_lower == "weekly":
        days = _normalize_recurrence_days(recurrence_days) or [_rrule_weekday_for_start(start)]
        parts.append("BYDAY=" + ",".join(days))
    if recurrence_until:
        until = recurrence_until.replace("-", "")
        if "T" in until:
            until = until.replace(":", "")
        parts.append(f"UNTIL={until}")
    if recurrence_count:
        parts.append(f"COUNT={int(recurrence_count)}")
    return ";".join(parts)


def _rrule_part(rrule: str, key: str) -> str:
    prefix = key.upper() + "="
    for part in (rrule or "").split(";"):
        if part.upper().startswith(prefix):
            return part[len(prefix):]
    return ""


def _recurrence_fields_from_rrule(rrule: str, start: str, recurrence_until: str = "") -> dict:
    if not rrule:
        return {}
    fields: dict = {
        "type": "recurring",
        "rrule": rrule,
        "startRecur": start[:10],
    }
    until = (recurrence_until or _rrule_part(rrule, "UNTIL")).replace("-", "")
    if len(until) >= 8:
        fields["endRecur"] = until[:4] + "-" + until[4:6] + "-" + until[6:8]
    byday = _rrule_part(rrule, "BYDAY")
    if byday:
        fields["daysOfWeek"] = [
            _RRULE_TO_FULLCALENDAR_DAY[d]
            for d in byday.split(",")
            if d in _RRULE_TO_FULLCALENDAR_DAY
        ]
    return fields


def _safe_filename(name: str) -> str:
    """Create a filesystem-safe version of a string."""
    # Normalize unicode
    name = unicodedata.normalize("NFC", name)
    # Remove/replace unsafe characters
    name = re.sub(r'[<>:"/\\|?*]', "", name)
    name = name.strip(". ")
    return name[:80] or "event"


# ---------------------------------------------------------------------------
# Event Markdown I/O (Full Calendar format)
# ---------------------------------------------------------------------------

def _event_filename(date_str: str, title: str) -> str:
    """Generate filename: 'YYYY-MM-DD title.md'."""
    # Extract date part (handle full ISO datetime)
    date_part = date_str[:10] if len(date_str) >= 10 else date_str
    safe_title = _safe_filename(title)
    return f"{date_part} {safe_title}.md"


def _event_reminder_to_checklist_item(reminder: dict, title: str) -> dict:
    minutes = reminder.get("minutes_before")
    if minutes in (None, ""):
        raise ValueError("Event reminder is missing minutes_before.")
    minutes = int(minutes)
    message = reminder.get("message") or f"In {minutes} Minuten: {title}"
    return {
        "id": f"rem-{uuid.uuid4().hex[:8]}",
        "trigger": "relative",
        "trigger_offset": f"-{minutes}m",
        "message": message,
        "notify": reminder.get("notify", True),
        "source": "event_reminder",
    }


def _write_event_md(filepath: str, event: dict) -> None:
    """Write an event as a Full Calendar compatible Markdown file."""
    start_str = event["start"]
    # Parse start datetime
    try:
        start_dt = datetime.fromisoformat(start_str)
        date_str = start_dt.strftime("%Y-%m-%d")
        start_time = start_dt.strftime("%H:%M")
    except ValueError:
        date_str = start_str[:10]
        start_time = None

    # Parse end datetime
    end_time = None
    end_date = None
    if event.get("end"):
        try:
            end_dt = datetime.fromisoformat(event["end"])
            end_time = end_dt.strftime("%H:%M")
            end_date_str = end_dt.strftime("%Y-%m-%d")
            if end_date_str != date_str:
                end_date = end_date_str
        except ValueError:
            pass

    all_day = start_time is None or start_time == "00:00" and end_time in (None, "00:00", "23:59")

    recurrence = event.get("recurrence", {}) or {}
    rrule = recurrence.get("rrule") or event.get("rrule") or ""

    # Build frontmatter
    fm = {
        "title": event["title"],
        "date": date_str,
        "allDay": all_day,
        "type": "recurring" if rrule else "single",
    }
    if not all_day and start_time:
        fm["startTime"] = start_time
    if not all_day and end_time:
        fm["endTime"] = end_time
    if end_date:
        fm["endDate"] = end_date
    if event.get("location"):
        fm["location"] = event["location"]
    if event.get("completed"):
        fm["completed"] = event["completed"]
    if event.get("reminders"):
        fm["reminders"] = event["reminders"]
    if rrule:
        fm.update(_recurrence_fields_from_rrule(
            rrule,
            event["start"],
            recurrence.get("until") or "",
        ))

    # Checklist automation config goes into frontmatter (scheduler reads it)
    if event.get("checklist"):
        fm["checklist"] = event["checklist"]

    # Build body
    body_parts = []
    if event.get("description"):
        body_parts.append(event["description"])

    # Human-readable checklist in body
    if event.get("checklist"):
        body_parts.append("\n## Checkliste")
        for item in event["checklist"]:
            label = item.get("message") or item.get("script", "")
            offset = item.get("trigger_offset", "")
            if offset:
                label += f" ({offset})"
            body_parts.append(f"- [ ] {label}")

    content = f"---\n{yaml.dump(fm, allow_unicode=True, default_flow_style=False).rstrip()}\n---\n"
    if body_parts:
        content += "\n" + "\n".join(body_parts) + "\n"

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)


def _read_event_md(filepath: str) -> dict | None:
    """Parse a Full Calendar event .md file into a dict."""
    try:
        with open(filepath, encoding="utf-8") as f:
            text = f.read()
    except Exception:
        return None

    # Parse frontmatter
    if not text.lstrip().startswith("---"):
        return None
    parts = text.split("---", 2)
    if len(parts) < 3:
        return None
    try:
        fm = yaml.safe_load(parts[1]) or {}
    except Exception:
        return None

    body = parts[2].strip()

    # Reconstruct event dict
    date_str = fm.get("date", "")
    start_time = fm.get("startTime", "")
    end_time = fm.get("endTime", "")

    if start_time and date_str:
        start = f"{date_str}T{start_time}:00"
    else:
        start = date_str

    end = ""
    if end_time:
        end_date = fm.get("endDate", date_str)
        end = f"{end_date}T{end_time}:00"

    # Extract description (body minus checklist section)
    description = body
    if "\n## Checkliste" in body:
        description = body.split("\n## Checkliste")[0].strip()

    event = {
        "id": os.path.basename(filepath),
        "title": fm.get("title", ""),
        "start": start,
        "end": end,
        "description": description,
        "location": fm.get("location", ""),
        "reminders": fm.get("reminders", []),
        "checklist": fm.get("checklist", []),
        "type": fm.get("type", "single"),
        "rrule": fm.get("rrule", ""),
        "daysOfWeek": fm.get("daysOfWeek", []),
        "startRecur": fm.get("startRecur", ""),
        "endRecur": fm.get("endRecur", ""),
    }
    return event


def _find_event_file(calendar_dir: str, event_id: str) -> str | None:
    """Find an event file by ID (filename with or without .md)."""
    if not event_id.endswith(".md"):
        event_id += ".md"
    filepath = os.path.join(calendar_dir, event_id)
    if os.path.exists(filepath):
        return filepath
    # Fuzzy: search by partial match
    if os.path.isdir(calendar_dir):
        for f in os.listdir(calendar_dir):
            if event_id.lower() in f.lower():
                return os.path.join(calendar_dir, f)
    return None


# ---------------------------------------------------------------------------
# Tasks Markdown I/O (Obsidian Tasks emoji format)
# ---------------------------------------------------------------------------

_PRIORITY_TO_EMOJI = {
    "highest": "\U0001f53a",  # 🔺
    "high": "\u23eb",         # ⏫
    "medium": "\U0001f53c",   # 🔼
    "low": "\U0001f53d",      # 🔽
    "lowest": "\u23ec",       # ⏬
}

_EMOJI_TO_PRIORITY = {v: k for k, v in _PRIORITY_TO_EMOJI.items()}

# Regex for parsing a task line
_TASK_RE = re.compile(
    r"^- \[([ xX\-])\] (.+)$"
)


def _make_task_id(filepath: str, line_no: int) -> str:
    """Generate a stable, short task ID from filepath + line number via MD5."""
    return hashlib.md5(f"{filepath}:{line_no}".encode()).hexdigest()[:8]


def _task_to_line(task: dict) -> str:
    """Convert a task dict to an Obsidian Tasks emoji line."""
    status = task.get("status", "pending")
    if status == "completed":
        checkbox = "[x]"
    elif status == "cancelled":
        checkbox = "[-]"
    else:
        checkbox = "[ ]"

    parts = [f"- {checkbox} {task['title']}"]

    # Priority emoji
    prio = task.get("priority", "")
    if prio in _PRIORITY_TO_EMOJI:
        parts.append(_PRIORITY_TO_EMOJI[prio])

    # Recurrence
    if task.get("recurrence"):
        parts.append(f"\U0001f501 {task['recurrence']}")  # 🔁

    # Scheduled date (for reminders)
    if task.get("scheduled"):
        parts.append(f"\u23f3 {task['scheduled']}")  # ⏳

    # Start date
    if task.get("start_date"):
        parts.append(f"\U0001f6eb {task['start_date']}")  # 🛫

    # Due date
    if task.get("due_date"):
        parts.append(f"\U0001f4c5 {task['due_date']}")  # 📅

    # Created date
    if task.get("created_at"):
        created = task["created_at"][:10]
        parts.append(f"\u2795 {created}")  # ➕

    # Done date
    if status == "completed" and task.get("completed_at"):
        done = task["completed_at"][:10]
        parts.append(f"\u2705 {done}")  # ✅

    # Cancelled date
    if status == "cancelled" and task.get("cancelled_at"):
        parts.append(f"\u274c {task['cancelled_at'][:10]}")  # ❌

    return " ".join(parts)


def _line_to_task(line: str) -> dict | None:
    """Parse an Obsidian Tasks emoji line into a task dict."""
    m = _TASK_RE.match(line.strip())
    if not m:
        return None

    check = m.group(1)
    rest = m.group(2)

    if check in ("x", "X"):
        status = "completed"
    elif check == "-":
        status = "cancelled"
    else:
        status = "pending"

    task = {"status": status, "is_reminder": False}

    # Extract emojis and their values from the rest
    # Work backwards: split off emoji+value pairs
    title_parts = []
    tokens = rest.split()
    i = 0
    while i < len(tokens):
        token = tokens[i]
        # Check if this token is an emoji marker with a following value
        if token in ("\U0001f53a", "\u23eb", "\U0001f53c", "\U0001f53d", "\u23ec"):
            # Priority (standalone, no value)
            task["priority"] = _EMOJI_TO_PRIORITY.get(token, "")
            i += 1
        elif token == "\U0001f501" and i + 1 < len(tokens):  # 🔁
            # Recurrence: collect all following words until next emoji
            i += 1
            rec_parts = []
            while i < len(tokens) and not _is_task_emoji(tokens[i]):
                rec_parts.append(tokens[i])
                i += 1
            task["recurrence"] = " ".join(rec_parts)
        elif token == "\u23f3" and i + 1 < len(tokens):  # ⏳ scheduled
            task["scheduled"] = tokens[i + 1]
            task["is_reminder"] = True
            i += 2
        elif token == "\U0001f6eb" and i + 1 < len(tokens):  # 🛫 start
            task["start_date"] = tokens[i + 1]
            i += 2
        elif token == "\U0001f4c5" and i + 1 < len(tokens):  # 📅 due
            task["due_date"] = tokens[i + 1]
            i += 2
        elif token == "\u2795" and i + 1 < len(tokens):  # ➕ created
            task["created_at"] = tokens[i + 1]
            i += 2
        elif token == "\u2705" and i + 1 < len(tokens):  # ✅ done
            task["completed_at"] = tokens[i + 1]
            i += 2
        elif token == "\u274c" and i + 1 < len(tokens):  # ❌ cancelled
            task["cancelled_at"] = tokens[i + 1]
            i += 2
        else:
            title_parts.append(token)
            i += 1

    task["title"] = " ".join(title_parts)
    return task


def _is_task_emoji(token: str) -> bool:
    """Check if token is a known Obsidian Tasks emoji marker."""
    return token in (
        "\U0001f53a", "\u23eb", "\U0001f53c", "\U0001f53d", "\u23ec",  # priorities
        "\U0001f501",  # 🔁 recurrence
        "\u23f3",      # ⏳ scheduled
        "\U0001f6eb",  # 🛫 start
        "\U0001f4c5",  # 📅 due
        "\u2795",      # ➕ created
        "\u2705",      # ✅ done
        "\u274c",      # ❌ cancelled
    )


def _read_tasks_md(path: str) -> list[dict]:
    """Read all tasks from a tasks.md file."""
    if not os.path.exists(path):
        return []
    tasks = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            task = _line_to_task(line)
            if task:
                task["id"] = _make_task_id("tasks.md", line_no)
                tasks.append(task)
    return tasks


def _write_tasks_md(path: str, tasks: list[dict]) -> None:
    """Write all tasks to a tasks.md file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = [_task_to_line(t) for t in tasks]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n" if lines else "")


def _find_tasks_in_workspace(workspace_dir: str) -> list[tuple[dict, str]]:
    """Scan all .md files in workspace (excluding tasks.md and calendar/) for task lines.

    Returns list of (task_dict, source_file) tuples.
    """
    return [(t, rel) for t, rel, _ln in _iter_workspace_note_tasks(workspace_dir)]


def _iter_workspace_note_tasks(workspace_dir: str):
    """Yield (task_dict, source_file_relpath, line_no) for every task line in workspace notes."""
    if not os.path.isdir(workspace_dir):
        return
    for root, _dirs, files in os.walk(workspace_dir):
        # Skip calendar/ directory
        if root.endswith("calendar") or "/calendar/" in root:
            continue
        for fname in files:
            if not fname.endswith(".md"):
                continue
            # Skip the canonical tasks.md
            if fname == "tasks.md" and root == workspace_dir:
                continue
            path = os.path.join(root, fname)
            rel = os.path.relpath(path, workspace_dir)
            try:
                with open(path, encoding="utf-8") as f:
                    for line_no, line in enumerate(f):
                        task = _line_to_task(line)
                        if task:
                            task["id"] = _make_task_id(rel, line_no)
                            yield task, rel, line_no
            except Exception:
                continue


def _scan_file_for_id(file_path: str, rel: str, task_id: str) -> int | None:
    """Return the current line number whose ID matches, or None.

    Re-derives the ID for every task line so the result reflects the *current*
    file state (line numbers can shift if other tasks were edited above).
    """
    try:
        with open(file_path, encoding="utf-8") as f:
            for line_no, line in enumerate(f):
                if _make_task_id(rel, line_no) != task_id:
                    continue
                if _line_to_task(line) is not None:
                    return line_no
    except Exception:
        return None
    return None


def _find_task_location_by_id(user_id: str, session_dir: str, task_id: str):
    """Find a task by its stable ID across tasks.md and all workspace note files.

    Returns (file_path, rel, line_no) or (None, None, None) if not found.
    The line number reflects the *current* file state, not the state at
    list-tasks time.
    """
    if not task_id:
        return None, None, None

    tasks_md = _tasks_path(user_id, session_dir)
    if os.path.exists(tasks_md):
        line_no = _scan_file_for_id(tasks_md, "tasks.md", task_id)
        if line_no is not None:
            return tasks_md, "tasks.md", line_no

    workspace = _workspace_dir(user_id, session_dir)
    if os.path.isdir(workspace):
        for root, _dirs, files in os.walk(workspace):
            if root.endswith("calendar") or "/calendar/" in root:
                continue
            for fname in files:
                if not fname.endswith(".md"):
                    continue
                if fname == "tasks.md" and root == workspace:
                    continue
                path = os.path.join(root, fname)
                rel = os.path.relpath(path, workspace)
                line_no = _scan_file_for_id(path, rel, task_id)
                if line_no is not None:
                    return path, rel, line_no

    return None, None, None


def _toggle_complete_in_file(file_path: str, line_no: int) -> None:
    """Toggle the [ ] checkbox on the given line of the file to [x] (idempotent)."""
    with open(file_path, encoding="utf-8") as f:
        lines = f.readlines()
    if line_no < 0 or line_no >= len(lines):
        raise ValueError(f"Line {line_no} not found in {file_path}")
    stripped = lines[line_no].rstrip("\n")
    m = _TASK_RE.match(stripped)
    if not m:
        raise ValueError(f"Line {line_no} of {file_path} is not a task line")
    if m.group(1).lower() == "x":
        return
    rest = m.group(2)
    if "\u2705" not in rest:
        today = datetime.now().strftime("%Y-%m-%d")
        rest = f"{rest} \u2705 {today}"
    lines[line_no] = f"- [x] {rest}\n"
    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def _delete_line_in_file(file_path: str, line_no: int) -> None:
    """Remove the given line from the file."""
    with open(file_path, encoding="utf-8") as f:
        lines = f.readlines()
    if line_no < 0 or line_no >= len(lines):
        raise ValueError(f"Line {line_no} not found in {file_path}")
    lines.pop(line_no)
    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(lines)


# ---------------------------------------------------------------------------
# Calendar commands
# ---------------------------------------------------------------------------

def cmd_add_event(args) -> None:
    checklist = []
    if args.checklist:
        try:
            checklist = json.loads(_strip_quotes(args.checklist))
        except json.JSONDecodeError:
            _out({"success": False, "error": "Invalid checklist JSON."})
            return

    reminders = []
    if args.reminders:
        try:
            reminders = json.loads(_strip_quotes(args.reminders))
        except json.JSONDecodeError:
            _out({"success": False, "error": "Invalid reminders JSON."})
            return

    for item in checklist:
        if "id" not in item:
            item["id"] = f"chk-{uuid.uuid4().hex[:8]}"

    try:
        reminder_checklist = [_event_reminder_to_checklist_item(rem, args.title) for rem in reminders]
    except (TypeError, ValueError) as e:
        _out({"success": False, "error": str(e)})
        return

    checklist.extend(reminder_checklist)

    try:
        rrule = _build_rrule(
            args.recurrence or "none",
            args.start,
            args.recurrence_days or "",
            args.recurrence_until or "",
            args.recurrence_count,
        )
    except ValueError as e:
        _out({"success": False, "error": str(e)})
        return

    event = {
        "title": args.title,
        "start": args.start,
        "end": args.end or "",
        "description": args.description or "",
        "location": args.location or "",
        "reminders": reminders,
        "checklist": checklist,
    }
    if rrule:
        event["recurrence"] = {"rrule": rrule, "until": args.recurrence_until or ""}

    cal_dir = _calendar_dir(args.user_id, args.session_dir)
    filename = _event_filename(args.start, args.title)
    filepath = os.path.join(cal_dir, filename)

    # Avoid overwriting: append number if needed
    if os.path.exists(filepath):
        base, ext = os.path.splitext(filename)
        n = 2
        while os.path.exists(os.path.join(cal_dir, f"{base} {n}{ext}")):
            n += 1
        filename = f"{base} {n}{ext}"
        filepath = os.path.join(cal_dir, filename)

    _write_event_md(filepath, event)

    msg = f"Event '{args.title}' added"
    if rrule:
        msg += " as recurring event"
    if checklist:
        msg += f" with {len(checklist)} checklist items"
    if reminders:
        msg += f" and {len(reminders)} reminders"
    _out({"success": True, "message": msg + ".", "event_id": filename})


def cmd_list_events(args) -> None:
    cal_dir = _calendar_dir(args.user_id, args.session_dir)
    events = []
    if os.path.isdir(cal_dir):
        for f in sorted(os.listdir(cal_dir), reverse=True):
            if f.endswith(".md"):
                ev = _read_event_md(os.path.join(cal_dir, f))
                if ev:
                    events.append(ev)
    limit = args.limit or 10
    _out({"success": True, "events": events[:limit], "total": len(events)})


def cmd_delete_event(args) -> None:
    cal_dir = _calendar_dir(args.user_id, args.session_dir)
    filepath = _find_event_file(cal_dir, args.event_id)
    if not filepath:
        _out({"success": False, "error": f"Event not found: {args.event_id}"})
        return
    os.remove(filepath)
    _out({"success": True, "message": f"Event deleted: {os.path.basename(filepath)}"})


# ---------------------------------------------------------------------------
# Task commands
# ---------------------------------------------------------------------------

def _find_project_note(workspace_dir: str, project: str) -> str | None:
    """Search for a project note by name in workspace/wiki/topics/ and workspace root.

    Returns the file path relative to workspace_dir, or None.
    """
    candidates: list[str] = []
    slug = project.lower().replace(" ", "-").replace("_", "-")
    candidates.append(os.path.join("wiki", "topics", f"{slug}.md"))
    candidates.append(f"{slug}.md")
    candidates.append(os.path.join("wiki", "topics", f"{project}.md"))
    candidates.append(f"{project}.md")
    # Also try with underscores
    uslug = project.lower().replace(" ", "_").replace("-", "_")
    candidates.append(os.path.join("wiki", "topics", f"{uslug}.md"))
    candidates.append(f"{uslug}.md")

    for rel in candidates:
        abspath = os.path.join(workspace_dir, rel)
        if os.path.isfile(abspath):
            return rel
    return None


def cmd_add_task(args) -> None:
    task = {
        "title": args.title,
        "due_date": args.due_date or "",
        "priority": args.priority or "",
        "status": "pending",
        "created_at": datetime.now().strftime("%Y-%m-%d"),
    }

    workspace = _workspace_dir(args.user_id, args.session_dir)
    task_line = _task_to_line(task)

    # If --project is given, try to append to that project note
    if args.project:
        note_rel = _find_project_note(workspace, args.project)
        if note_rel:
            note_path = os.path.join(workspace, note_rel)
            with open(note_path, "a", encoding="utf-8") as f:
                f.write(f"\n{task_line}")
            _out({
                "success": True,
                "message": f"Task '{args.title}' added to {note_rel}.",
                "task_id": args.title,
                "project_note": note_rel,
            })
            return

    # Fallback: append to tasks.md
    path = _tasks_path(args.user_id, args.session_dir)
    tasks = _read_tasks_md(path)
    tasks.append(task)
    _write_tasks_md(path, tasks)

    # Re-read to get the stable task ID
    tasks = _read_tasks_md(path)
    task_id = tasks[-1]["id"] if tasks else args.title

    # Store task reminders in scheduler_state.json (outside workspace)
    if args.reminders:
        try:
            reminders = json.loads(_strip_quotes(args.reminders))
        except json.JSONDecodeError:
            reminders = []
        if reminders:
            state_path = os.path.join(args.session_dir, args.user_id, "scheduler_state.json")
            state = {}
            if os.path.exists(state_path):
                try:
                    with open(state_path, encoding="utf-8") as f:
                        state = json.load(f)
                except Exception:
                    pass
            task_reminders = state.setdefault("task_reminders", {})
            task_key = args.title.lower()
            for rem in reminders:
                if "fired" not in rem:
                    rem["fired"] = False
            task_reminders[task_key] = reminders
            os.makedirs(os.path.dirname(state_path), exist_ok=True)
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, ensure_ascii=False)

    _out({"success": True, "message": f"Task '{args.title}' added.", "task_id": task_id})


def cmd_list_tasks(args) -> None:
    path = _tasks_path(args.user_id, args.session_dir)
    tasks = _read_tasks_md(path)
    workspace = _workspace_dir(args.user_id, args.session_dir)

    # Also scan project notes for tasks
    note_tasks = _find_tasks_in_workspace(workspace)
    note_count = len(note_tasks)
    for task, source in note_tasks:
        task["source"] = source
        tasks.append(task)

    # Filter out reminders
    tasks = [t for t in tasks if not t.get("is_reminder")]

    status_filter = args.status or "pending"
    if status_filter != "all":
        tasks = [t for t in tasks if t.get("status") == status_filter]

    # Filter by project if specified
    if args.project:
        project_slug = args.project.lower().replace(" ", "-").replace("_", "-")
        filtered = []
        for t in tasks:
            src = t.get("source", "")
            if project_slug in src.lower().replace("\\", "/"):
                filtered.append(t)
        tasks = filtered

    limit = args.limit or 10
    # Format for display
    display = []
    for t in tasks[:limit]:
        item = {
            "id": t.get("id", ""),
            "title": t["title"],
            "status": t["status"],
            "due_date": t.get("due_date", ""),
            "priority": t.get("priority", ""),
        }
        if t.get("source"):
            item["source"] = t["source"]
        display.append(item)
    _out({"success": True, "tasks": display, "total": len(tasks), "note_tasks": note_count})


def cmd_complete_task(args) -> None:
    file_path, _rel, line_no = _find_task_location_by_id(
        args.user_id, args.session_dir, args.task_id
    )
    if file_path is not None:
        _toggle_complete_in_file(file_path, line_no)
        _out({"success": True, "message": "Task marked as completed."})
        return

    # Fall back to title substring match in tasks.md (legacy).
    path = _tasks_path(args.user_id, args.session_dir)
    tasks = _read_tasks_md(path)
    for t in tasks:
        if (t.get("title", "").lower() == args.task_id.lower()
                or args.task_id.lower() in t.get("title", "").lower()):
            t["status"] = "completed"
            t["completed_at"] = datetime.now().strftime("%Y-%m-%d")
            _write_tasks_md(path, tasks)
            _out({"success": True, "message": "Task marked as completed."})
            return

    _out({"success": False, "error": f"Task not found: {args.task_id}"})


def cmd_delete_task(args) -> None:
    file_path, _rel, line_no = _find_task_location_by_id(
        args.user_id, args.session_dir, args.task_id
    )
    if file_path is not None:
        _delete_line_in_file(file_path, line_no)
        if file_path == _tasks_path(args.user_id, args.session_dir):
            remaining = sum(1 for _ in _read_tasks_md(file_path))
        else:
            with open(file_path, encoding="utf-8") as f:
                remaining = sum(1 for line in f if _line_to_task(line))
        _out({"success": True, "message": "Task deleted.", "remaining": remaining})
        return

    # Fall back to title substring match in tasks.md (legacy).
    path = _tasks_path(args.user_id, args.session_dir)
    tasks = _read_tasks_md(path)
    before = len(tasks)
    tasks = [t for t in tasks if not (
        t.get("title", "").lower() == args.task_id.lower()
        or args.task_id.lower() in t.get("title", "").lower()
    )]
    if len(tasks) == before:
        _out({"success": False, "error": f"Task not found: {args.task_id}"})
        return
    _write_tasks_md(path, tasks)
    _out({"success": True, "message": "Task deleted.", "remaining": len(tasks)})


# ---------------------------------------------------------------------------
# Reminder commands (stored as scheduled tasks in tasks.md)
# ---------------------------------------------------------------------------

def _parse_fire_at(fire_at: str, now: Optional[datetime] = None) -> datetime:
    """Parse ISO8601 or relative time ('10m', '2h', '1d').

    Relative offsets are added to *now*, which should be the user's local
    wall-clock time so the stored fire-at matches the scheduler's frame.
    """
    fire_at = fire_at.strip()
    m = re.match(r"^(\d+)\s*(m|min|h|d)$", fire_at.lower())
    if m:
        amount = int(m.group(1))
        unit = m.group(2)
        if now is None:
            now = datetime.now()
        if unit in ("m", "min"):
            return now + timedelta(minutes=amount)
        elif unit == "h":
            return now + timedelta(hours=amount)
        elif unit == "d":
            return now + timedelta(days=amount)
    return datetime.fromisoformat(fire_at)


def cmd_add_reminder(args) -> None:
    if not args.fire_at:
        _out({"success": False, "error": "fire_at is required."})
        return
    if not args.message:
        _out({"success": False, "error": "message is required."})
        return

    try:
        now = _local_now(args.user_id, args.session_dir)
        fire_at = _parse_fire_at(args.fire_at, now=now)
    except Exception as e:
        _out({"success": False, "error": f"Invalid fire_at format: {e}"})
        return

    # Weekday-specific recurrence (e.g. "mo,mi", "monday,wednesday",
    # "FREQ=WEEKLY;BYDAY=MO,WE"). The tasks.md reminder engine only understands
    # daily/weekly/monthly cadences and would silently turn anything else into a
    # daily reminder. Redirect these to a recurring calendar event whose
    # reminders fire on every matching weekday via the scheduler.
    weekdays = _parse_weekdays_from_recurrence(args.recurrence)
    if weekdays:
        rrule = f"FREQ=WEEKLY;BYDAY={','.join(weekdays)}"
        label = args.label or "Reminder"
        start_iso = fire_at.strftime("%Y-%m-%dT%H:%M:%S")
        event = {
            "title": label,
            "start": start_iso,
            "end": "",
            "description": args.message,
            "location": "",
            "reminders": [{"minutes_before": 0, "message": args.message, "notify": True}],
            "checklist": [],
            "recurrence": {"rrule": rrule, "until": ""},
        }
        cal_dir = _calendar_dir(args.user_id, args.session_dir)
        filename = _event_filename(start_iso, label)
        filepath = os.path.join(cal_dir, filename)
        if os.path.exists(filepath):
            base, ext = os.path.splitext(filename)
            n = 2
            while os.path.exists(os.path.join(cal_dir, f"{base} {n}{ext}")):
                n += 1
            filename = f"{base} {n}{ext}"
            filepath = os.path.join(cal_dir, filename)
        _write_event_md(filepath, event)
        _out({
            "success": True,
            "message": (
                f"Reminder als wiederkehrender Termin angelegt "
                f"(Wochentage: {','.join(weekdays)}, "
                f"{fire_at.strftime('%H:%M')} Uhr)."
            ),
            "event_id": filename,
            "redirected_to_event": True,
        })
        return

    recurrence = (args.recurrence or "").strip()
    recurrence_str = ""
    if recurrence and recurrence.lower() not in ("none", "once", "no", "never"):
        # Check if it's an RRULE string
        if recurrence.upper().startswith("FREQ="):
            recurrence_str = recurrence
        else:
            recurrence_lower = recurrence.lower()
            if recurrence_lower == "daily":
                recurrence_str = "every day"
            elif recurrence_lower == "weekly":
                recurrence_str = "every week"
            elif recurrence_lower == "monthly":
                recurrence_str = "every month"
            else:
                recurrence_str = f"every {recurrence_lower.rstrip('ly')}"

    label = args.label or "Reminder"
    title = f"\U0001f514 {label}: {args.message}"  # 🔔

    task = {
        "title": title,
        "scheduled": fire_at.strftime("%Y-%m-%dT%H:%M"),
        "status": "pending",
        "created_at": datetime.now().strftime("%Y-%m-%d"),
        "is_reminder": True,
    }
    if recurrence_str:
        task["recurrence"] = recurrence_str

    path = _tasks_path(args.user_id, args.session_dir)
    tasks = _read_tasks_md(path)
    tasks.append(task)
    _write_tasks_md(path, tasks)

    _out({
        "success": True,
        "message": f"Reminder scheduled for {fire_at.strftime('%d.%m.%Y %H:%M')}",
        "reminder_id": title,
    })


def cmd_list_reminders(args) -> None:
    path = _tasks_path(args.user_id, args.session_dir)
    tasks = _read_tasks_md(path)
    workspace = _workspace_dir(args.user_id, args.session_dir)

    note_tasks = _find_tasks_in_workspace(workspace)
    for task, source in note_tasks:
        task["source"] = source
        tasks.append(task)

    reminders = [t for t in tasks if t.get("is_reminder") and t.get("status") == "pending"]
    display = []
    for r in reminders:
        item = {
            "title": r["title"],
            "scheduled": r.get("scheduled", ""),
            "recurrence": r.get("recurrence", ""),
        }
        if r.get("source"):
            item["source"] = r["source"]
        display.append(item)
    _out({"success": True, "reminders": display, "total": len(reminders)})


def cmd_delete_reminder(args) -> None:
    path = _tasks_path(args.user_id, args.session_dir)
    tasks = _read_tasks_md(path)
    before = len(tasks)
    tasks = [t for t in tasks if not (
        t.get("is_reminder")
        and args.reminder_id.lower() in t.get("title", "").lower()
    )]
    if len(tasks) == before:
        _out({"success": False, "error": f"Reminder not found: {args.reminder_id}"})
        return
    _write_tasks_md(path, tasks)
    _out({"success": True, "message": "Reminder deleted."})


# ---------------------------------------------------------------------------
# Job commands (scheduler-internal JSON, not in Obsidian vault)
# ---------------------------------------------------------------------------

def cmd_add_job(args) -> None:
    ok, err = _validate_schedule(args.schedule)
    if not ok:
        _out({"success": False, "error": f"Ungültiger Schedule: {err}"})
        return
    if not args.script and not args.instruction:
        _out({"success": False, "error": "Job braucht --script oder --instruction."})
        return
    params: dict = {}
    if args.params:
        try:
            parsed = json.loads(args.params)
            if not isinstance(parsed, dict):
                raise ValueError("params must be a JSON object")
            params = parsed
        except (ValueError, TypeError) as e:
            _out({"success": False, "error": f"Ungültige --params (JSON-Objekt erwartet): {e}"})
            return
    if args.no_notify:
        notify: bool | str = False
    elif args.notify_on_error:
        notify = "error"
    elif args.notify_on_output:
        notify = "output_only"
    else:
        # Script jobs stay silent on a quiet day; instruction jobs always deliver.
        notify = "output_only" if args.script else True
    job = {
        "id": f"job-{uuid.uuid4().hex[:8]}",
        "name": args.name,
        "schedule": args.schedule,
        "instruction": args.instruction,
        "script": args.script,
        "params": params,
        "notify": notify,
        "enabled": True,
        "created_at": datetime.now().isoformat(),
        "last_run": "",
        "last_result": "",
    }
    path = _jobs_path(args.user_id, args.session_dir)
    jobs = _load_json(path)
    jobs.append(job)
    _save_json(path, jobs)
    _out({"success": True, "message": f"Job '{args.name}' scheduled ({args.schedule}).", "job_id": job["id"]})


def cmd_list_jobs(args) -> None:
    path = _jobs_path(args.user_id, args.session_dir)
    jobs = _load_json(path)
    _out({"success": True, "jobs": jobs, "total": len(jobs)})


def cmd_delete_job(args) -> None:
    path = _jobs_path(args.user_id, args.session_dir)
    jobs = _load_json(path)
    before = len(jobs)
    jobs = [j for j in jobs if j.get("id") != args.job_id]
    if len(jobs) == before:
        _out({"success": False, "error": "Job not found."})
        return
    _save_json(path, jobs)
    _out({"success": True, "message": "Job deleted.", "remaining": len(jobs)})


def cmd_toggle_job(args) -> None:
    path = _jobs_path(args.user_id, args.session_dir)
    jobs = _load_json(path)
    found = False
    state = ""
    for j in jobs:
        if j.get("id") == args.job_id:
            j["enabled"] = not j.get("enabled", True)
            found = True
            state = "enabled" if j["enabled"] else "disabled"
            break
    if not found:
        _out({"success": False, "error": "Job not found."})
        return
    _save_json(path, jobs)
    _out({"success": True, "message": f"Job {state}."})


def cmd_run_job(args) -> None:
    path = _jobs_path(args.user_id, args.session_dir)
    jobs = _load_json(path)
    job = None
    for j in jobs:
        if j.get("id") == args.job_id:
            job = j
            break
    if not job:
        _out({"success": False, "error": "Job not found."})
        return
    job["force_run"] = True
    _save_json(path, jobs)
    _out({
        "success": True,
        "message": f"Job '{job.get('name', args.job_id)}' wird beim nächsten Scheduler-Tick ausgeführt.",
        "instruction": job.get("instruction", ""),
    })


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    def _base(p):
        p.add_argument("--user-id", default=os.environ.get("PAWLIA_USER_ID"))
        p.add_argument("--session-dir", default=os.environ.get("PAWLIA_SESSION_DIR"))

    # add-event
    p = sub.add_parser("add-event")
    _base(p)
    p.add_argument("--title", required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end")
    p.add_argument("--description")
    p.add_argument("--location")
    p.add_argument("--checklist", help="JSON array of checklist items")
    p.add_argument("--reminders", help="JSON array of event reminders")
    p.add_argument("--recurrence", default="none", help="none, daily, weekly, monthly, yearly, or RRULE string (e.g. FREQ=WEEKLY;BYDAY=MO,WE)")
    p.add_argument("--recurrence-days", help="Weekly days, e.g. 'TU,TH' or 'dienstag donnerstag'")
    p.add_argument("--recurrence-until", help="Last recurrence date as YYYY-MM-DD")
    p.add_argument("--recurrence-count", type=int, help="Maximum number of occurrences")

    # list-events
    p = sub.add_parser("list-events")
    _base(p)
    p.add_argument("--limit", type=int)

    # delete-event
    p = sub.add_parser("delete-event")
    _base(p)
    p.add_argument("--event-id", required=True, help="Event filename (with or without .md)")

    # add-task
    p = sub.add_parser("add-task")
    _base(p)
    p.add_argument("--title", required=True)
    p.add_argument("--due-date")
    p.add_argument("--priority", choices=["highest", "high", "medium", "low", "lowest"])
    p.add_argument("--description")
    p.add_argument("--project", help="Project name — appends task to matching project note if found")
    # Note: --reminders is accepted but task reminders are now managed via
    # scheduler_state.json, not embedded in the task line itself.
    p.add_argument("--reminders", help="JSON array of reminder rules (stored in scheduler state)")

    # list-tasks
    p = sub.add_parser("list-tasks")
    _base(p)
    p.add_argument("--status", default="pending")
    p.add_argument("--limit", type=int)
    p.add_argument("--project", help="Filter by project note name")

    # complete-task
    p = sub.add_parser("complete-task")
    _base(p)
    p.add_argument("--task-id", required=True, help="Task ID (from list-tasks) or title/substring")

    # delete-task
    p = sub.add_parser("delete-task")
    _base(p)
    p.add_argument("--task-id", required=True, help="Task ID (from list-tasks) or title/substring")

    # add-reminder
    p = sub.add_parser("add-reminder")
    _base(p)
    p.add_argument("--fire-at", required=True, help="ISO8601 or relative ('10m', '2h', '1d')")
    p.add_argument("--message", required=True)
    p.add_argument("--label")
    p.add_argument("--recurrence", help="none, daily, weekly, monthly, or RRULE string (e.g. FREQ=WEEKLY;BYDAY=MO,WE)")

    # list-reminders
    p = sub.add_parser("list-reminders")
    _base(p)

    # delete-reminder
    p = sub.add_parser("delete-reminder")
    _base(p)
    p.add_argument("--reminder-id", required=True, help="Reminder title (or substring)")

    # add-job
    p = sub.add_parser("add-job")
    _base(p)
    p.add_argument("--name", required=True)
    p.add_argument("--instruction", default="",
                   help="Natural-language instruction run via the LLM (trivial jobs)")
    p.add_argument("--script", default="",
                   help="Script in workspace/.scripts/ run deterministically (preferred)")
    p.add_argument("--params", default="",
                   help="JSON dict passed to the script as AUTOMATION_PARAMS")
    p.add_argument("--schedule", required=True)
    p.add_argument("--no-notify", action="store_true",
                   help="Never deliver output (failures are still surfaced)")
    p.add_argument("--notify-on-error", action="store_true",
                   help="Deliver only on failure")
    p.add_argument("--notify-on-output", action="store_true",
                   help="Deliver only when there is output (silent on empty) — default for --script")

    # list-jobs
    p = sub.add_parser("list-jobs")
    _base(p)

    # delete-job
    p = sub.add_parser("delete-job")
    _base(p)
    p.add_argument("--job-id", required=True)

    # toggle-job
    p = sub.add_parser("toggle-job")
    _base(p)
    p.add_argument("--job-id", required=True)

    # run-job
    p = sub.add_parser("run-job")
    _base(p)
    p.add_argument("--job-id", required=True, help="Job ID to trigger manually")

    args = parser.parse_args()

    if not args.user_id or not args.session_dir:
        print(json.dumps({"success": False, "error": "user-id and session-dir required (via args or env vars)."}))
        sys.exit(1)

    dispatch = {
        "add-event": cmd_add_event,
        "list-events": cmd_list_events,
        "delete-event": cmd_delete_event,
        "add-task": cmd_add_task,
        "list-tasks": cmd_list_tasks,
        "complete-task": cmd_complete_task,
        "delete-task": cmd_delete_task,
        "add-reminder": cmd_add_reminder,
        "list-reminders": cmd_list_reminders,
        "delete-reminder": cmd_delete_reminder,
        "add-job": cmd_add_job,
        "list-jobs": cmd_list_jobs,
        "delete-job": cmd_delete_job,
        "toggle-job": cmd_toggle_job,
        "run-job": cmd_run_job,
    }

    fn = dispatch.get(args.cmd)
    if not fn:
        print(json.dumps({"success": False, "error": f"Unknown subcommand: {args.cmd}"}))
        sys.exit(1)

    try:
        fn(args)
    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
