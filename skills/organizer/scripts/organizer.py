"""
Organizer script – the single entry point for all time/planning operations:
calendar events (with checklists), tasks (with reminders), simple reminders,
and scheduled automation jobs.

Usage:
  python organizer.py <subcommand> --user-id <id> --session-dir <dir> [options]

Subcommands:
  add-event, list-events, delete-event
  add-task, list-tasks, complete-task, delete-task
  add-reminder, list-reminders, delete-reminder
  add-job, list-jobs, delete-job, toggle-job
"""

import argparse
import json
import os
import re
import sys
import uuid
from datetime import datetime, timedelta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _user_dir(user_id: str, session_dir: str) -> str:
    path = os.path.join(session_dir, user_id)
    os.makedirs(path, exist_ok=True)
    return path


def _calendar_path(user_id: str, session_dir: str) -> str:
    d = os.path.join(_user_dir(user_id, session_dir), "calendar")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "events.json")


def _tasks_path(user_id: str, session_dir: str) -> str:
    d = os.path.join(_user_dir(user_id, session_dir), "tasks")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "tasks.json")


def _reminders_path(user_id: str, session_dir: str) -> str:
    return os.path.join(_user_dir(user_id, session_dir), "reminders.json")


def _jobs_path(user_id: str, session_dir: str) -> str:
    d = os.path.join(_user_dir(user_id, session_dir), "automations")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "jobs.json")


def _load(path: str):
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _save(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _out(data) -> None:
    print(json.dumps(data, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Calendar commands
# ---------------------------------------------------------------------------

def cmd_add_event(args) -> None:
    # Parse checklist from JSON string if provided
    checklist = []
    if args.checklist:
        try:
            checklist = json.loads(args.checklist)
        except json.JSONDecodeError:
            _out({"success": False, "error": "Invalid checklist JSON."})
            return

    # Ensure each checklist item has an id and status
    for item in checklist:
        if "id" not in item:
            item["id"] = f"chk-{uuid.uuid4().hex[:8]}"
        if "status" not in item:
            item["status"] = "pending"
        if "result" not in item:
            item["result"] = None

    event = {
        "id": str(uuid.uuid4()),
        "title": args.title,
        "start": args.start,
        "end": args.end or "",
        "description": args.description or "",
        "location": args.location or "",
        "checklist": checklist,
        "created_at": datetime.now().isoformat(),
    }
    path = _calendar_path(args.user_id, args.session_dir)
    events = _load(path)
    events.append(event)
    _save(path, events)

    msg = f"Event '{args.title}' added"
    if checklist:
        msg += f" with {len(checklist)} checklist items"
    _out({"success": True, "message": msg + ".", "event_id": event["id"]})


def cmd_list_events(args) -> None:
    path = _calendar_path(args.user_id, args.session_dir)
    events = _load(path)
    events.sort(key=lambda x: x.get("start", ""), reverse=True)
    limit = args.limit or 10
    _out({"success": True, "events": events[:limit], "total": len(events)})


def cmd_delete_event(args) -> None:
    path = _calendar_path(args.user_id, args.session_dir)
    events = _load(path)
    before = len(events)
    events = [e for e in events if e.get("id") != args.event_id]
    if len(events) == before:
        _out({"success": False, "error": "Event not found."})
        return
    _save(path, events)
    _out({"success": True, "message": "Event deleted.", "remaining": len(events)})


# ---------------------------------------------------------------------------
# Task commands
# ---------------------------------------------------------------------------

def cmd_add_task(args) -> None:
    # Parse reminders from JSON string if provided
    reminders = []
    if args.reminders:
        try:
            reminders = json.loads(args.reminders)
        except json.JSONDecodeError:
            _out({"success": False, "error": "Invalid reminders JSON."})
            return

    # Ensure each reminder has required fields
    for rem in reminders:
        if "fired" not in rem:
            rem["fired"] = False
        if "offset" not in rem:
            rem["offset"] = "-1d"  # default: 1 day before

    task = {
        "id": str(uuid.uuid4()),
        "title": args.title,
        "due_date": args.due_date or "",
        "priority": args.priority or "medium",
        "description": args.description or "",
        "status": "pending",
        "reminders": reminders,
        "created_at": datetime.now().isoformat(),
    }
    path = _tasks_path(args.user_id, args.session_dir)
    tasks = _load(path)
    tasks.append(task)
    _save(path, tasks)

    msg = f"Task '{args.title}' added"
    if reminders:
        msg += f" with {len(reminders)} reminders"
    _out({"success": True, "message": msg + ".", "task_id": task["id"]})


def cmd_list_tasks(args) -> None:
    path = _tasks_path(args.user_id, args.session_dir)
    tasks = _load(path)
    status_filter = args.status or "pending"
    if status_filter != "all":
        tasks = [t for t in tasks if t.get("status") == status_filter]
    tasks.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    limit = args.limit or 10
    _out({"success": True, "tasks": tasks[:limit], "total": len(tasks)})


def cmd_complete_task(args) -> None:
    path = _tasks_path(args.user_id, args.session_dir)
    tasks = _load(path)
    found = False
    for t in tasks:
        if t.get("id") == args.task_id:
            t["status"] = "completed"
            t["completed_at"] = datetime.now().isoformat()
            found = True
            break
    if not found:
        _out({"success": False, "error": "Task not found."})
        return
    _save(path, tasks)
    _out({"success": True, "message": "Task marked as completed."})


def cmd_delete_task(args) -> None:
    path = _tasks_path(args.user_id, args.session_dir)
    tasks = _load(path)
    before = len(tasks)
    tasks = [t for t in tasks if t.get("id") != args.task_id]
    if len(tasks) == before:
        _out({"success": False, "error": "Task not found."})
        return
    _save(path, tasks)
    _out({"success": True, "message": "Task deleted.", "remaining": len(tasks)})


# ---------------------------------------------------------------------------
# Simple reminder commands (replaces the old ReminderTool)
# ---------------------------------------------------------------------------

def _parse_fire_at(fire_at: str) -> datetime:
    """Parse ISO8601 or relative time ('10m', '2h', '1d')."""
    fire_at = fire_at.strip()
    m = re.match(r"^(\d+)\s*(m|min|h|d)$", fire_at.lower())
    if m:
        amount = int(m.group(1))
        unit = m.group(2)
        now = datetime.now()
        if unit in ("m", "min"):
            return now + timedelta(minutes=amount)
        elif unit == "h":
            return now + timedelta(hours=amount)
        elif unit == "d":
            return now + timedelta(days=amount)
    return datetime.fromisoformat(fire_at)


def cmd_add_reminder(args) -> None:
    fire_at_str = args.fire_at
    if not fire_at_str:
        _out({"success": False, "error": "fire_at is required."})
        return
    message = args.message or ""
    if not message:
        _out({"success": False, "error": "message is required."})
        return

    try:
        fire_at = _parse_fire_at(fire_at_str)
    except Exception as e:
        _out({"success": False, "error": f"Invalid fire_at format: {e}"})
        return

    recurrence = (args.recurrence or "none").strip().lower()
    if recurrence not in ("none", "daily", "weekly", "monthly"):
        recurrence = "none"

    reminder = {
        "id": str(uuid.uuid4()),
        "user_id": args.user_id,
        "fire_at": fire_at.isoformat(),
        "message": message,
        "label": args.label or "Reminder",
        "recurrence": recurrence,
        "fired": False,
        "created_at": datetime.now().isoformat(),
    }
    path = _reminders_path(args.user_id, args.session_dir)
    reminders = _load(path)
    reminders.append(reminder)
    _save(path, reminders)
    _out({
        "success": True,
        "message": f"Reminder scheduled for {fire_at.strftime('%d.%m.%Y %H:%M')}",
        "reminder_id": reminder["id"],
    })


def cmd_list_reminders(args) -> None:
    path = _reminders_path(args.user_id, args.session_dir)
    reminders = _load(path)
    pending = [r for r in reminders if not r.get("fired")]
    _out({"success": True, "reminders": pending, "total": len(pending)})


def cmd_delete_reminder(args) -> None:
    path = _reminders_path(args.user_id, args.session_dir)
    reminders = _load(path)
    before = len(reminders)
    reminders = [r for r in reminders if r.get("id") != args.reminder_id]
    if len(reminders) == before:
        _out({"success": False, "error": "Reminder not found."})
        return
    _save(path, reminders)
    _out({"success": True, "message": "Reminder deleted."})


# ---------------------------------------------------------------------------
# Job commands (scheduled automation scripts)
# ---------------------------------------------------------------------------

def cmd_add_job(args) -> None:
    job = {
        "id": f"job-{uuid.uuid4().hex[:8]}",
        "name": args.name,
        "script": args.script,
        "schedule": args.schedule,
        "params": json.loads(args.params) if args.params else {},
        "notify": not args.no_notify,
        "enabled": True,
        "created_at": datetime.now().isoformat(),
        "last_run": "",
        "last_result": "",
    }
    path = _jobs_path(args.user_id, args.session_dir)
    jobs = _load(path)
    jobs.append(job)
    _save(path, jobs)
    _out({"success": True, "message": f"Job '{args.name}' scheduled ({args.schedule}).", "job_id": job["id"]})


def cmd_list_jobs(args) -> None:
    path = _jobs_path(args.user_id, args.session_dir)
    jobs = _load(path)
    _out({"success": True, "jobs": jobs, "total": len(jobs)})


def cmd_delete_job(args) -> None:
    path = _jobs_path(args.user_id, args.session_dir)
    jobs = _load(path)
    before = len(jobs)
    jobs = [j for j in jobs if j.get("id") != args.job_id]
    if len(jobs) == before:
        _out({"success": False, "error": "Job not found."})
        return
    _save(path, jobs)
    _out({"success": True, "message": "Job deleted.", "remaining": len(jobs)})


def cmd_toggle_job(args) -> None:
    path = _jobs_path(args.user_id, args.session_dir)
    jobs = _load(path)
    found = False
    for j in jobs:
        if j.get("id") == args.job_id:
            j["enabled"] = not j.get("enabled", True)
            found = True
            state = "enabled" if j["enabled"] else "disabled"
            break
    if not found:
        _out({"success": False, "error": "Job not found."})
        return
    _save(path, jobs)
    _out({"success": True, "message": f"Job {state}."})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    def _base(p):
        p.add_argument("--user-id", required=True)
        p.add_argument("--session-dir", required=True)

    # add-event (now with --checklist)
    p = sub.add_parser("add-event")
    _base(p)
    p.add_argument("--title", required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end")
    p.add_argument("--description")
    p.add_argument("--location")
    p.add_argument("--checklist", help="JSON array of checklist items")

    # list-events
    p = sub.add_parser("list-events")
    _base(p)
    p.add_argument("--limit", type=int)

    # delete-event
    p = sub.add_parser("delete-event")
    _base(p)
    p.add_argument("--event-id", required=True)

    # add-task (now with --reminders)
    p = sub.add_parser("add-task")
    _base(p)
    p.add_argument("--title", required=True)
    p.add_argument("--due-date")
    p.add_argument("--priority")
    p.add_argument("--description")
    p.add_argument("--reminders", help="JSON array of reminder rules")

    # list-tasks
    p = sub.add_parser("list-tasks")
    _base(p)
    p.add_argument("--status", default="pending")
    p.add_argument("--limit", type=int)

    # complete-task
    p = sub.add_parser("complete-task")
    _base(p)
    p.add_argument("--task-id", required=True)

    # delete-task
    p = sub.add_parser("delete-task")
    _base(p)
    p.add_argument("--task-id", required=True)

    # add-reminder
    p = sub.add_parser("add-reminder")
    _base(p)
    p.add_argument("--fire-at", required=True, help="ISO8601 or relative ('10m', '2h', '1d')")
    p.add_argument("--message", required=True)
    p.add_argument("--label")
    p.add_argument("--recurrence", choices=["none", "daily", "weekly", "monthly"])

    # list-reminders
    p = sub.add_parser("list-reminders")
    _base(p)

    # delete-reminder
    p = sub.add_parser("delete-reminder")
    _base(p)
    p.add_argument("--reminder-id", required=True)

    # add-job
    p = sub.add_parser("add-job")
    _base(p)
    p.add_argument("--name", required=True)
    p.add_argument("--script", required=True)
    p.add_argument("--schedule", required=True, help="'HH:MM' or 'interval:Nm/Nh'")
    p.add_argument("--params", help="JSON object of script params")
    p.add_argument("--no-notify", action="store_true")

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

    args = parser.parse_args()

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
