"""
Organizer script – calendar events and tasks.

Usage:
  python organizer.py <subcommand> --user-id <id> --session-dir <dir> [options]

Subcommands:
  add-event, list-events, delete-event
  add-task, list-tasks, complete-task, delete-task
"""

import argparse
import json
import os
import sys
import uuid
from datetime import datetime


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


def _load(path: str):
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _save(path: str, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _out(data) -> None:
    print(json.dumps(data, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Calendar commands
# ---------------------------------------------------------------------------

def cmd_add_event(args) -> None:
    event = {
        "id": str(uuid.uuid4()),
        "title": args.title,
        "start": args.start,
        "end": args.end or "",
        "description": args.description or "",
        "location": args.location or "",
        "created_at": datetime.now().isoformat(),
    }
    path = _calendar_path(args.user_id, args.session_dir)
    events = _load(path)
    events.append(event)
    _save(path, events)
    _out({"success": True, "message": f"Event '{args.title}' added.", "event_id": event["id"]})


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
    task = {
        "id": str(uuid.uuid4()),
        "title": args.title,
        "due_date": args.due_date or "",
        "priority": args.priority or "medium",
        "description": args.description or "",
        "status": "pending",
        "created_at": datetime.now().isoformat(),
    }
    path = _tasks_path(args.user_id, args.session_dir)
    tasks = _load(path)
    tasks.append(task)
    _save(path, tasks)
    _out({"success": True, "message": f"Task '{args.title}' added.", "task_id": task["id"]})


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
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    def _base(p):
        p.add_argument("--user-id", required=True)
        p.add_argument("--session-dir", required=True)

    # add-event
    p = sub.add_parser("add-event")
    _base(p)
    p.add_argument("--title", required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end")
    p.add_argument("--description")
    p.add_argument("--location")

    # list-events
    p = sub.add_parser("list-events")
    _base(p)
    p.add_argument("--limit", type=int)

    # delete-event
    p = sub.add_parser("delete-event")
    _base(p)
    p.add_argument("--event-id", required=True)

    # add-task
    p = sub.add_parser("add-task")
    _base(p)
    p.add_argument("--title", required=True)
    p.add_argument("--due-date")
    p.add_argument("--priority")
    p.add_argument("--description")

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

    args = parser.parse_args()

    # argparse stores --user-id as user_id, --session-dir as session_dir
    dispatch = {
        "add-event": cmd_add_event,
        "list-events": cmd_list_events,
        "delete-event": cmd_delete_event,
        "add-task": cmd_add_task,
        "list-tasks": cmd_list_tasks,
        "complete-task": cmd_complete_task,
        "delete-task": cmd_delete_task,
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
