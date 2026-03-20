---
name: organizer
description: Personal planner for reminders, calendar events, tasks, and scheduled automations. Use when the user wants to be reminded of something, plan a personal event/appointment, manage personal tasks, or schedule recurring automation scripts.
license: MIT
metadata:
  author: Christian Ulrich
  version: "2.0"
---

# Organizer

The single entry point for ALL time-related and planning operations:
- **Simple reminders** ("remind me in 10 min about the pizza")
- **Calendar events** with automation checklists
- **Tasks** with automatic reminder rules
- **Scheduled jobs** (recurring script execution)

## Instructions

Run the script with the appropriate subcommand. Always pass `--user-id` and `--session-dir` from the context.

### Simple Reminders

For quick reminders like "remind me in 10 minutes" or "remind me tomorrow at 9".

Add a reminder:
```
python <scripts_dir>/organizer.py add-reminder --user-id <user_id> --session-dir "<session_dir>" --fire-at "<time>" --message "<message>" [--label "<label>"] [--recurrence none|daily|weekly|monthly]
```

- `--fire-at`: ISO8601 datetime or relative time: `"10m"`, `"2h"`, `"1d"`
- `--recurrence`: optional, for repeating reminders

List pending reminders:
```
python <scripts_dir>/organizer.py list-reminders --user-id <user_id> --session-dir "<session_dir>"
```

Delete a reminder:
```
python <scripts_dir>/organizer.py delete-reminder --user-id <user_id> --session-dir "<session_dir>" --reminder-id "<id>"
```

### Calendar

Add an event:
```
python <scripts_dir>/organizer.py add-event --user-id <user_id> --session-dir "<session_dir>" --title "<title>" --start "<ISO8601>" [--end "<ISO8601>"] [--description "<desc>"] [--location "<loc>"] [--checklist '<JSON>']
```

The `--checklist` parameter accepts a JSON array of automation items. Each item can reference a script that the system executes automatically at the right time — no LLM needed at runtime.

**Checklist item format:**
```json
[
  {
    "script": "route_plan.py",
    "trigger": "relative",
    "trigger_offset": "-90m",
    "params": {"from": "home", "to": "Destination"},
    "notify": true
  },
  {
    "script": "",
    "trigger": "relative",
    "trigger_offset": "-1d",
    "message": "Morgen: {title} in {location}. Unterlagen vorbereiten!"
  },
  {
    "script": "check_traffic.py",
    "trigger": "relative",
    "trigger_offset": "-60m",
    "params": {"destination": "Magdeburg"},
    "notify": true
  }
]
```

- `trigger`: `"relative"` (offset from event start), `"on_create"` (immediately), `"absolute"` (fixed time)
- `trigger_offset`: e.g. `"-2h"`, `"-1d"`, `"-30m"` (negative = before event)
- `script`: path to automation script (resolved from user automations dir or global scripts)
- `message`: plain text notification (when no script needed)
- `params`: passed to the script as AUTOMATION_PARAMS env var (JSON)
- `notify`: whether to send the result to the user (default: true)

**IMPORTANT:** When creating events with a location, ALWAYS create a checklist with appropriate reminders and preparation steps. Think about what the user needs (route, departure time, documents, etc.) and plan it as checklist items.

List events:
```
python <scripts_dir>/organizer.py list-events --user-id <user_id> --session-dir "<session_dir>" [--limit <n>]
```

Delete an event:
```
python <scripts_dir>/organizer.py delete-event --user-id <user_id> --session-dir "<session_dir>" --event-id "<id>"
```

### Tasks

Add a task:
```
python <scripts_dir>/organizer.py add-task --user-id <user_id> --session-dir "<session_dir>" --title "<title>" [--due-date "YYYY-MM-DD"] [--priority high|medium|low] [--description "<desc>"] [--reminders '<JSON>']
```

The `--reminders` parameter accepts a JSON array of reminder rules. The system fires them automatically based on the due date.

**Reminder format:**
```json
[
  {"offset": "-3d", "message": "In 3 Tagen fällig: {title}"},
  {"offset": "-1d", "message": "Morgen fällig: {title}"},
  {"offset": "-2h", "message": "In 2 Stunden fällig: {title}"}
]
```

- `offset`: relative to due_date, e.g. `"-3d"`, `"-1d"`, `"-2h"`
- `message`: notification text (`{title}` and `{due_date}` are replaced)

**IMPORTANT:** When creating tasks with a due date, ALWAYS add appropriate reminders based on priority:
- high: 3d, 1d, 2h before
- medium: 1d, 2h before
- low: 2h before

List tasks:
```
python <scripts_dir>/organizer.py list-tasks --user-id <user_id> --session-dir "<session_dir>" [--status pending|completed|all] [--limit <n>]
```

Complete a task:
```
python <scripts_dir>/organizer.py complete-task --user-id <user_id> --session-dir "<session_dir>" --task-id "<id>"
```

Delete a task:
```
python <scripts_dir>/organizer.py delete-task --user-id <user_id> --session-dir "<session_dir>" --task-id "<id>"
```

### Scheduled Jobs (Automation)

For recurring automated tasks ("every day at 16:00 create a report"), first write the automation script, then register it as a job.

Add a job:
```
python <scripts_dir>/organizer.py add-job --user-id <user_id> --session-dir "<session_dir>" --name "<name>" --script "<script_path>" --schedule "<schedule>" [--params '<JSON>'] [--no-notify]
```

**Schedule formats:**
- `"16:00"` — daily at 16:00
- `"interval:30m"` — every 30 minutes
- `"interval:2h"` — every 2 hours

**Workflow for creating a scheduled job:**
1. Write the automation script to `session/<user_id>/automations/<script_name>.py`
2. The script receives params via `AUTOMATION_PARAMS` env var
3. The script's stdout becomes the notification message
4. Register the job with `add-job`

List jobs:
```
python <scripts_dir>/organizer.py list-jobs --user-id <user_id> --session-dir "<session_dir>"
```

Delete a job:
```
python <scripts_dir>/organizer.py delete-job --user-id <user_id> --session-dir "<session_dir>" --job-id "<id>"
```

Toggle a job (enable/disable):
```
python <scripts_dir>/organizer.py toggle-job --user-id <user_id> --session-dir "<session_dir>" --job-id "<id>"
```

## Output

The script outputs JSON. Parse it and report the result naturally to the user. On error, report the `error` field.
