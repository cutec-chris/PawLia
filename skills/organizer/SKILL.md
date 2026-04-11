---
name: organizer
description: "Personal planner for reminders, calendar events, and tasks. Use when the user wants to: be reminded of something, plan an event/appointment, or manage personal tasks."
license: MIT
metadata:
  author: Christian Ulrich
  version: "3.0"
---

# Organizer

Obsidian-native storage for all time-related and planning operations.
The workspace directory functions as an Obsidian vault:

- **Events** → `workspace/calendar/<YYYY-MM-DD> <title>.md` (Full Calendar plugin format)
- **Tasks** → `workspace/tasks.md` (Obsidian Tasks emoji format)
- **Reminders** → `workspace/tasks.md` (scheduled tasks with 🔔 prefix)
- **Jobs** → `automations/jobs.json` (scheduler-internal, not in vault)

## Instructions

Run the script with the appropriate subcommand. The `--user-id` and `--session-dir` arguments are automatically provided via environment variables — do NOT pass them manually.

### Simple Reminders

For quick reminders like "remind me in 10 minutes" or "remind me tomorrow at 9".
Reminders are stored as scheduled tasks in `tasks.md` with a 🔔 prefix.

Add a reminder:
```
python <scripts_dir>/organizer.py add-reminder --fire-at "<time>" --message "<message>" [--label "<label>"] [--recurrence none|daily|weekly|monthly]
```

- `--fire-at`: ISO8601 datetime or relative time: `"10m"`, `"2h"`, `"1d"`
- `--recurrence`: optional, for repeating reminders (uses 🔁 emoji)

List pending reminders:
```
python <scripts_dir>/organizer.py list-reminders
```

Delete a reminder:
```
python <scripts_dir>/organizer.py delete-reminder --reminder-id "<title substring>"
```

### Calendar

Events are stored as individual Markdown files with Full Calendar compatible YAML frontmatter.

Add an event:
```
python <scripts_dir>/organizer.py add-event --title "<title>" --start "<ISO8601>" [--end "<ISO8601>"] [--description "<desc>"] [--location "<loc>"] [--checklist '<JSON>']
```

The `--checklist` parameter accepts a JSON array of automation items stored in frontmatter. Each item can reference a script that the system executes automatically at the right time.

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
  }
]
```

- `trigger`: `"relative"` (offset from event start), `"on_create"` (immediately), `"absolute"` (fixed time)
- `trigger_offset`: e.g. `"-2h"`, `"-1d"`, `"-30m"` (negative = before event)
- `script`: path to automation script
- `message`: plain text notification (when no script needed)
- `params`: passed to the script as AUTOMATION_PARAMS env var (JSON)
- `notify`: whether to send the result to the user (default: true)

**IMPORTANT:** When creating events with a location, ALWAYS create a checklist with appropriate reminders and preparation steps.

List events:
```
python <scripts_dir>/organizer.py list-events [--limit <n>]
```

Delete an event:
```
python <scripts_dir>/organizer.py delete-event --event-id "<filename>"
```

The `--event-id` is the filename (with or without `.md`), e.g. `"2026-04-10 Teammeeting"`.

### Tasks

Tasks are stored in Obsidian Tasks emoji format in `workspace/tasks.md`.

Add a task:
```
python <scripts_dir>/organizer.py add-task --title "<title>" [--due-date "YYYY-MM-DD"] [--priority highest|high|medium|low|lowest] [--description "<desc>"] [--reminders '<JSON>']
```

Priority maps to Obsidian Tasks emojis: 🔺 highest, ⏫ high, 🔼 medium, 🔽 low, ⏬ lowest.

The `--reminders` parameter stores reminder rules in `scheduler_state.json` (outside the vault). Format:
```json
[
  {"offset": "-3d", "message": "In 3 Tagen fällig: {title}"},
  {"offset": "-1d", "message": "Morgen fällig: {title}"},
  {"offset": "-2h", "message": "In 2 Stunden fällig: {title}"}
]
```

**IMPORTANT:** When creating tasks with a due date, ALWAYS add appropriate reminders based on priority:
- highest/high: 3d, 1d, 2h before
- medium: 1d, 2h before
- low/lowest: 2h before

List tasks:
```
python <scripts_dir>/organizer.py list-tasks [--status pending|completed|all] [--limit <n>]
```

Complete a task:
```
python <scripts_dir>/organizer.py complete-task --task-id "<title or substring>"
```

Delete a task:
```
python <scripts_dir>/organizer.py delete-task --task-id "<title or substring>"
```

### Scheduled Jobs (Automation)

Jobs are stored in `automations/jobs.json` (outside the vault) since they are scheduler-internal automation config.

Add a job:
```
python <scripts_dir>/organizer.py add-job --name "<name>" --script "<script_path>" --schedule "<schedule>" [--params '<JSON>'] [--no-notify]
```

**Schedule formats:**
- `"16:00"` — daily at 16:00
- `"interval:30m"` — every 30 minutes
- `"interval:2h"` — every 2 hours
- `"weekly:DOW:HH:MM"` — weekly on day-of-week (0=Mon)
- `"monthly:DD:HH:MM"` — monthly on day

List jobs:
```
python <scripts_dir>/organizer.py list-jobs
```

Delete a job:
```
python <scripts_dir>/organizer.py delete-job --job-id "<id>"
```

Toggle a job (enable/disable):
```
python <scripts_dir>/organizer.py toggle-job --job-id "<id>"
```

## Output

The script outputs JSON. Parse it and report the result naturally to the user. On error, report the `error` field.
