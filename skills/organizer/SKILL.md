---
name: organizer
description: Manage calendar events and tasks. Use for adding, listing, completing, or deleting events and tasks.
license: MIT
metadata:
  author: Christian Ulrich
  version: "1.0"
---

# Organizer

Manages the user's calendar events and tasks using the organizer script.

## Instructions

Run the script with the appropriate subcommand. Always pass `--user-id` and `--session-dir` from the context.

### Calendar

Add an event:
```
python <scripts_dir>/organizer.py add-event --user-id <user_id> --session-dir "<session_dir>" --title "<title>" --start "<ISO8601>" [--end "<ISO8601>"] [--description "<desc>"] [--location "<loc>"]
```

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
python <scripts_dir>/organizer.py add-task --user-id <user_id> --session-dir "<session_dir>" --title "<title>" [--due-date "YYYY-MM-DD"] [--priority high|medium|low] [--description "<desc>"]
```

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

## Output

The script outputs JSON. Parse it and report the result naturally to the user. On error, report the `error` field.
