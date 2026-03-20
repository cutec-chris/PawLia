---
name: automation
description: "Write and schedule automation scripts. Use when the user wants something to happen automatically or repeatedly (e.g. 'show my tasks every 5 minutes', 'send me a daily report at 16:00', 'check the weather every hour'). This skill writes the script and registers the scheduled job."
license: MIT
metadata:
  author: Christian Ulrich
  version: "1.0"
---

# Automation

Writes automation scripts and registers them as scheduled jobs. The system then executes them automatically — no LLM needed at runtime.

## Instructions

When the user wants something to happen automatically or on a schedule, follow these steps:

### Step 1: Write the automation script

Write a Python script to the user's automations directory:

```
session/<user_id>/automations/<script_name>.py
```

Use the bash tool to write the file. The script:
- Receives parameters via `AUTOMATION_PARAMS` env var (JSON string)
- Should print its result to stdout (this becomes the notification message)
- Exit code 0 = success, non-zero = failure
- Keep output concise (1-5 lines)

**Example script** (`show_tasks.py`):
```python
import json
import os

session_dir = os.environ.get("AUTOMATION_SESSION_DIR", "session")
params = json.loads(os.environ.get("AUTOMATION_PARAMS", "{}"))
user_id = params.get("user_id", "")

tasks_path = os.path.join(session_dir, user_id, "tasks", "tasks.json")
if not os.path.exists(tasks_path):
    print("Keine offenen Aufgaben.")
else:
    with open(tasks_path, "r") as f:
        tasks = json.load(f)
    pending = [t for t in tasks if t.get("status") == "pending"]
    if not pending:
        print("Alle Aufgaben erledigt!")
    else:
        for t in pending:
            prio = t.get("priority", "medium")
            due = t.get("due_date", "")
            line = f"- [{prio}] {t['title']}"
            if due:
                line += f" (fällig: {due})"
            print(line)
```

**IMPORTANT:**
- The session directory is available at: `<session_dir>`
- The user ID is: `<user_id>`
- Use these values directly in the script, not as placeholders
- Scripts must be self-contained (no imports from pawlia)
- Scripts can read JSON files from the session directory for data

### Step 2: Register the job

Use the organizer script to register the job:

```bash
python <scripts_dir>/../organizer/scripts/organizer.py add-job \
  --user-id <user_id> \
  --session-dir "<session_dir>" \
  --name "<descriptive name>" \
  --script "<script_filename>.py" \
  --schedule "<schedule>"
```

**Schedule formats:**
- `"16:00"` — daily at 16:00
- `"interval:5m"` — every 5 minutes
- `"interval:1h"` — every hour
- `"weekly:0:09:00"` — weekly Monday at 09:00 (0=Mon..6=Sun)
- `"monthly:1:10:00"` — monthly on the 1st at 10:00

### Managing existing jobs

List jobs:
```bash
python <scripts_dir>/../organizer/scripts/organizer.py list-jobs --user-id <user_id> --session-dir "<session_dir>"
```

Delete a job:
```bash
python <scripts_dir>/../organizer/scripts/organizer.py delete-job --user-id <user_id> --session-dir "<session_dir>" --job-id "<id>"
```

Toggle a job (enable/disable):
```bash
python <scripts_dir>/../organizer/scripts/organizer.py toggle-job --user-id <user_id> --session-dir "<session_dir>" --job-id "<id>"
```

## Output

After creating the script and registering the job, confirm to the user:
- What the script does
- How often it runs
- That they can ask to list, disable, or delete it later
