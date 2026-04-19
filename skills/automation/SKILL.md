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

Write a Python script to the user's automations directory.
Use the files skill or bash tool to write the script to the `automations/` subdirectory in the user's workspace.

The script:
- Receives parameters via `AUTOMATION_PARAMS` env var (JSON string)
- Receives user context via `PAWLIA_USER_ID` and `PAWLIA_SESSION_DIR` env vars
- Should print its result to stdout (this becomes the notification message)
- Exit code 0 = success, non-zero = failure
- Keep output concise (1-5 lines)

**Example script** (`show_tasks.py`):
```python
import json
import os

session_dir = os.environ.get("PAWLIA_SESSION_DIR", "session")
user_id = os.environ.get("PAWLIA_USER_ID", "")

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
- Always use `os.environ.get("PAWLIA_USER_ID")` and `os.environ.get("PAWLIA_SESSION_DIR")` in scripts
- Scripts must be self-contained (no imports from pawlia)
- Scripts can read JSON files from the session directory for data

### Optional: LLM-Schritt im Automations-Skript

Wenn ein Schritt zwingend das Sprachmodell braucht (Texte zusammenfassen,
natürlichsprachliche Ausgabe erzeugen), **niemals** das Modell direkt
importieren — stattdessen den mitgelieferten Harness als Subprocess aufrufen.
Der Harness garantiert eine Retry/Nudge-Schleife bis ein nicht-leeres
Ergebnis vorliegt (oder mit Non-Zero-Exit hart fehlschlägt — kein stilles
Leerergebnis).

Der Harness liegt unter `<skills_dir>/automation/scripts/llm.py`.

**Beispiel** (`news_digest.py`, fasst ein paar Schlagzeilen mit dem LLM zu
einer kurzen Nachricht zusammen):

```python
import subprocess
import sys

SOURCES = [
    "Regierung beschließt neues Energiegesetz",
    "Bahn kündigt Streik für Donnerstag an",
    "Warnstreik im öffentlichen Dienst beendet",
]

prompt = (
    "Fasse die folgenden Schlagzeilen in zwei Sätzen auf Deutsch zusammen:\n\n"
    + "\n".join(f"- {s}" for s in SOURCES)
)

result = subprocess.run(
    [sys.executable, "<skills_dir>/automation/scripts/llm.py",
     "--prompt", prompt,
     "--retries", "4",
     "--min-chars", "20"],
    capture_output=True, text=True, timeout=90,
)

if result.returncode != 0:
    print(f"LLM-Schritt fehlgeschlagen: {result.stderr.strip()}", file=sys.stderr)
    sys.exit(1)

print(result.stdout.strip())
```

Regeln für LLM-Schritte:
- Nur einsetzen, wenn ein deterministischer Schritt die Aufgabe nicht erledigen
  kann. Die Laufzeit von Automationen soll so reproduzierbar wie möglich sein.
- `--min-chars` setzen, damit offensichtlich leere/abgeschnittene Antworten
  einen Retry auslösen.
- Exit-Code prüfen und bei Fehler den Job mit Non-Zero-Exit beenden — so wird
  der Fehler sichtbar statt verschluckt.

### Step 1b: Skript testen (Pflicht, bevor der Job registriert wird)

Nach dem Schreiben das Skript einmal direkt ausführen und den Output prüfen:

```bash
AUTOMATION_PARAMS='{}' \
PAWLIA_USER_ID="<user_id>" \
PAWLIA_SESSION_DIR="<session_dir>" \
python <path_to_script>
```

Erst wenn der Testlauf einen sinnvollen, nicht-leeren stdout liefert, wird der
Job registriert. Ein nicht-getestetes Skript wird nicht registriert.

### Step 2: Register the job

Use the organizer script to register the job:

```bash
python <scripts_dir>/../organizer/scripts/organizer.py add-job \
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
python <scripts_dir>/../organizer/scripts/organizer.py list-jobs
```

Delete a job:
```bash
python <scripts_dir>/../organizer/scripts/organizer.py delete-job --job-id "<id>"
```

Toggle a job (enable/disable):
```bash
python <scripts_dir>/../organizer/scripts/organizer.py toggle-job --job-id "<id>"
```

## Output

After creating the script and registering the job, confirm to the user:
- What the script does
- How often it runs
- That they can ask to list, disable, or delete it later
