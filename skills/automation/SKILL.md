---
name: automation
description: "Create and manage scheduled automations. Use when the user wants something to happen automatically on a schedule (e.g. 'send me a daily report at 16:00', 'check the weather every morning', 'remind me to stand up every hour')."
license: MIT
metadata:
  author: Christian Ulrich
  version: "2.0"
---

# Automation

Creates scheduled automations that are executed by the LLM at the specified times. No scripts needed — the LLM runs the instruction directly using its available skills.

## Two Approaches

### Variante A — Simple (for trivial tasks)

When the task is simple and can be done with existing skills in one step:

1. Register the job directly with a natural-language instruction

Example: "Zeige offene Aufgaben", "Was steht heute im Kalender?", "Wie spät ist es"

### Variante B — Skill + Automation (for non-trivial tasks)

When the task requires API calls, data processing, complex logic, or needs to be reliable:

1. **First**: Use the skill-creator to build a dedicated skill for the task
2. **Then**: Register an automation that calls the skill via a simple instruction

Example: "Erstelle einen Wetterbericht" → Build weather skill first, then automate "Nutze den weather Skill"

### When to use which

- **Simple**: Single-step tasks that any existing skill can handle directly
- **Skill + Automation**: Everything else — API calls, multi-step logic, web scraping, data processing
- **In doubt**: Build a skill. A dedicated skill is always more robust than a fragile instruction.

## Instructions

### Step 1 (only for Variante B): Build the skill

If the task is non-trivial, use the skill-creator skill first:

```
Call skill-creator with: "Baue einen Skill der [task description]"
```

Wait for the skill to be created and confirmed before proceeding.

### Step 2: Register the automation

Use the organizer script to register the job:

```bash
python <scripts_dir>/../organizer/scripts/organizer.py add-job \
  --name "<descriptive name>" \
  --schedule "<schedule>" \
  --instruction "<instruction>"
```

**Schedule-Formate:**
- `"16:00"` — täglich um 16:00
- `"interval:5m"` — alle 5 Minuten (m=Minuten, h=Stunden, d=Tage)
- `"interval:1h"` — jede Stunde
- `"weekly:0:09:00"` — wöchentlich Montag 09:00 (0=Mo..6=So)
- `"monthly:1:10:00"` — monatlich am 1. um 10:00
- `"30 8 * * 1,3"` — Cron-Syntax: Minute 30, Stunde 8, Mo+Mi (5 Felder mit Leerzeichen)

**Instruction examples:**
- Simple: `"Lies die Datei workspace/tasks.md und zeige die offenen Aufgaben"`
- Skill call: `"Nutze den weather Skill um mir das aktuelle Wetter für Hamburg zu zeigen"`
- Multi-step: `"Prüfe Kalender und Aufgaben für heute, dann fasse beides kurz zusammen"`

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

### Manual trigger (for testing)

To test a job immediately without waiting for the scheduled time:
```bash
python <scripts_dir>/../organizer/scripts/organizer.py run-job --job-id "<id>"
```

This flags the job for immediate execution on the next scheduler tick (within 60 seconds). Use this after the skill-creator has modified a skill to verify the automation works correctly.

## Output

After registering, confirm to the user:
- What the automation does (the instruction)
- How often it runs (the schedule)
- If a skill was built: mention the skill name
- That they can ask to list, disable, delete, or manually trigger it later
