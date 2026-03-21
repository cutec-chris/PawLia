# Automation & Task Management

PawLia's automation system follows the principle **"LLM plans, system executes"**. The LLM creates events, tasks and jobs with concrete automation rules. The system then processes them automatically at the right time — no LLM needed at runtime.

All notifications (reminders, script results, etc.) are routed through the LLM for a personalized response before delivery. If the LLM fails (timeout is controlled by the provider's `timeout` setting in config), the raw message is delivered as fallback.

## Overview

| Feature | Storage | Trigger | LLM involved? |
|---------|---------|---------|---------------|
| Simple reminders | `reminders.json` | Fixed time or relative (`10m`, `2h`) | Only for formatting output |
| Event checklists | `calendar/events.json` → `checklist[]` | Relative to event start | Only for formatting output |
| Task reminders | `tasks/tasks.json` → `reminders[]` | Relative to due date | Only for formatting output |
| Scheduled jobs | `automations/jobs.json` | Cron-like schedule | Only for formatting output |

## Simple Reminders

Quick, one-off or recurring reminders.

```
"Erinnere mich in 10 Minuten an die Pizza"
```

The LLM creates a reminder via the organizer skill:

```bash
python organizer.py add-reminder --user-id <id> --session-dir <dir> \
  --fire-at "10m" --message "Pizza aus dem Ofen!" --label "Pizza"
```

### Data model (`session/<user>/reminders.json`)

```json
{
  "id": "uuid",
  "user_id": "cli_user",
  "fire_at": "2026-03-20T18:30:00",
  "message": "Pizza aus dem Ofen!",
  "label": "Pizza",
  "recurrence": "none",
  "fired": false,
  "created_at": "2026-03-20T18:20:00"
}
```

- `fire_at` — absolute ISO8601 datetime (relative times like `10m` are resolved at creation)
- `recurrence` — `none`, `daily`, `weekly`, or `monthly`
- `fired` — set to `true` after delivery (recurring reminders update `fire_at` instead)

## Event Checklists

Events can have a checklist of automated preparation steps. Each item is either a **script** that gets executed or a **plain notification**.

```
"Termin am Freitag 14 Uhr, Kundenpräsentation in Hamburg"
```

The LLM creates the event with a checklist:

```json
{
  "title": "Kundenpräsentation Hamburg",
  "start": "2026-03-21T14:00:00",
  "location": "Hamburg Innenstadt",
  "checklist": [
    {
      "id": "chk-a1b2c3d4",
      "script": "",
      "trigger": "relative",
      "trigger_offset": "-1d",
      "message": "Morgen: Kundenpräsentation in Hamburg. Unterlagen vorbereiten!",
      "status": "pending"
    },
    {
      "id": "chk-e5f6g7h8",
      "script": "route_plan.py",
      "trigger": "relative",
      "trigger_offset": "-90m",
      "params": {"from": "home", "to": "Hamburg Innenstadt"},
      "notify": true,
      "status": "pending"
    },
    {
      "id": "chk-i9j0k1l2",
      "script": "check_traffic.py",
      "trigger": "relative",
      "trigger_offset": "-60m",
      "params": {"destination": "Hamburg"},
      "notify": true,
      "status": "pending"
    }
  ]
}
```

### Checklist item fields

| Field | Description |
|-------|-------------|
| `script` | Path to automation script. Empty = pure notification. |
| `trigger` | `relative` (offset from event start), `on_create` (immediately), `absolute` (fixed time) |
| `trigger_offset` | e.g. `-2h`, `-1d`, `-30m` (negative = before event) |
| `message` | Plain text notification. Supports placeholders: `{title}`, `{location}`, `{start}`, `{description}` |
| `params` | JSON object passed to the script via `AUTOMATION_PARAMS` env var |
| `notify` | Whether to send the result to the user (default: `true`) |
| `status` | `pending` → `done` or `failed` |
| `result` | Script stdout (on success) or stderr (on failure), set after execution |

### Execution timeline example

```
T-1d     📋 "Morgen: Kundenpräsentation in Hamburg. Unterlagen vorbereiten!"
T-90m    📋 route_plan.py → "ICE 1523 ab 11:15 Hbf, Ankunft 13:20, Gleis 8"
T-60m    📋 check_traffic.py → "Keine Verspätungen, alles planmäßig"
T-15m    📅 Standard event notification (built-in)
```

## Task Reminders

Tasks with a due date get automatic reminders based on priority.

```json
{
  "title": "Bericht schreiben",
  "due_date": "2026-03-22",
  "priority": "high",
  "status": "pending",
  "reminders": [
    {"offset": "-3d", "message": "In 3 Tagen fällig: {title}", "fired": false},
    {"offset": "-1d", "message": "Morgen fällig: {title}", "fired": false},
    {"offset": "-2h", "message": "In 2 Stunden fällig: {title}", "fired": false}
  ]
}
```

The LLM sets the reminder strategy when creating the task. Suggested defaults:

| Priority | Reminders |
|----------|-----------|
| high | 3d, 1d, 2h before |
| medium | 1d, 2h before |
| low | 2h before |

### Reminder fields

| Field | Description |
|-------|-------------|
| `offset` | Relative to `due_date`, e.g. `-3d`, `-1d`, `-2h` |
| `message` | Notification text. Placeholders: `{title}`, `{due_date}` |
| `fired` | Set to `true` after delivery |

## Scheduled Jobs

For recurring automated tasks the LLM writes a script and registers it as a job.

### Workflow

1. User: *"Erstelle mir jeden Tag um 16 Uhr eine Zusammenfassung"*
2. LLM writes a Python script → `session/<user>/automations/daily_report.py`
3. LLM registers the job via organizer:

```bash
python organizer.py add-job --user-id <id> --session-dir <dir> \
  --name "Tagesbericht" --script "daily_report.py" --schedule "16:00"
```

4. Every day at 16:00, the scheduler executes the script and sends the output as notification.

### Data model (`session/<user>/automations/jobs.json`)

```json
{
  "id": "job-a1b2c3d4",
  "name": "Tagesbericht",
  "script": "daily_report.py",
  "schedule": "16:00",
  "params": {},
  "notify": true,
  "enabled": true,
  "created_at": "2026-03-20T12:00:00",
  "last_run": "2026-03-20T16:00:00",
  "last_result": "success"
}
```

### Schedule formats

| Format | Description | Example |
|--------|-------------|---------|
| `HH:MM` | Daily at that time | `16:00` |
| `interval:Nm` | Every N minutes | `interval:30m` |
| `interval:Nh` | Every N hours | `interval:2h` |
| `weekly:DOW:HH:MM` | Weekly (0=Mon..6=Sun) | `weekly:4:09:00` |
| `monthly:DD:HH:MM` | Monthly on day DD | `monthly:1:10:00` |

### Writing automation scripts

Scripts are plain Python (or Node.js / Bash) files stored in `session/<user>/automations/`.

**Input:** Parameters are passed via the `AUTOMATION_PARAMS` environment variable as JSON.

```python
import json, os

params = json.loads(os.environ.get("AUTOMATION_PARAMS", "{}"))
job_name = params.get("job_name", "")
user_id = params.get("user_id", "")
```

**Output:** The script's stdout becomes the notification message. Keep it concise.

```python
print("Heute 3 neue E-Mails, 2 offene Tasks, keine Termine.")
```

**Exit code:** 0 = success, non-zero = failure (stderr is sent as error notification).

### Script resolution order

1. `session/<user>/automations/<script>` — user-specific scripts
2. `scripts/<script>` — global project scripts
3. `skills/*/scripts/<script>` — skill scripts

## Notification Pipeline

All notifications pass through this pipeline:

```
Trigger fires (reminder / checklist / job)
        │
        ▼
  Scheduler._notify(user_id, raw_message)
        │
        ▼
  LLM Formatter (30s timeout)
        ├─ success → personalized message
        └─ failure → raw message as fallback
        │
        ▼
  Interface callbacks (CLI / Telegram / Matrix / Webhook)
```

The LLM receives the raw data and the user's context (memory, preferences) to produce a natural, personalized message.

**Example:**
- Raw: `📋 Kundenpräsentation Hamburg: ICE 1523 ab 11:15 Hbf, Ankunft 13:20, Gleis 8`
- LLM: `Für deine Präsentation in Hamburg — nimm den ICE 1523 um 11:15 vom Hauptbahnhof, du bist um 13:20 da (Gleis 8).`

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                      Scheduler                            │
│                 (60s check interval)                      │
│                                                           │
│  ── High priority (every tick) ──────────────────────    │
│                                                           │
│  ┌──────────┐ ┌───────────┐ ┌────────────────┐          │
│  │ Reminders│ │ Checklist │ │ Task Reminders │          │
│  │          │ │ Processor │ │   Processor    │          │
│  └────┬─────┘ └─────┬─────┘ └──────┬─────────┘          │
│       │             │              │                      │
│       │    ┌────────┴────────┐     │                      │
│       │    │  Script Executor│     │                      │
│       │    └────────┬────────┘     │                      │
│       │             │              │                      │
│  ┌────┴─────────────┴──────────────┴──────────────────┐  │
│  │            _notify (LLM formatter)                 │  │
│  └────────────────────┬───────────────────────────────┘  │
│                       │                                   │
│  ┌────────────────────┴───────────────────────────────┐  │
│  │         Job Runner (cron-like schedule)             │  │
│  └────────────────────────────────────────────────────┘  │
│                                                           │
│  ── Low priority (all users idle 20min + LLM free) ──   │
│                                                           │
│  ┌────────────────────────────────────────────────────┐  │
│  │  Background Tasks (deferred agent.run via /background)│
│  └────────────────────┬───────────────────────────────┘  │
│                       │                                   │
│  ┌────────────────────┴───────────────────────────────┐  │
│  │  Memory Indexer (LightRAG knowledge graph)         │  │
│  └────────────────────────────────────────────────────┘  │
│                                                           │
└───────────────────────┬──────────────────────────────────┘
                        │
            ┌───────────┼───────────┐
            ▼           ▼           ▼
          CLI       Telegram     Matrix
```

## File structure

```
session/<user>/
├── reminders.json              # Simple reminders
├── calendar/
│   └── events.json             # Events with checklist[]
├── tasks/
│   └── tasks.json              # Tasks with reminders[]
├── automations/
│   ├── jobs.json               # Scheduled job definitions
│   ├── daily_report.py         # User automation scripts
│   └── check_traffic.py        # (written by LLM)
├── background_tasks/
│   └── <task_id>.json          # Deferred agent.run() tasks
├── memory_index/               # LightRAG knowledge graph index
│   └── indexed_files.json      # Tracks which chat logs have been indexed
└── researches/                 # Research skill output
```

## Related modules

| Module | Role |
|--------|------|
| [`pawlia/scheduler.py`](../pawlia/scheduler.py) | Main loop, notification pipeline with LLM formatting, priority gate |
| [`pawlia/automation.py`](../pawlia/automation.py) | Script executor, checklist/job/task processors |
| [`pawlia/background_tasks.py`](../pawlia/background_tasks.py) | Background task queue (deferred agent.run) |
| [`pawlia/memory_indexer.py`](../pawlia/memory_indexer.py) | LightRAG-based knowledge graph indexing of chat logs |
| [`skills/organizer/`](../skills/organizer/) | LLM-facing skill for creating events, tasks, reminders, jobs |
| [`pawlia/app.py`](../pawlia/app.py) | Wires LLM formatter and app reference into scheduler |
