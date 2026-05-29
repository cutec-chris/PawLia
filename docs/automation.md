# Automation & Task Management

PawLia's automation system follows the principle **"LLM plans, system executes"**. The LLM creates events, tasks and jobs with concrete automation rules. The system then processes them automatically at the right time — no LLM needed at runtime.

All data visible to the user is stored as **Obsidian-compatible Markdown** in `workspace/`. The workspace directory functions as an [Obsidian](https://obsidian.md) vault — events use the [Full Calendar](https://github.com/obsidian-community/obsidian-full-calendar) plugin format, tasks use the [Obsidian Tasks](https://github.com/obsidian-tasks-group/obsidian-tasks) emoji format.

Internal scheduler state (notification flags, checklist execution status) is stored separately in `scheduler_state.json` outside the vault.

All notifications (reminders, script results, etc.) are routed through the LLM for a personalized response before delivery. If the LLM fails, the raw message is delivered as fallback.

## Overview

| Feature | Storage | Trigger | LLM involved? |
|---------|---------|---------|---------------|
| Simple reminders | `workspace/tasks.md` (🔔 prefix + ⏳ scheduled date) | Fixed time or relative (`10m`, `2h`) | Only for formatting output |
| Calendar events | `workspace/calendar/<date> <title>.md` (Full Calendar frontmatter) | 15 min before start | Only for formatting output |
| Event reminders | Event frontmatter `reminders:` + derived `checklist:` items | Relative to event start | Only for formatting output |
| Event checklists | Event frontmatter `checklist:` + `scheduler_state.json` | Relative to event start | Only for formatting output |
| Task reminders | `workspace/tasks.md` (📅 due date) + `scheduler_state.json` | Relative to due date | Only for formatting output |
| Scheduled jobs | `automations/jobs.json` | Cron-like schedule | Only for formatting output |

## Simple Reminders

Quick, one-off or recurring reminders. Stored as scheduled tasks in `workspace/tasks.md` with a 🔔 prefix, compatible with the Obsidian Tasks plugin.

```
"Erinnere mich in 10 Minuten an die Pizza"
```

The LLM creates a reminder via the organizer skill:

```bash
python organizer.py add-reminder --fire-at "10m" --message "Pizza aus dem Ofen!" --label "Pizza"
```

### Format in `workspace/tasks.md`

```markdown
- [ ] 🔔 Pizza: Pizza aus dem Ofen! ⏳ 2026-03-20T18:30 ➕ 2026-03-20
```

When fired, the scheduler marks it as done:
```markdown
- [x] 🔔 Pizza: Pizza aus dem Ofen! ⏳ 2026-03-20T18:30 ➕ 2026-03-20 ✅ 2026-03-20
```

Recurring reminders use the 🔁 emoji and get rescheduled instead of completed:
```markdown
- [ ] 🔔 Daily: Wasser trinken! 🔁 every day ⏳ 2026-03-21T09:00 ➕ 2026-03-20
```

### Emoji reference

| Emoji | Meaning |
|-------|---------|
| 🔔 | Reminder (prefix in title) |
| ⏳ | Scheduled date/time |
| 🔁 | Recurrence (`every day`, `every week`, `every month`) |
| ➕ | Created date |
| ✅ | Completed date |

## Calendar Events

Events are stored as individual Markdown files with YAML frontmatter compatible with the Obsidian [Full Calendar](https://github.com/obsidian-community/obsidian-full-calendar) plugin.

```
"Termin am Freitag 14 Uhr, Kundenpräsentation in Hamburg"
```

The LLM creates the event via the organizer skill. Normal user-facing reminders belong in `reminders`; automation and preparation steps belong in `checklist`.

### File format (`workspace/calendar/2026-03-21 Kundenpräsentation Hamburg.md`)

```yaml
---
title: Kundenpräsentation Hamburg
allDay: false
date: '2026-03-21'
endDate: null
startTime: '14:00'
endTime: '16:00'
location: Hamburg Innenstadt
type: single
reminders:
- minutes_before: 60
  message: 'In 60 Minuten: {title} in {location}.'
checklist:
- id: chk-a1b2c3d4
  message: 'Morgen: {title} in {location}. Unterlagen vorbereiten!'
  trigger: relative
  trigger_offset: '-1d'
- id: chk-e5f6g7h8
  notify: true
  params:
    from: home
    to: Hamburg Innenstadt
  script: route_plan.py
  trigger: relative
  trigger_offset: '-90m'
---

Kundenpräsentation für Projekt Alpha.

## Checkliste
- [ ] Morgen: {title} in {location}. Unterlagen vorbereiten! (-1d)
- [ ] route_plan.py (-90m)
```

The frontmatter contains both standard Full Calendar fields and the event-specific automation config. `reminders` represent the user-visible event reminders. `checklist` contains automation/preparation items. The body is for human-readable notes — the checklist in the body is informational, the scheduler reads only from frontmatter.

### Full Calendar frontmatter fields

| Field | Type | Description |
|-------|------|-------------|
| `title` | string | Event title |
| `date` | string | Start date (`YYYY-MM-DD`) |
| `startTime` | string | Start time (`HH:MM`) — omit for all-day |
| `endTime` | string | End time (`HH:MM`) |
| `endDate` | string/null | End date if multi-day |
| `allDay` | boolean | All-day event flag |
| `type` | string | `single` (or `recurring` with `rrule`) |
| `location` | string | Location |
| `completed` | boolean/null | For task-type events |

### Event notifications

The scheduler automatically notifies 15 minutes before an event starts. The notification state is tracked in `scheduler_state.json` (not in the .md file).

### Event reminders

Event reminders are stored directly in the event frontmatter as `reminders:`. At write time, the organizer also derives scheduler-compatible checklist entries from them so the existing checklist processor can fire them.

```yaml
reminders:
- minutes_before: 40
  message: In 40 Minuten: Parcours beginnt um 17:00 Uhr.
  notify: true
```

Use `reminders` for plain notifications tied to the event. Use `checklist` only when the event needs preparation steps or scripts.

### Event ID

The event ID is the filename (e.g. `2026-03-21 Kundenpräsentation Hamburg.md`). Delete and list operations use this filename (`.md` extension optional).

## Event Checklists

Events can have a checklist of automated preparation steps in their frontmatter. Each item is either a **script** that gets executed or a **plain notification**.

### Checklist item fields

| Field | Description |
|-------|-------------|
| `id` | Unique identifier (e.g. `chk-a1b2c3d4`) |
| `script` | Path to automation script. Empty = pure notification. |
| `trigger` | `relative` (offset from event start), `on_create` (immediately), `absolute` (fixed time) |
| `trigger_offset` | e.g. `-2h`, `-1d`, `-30m` (negative = before event) |
| `message` | Plain text notification. Supports placeholders: `{title}`, `{location}`, `{start}`, `{description}` |
| `params` | Object passed to the script via `AUTOMATION_PARAMS` env var |
| `notify` | Whether to send the result to the user (default: `true`) |

### Checklist state tracking

Checklist execution state (done/failed, result, timestamp) is stored in `scheduler_state.json`, not in the event file:

```json
{
  "checklist_state": {
    "2026-03-21 Kundenpräsentation Hamburg.md": {
      "chk-a1b2c3d4": {"status": "done", "executed_at": "2026-03-20T14:00:00"},
      "chk-e5f6g7h8": {"status": "done", "result": "ICE 1523 ab 11:15...", "executed_at": "2026-03-21T12:30:00"}
    }
  }
}
```

### Execution timeline example

```
T-1d     📋 "Morgen: Kundenpräsentation in Hamburg. Unterlagen vorbereiten!"
T-90m    📋 route_plan.py → "ICE 1523 ab 11:15 Hbf, Ankunft 13:20, Gleis 8"
T-60m    📋 check_traffic.py → "Keine Verspätungen, alles planmäßig"
T-15m    📅 Standard event notification (built-in)
```

## Tasks

Tasks are stored in Obsidian [Tasks](https://github.com/obsidian-tasks-group/obsidian-tasks) emoji format in `workspace/tasks.md`.

```
"Erstell eine Aufgabe: Bericht schreiben, fällig Freitag, Priorität hoch"
```

### Format in `workspace/tasks.md`

```markdown
- [ ] Bericht schreiben ⏫ 📅 2026-03-22 ➕ 2026-03-19
- [ ] Server migrieren 🔼 📅 2026-04-15 ➕ 2026-04-01
- [x] Backup prüfen 🔽 📅 2026-03-20 ➕ 2026-03-18 ✅ 2026-03-19
```

### Task emoji reference

| Emoji | Meaning |
|-------|---------|
| 🔺 | Highest priority |
| ⏫ | High priority |
| 🔼 | Medium priority |
| 🔽 | Low priority |
| ⏬ | Lowest priority |
| 📅 | Due date |
| 🛫 | Start date |
| ⏳ | Scheduled date |
| ➕ | Created date |
| ✅ | Completed date |
| ❌ | Cancelled date |
| 🔁 | Recurrence |

### Task ID

Tasks are identified by their title (or a substring match). Complete and delete operations use title matching.

### Task reminders

Reminder rules are stored in `scheduler_state.json` (not in the task line). The LLM sets them when creating a task:

```json
{
  "task_reminders": {
    "bericht schreiben": [
      {"offset": "-3d", "message": "In 3 Tagen fällig: {title}", "fired": false},
      {"offset": "-1d", "message": "Morgen fällig: {title}", "fired": false},
      {"offset": "-2h", "message": "In 2 Stunden fällig: {title}", "fired": false}
    ]
  }
}
```

Suggested reminder defaults by priority:

| Priority | Reminders |
|----------|-----------|
| highest/high | 3d, 1d, 2h before |
| medium | 1d, 2h before |
| low/lowest | 2h before |

## Scheduled Jobs

For recurring automated tasks the LLM registers a natural-language instruction as a job. Jobs are stored in `automations/jobs.json` (outside the workspace) since they are scheduler-internal automation config. When due, the instruction runs through the normal agent pipeline, including skills.

Job CRUD lives in the dedicated **`automation`** skill (it used to be part of `organizer`).

### Workflow

1. User: *"Erstelle mir jeden Tag um 16 Uhr eine Zusammenfassung"*
2. LLM registers the instruction via the `automation` skill:

```bash
python <scripts_dir>/automation.py add-job --name "Tagesbericht" \
    --instruction "Erstelle eine kurze Tageszusammenfassung" \
    --schedule "16:00"
```

3. Every day at 16:00, the scheduler runs the instruction through the normal agent/skill dispatcher and sends the output as notification.

### Data model (`session/<user>/automations/jobs.json`)

```json
{
  "id": "job-a1b2c3d4",
  "name": "Tagesbericht",
  "instruction": "Erstelle eine kurze Tageszusammenfassung",
  "schedule": "16:00",
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

### Manual trigger

To test a job immediately without waiting for its schedule, use the `run-job` command:

```bash
python organizer.py run-job --job-id "job-a1b2c3d4"
```

This sets a `force_run` flag on the job. The scheduler executes it on the next tick (within 60 seconds), regardless of the configured schedule. Use this after the skill-creator has built or fixed a skill to verify the automation works end-to-end.

### Writing checklist scripts

Checklist scripts are plain Python (or Node.js / Bash) files stored in `session/<user>/automations/` or `workspace/.scripts/`.

**Input:** Parameters are passed via the `AUTOMATION_PARAMS` environment variable as JSON.
User context is available via `PAWLIA_USER_ID` and `PAWLIA_SESSION_DIR` environment variables.

```python
import json, os

params = json.loads(os.environ.get("AUTOMATION_PARAMS", "{}"))
job_name = params.get("job_name", "")
user_id = os.environ.get("PAWLIA_USER_ID", "")
session_dir = os.environ.get("PAWLIA_SESSION_DIR", "")
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

## Workspace Git Sync

The workspace can be kept in a Git repository for syncing (e.g. to a remote for backup or multi-device access). Enable in `config.yaml`:

```yaml
workspace:
  git:
    enabled: true
    daily_squash_time: "23:00"     # squash all daily commits into one
    weekly_squash_day: 6           # 0=Mon..6=Sun (default: Sunday)
    weekly_squash_time: "23:30"    # squash all weekly commits into one
    push: false                    # push to remote after squash
```

### How it works

1. **Auto-commit** — every scheduler tick (60s), uncommitted changes are committed. Throttled to max 1 commit per 5 minutes to avoid noise.
2. **Daily squash** — at the configured time (default 23:00), all commits from today are squashed into one `Daily: YYYY-MM-DD` commit.
3. **Weekly squash** — on the configured day (default Sunday) at the configured time (default 23:30), all commits from this week are squashed into one `Week: YYYY-Www` commit.
4. **Push** — if `push: true` and a remote is configured, pushes after each squash using `--force-with-lease`.

### Setting up a remote

```bash
cd session/<user>/workspace
git remote add origin git@github.com:you/vault.git
```

The scheduler will auto-push after squash if `push: true`.

### What gets committed

Everything in `workspace/` — wiki pages, calendar events, tasks, identity files, memory logs. Internal state files (`scheduler_state.json`, `automations/jobs.json`) live outside the workspace and are not tracked.

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
│  workspace/tasks.md        workspace/calendar/*.md        │
│  ┌──────────┐              ┌───────────┐                  │
│  │ Reminders│              │  Events   │                  │
│  │  (🔔⏳)  │              │(frontmatter│                 │
│  └────┬─────┘              └─────┬─────┘                  │
│       │                          │                        │
│       │  ┌───────────┐ ┌────────┴────────┐                │
│       │  │ Task Rem. │ │  Checklist Proc │                │
│       │  │ Processor │ │  (scripts)      │                │
│       │  └─────┬─────┘ └────────┬────────┘                │
│       │        │                │                         │
│  ┌────┴────────┴────────────────┴──────────────────────┐  │
│  │       scheduler_state.json (flags, results)         │  │
│  └─────────────────────┬──────────────────────────────┘  │
│                        │                                  │
│  ┌─────────────────────┴──────────────────────────────┐  │
│  │          _notify (LLM formatter)                    │  │
│  └─────────────────────┬──────────────────────────────┘  │
│                        │                                  │
│  ┌─────────────────────┴──────────────────────────────┐  │
│  │   automations/jobs.json → Job Runner (cron-like)    │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                           │
│  ── Low priority (idle + LLM free) ──────────────────    │
│                                                           │
│  ┌─────────────────────────────────────────────────────┐  │
│  │  Background Tasks (deferred agent.run)              │  │
│  └─────────────────────┬──────────────────────────────┘  │
│  ┌─────────────────────┴──────────────────────────────┐  │
│  │  Memory Indexer / Dream Wiki                        │  │
│  └─────────────────────────────────────────────────────┘  │
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
├── workspace/                       # Obsidian vault
│   ├── calendar/                    # Full Calendar plugin events
│   │   ├── 2026-03-21 Meeting.md    # one .md per event
│   │   └── 2026-03-22 Standup.md
│   ├── tasks.md                     # Obsidian Tasks emoji format (incl. reminders)
│   ├── wiki/                        # Dream Wiki knowledge base
│   │   ├── index.md
│   │   ├── log.md
│   │   └── topics/
│   ├── memory/                      # Daily chat logs + memory.md + context_summary.md
│   ├── research/                    # Per-project document collections (researcher skill)
│   ├── soul.md / identity.md / user.md / memory.md / bootstrap.md  # Identity files
│   └── skills/                      # Optional workspace-local skills
├── scheduler_state.json             # Internal: notified/fired/checklist state
├── automations/
│   └── jobs.json                    # Scheduled job definitions (automation skill)
└── memory_index/                    # RAG backend index (outside the vault)
    └── dreamed_files.json
```

`/background` tasks are kept inside `automations/` alongside the jobs file — there is no separate `background_tasks/` directory.

## Related modules

| Module | Role |
|--------|------|
| [`pawlia/scheduler.py`](../pawlia/scheduler.py) | Main loop, reads workspace Markdown, manages scheduler_state.json |
| [`pawlia/automation.py`](../pawlia/automation.py) | Script executor, checklist/job/task processors |
| [`pawlia/background_tasks.py`](../pawlia/background_tasks.py) | Background task queue (deferred agent.run) |
| [`pawlia/memory_indexer.py`](../pawlia/memory_indexer.py) | Memory indexing / Dream Wiki processing |
| [`pawlia/dream_wiki.py`](../pawlia/dream_wiki.py) | Dream Wiki backend (Karpathy's LLM Wiki) |
| [`skills/organizer/`](../skills/organizer/) | LLM-facing skill for creating events, tasks, reminders, jobs |
| [`pawlia/app.py`](../pawlia/app.py) | Wires LLM formatter and app reference into scheduler |
