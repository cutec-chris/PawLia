---
name: automation
description: "Create and manage scheduled automations. Use when the user wants something to happen automatically on a schedule (e.g. 'send me a daily report at 16:00', 'check the weather every morning', 'warn me about thunderstorms every hour', 'tell me when my train is delayed')."
license: MIT
metadata:
  author: Christian Ulrich
  version: "3.0"
---

# Automation

Creates scheduled automations the scheduler runs at the given times.

**Default to a script.** An automation is a deterministic script the scheduler
executes. The script decides for itself whether there is anything worth saying:
when nothing is up, it prints nothing and the user hears nothing; when something
*is* up, it prints the message (that text becomes the notification). This is the
key behaviour for monitoring jobs (thunderstorm alert, train-delay watch) — they
must stay **silent on a quiet day** instead of pinging "all ok" every hour.

The LLM is a *step inside the script* (via the harness `llm_call()`), used only
when there is genuinely something to phrase or curate — never the thing that runs
every tick and decides whether to notify.

A bare natural-language `--instruction` (LLM every tick, always replies) still
exists for trivial one-off jobs, but is the exception, not the default.

## How a job stays silent

`notify` modes (set via the CLI flags below):

- `output_only` *(default for `--script`)* — deliver only when the script prints
  something. Empty output → nothing sent. **This is what makes monitors silent.**
- `True` *(default for `--instruction`)* — always deliver (empty becomes "erledigt").
- `error` — deliver only when the job fails.
- `False` — never deliver on success.

Failures (script exits non-zero / raises) are **always** surfaced, in every mode —
a broken monitor must not fail silently.

A script reports by printing to stdout. To stay silent it simply prints nothing
(or the marker `PAWLIA_SILENT`).

## The harness

Job scripts import `pawlia.automation_harness` (always importable inside a job;
the scheduler puts it on `PYTHONPATH`):

```python
from pawlia.automation_harness import get_params, emit, silent, llm_call, log

get_params() -> dict      # the job's --params, e.g. {"city": "Magdeburg"}
emit(text)                # report text → becomes the notification (empty = silent)
silent()                  # say nothing (documents intent; a no-op)
log(msg)                  # diagnostics to stderr (never sent to the user)
llm_call(prompt, system=None, model=None) -> str   # one robust LLM call, retries
```

Rules: do the deterministic check first; call `emit()` only when there is
something to report; use `llm_call()` sparingly; let exceptions propagate so
failures are loud.

## Two patterns

### A — Silent monitor (Gewitter / Bahn): gate, mostly silent, little/no LLM

```python
#!/usr/bin/env python
from pawlia.automation_harness import get_params, emit, silent, log
import urllib.request, json

p = get_params()
city = p.get("city", "Magdeburg")
# ... fetch forecast from a weather API ...
data = json.loads(urllib.request.urlopen(f"https://api.example/forecast?q={city}").read())

if not data.get("thunderstorm"):
    silent()                      # quiet day → nothing sent
else:
    emit(f"⛈ Gewitterwarnung {city}: {data['summary']} ab {data['start']}")
```

### B — LLM-curated digest (Morgenbericht): gather, then let the LLM shape it

```python
#!/usr/bin/env python
import sys
from pawlia.automation_harness import get_params, emit, llm_call

messages = collect_all_messages()         # deterministic data gathering
if not messages:
    sys.exit(0)                           # nothing to report → silent
report = llm_call(
    system="Du bist ein knapper Nachrichten-Kurator. Fasse zusammen, gruppiere, priorisiere.",
    prompt="Erstelle einen aufgeräumten Morgenbericht aus:\n\n" + "\n".join(messages),
)
emit(report)
```

## Building a job

### Step 1 — build the script (use skill-creator for anything non-trivial)

Trivial gate you can write inline → write it to `workspace/.scripts/<name>.py`.
Anything with API calls, parsing, or LLM curation → ask skill-creator to build it
(it knows the harness skeleton and will test the script before you register it):

```
Call skill-creator with: "Baue ein Automations-Skript das [task]. Nutze das
pawlia.automation_harness (get_params/emit/silent/llm_call) und bleib still wenn
nichts zu melden ist."
```

The script must live under `workspace/.scripts/` (the only place the scheduler
resolves job scripts from, alongside the legacy `automations/`).

### Step 2 — register the job

```bash
python <scripts_dir>/../organizer/scripts/organizer.py add-job \
  --name "<descriptive name>" \
  --schedule "<schedule>" \
  --script "<name>.py" \
  [--params '{"city":"Magdeburg"}']
```

For a trivial LLM job instead of a script:

```bash
python <scripts_dir>/../organizer/scripts/organizer.py add-job \
  --name "<name>" --schedule "<schedule>" --instruction "<instruction>"
```

**Notify flags** (override the per-kind default):
- `--notify-on-output` — deliver only when there is output (silent on empty). Default for `--script`.
- `--notify-on-error` — deliver only on failure.
- `--no-notify` — never deliver on success (failures still surface).

**Schedule formats:**
- `"16:00"` — daily at 16:00
- `"interval:5m"` — every 5 minutes (m/h/d)
- `"interval:1h"` — every hour
- `"weekly:0:09:00"` — weekly Monday 09:00 (0=Mon..6=Sun)
- `"monthly:1:10:00"` — monthly on the 1st at 10:00
- `"30 8 * * 1,3"` — cron syntax (5 space-separated fields)

### Managing existing jobs

```bash
python <scripts_dir>/../organizer/scripts/organizer.py list-jobs
python <scripts_dir>/../organizer/scripts/organizer.py delete-job --job-id "<id>"
python <scripts_dir>/../organizer/scripts/organizer.py toggle-job --job-id "<id>"
```

### Manual trigger (for testing)

```bash
python <scripts_dir>/../organizer/scripts/organizer.py run-job --job-id "<id>"
```

Flags the job for immediate execution on the next scheduler tick (within 60s).
Use this after building/modifying the script to verify the job behaves: confirm
the silent case sends nothing and the alert case sends exactly one message.

## Output

After registering, confirm to the user:
- What the automation does and whether it's a script or an instruction
- How often it runs (the schedule)
- That it stays silent unless there is something to report (for monitors)
- If a skill/script was built: mention its name
- That they can list, disable, delete, or manually trigger it later
