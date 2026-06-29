"""Automation runtime helpers — offset parsing, scheduler state, the script
sandbox, job due-logic and the small factory/interpolation helpers.

The cron *validation* and ``_cron_matches`` are covered in
test_scheduler_datemath.py; here we pin the surrounding machinery that decides
*when* and *whether* a job runs and *what* it is allowed to run. ``_is_due`` is
fed an explicit ``now`` so the assertions are deterministic.
"""

import asyncio
import os
from datetime import datetime, timedelta

import pytest

import json

from pawlia.automation import (
    ChecklistProcessor,
    JobRunner,
    ScriptExecutor,
    SILENT_SENTINEL,
    _load_state,
    _parse_offset,
    _save_state,
    _strip_sentinel,
    _success_notification,
    create_checklist_item,
    create_job,
)


# ---- _parse_offset ---------------------------------------------------------
@pytest.mark.parametrize("text,expected", [
    ("-90m", timedelta(minutes=-90)),
    ("+30m", timedelta(minutes=30)),
    ("2h", timedelta(hours=2)),
    ("-1d", timedelta(days=-1)),
    (" 15m ", timedelta(minutes=15)),
])
def test_parse_offset_handles_signed_units(text, expected):
    assert _parse_offset(text) == expected


@pytest.mark.parametrize("bad", ["90", "5w", "abc", ""])
def test_parse_offset_rejects_unknown_units(bad):
    with pytest.raises(ValueError):
        _parse_offset(bad)


# ---- scheduler state -------------------------------------------------------
def test_state_roundtrips_through_disk(tmp_path):
    _save_state(str(tmp_path), "u1", {"job-1": "2026-06-04T09:00:00"})
    assert _load_state(str(tmp_path), "u1") == {"job-1": "2026-06-04T09:00:00"}


def test_load_state_defaults_to_empty_when_absent(tmp_path):
    assert _load_state(str(tmp_path), "nobody") == {}


def test_load_state_tolerates_corrupt_file(tmp_path):
    path = tmp_path / "u1" / "scheduler_state.json"
    path.parent.mkdir(parents=True)
    path.write_text("{ not json", encoding="utf-8")
    assert _load_state(str(tmp_path), "u1") == {}


# ---- ScriptExecutor sandbox ------------------------------------------------
def test_is_allowed_path_accepts_user_automations_dir(tmp_path):
    script = tmp_path / "u1" / "automations" / "do.py"
    script.parent.mkdir(parents=True)
    script.write_text("print('hi')", encoding="utf-8")
    assert ScriptExecutor._is_allowed_path(str(script), "u1", str(tmp_path)) is True


def test_is_allowed_path_rejects_arbitrary_paths(tmp_path):
    script = tmp_path / "evil.py"
    script.write_text("print('pwn')", encoding="utf-8")
    assert ScriptExecutor._is_allowed_path(str(script), "u1", str(tmp_path)) is False


def test_run_executes_an_allowed_script_and_captures_stdout(tmp_path):
    script = tmp_path / "u1" / "automations" / "ok.py"
    script.parent.mkdir(parents=True)
    script.write_text("print('hello from script')", encoding="utf-8")

    result = asyncio.run(ScriptExecutor.run(
        str(script), user_id="u1", session_dir=str(tmp_path)))

    assert result["success"] is True
    assert "hello from script" in result["output"]


def test_run_surfaces_nonzero_exit_as_failure(tmp_path):
    script = tmp_path / "u1" / "automations" / "boom.py"
    script.parent.mkdir(parents=True)
    script.write_text("import sys; sys.stderr.write('nope'); sys.exit(3)", encoding="utf-8")

    result = asyncio.run(ScriptExecutor.run(
        str(script), user_id="u1", session_dir=str(tmp_path)))

    assert result["success"] is False
    assert "nope" in result["error"]


def test_run_refuses_a_script_outside_the_sandbox(tmp_path):
    script = tmp_path / "evil.py"
    script.write_text("print('pwn')", encoding="utf-8")

    result = asyncio.run(ScriptExecutor.run(
        str(script), user_id="u1", session_dir=str(tmp_path)))

    assert result["success"] is False
    assert "erlaubten" in result["error"]  # German sandbox message


def test_run_reports_missing_script(tmp_path):
    result = asyncio.run(ScriptExecutor.run(str(tmp_path / "ghost.py")))
    assert result["success"] is False
    assert "not found" in result["error"].lower()


def test_run_passes_params_via_env(tmp_path):
    script = tmp_path / "u1" / "automations" / "echo.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "import os; print(os.environ.get('AUTOMATION_PARAMS', 'MISSING'))",
        encoding="utf-8",
    )
    result = asyncio.run(ScriptExecutor.run(
        str(script), params={"k": "v"}, user_id="u1", session_dir=str(tmp_path)))
    assert result["success"] is True
    assert '"k": "v"' in result["output"]


# ---- JobRunner._is_due -----------------------------------------------------
NOW = datetime(2026, 6, 4, 9, 0, 0)  # a fixed reference instant


def test_is_due_false_without_schedule():
    assert JobRunner._is_due({}, NOW) is False


def test_is_due_interval_fires_on_first_run():
    assert JobRunner._is_due({"schedule": "interval:30m"}, NOW) is True


def test_is_due_interval_waits_for_the_window():
    recent = (NOW - timedelta(minutes=10)).isoformat()
    assert JobRunner._is_due(
        {"schedule": "interval:30m", "last_run": recent}, NOW) is False


def test_is_due_interval_fires_after_the_window():
    old = (NOW - timedelta(minutes=40)).isoformat()
    assert JobRunner._is_due(
        {"schedule": "interval:30m", "last_run": old}, NOW) is True


def test_is_due_interval_rejects_bad_offset():
    assert JobRunner._is_due({"schedule": "interval:5w"}, NOW) is False


def test_is_due_daily_fires_in_the_minute_window():
    schedule = f"{NOW.hour}:{NOW.minute:02d}"
    assert JobRunner._is_due({"schedule": schedule}, NOW) is True


def test_is_due_daily_skips_other_times():
    schedule = f"{(NOW.hour + 1) % 24}:00"
    assert JobRunner._is_due({"schedule": schedule}, NOW) is False


def test_is_due_daily_blocked_when_just_run():
    schedule = f"{NOW.hour}:{NOW.minute:02d}"
    just_ran = NOW.isoformat()
    assert JobRunner._is_due({"schedule": schedule, "last_run": just_ran}, NOW) is False


def test_is_due_weekly_fires_on_the_right_weekday():
    schedule = f"weekly:{NOW.weekday()}:{NOW.hour}:{NOW.minute}"
    assert JobRunner._is_due({"schedule": schedule}, NOW) is True


def test_is_due_weekly_skips_wrong_weekday():
    schedule = f"weekly:{(NOW.weekday() + 1) % 7}:{NOW.hour}:{NOW.minute}"
    assert JobRunner._is_due({"schedule": schedule}, NOW) is False


def test_is_due_monthly_fires_on_the_right_day():
    schedule = f"monthly:{NOW.day}:{NOW.hour}:{NOW.minute}"
    assert JobRunner._is_due({"schedule": schedule}, NOW) is True


def test_is_due_cron_matches_minute_and_hour():
    schedule = f"{NOW.minute} {NOW.hour} * * *"
    assert JobRunner._is_due({"schedule": schedule}, NOW) is True


# ---- ChecklistProcessor._interpolate --------------------------------------
def test_interpolate_fills_known_event_fields():
    msg = "Termin {title} um {start} in {location}"
    event = {"title": "Standup", "start": "09:00", "location": "Raum 1"}
    assert ChecklistProcessor._interpolate(msg, event) == "Termin Standup um 09:00 in Raum 1"


def test_interpolate_blanks_missing_fields():
    assert ChecklistProcessor._interpolate("Ort: {location}", {}) == "Ort: "


# ---- ChecklistProcessor skips event_reminder items -------------------------
@pytest.mark.asyncio
async def test_checklist_processor_skips_event_reminder_items(tmp_path):
    """Reminders derived from event frontmatter are fired recurrence-aware by
    the scheduler's _check_events, so ChecklistProcessor must not also fire
    them (would double-notify)."""
    import yaml

    user = "u1"
    cal_dir = tmp_path / user / "workspace" / "calendar"
    cal_dir.mkdir(parents=True)
    start = datetime.now() - timedelta(hours=1)
    fm = {
        "title": "Parcours",
        "date": start.strftime("%Y-%m-%d"),
        "startTime": start.strftime("%H:%M"),
        "allDay": False,
        "type": "single",
        "checklist": [
            {"id": "rem-1", "trigger": "relative", "trigger_offset": "-40m",
             "message": "sollte NICHT feuern", "notify": True, "source": "event_reminder"},
            {"id": "chk-1", "trigger": "relative", "trigger_offset": "-40m",
             "message": "Prep: Unterlagen", "notify": True},
        ],
    }
    (cal_dir / "2026-05-19 Parcours.md").write_text(
        f"---\n{yaml.dump(fm, allow_unicode=True, default_flow_style=False).rstrip()}\n---\n",
        encoding="utf-8",
    )

    notifications = []

    async def capture(uid, msg):
        notifications.append(msg)

    proc = ChecklistProcessor(str(tmp_path), capture)
    await proc.process_user(user)

    assert any("Prep: Unterlagen" in n for n in notifications)
    assert not any("sollte NICHT feuern" in n for n in notifications)


# ---- factory helpers -------------------------------------------------------
def test_create_checklist_item_has_prefixed_id_and_defaults():
    item = create_checklist_item(script="x.py", message="hi")
    assert item["id"].startswith("chk-")
    assert item["script"] == "x.py"
    assert item["trigger"] == "relative"
    assert item["params"] == {}
    assert item["notify"] is True


def test_create_job_has_prefixed_id_and_is_enabled():
    job = create_job("Daily", "9:00", "do the thing", notify="error")
    assert job["id"].startswith("job-")
    assert job["name"] == "Daily"
    assert job["schedule"] == "9:00"
    assert job["instruction"] == "do the thing"
    assert job["notify"] == "error"
    assert job["enabled"] is True
    assert job["last_run"] == ""


def test_create_job_ids_are_unique():
    assert create_job("a", "9:00", "x")["id"] != create_job("b", "9:00", "y")["id"]


# ---- _strip_sentinel / _success_notification ------------------------------
@pytest.mark.parametrize("raw,expected", [
    ("", ""),
    ("   \n ", ""),
    (SILENT_SENTINEL, ""),
    (f"  {SILENT_SENTINEL}  ", ""),
    ("SILENT", ""),            # bare word an LLM job emits
    ("  silent\n", ""),        # case-insensitive + whitespace
    ("‎", ""),            # U+200E LEFT-TO-RIGHT MARK — model's "say nothing" reply
    ("‎​  ", ""),    # only zero-width / format chars + whitespace
    ("‎SILENT‎", ""),  # marker wrapped in invisible chars
    ("Gewitter um 16:00", "Gewitter um 16:00"),
    (f"{SILENT_SENTINEL} aber doch was", f"{SILENT_SENTINEL} aber doch was"),  # only a *bare* marker silences
    ("SILENT, aber doch was", "SILENT, aber doch was"),  # not a bare marker
])
def test_strip_sentinel(raw, expected):
    assert _strip_sentinel(raw) == expected


@pytest.mark.parametrize("notify,output,expected", [
    (True, "", "erledigt"),            # always deliver; empty becomes erledigt
    (True, "hi", "hi"),
    (True, "SILENT", None),            # explicit silence wins over notify=True (LLM job)
    (True, SILENT_SENTINEL, None),     # ... same for the canonical marker
    (True, "‎", None),           # U+200E invisible reply → no "alert without text"
    (True, "   ", None),               # whitespace-only carries no message
    ("output_only", "", None),         # silent on empty — the monitor case
    ("output_only", SILENT_SENTINEL, None),
    ("output_only", "SILENT", None),   # bare LLM marker also silences
    ("output_only", "‎", None),  # invisible reply silences here too
    ("output_only", "Warnung", "Warnung"),
    ("error", "hi", None),             # nothing on success
    (False, "hi", None),
])
def test_success_notification(notify, output, expected):
    assert _success_notification(notify, output) == expected


# ---- create_job script defaults -------------------------------------------
def test_create_job_script_defaults_to_output_only():
    job = create_job("Gewitter", "interval:1h", script="gewitter.py", params={"city": "X"})
    assert job["script"] == "gewitter.py"
    assert job["params"] == {"city": "X"}
    assert job["notify"] == "output_only"   # script jobs stay silent by default
    assert job["instruction"] == ""


def test_create_job_instruction_defaults_to_true():
    job = create_job("Daily", "9:00", instruction="do it")
    assert job["notify"] is True
    assert job["script"] == ""


# ---- JobRunner script branch (end to end) ---------------------------------
def _write_script_job(tmp_path, user, body, *, notify=None, params=None):
    """Drop a script in automations/ and register a due job pointing at it."""
    autodir = tmp_path / user / "automations"
    autodir.mkdir(parents=True, exist_ok=True)
    (autodir / "mon.py").write_text(body, encoding="utf-8")
    job = create_job("Monitor", "interval:1m", script="mon.py",
                     notify=notify, params=params)
    (autodir / "jobs.json").write_text(json.dumps([job]), encoding="utf-8")
    return job


async def _run_jobs(tmp_path, user):
    notifications = []

    async def capture(uid, msg):
        notifications.append(msg)

    await JobRunner(str(tmp_path), capture, app=None).process_user(user)
    return notifications


@pytest.mark.asyncio
async def test_script_job_stays_silent_on_empty_output(tmp_path):
    _write_script_job(tmp_path, "u1", "pass\n")  # prints nothing
    assert await _run_jobs(tmp_path, "u1") == []


@pytest.mark.asyncio
async def test_script_job_stays_silent_on_sentinel(tmp_path):
    _write_script_job(tmp_path, "u1", f"print({SILENT_SENTINEL!r})\n")
    assert await _run_jobs(tmp_path, "u1") == []


@pytest.mark.asyncio
async def test_script_job_notifies_on_real_output(tmp_path):
    _write_script_job(tmp_path, "u1", "print('Gewitterwarnung 16:00')\n")
    notes = await _run_jobs(tmp_path, "u1")
    assert len(notes) == 1
    assert "Gewitterwarnung 16:00" in notes[0]


@pytest.mark.asyncio
async def test_script_job_failure_is_always_loud(tmp_path):
    # Even with output_only (the silent default), a crash must surface.
    _write_script_job(tmp_path, "u1", "import sys; sys.exit(2)\n")
    notes = await _run_jobs(tmp_path, "u1")
    assert len(notes) == 1
    assert "fehlgeschlagen" in notes[0]


@pytest.mark.asyncio
async def test_script_job_receives_params(tmp_path):
    body = (
        "import os, json\n"
        "p = json.loads(os.environ['AUTOMATION_PARAMS'])\n"
        "print(f\"Stadt {p['city']}\")\n"
    )
    _write_script_job(tmp_path, "u1", body, params={"city": "Magdeburg"})
    notes = await _run_jobs(tmp_path, "u1")
    assert any("Stadt Magdeburg" in n for n in notes)
