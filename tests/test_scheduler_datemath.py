"""Recurrence & calendar contracts: date math + RRULE generation/roundtrip.

Brings together the recurrence logic that was scattered across the scheduler,
the organizer skill and the Radicale CalDAV bridge:
- ``_next_occurrence`` advances a datetime by a natural-language cadence;
- the organizer skill writes Full-Calendar recurring frontmatter (RRULE +
  daysOfWeek) and per-event reminders;
- the Radicale storage round-trips an RRULE between Markdown and iCal.

All deterministic, no LLM. (The scheduler's *firing* behavior — due reminders,
event notifications — stays in test_scheduler.py as a system test.)
"""

import argparse
import importlib.util
import io
import json
import sys
from datetime import datetime
from pathlib import Path

import pytest
import yaml

from pawlia.scheduler import _next_occurrence

REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# _next_occurrence — natural-language cadence math
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cadence,expected", [
    ("every day", datetime(2026, 3, 16, 10, 0)),
    ("every week", datetime(2026, 3, 22, 10, 0)),
    ("every month", datetime(2026, 4, 15, 10, 0)),
    ("unknown", datetime(2026, 3, 16, 10, 0)),  # falls back to daily
])
def test_next_occurrence_advances_by_cadence(cadence, expected):
    assert _next_occurrence(datetime(2026, 3, 15, 10, 0), cadence) == expected


def test_next_occurrence_monthly_wraps_the_year():
    assert _next_occurrence(datetime(2026, 12, 15, 10, 0), "every month") == \
        datetime(2027, 1, 15, 10, 0)


def test_next_occurrence_monthly_clamps_day_overflow():
    result = _next_occurrence(datetime(2026, 1, 31, 10, 0), "every month")
    assert (result.month, result.day) == (2, 28)


# ---------------------------------------------------------------------------
# organizer skill — Full Calendar recurring frontmatter + reminders
# ---------------------------------------------------------------------------
def _load_organizer():
    path = REPO / "skills" / "organizer" / "scripts" / "organizer.py"
    spec = importlib.util.spec_from_file_location("organizer_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _event_args(tmp_path, **overrides):
    base = dict(
        user_id="u1", session_dir=str(tmp_path), title="Parcours",
        start="2026-05-19T17:00:00", end="2026-05-19T18:30:00",
        description="Training", location="", checklist="", reminders="",
        recurrence="none", recurrence_days="", recurrence_until="",
        recurrence_count=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_organizer_writes_weekly_rrule_with_explicit_days(tmp_path):
    mod = _load_organizer()
    mod.cmd_add_event(_event_args(
        tmp_path, recurrence="weekly", recurrence_days="TU,TH",
        recurrence_until="2026-12-31",
    ))
    path = tmp_path / "u1" / "workspace" / "calendar" / "2026-05-19 Parcours.md"
    fm = yaml.safe_load(path.read_text(encoding="utf-8").split("---", 2)[1])

    assert fm["type"] == "recurring"
    assert fm["rrule"] == "FREQ=WEEKLY;BYDAY=TU,TH;UNTIL=20261231"
    assert fm["daysOfWeek"] == [2, 4]
    assert fm["startRecur"] == "2026-05-19"
    assert fm["endRecur"] == "2026-12-31"


def test_organizer_weekly_event_defaults_to_the_start_weekday(tmp_path):
    mod = _load_organizer()
    mod.cmd_add_event(_event_args(
        tmp_path, title="Dienstagstraining", description="", recurrence="weekly",
    ))
    path = tmp_path / "u1" / "workspace" / "calendar" / "2026-05-19 Dienstagstraining.md"
    fm = yaml.safe_load(path.read_text(encoding="utf-8").split("---", 2)[1])

    assert fm["rrule"] == "FREQ=WEEKLY;BYDAY=TU"
    assert fm["daysOfWeek"] == [2]


def test_organizer_persists_reminders_as_a_scheduler_checklist(tmp_path):
    mod = _load_organizer()
    mod.cmd_add_event(_event_args(
        tmp_path, reminders='[{"minutes_before": 40, "message": "Bereitmachen"}]',
    ))
    path = tmp_path / "u1" / "workspace" / "calendar" / "2026-05-19 Parcours.md"
    fm = yaml.safe_load(path.read_text(encoding="utf-8").split("---", 2)[1])

    assert fm["reminders"] == [{"minutes_before": 40, "message": "Bereitmachen"}]
    assert fm["checklist"][0]["source"] == "event_reminder"
    assert fm["checklist"][0]["trigger_offset"] == "-40m"


# ---------------------------------------------------------------------------
# add-reminder with weekday-specific recurrence → recurring calendar event
# ---------------------------------------------------------------------------
def _reminder_args(tmp_path, **overrides):
    base = dict(
        user_id="u1", session_dir=str(tmp_path),
        fire_at="2026-06-15T13:48",
        message="RB40 Check -30min Rückweg",
        label="RB40 Check",
        recurrence="",
    )
    base.update(overrides)
    return argparse.Namespace(**base)


@pytest.mark.parametrize("recurrence,expected_byday", [
    ("mo,mi", "MO,WE"),
    ("monday,wednesday", "MO,WE"),
    ("MO/MI", "MO,WE"),
    ("montag mittwoch", "MO,WE"),
    ("FREQ=WEEKLY;BYDAY=MO,WE", "MO,WE"),
    ("every monday and wednesday", "MO,WE"),
])
def test_add_reminder_redirects_weekday_recurrence_to_event(tmp_path, capsys, recurrence, expected_byday):
    mod = _load_organizer()
    mod.cmd_add_reminder(_reminder_args(tmp_path, recurrence=recurrence))

    result = json.loads(capsys.readouterr().out)
    assert result["success"] is True
    assert result["redirected_to_event"] is True

    path = tmp_path / "u1" / "workspace" / "calendar" / "2026-06-15 RB40 Check.md"
    assert path.exists()
    fm = yaml.safe_load(path.read_text(encoding="utf-8").split("---", 2)[1])
    assert fm["type"] == "recurring"
    assert fm["rrule"] == f"FREQ=WEEKLY;BYDAY={expected_byday}"
    assert fm["reminders"] == [
        {"minutes_before": 0, "message": "RB40 Check -30min Rückweg", "notify": True},
    ]
    # No tasks.md reminder may be created for the redirected case.
    assert not (tmp_path / "u1" / "workspace" / "tasks.md").exists()


@pytest.mark.parametrize("recurrence", [
    "", "none", "daily", "weekly", "monthly", "every week", "every day",
])
def test_add_reminder_keeps_plain_cadence_as_task(tmp_path, capsys, recurrence):
    mod = _load_organizer()
    mod.cmd_add_reminder(_reminder_args(tmp_path, recurrence=recurrence))

    result = json.loads(capsys.readouterr().out)
    assert result["success"] is True
    assert "redirected_to_event" not in result
    # No calendar event created.
    assert not list((tmp_path / "u1" / "workspace" / "calendar").glob("*.md"))
    # A tasks.md reminder exists.
    md = (tmp_path / "u1" / "workspace" / "tasks.md").read_text(encoding="utf-8")
    assert "RB40 Check -30min Rückweg" in md


def test_organizer_run_job_forces_run_and_returns_the_instruction(tmp_path, capsys):
    import json
    mod = _load_organizer()
    mod.cmd_add_job(argparse.Namespace(
        user_id="u1", session_dir=str(tmp_path), name="Morgenbericht",
        schedule="08:00", instruction="Erstelle einen Morgenbericht",
        script="", params="", no_notify=False, notify_on_error=False,
        notify_on_output=False,
    ))
    job_id = json.loads(capsys.readouterr().out)["job_id"]

    mod.cmd_run_job(argparse.Namespace(
        user_id="u1", session_dir=str(tmp_path), job_id=job_id,
    ))
    result = json.loads(capsys.readouterr().out)

    assert result["success"] is True
    assert result["instruction"] == "Erstelle einen Morgenbericht"
    jobs = mod._load_json(str(tmp_path / "u1" / "automations" / "jobs.json"))
    assert jobs[0].get("force_run") is True


# ---------------------------------------------------------------------------
# Radicale CalDAV bridge — RRULE survives the Markdown <-> iCal roundtrip
# ---------------------------------------------------------------------------
def _load_radicale_storage():
    pytest.importorskip("radicale")
    pytest.importorskip("icalendar")
    pytest.importorskip("vobject")
    path = REPO / "radicale" / "picoclaw_storage.py"
    spec = importlib.util.spec_from_file_location("picoclaw_storage", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_markdown_recurring_event_exports_its_rrule_to_ical(tmp_path):
    mod = _load_radicale_storage()
    path = tmp_path / "2026-05-19 Parcours.md"
    path.write_text(
        "---\ntitle: Parcours\ndate: '2026-05-19'\nstartTime: '17:00'\n"
        "endTime: '18:30'\nallDay: false\ntype: recurring\n"
        "rrule: FREQ=WEEKLY;BYDAY=TU,TH\n---\n\nTraining\n",
        encoding="utf-8",
    )
    ical = mod._md_to_ical(str(path), "2026-05-19 Parcours.ics")

    assert ical is not None
    assert "RRULE:FREQ=WEEKLY;BYDAY=TU,TH" in ical


def test_ical_recurring_event_imports_fullcalendar_fields():
    mod = _load_radicale_storage()
    ical = "\r\n".join([
        "BEGIN:VCALENDAR", "VERSION:2.0", "BEGIN:VEVENT", "UID:event-1",
        "SUMMARY:Parcours", "DTSTART:20260519T170000", "DTEND:20260519T183000",
        "RRULE:FREQ=WEEKLY;BYDAY=TU,TH;UNTIL=20261231", "END:VEVENT",
        "END:VCALENDAR", "",
    ])
    content, _meta = mod._ical_to_md(ical)
    fm = yaml.safe_load(content.split("---", 2)[1])

    assert fm["type"] == "recurring"
    assert fm["rrule"] == "FREQ=WEEKLY;BYDAY=TU,TH;UNTIL=20261231"
    assert fm["daysOfWeek"] == [2, 4]


# ---------------------------------------------------------------------------
# schedule validation — single source of truth in pawlia.automation
# ---------------------------------------------------------------------------
def _load_automation():
    spec = importlib.util.spec_from_file_location(
        "pawlia_automation", REPO / "pawlia" / "automation.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("schedule", [
    "16:00", "08:30", "0:0",
    "interval:30m", "interval:2h", "interval:1d",
    "weekly:0:09:00", "weekly:6:23:59",
    "monthly:1:10:00", "monthly:31:00:00",
    "* * * * *", "30 8 * * 1,3", "*/5 * * * *",
    "0 0 1,15 * *", "0 9-17 * * 1-5",
])
def test_validate_schedule_accepts_well_formed_inputs(schedule):
    mod = _load_automation()
    ok, err = mod.validate_schedule(schedule)
    assert ok is True, f"expected '{schedule}' to be valid, got error: {err!r}"


@pytest.mark.parametrize("schedule", [
    "", "   ",
    "interval:xyz", "interval:",
    "weekly:7:00:00", "weekly:0:24:00", "weekly:0:00:60",
    "monthly:0:00:00", "monthly:32:00:00",
    "99:99", "25:00", "12:60",
    "* * * *",        # too few fields
    "* * * * * *",    # too many fields
    "60 * * * *",     # minute out of range
    "* 24 * * *",     # hour out of range
    "* * 32 * *",     # day out of range
    "* * 0 * *",      # day=0 out of range
    "* * * 13 *",     # month=13 out of range
    "* * * 0 *",      # month=0 out of range
    "* * * * 7",      # dow=7 out of range
    "0 9-99 * * *",   # hour range overflow
    "0 0 1,40 * *",   # day list overflow
    "*/0 * * * *",    # zero step
    "*/100 * * * *",  # step larger than field
    "abc * * * *",    # non-numeric
])
def test_validate_schedule_rejects_malformed_inputs(schedule):
    mod = _load_automation()
    ok, _err = mod.validate_schedule(schedule)
    assert ok is False, f"expected '{schedule}' to be rejected"


def test_cron_matches_at_exact_minute():
    mod = _load_automation()
    now = datetime(2026, 6, 4, 8, 30)  # Thu 4 Jun 2026
    assert mod._cron_matches("30 8 * * *", now) is True
    assert mod._cron_matches("0 8 * * *", now) is False


def test_cron_matches_day_of_week_with_zero_based_sunday():
    mod = _load_automation()
    # weekday(): Mon=0..Sun=6; cron dow: Sun=0..Sat=6
    # Thu Jun 4 2026 -> weekday=3 -> cron dow=(3+1)%7=4
    thursday = datetime(2026, 6, 4, 8, 30)
    assert mod._cron_matches("30 8 * * 4", thursday) is True
    assert mod._cron_matches("30 8 * * 1,3", thursday) is False  # Sun, Tue


def test_organizer_add_job_rejects_bad_cron(tmp_path, capsys):
    import json
    mod = _load_organizer()
    with pytest.raises(SystemExit):
        mod.cmd_add_job(argparse.Namespace(
            user_id="u1", session_dir=str(tmp_path), name="Bad",
            schedule="60 * * * *", instruction="x",
            script="", params="", no_notify=False, notify_on_error=False,
            notify_on_output=False,
        ))
    result = json.loads(capsys.readouterr().out)
    assert result["success"] is False
    assert "0-59" in result["error"] or "Minute" in result["error"]


def test_organizer_add_job_accepts_cron(tmp_path, capsys):
    mod = _load_organizer()
    mod.cmd_add_job(argparse.Namespace(
        user_id="u1", session_dir=str(tmp_path), name="WeekdayMorning",
        schedule="30 8 * * 1-5", instruction="Morgenbericht",
        script="", params="", no_notify=False, notify_on_error=False,
        notify_on_output=False,
    ))
    result = json.loads(capsys.readouterr().out)
    assert result["success"] is True
    jobs = mod._load_json(str(tmp_path / "u1" / "automations" / "jobs.json"))
    assert jobs[0]["schedule"] == "30 8 * * 1-5"


# ---------------------------------------------------------------------------
# organizer note-task complete / delete — audit problem #1
# ---------------------------------------------------------------------------
def _seed_note_workspace(tmp_path) -> dict:
    """Create a workspace with one memory note containing 4 task lines."""
    user = "u1"
    ws = tmp_path / user / "workspace"
    mem = ws / "memory"
    mem.mkdir(parents=True)
    note = mem / "2026-03-25.md"
    note.write_text(
        "# 25. März\n\n"
        "- [ ] Initiales Konzept erstellen\n"
        "- [ ] Hardware-Auswahl\n"
        "- [ ] Budgetplanung\n"
        "- [ ] Zeitplanung\n",
        encoding="utf-8",
    )
    return {
        "session": str(tmp_path),
        "user": user,
        "note": note,
    }


def _capture(mod, fn, *args, **kwargs):
    """Run a cmd_* function, capture its JSON stdout, and return the parsed dict.

    Returns (parsed_dict, exit_code). exit_code is None if the call exited
    cleanly (success or no _out call), or the integer passed to sys.exit.
    """
    buf = io.StringIO()
    real = sys.stdout
    sys.stdout = buf
    exit_code = None
    try:
        try:
            fn(*args, **kwargs)
        except SystemExit as e:
            exit_code = e.code
    finally:
        sys.stdout = real
    return json.loads(buf.getvalue()), exit_code


def test_complete_task_toggles_note_task_in_source_file(tmp_path, capsys):
    mod = _load_organizer()
    seed = _seed_note_workspace(tmp_path)
    listing, _ = _capture(mod, mod.cmd_list_tasks, argparse.Namespace(
        user_id=seed["user"], session_dir=seed["session"],
        status="all", limit=100, project="",
    ))
    target = next(t for t in listing["tasks"]
                  if t["title"] == "Initiales Konzept erstellen")
    assert target["source"] == "memory/2026-03-25.md"

    result, _ = _capture(mod, mod.cmd_complete_task, argparse.Namespace(
        user_id=seed["user"], session_dir=seed["session"],
        task_id=target["id"],
    ))
    assert result["success"] is True

    after = seed["note"].read_text(encoding="utf-8")
    assert "- [x] Initiales Konzept erstellen ✅" in after
    # The other tasks in the same file must be untouched
    assert "- [ ] Hardware-Auswahl\n" in after
    assert "- [ ] Budgetplanung\n" in after


def test_complete_task_is_idempotent_on_note_task(tmp_path):
    mod = _load_organizer()
    seed = _seed_note_workspace(tmp_path)
    listing, _ = _capture(mod, mod.cmd_list_tasks, argparse.Namespace(
        user_id=seed["user"], session_dir=seed["session"],
        status="all", limit=100, project="",
    ))
    target = next(t for t in listing["tasks"] if t["title"] == "Hardware-Auswahl")
    _capture(mod, mod.cmd_complete_task, argparse.Namespace(
        user_id=seed["user"], session_dir=seed["session"],
        task_id=target["id"],
    ))
    # Second call should be a no-op (no double-✅).
    _capture(mod, mod.cmd_complete_task, argparse.Namespace(
        user_id=seed["user"], session_dir=seed["session"],
        task_id=target["id"],
    ))
    after = seed["note"].read_text(encoding="utf-8")
    assert after.count("✅") == 1, after


def test_delete_task_removes_note_task_line(tmp_path):
    mod = _load_organizer()
    seed = _seed_note_workspace(tmp_path)
    listing, _ = _capture(mod, mod.cmd_list_tasks, argparse.Namespace(
        user_id=seed["user"], session_dir=seed["session"],
        status="all", limit=100, project="",
    ))
    target = next(t for t in listing["tasks"] if t["title"] == "Budgetplanung")

    result, _ = _capture(mod, mod.cmd_delete_task, argparse.Namespace(
        user_id=seed["user"], session_dir=seed["session"],
        task_id=target["id"],
    ))
    assert result["success"] is True
    assert result["remaining"] == 3

    after = seed["note"].read_text(encoding="utf-8")
    assert "Budgetplanung" not in after
    assert "Initiales Konzept" in after
    assert "Hardware-Auswahl" in after
    assert "Zeitplanung" in after


def test_complete_and_delete_fall_back_to_tasks_md(tmp_path):
    mod = _load_organizer()
    # tasks.md only, no workspace notes
    user = "u1"
    (tmp_path / user / "workspace").mkdir(parents=True)
    add, _ = _capture(mod, mod.cmd_add_task, argparse.Namespace(
        user_id=user, session_dir=str(tmp_path), title="In Tasks",
        due_date="", priority="", project="", reminders="", description="",
    ))
    tid = add["task_id"]

    c, _ = _capture(mod, mod.cmd_complete_task, argparse.Namespace(
        user_id=user, session_dir=str(tmp_path), task_id=tid,
    ))
    assert c["success"] is True
    md = (tmp_path / user / "workspace" / "tasks.md").read_text(encoding="utf-8")
    assert "- [x] In Tasks" in md

    add, _ = _capture(mod, mod.cmd_add_task, argparse.Namespace(
        user_id=user, session_dir=str(tmp_path), title="Auch In Tasks",
        due_date="", priority="", project="", reminders="", description="",
    ))
    d, _ = _capture(mod, mod.cmd_delete_task, argparse.Namespace(
        user_id=user, session_dir=str(tmp_path), task_id=add["task_id"],
    ))
    assert d["success"] is True
    md = (tmp_path / user / "workspace" / "tasks.md").read_text(encoding="utf-8")
    assert "Auch In Tasks" not in md


def test_complete_task_still_falls_back_to_title_substring_in_tasks_md(tmp_path):
    """Legacy callers pass a title substring instead of a stable ID."""
    mod = _load_organizer()
    user = "u1"
    (tmp_path / user / "workspace").mkdir(parents=True)
    _capture(mod, mod.cmd_add_task, argparse.Namespace(
        user_id=user, session_dir=str(tmp_path), title="Einkaufen",
        due_date="", priority="", project="", reminders="", description="",
    ))
    result, _ = _capture(mod, mod.cmd_complete_task, argparse.Namespace(
        user_id=user, session_dir=str(tmp_path), task_id="Einkaufen",
    ))
    assert result["success"] is True
    md = (tmp_path / user / "workspace" / "tasks.md").read_text(encoding="utf-8")
    assert "- [x] Einkaufen" in md


def test_complete_task_unknown_id_still_reports_not_found(tmp_path):
    mod = _load_organizer()
    user = "u1"
    (tmp_path / user / "workspace").mkdir(parents=True)
    result, _ = _capture(mod, mod.cmd_complete_task, argparse.Namespace(
        user_id=user, session_dir=str(tmp_path), task_id="deadbeef",
    ))
    assert result["success"] is False
    assert "deadbeef" in result["error"]


def test_complete_task_rejects_id_whose_line_shifted_above(tmp_path):
    """The stable ID is derived from (filepath, line_no). If other tasks
    above the target are deleted between list-tasks and complete-task,
    the ID no longer points to the task — the user must re-list."""
    mod = _load_organizer()
    seed = _seed_note_workspace(tmp_path)
    listing, _ = _capture(mod, mod.cmd_list_tasks, argparse.Namespace(
        user_id=seed["user"], session_dir=seed["session"],
        status="all", limit=100, project="",
    ))
    target = next(t for t in listing["tasks"] if t["title"] == "Zeitplanung")
    stale_id = target["id"]

    # Manually delete the three tasks above Zeitplanung so its line shifts up.
    note_text = seed["note"].read_text(encoding="utf-8")
    stripped = "\n".join(
        line for line in note_text.splitlines()
        if "Zeitplanung" in line
    )
    seed["note"].write_text("# 25. März\n\n" + stripped + "\n", encoding="utf-8")

    result, exit_code = _capture(mod, mod.cmd_complete_task, argparse.Namespace(
        user_id=seed["user"], session_dir=seed["session"],
        task_id=stale_id,
    ))
    assert result["success"] is False
    assert stale_id in result["error"]
    assert exit_code == 1
