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


def test_organizer_run_job_forces_run_and_returns_the_instruction(tmp_path, capsys):
    import json
    mod = _load_organizer()
    mod.cmd_add_job(argparse.Namespace(
        user_id="u1", session_dir=str(tmp_path), name="Morgenbericht",
        schedule="08:00", instruction="Erstelle einen Morgenbericht",
        no_notify=False, notify_on_error=False,
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
