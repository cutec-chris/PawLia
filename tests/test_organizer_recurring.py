import argparse
import importlib.util
from pathlib import Path

import yaml


def _load_organizer():
    path = Path(__file__).resolve().parents[1] / "skills" / "organizer" / "scripts" / "organizer.py"
    spec = importlib.util.spec_from_file_location("organizer_script", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_add_event_writes_fullcalendar_recurring_frontmatter(tmp_path):
    mod = _load_organizer()
    args = argparse.Namespace(
        user_id="u1",
        session_dir=str(tmp_path),
        title="Parcours",
        start="2026-05-19T17:00:00",
        end="2026-05-19T18:30:00",
        description="Training",
        location="",
        checklist="",
        reminders="",
        recurrence="weekly",
        recurrence_days="TU,TH",
        recurrence_until="2026-12-31",
        recurrence_count=None,
    )

    mod.cmd_add_event(args)

    event_path = tmp_path / "u1" / "workspace" / "calendar" / "2026-05-19 Parcours.md"
    raw = event_path.read_text(encoding="utf-8")
    fm = yaml.safe_load(raw.split("---", 2)[1])

    assert fm["type"] == "recurring"
    assert fm["rrule"] == "FREQ=WEEKLY;BYDAY=TU,TH;UNTIL=20261231"
    assert fm["daysOfWeek"] == [2, 4]
    assert fm["startRecur"] == "2026-05-19"
    assert fm["endRecur"] == "2026-12-31"


def test_add_weekly_event_defaults_to_start_weekday(tmp_path):
    mod = _load_organizer()
    args = argparse.Namespace(
        user_id="u1",
        session_dir=str(tmp_path),
        title="Dienstagstraining",
        start="2026-05-19T17:00:00",
        end="2026-05-19T18:30:00",
        description="",
        location="",
        checklist="",
        reminders="",
        recurrence="weekly",
        recurrence_days="",
        recurrence_until="",
        recurrence_count=None,
    )

    mod.cmd_add_event(args)

    event_path = tmp_path / "u1" / "workspace" / "calendar" / "2026-05-19 Dienstagstraining.md"
    fm = yaml.safe_load(event_path.read_text(encoding="utf-8").split("---", 2)[1])

    assert fm["rrule"] == "FREQ=WEEKLY;BYDAY=TU"
    assert fm["daysOfWeek"] == [2]


def test_add_event_persists_event_reminders_and_scheduler_checklist(tmp_path):
    mod = _load_organizer()
    args = argparse.Namespace(
        user_id="u1",
        session_dir=str(tmp_path),
        title="Parcours",
        start="2026-05-19T17:00:00",
        end="2026-05-19T18:30:00",
        description="Training",
        location="",
        checklist="",
        reminders='[{"minutes_before": 40, "message": "Bereitmachen"}]',
        recurrence="none",
        recurrence_days="",
        recurrence_until="",
        recurrence_count=None,
    )

    mod.cmd_add_event(args)

    event_path = tmp_path / "u1" / "workspace" / "calendar" / "2026-05-19 Parcours.md"
    raw = event_path.read_text(encoding="utf-8")
    fm = yaml.safe_load(raw.split("---", 2)[1])

    assert fm["reminders"] == [{"minutes_before": 40, "message": "Bereitmachen"}]
    assert len(fm["checklist"]) == 1
    assert fm["checklist"][0]["source"] == "event_reminder"
    assert fm["checklist"][0]["trigger_offset"] == "-40m"

    listed = mod._read_event_md(str(event_path))
    assert listed is not None
    assert listed["reminders"] == [{"minutes_before": 40, "message": "Bereitmachen"}]


def test_run_job_sets_force_run_and_returns_instruction(tmp_path, capsys):
    mod = _load_organizer()

    # Add a job first
    add_args = argparse.Namespace(
        user_id="u1",
        session_dir=str(tmp_path),
        name="Morgenbericht",
        schedule="08:00",
        instruction="Erstelle einen Morgenbericht",
        no_notify=False,
        notify_on_error=False,
    )
    mod.cmd_add_job(add_args)

    # Capture the job ID from stdout
    captured = capsys.readouterr()
    import json
    add_result = json.loads(captured.out)
    job_id = add_result["job_id"]

    # Run the job manually
    run_args = argparse.Namespace(
        user_id="u1",
        session_dir=str(tmp_path),
        job_id=job_id,
    )
    mod.cmd_run_job(run_args)

    # Check stdout
    captured = capsys.readouterr()
    run_result = json.loads(captured.out)
    assert run_result["success"] is True
    assert "Morgenbericht" in run_result["message"]
    assert run_result["instruction"] == "Erstelle einen Morgenbericht"

    # Verify jobs.json has force_run set
    jobs = mod._load_json(str(tmp_path / "u1" / "automations" / "jobs.json"))
    assert len(jobs) == 1
    assert jobs[0].get("force_run") is True
