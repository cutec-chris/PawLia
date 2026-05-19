import importlib.util
from pathlib import Path

import pytest
import yaml


pytest.importorskip("radicale")
pytest.importorskip("icalendar")
pytest.importorskip("vobject")


def _load_storage():
    path = Path(__file__).resolve().parents[1] / "radicale" / "picoclaw_storage.py"
    spec = importlib.util.spec_from_file_location("picoclaw_storage", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_markdown_recurring_event_exports_rrule_to_ical(tmp_path):
    mod = _load_storage()
    event_path = tmp_path / "2026-05-19 Parcours.md"
    event_path.write_text(
        "---\n"
        "title: Parcours\n"
        "date: '2026-05-19'\n"
        "startTime: '17:00'\n"
        "endTime: '18:30'\n"
        "allDay: false\n"
        "type: recurring\n"
        "rrule: FREQ=WEEKLY;BYDAY=TU,TH\n"
        "---\n"
        "\nTraining\n",
        encoding="utf-8",
    )

    ical = mod._md_to_ical(str(event_path), "2026-05-19 Parcours.ics")

    assert ical is not None
    assert "RRULE:FREQ=WEEKLY;BYDAY=TU,TH" in ical


def test_ical_recurring_event_imports_fullcalendar_fields():
    mod = _load_storage()
    ical = "\r\n".join([
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "BEGIN:VEVENT",
        "UID:event-1",
        "SUMMARY:Parcours",
        "DTSTART:20260519T170000",
        "DTEND:20260519T183000",
        "RRULE:FREQ=WEEKLY;BYDAY=TU,TH;UNTIL=20261231",
        "END:VEVENT",
        "END:VCALENDAR",
        "",
    ])

    result = mod._ical_to_md(ical)

    assert result is not None
    content, _meta = result
    fm = yaml.safe_load(content.split("---", 2)[1])
    assert fm["type"] == "recurring"
    assert fm["rrule"] == "FREQ=WEEKLY;BYDAY=TU,TH;UNTIL=20261231"
    assert fm["daysOfWeek"] == [2, 4]
    assert fm["startRecur"] == "2026-05-19"
    assert fm["endRecur"] == "2026-12-31"
