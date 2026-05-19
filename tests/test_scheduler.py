"""Tests for pawlia.scheduler (Obsidian-native storage)."""

import asyncio
import json
import os
import tempfile
from datetime import datetime, timedelta

import pytest
import yaml

from pawlia.llm import LLMFactory
from pawlia.memory import MemoryManager
from pawlia.scheduler import Scheduler, _next_occurrence


def _write_event_md(cal_dir, filename, fm_dict, body=""):
    """Helper: write a Full Calendar event .md file."""
    os.makedirs(cal_dir, exist_ok=True)
    content = f"---\n{yaml.dump(fm_dict, allow_unicode=True, default_flow_style=False).rstrip()}\n---\n"
    if body:
        content += f"\n{body}\n"
    with open(os.path.join(cal_dir, filename), "w", encoding="utf-8") as f:
        f.write(content)


def _write_tasks_md(workspace_dir, lines):
    """Helper: write tasks.md with the given lines."""
    os.makedirs(workspace_dir, exist_ok=True)
    with open(os.path.join(workspace_dir, "tasks.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _read_tasks_md(workspace_dir):
    """Helper: read tasks.md lines."""
    with open(os.path.join(workspace_dir, "tasks.md"), encoding="utf-8") as f:
        return f.read().strip().split("\n")


class TestNextOccurrence:
    def test_daily(self):
        dt = datetime(2026, 3, 15, 10, 0)
        result = _next_occurrence(dt, "every day")
        assert result == datetime(2026, 3, 16, 10, 0)

    def test_weekly(self):
        dt = datetime(2026, 3, 15, 10, 0)
        result = _next_occurrence(dt, "every week")
        assert result == datetime(2026, 3, 22, 10, 0)

    def test_monthly(self):
        dt = datetime(2026, 3, 15, 10, 0)
        result = _next_occurrence(dt, "every month")
        assert result == datetime(2026, 4, 15, 10, 0)

    def test_monthly_year_wrap(self):
        dt = datetime(2026, 12, 15, 10, 0)
        result = _next_occurrence(dt, "every month")
        assert result == datetime(2027, 1, 15, 10, 0)

    def test_monthly_day_overflow(self):
        dt = datetime(2026, 1, 31, 10, 0)
        result = _next_occurrence(dt, "every month")
        assert result.month == 2
        assert result.day == 28

    def test_unknown_recurrence(self):
        dt = datetime(2026, 3, 15, 10, 0)
        result = _next_occurrence(dt, "unknown")
        assert result == datetime(2026, 3, 16, 10, 0)


class TestSchedulerReminders:
    """Reminders are stored as scheduled tasks in workspace/tasks.md."""

    @pytest.mark.asyncio
    async def test_fires_due_reminder(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = os.path.join(tmpdir, "test_user", "workspace")
            fire_at = (datetime.now() - timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M")
            _write_tasks_md(workspace, [
                f"- [ ] \U0001f514 Break: Take a break \u23f3 {fire_at} \u2795 2026-04-10"
            ])

            notifications = []

            async def capture(user_id, message):
                notifications.append((user_id, message))

            scheduler = Scheduler(tmpdir)
            scheduler.register(capture)
            await scheduler._check_all()

            assert len(notifications) == 1
            assert "test_user" == notifications[0][0]
            assert "Take a break" in notifications[0][1]

            # Should be marked as done [x]
            lines = _read_tasks_md(workspace)
            assert "[x]" in lines[0]

    @pytest.mark.asyncio
    async def test_skips_completed_reminder(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = os.path.join(tmpdir, "test_user", "workspace")
            fire_at = (datetime.now() - timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M")
            _write_tasks_md(workspace, [
                f"- [x] \U0001f514 Old: old reminder \u23f3 {fire_at} \u2705 2026-04-09"
            ])

            notifications = []

            async def capture(user_id, message):
                notifications.append((user_id, message))

            scheduler = Scheduler(tmpdir)
            scheduler.register(capture)
            await scheduler._check_all()

            assert len(notifications) == 0

    @pytest.mark.asyncio
    async def test_skips_future_reminder(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = os.path.join(tmpdir, "test_user", "workspace")
            fire_at = (datetime.now() + timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M")
            _write_tasks_md(workspace, [
                f"- [ ] \U0001f514 Future: future reminder \u23f3 {fire_at}"
            ])

            notifications = []

            async def capture(user_id, message):
                notifications.append((user_id, message))

            scheduler = Scheduler(tmpdir)
            scheduler.register(capture)
            await scheduler._check_all()

            assert len(notifications) == 0

    @pytest.mark.asyncio
    async def test_recurring_reminder_reschedules(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = os.path.join(tmpdir, "test_user", "workspace")
            fire_at = (datetime.now() - timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M")
            _write_tasks_md(workspace, [
                f"- [ ] \U0001f514 Daily: daily check \U0001f501 every day \u23f3 {fire_at}"
            ])

            notifications = []

            async def capture(user_id, message):
                notifications.append((user_id, message))

            scheduler = Scheduler(tmpdir)
            scheduler.register(capture)
            await scheduler._check_all()

            assert len(notifications) == 1

            # Should still be pending [ ] but with advanced date
            lines = _read_tasks_md(workspace)
            assert "[ ]" in lines[0]
            # The scheduled time should be in the future now
            import re
            m = re.search(r"\u23f3\s+([\d\-T:]+)", lines[0])
            assert m
            new_dt = datetime.fromisoformat(m.group(1))
            assert new_dt > datetime.now()


class TestSchedulerEvents:
    """Events are stored as .md files in workspace/calendar/."""

    @pytest.mark.asyncio
    async def test_upcoming_event_notified(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cal_dir = os.path.join(tmpdir, "test_user", "workspace", "calendar")
            start = datetime.now() + timedelta(minutes=10)
            _write_event_md(cal_dir, "2026-04-10 Meeting.md", {
                "title": "Meeting",
                "date": start.strftime("%Y-%m-%d"),
                "startTime": start.strftime("%H:%M"),
                "allDay": False,
                "type": "single",
                "location": "Room 42",
            })

            notifications = []

            async def capture(user_id, message):
                notifications.append((user_id, message))

            scheduler = Scheduler(tmpdir)
            scheduler.register(capture)
            await scheduler._check_all()

            assert len(notifications) == 1
            assert "Meeting" in notifications[0][1]
            assert "Room 42" in notifications[0][1]

            # State should track notified event
            state_path = os.path.join(tmpdir, "test_user", "scheduler_state.json")
            assert os.path.exists(state_path)
            with open(state_path) as f:
                state = json.load(f)
            assert "2026-04-10 Meeting.md" in state.get("notified_events", [])

    @pytest.mark.asyncio
    async def test_past_event_not_notified(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cal_dir = os.path.join(tmpdir, "test_user", "workspace", "calendar")
            start = datetime.now() - timedelta(hours=1)
            _write_event_md(cal_dir, "2026-04-10 Old.md", {
                "title": "Old meeting",
                "date": start.strftime("%Y-%m-%d"),
                "startTime": start.strftime("%H:%M"),
                "allDay": False,
                "type": "single",
            })

            notifications = []

            async def capture(user_id, message):
                notifications.append((user_id, message))

            scheduler = Scheduler(tmpdir)
            scheduler.register(capture)
            await scheduler._check_all()

            assert len(notifications) == 0

    @pytest.mark.asyncio
    async def test_far_future_event_not_notified(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cal_dir = os.path.join(tmpdir, "test_user", "workspace", "calendar")
            start = datetime.now() + timedelta(hours=3)
            _write_event_md(cal_dir, "2026-04-10 Far.md", {
                "title": "Far away",
                "date": start.strftime("%Y-%m-%d"),
                "startTime": start.strftime("%H:%M"),
                "allDay": False,
                "type": "single",
            })

            notifications = []

            async def capture(user_id, message):
                notifications.append((user_id, message))

            scheduler = Scheduler(tmpdir)
            scheduler.register(capture)
            await scheduler._check_all()

            assert len(notifications) == 0

    @pytest.mark.asyncio
    async def test_already_notified_event_skipped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cal_dir = os.path.join(tmpdir, "test_user", "workspace", "calendar")
            start = datetime.now() + timedelta(minutes=10)
            _write_event_md(cal_dir, "2026-04-10 Meeting.md", {
                "title": "Meeting",
                "date": start.strftime("%Y-%m-%d"),
                "startTime": start.strftime("%H:%M"),
                "allDay": False,
                "type": "single",
            })

            # Pre-set state as notified
            state_dir = os.path.join(tmpdir, "test_user")
            os.makedirs(state_dir, exist_ok=True)
            with open(os.path.join(state_dir, "scheduler_state.json"), "w") as f:
                json.dump({"notified_events": ["2026-04-10 Meeting.md"]}, f)

            notifications = []

            async def capture(user_id, message):
                notifications.append((user_id, message))

            scheduler = Scheduler(tmpdir)
            scheduler.register(capture)
            await scheduler._check_all()

            assert len(notifications) == 0


class TestSchedulerCallbacks:
    @pytest.mark.asyncio
    async def test_multiple_callbacks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = os.path.join(tmpdir, "test_user", "workspace")
            fire_at = (datetime.now() - timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M")
            _write_tasks_md(workspace, [
                f"- [ ] \U0001f514 Test: test \u23f3 {fire_at}"
            ])

            cb1_calls = []
            cb2_calls = []

            async def cb1(uid, msg):
                cb1_calls.append(msg)

            async def cb2(uid, msg):
                cb2_calls.append(msg)

            scheduler = Scheduler(tmpdir)
            scheduler.register(cb1)
            scheduler.register(cb2)
            await scheduler._check_all()

            assert len(cb1_calls) == 1
            assert len(cb2_calls) == 1

    @pytest.mark.asyncio
    async def test_callback_error_does_not_stop_others(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = os.path.join(tmpdir, "test_user", "workspace")
            fire_at = (datetime.now() - timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M")
            _write_tasks_md(workspace, [
                f"- [ ] \U0001f514 Test: test \u23f3 {fire_at}"
            ])

            cb2_calls = []

            async def bad_cb(uid, msg):
                raise RuntimeError("callback error")

            async def good_cb(uid, msg):
                cb2_calls.append(msg)

            scheduler = Scheduler(tmpdir)
            scheduler.register(bad_cb)
            scheduler.register(good_cb)
            await scheduler._check_all()

            assert len(cb2_calls) == 1

    @pytest.mark.asyncio
    async def test_no_session_dir(self):
        scheduler = Scheduler("/nonexistent/path")
        await scheduler._check_all()


class TestSchedulerSummarizationThreshold:
    def test_summary_threshold_reserves_prompt_overhead(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "providers": {
                    "test": {
                        "backend": "pawlia",
                        "apiBase": "http://example.test/v1",
                        "apiKey": "x",
                    }
                },
                "models": {
                    "m1": {
                        "model": "gpt-oss-test",
                        "provider": "test",
                        "context_size": 4096,
                        "summarize_at_fraction": 0.95,
                    }
                },
                "agents": {"default": "m1"},
            }

            memory = MemoryManager(tmpdir)
            session = memory.load_session("test_user")

            class _App:
                def __init__(self):
                    self.memory = memory
                    self.llm = LLMFactory(config)

                def _build_user_skills(self, user_id, disabled=None):
                    return {}

            scheduler = Scheduler(tmpdir, config=config)
            scheduler.set_app(_App())

            unclamped = scheduler._app.llm.summary_threshold_tokens("m1")
            clamped = scheduler._summary_threshold_for(session)

            assert clamped > 0
            assert clamped < unclamped
