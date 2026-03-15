"""Tests for pawlia.scheduler."""

import asyncio
import json
import os
import tempfile
from datetime import datetime, timedelta

import pytest

from pawlia.scheduler import Scheduler, _next_occurrence


class TestNextOccurrence:
    def test_daily(self):
        dt = datetime(2026, 3, 15, 10, 0)
        result = _next_occurrence(dt, "daily")
        assert result == datetime(2026, 3, 16, 10, 0)

    def test_weekly(self):
        dt = datetime(2026, 3, 15, 10, 0)
        result = _next_occurrence(dt, "weekly")
        assert result == datetime(2026, 3, 22, 10, 0)

    def test_monthly(self):
        dt = datetime(2026, 3, 15, 10, 0)
        result = _next_occurrence(dt, "monthly")
        assert result == datetime(2026, 4, 15, 10, 0)

    def test_monthly_year_wrap(self):
        dt = datetime(2026, 12, 15, 10, 0)
        result = _next_occurrence(dt, "monthly")
        assert result == datetime(2027, 1, 15, 10, 0)

    def test_monthly_day_overflow(self):
        # Jan 31 -> Feb has no 31st, should fall back to 28
        dt = datetime(2026, 1, 31, 10, 0)
        result = _next_occurrence(dt, "monthly")
        assert result.month == 2
        assert result.day == 28

    def test_unknown_recurrence(self):
        dt = datetime(2026, 3, 15, 10, 0)
        result = _next_occurrence(dt, "unknown")
        assert result == datetime(2026, 3, 16, 10, 0)


class TestSchedulerReminders:
    @pytest.mark.asyncio
    async def test_fires_due_reminder(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a due reminder
            user_dir = os.path.join(tmpdir, "test_user")
            os.makedirs(user_dir)
            reminders = [{
                "id": "r1",
                "fire_at": (datetime.now() - timedelta(minutes=5)).isoformat(),
                "message": "Take a break",
                "label": "Break",
                "recurrence": "none",
                "fired": False,
            }]
            path = os.path.join(user_dir, "reminders.json")
            with open(path, "w") as f:
                json.dump(reminders, f)

            # Set up scheduler
            notifications = []

            async def capture(user_id, message):
                notifications.append((user_id, message))

            scheduler = Scheduler(tmpdir)
            scheduler.register(capture)
            await scheduler._check_all()

            assert len(notifications) == 1
            assert notifications[0][0] == "test_user"
            assert "Break" in notifications[0][1]
            assert "Take a break" in notifications[0][1]

            # Reminder should be marked as fired
            with open(path) as f:
                data = json.load(f)
            assert data[0]["fired"] is True

    @pytest.mark.asyncio
    async def test_skips_fired_reminder(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            user_dir = os.path.join(tmpdir, "test_user")
            os.makedirs(user_dir)
            reminders = [{
                "id": "r1",
                "fire_at": (datetime.now() - timedelta(minutes=5)).isoformat(),
                "message": "old",
                "label": "Old",
                "recurrence": "none",
                "fired": True,
            }]
            with open(os.path.join(user_dir, "reminders.json"), "w") as f:
                json.dump(reminders, f)

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
            user_dir = os.path.join(tmpdir, "test_user")
            os.makedirs(user_dir)
            reminders = [{
                "id": "r1",
                "fire_at": (datetime.now() + timedelta(hours=2)).isoformat(),
                "message": "future",
                "label": "Future",
                "recurrence": "none",
                "fired": False,
            }]
            with open(os.path.join(user_dir, "reminders.json"), "w") as f:
                json.dump(reminders, f)

            notifications = []

            async def capture(user_id, message):
                notifications.append((user_id, message))

            scheduler = Scheduler(tmpdir)
            scheduler.register(capture)
            await scheduler._check_all()

            assert len(notifications) == 0

    @pytest.mark.asyncio
    async def test_recurring_reminder_advances(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            user_dir = os.path.join(tmpdir, "test_user")
            os.makedirs(user_dir)
            original_time = datetime.now() - timedelta(minutes=5)
            reminders = [{
                "id": "r1",
                "fire_at": original_time.isoformat(),
                "message": "daily check",
                "label": "Daily",
                "recurrence": "daily",
                "fired": False,
            }]
            path = os.path.join(user_dir, "reminders.json")
            with open(path, "w") as f:
                json.dump(reminders, f)

            notifications = []

            async def capture(user_id, message):
                notifications.append((user_id, message))

            scheduler = Scheduler(tmpdir)
            scheduler.register(capture)
            await scheduler._check_all()

            assert len(notifications) == 1

            # Should NOT be fired, but advanced
            with open(path) as f:
                data = json.load(f)
            assert data[0]["fired"] is False
            next_fire = datetime.fromisoformat(data[0]["fire_at"])
            assert next_fire > datetime.now()


class TestSchedulerEvents:
    @pytest.mark.asyncio
    async def test_upcoming_event_notified(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            user_dir = os.path.join(tmpdir, "test_user")
            cal_dir = os.path.join(user_dir, "calendar")
            os.makedirs(cal_dir)

            events = [{
                "id": "e1",
                "title": "Meeting",
                "start": (datetime.now() + timedelta(minutes=10)).isoformat(),
                "location": "Room 42",
            }]
            path = os.path.join(cal_dir, "events.json")
            with open(path, "w") as f:
                json.dump(events, f)

            notifications = []

            async def capture(user_id, message):
                notifications.append((user_id, message))

            scheduler = Scheduler(tmpdir)
            scheduler.register(capture)
            await scheduler._check_all()

            assert len(notifications) == 1
            assert "Meeting" in notifications[0][1]
            assert "Room 42" in notifications[0][1]

            # _notified flag set
            with open(path) as f:
                data = json.load(f)
            assert data[0]["_notified"] is True

    @pytest.mark.asyncio
    async def test_past_event_not_notified(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            user_dir = os.path.join(tmpdir, "test_user")
            cal_dir = os.path.join(user_dir, "calendar")
            os.makedirs(cal_dir)

            events = [{
                "id": "e1",
                "title": "Old meeting",
                "start": (datetime.now() - timedelta(hours=1)).isoformat(),
            }]
            with open(os.path.join(cal_dir, "events.json"), "w") as f:
                json.dump(events, f)

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
            user_dir = os.path.join(tmpdir, "test_user")
            cal_dir = os.path.join(user_dir, "calendar")
            os.makedirs(cal_dir)

            events = [{
                "id": "e1",
                "title": "Far away",
                "start": (datetime.now() + timedelta(hours=3)).isoformat(),
            }]
            with open(os.path.join(cal_dir, "events.json"), "w") as f:
                json.dump(events, f)

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
            user_dir = os.path.join(tmpdir, "test_user")
            cal_dir = os.path.join(user_dir, "calendar")
            os.makedirs(cal_dir)

            events = [{
                "id": "e1",
                "title": "Meeting",
                "start": (datetime.now() + timedelta(minutes=10)).isoformat(),
                "_notified": True,
            }]
            with open(os.path.join(cal_dir, "events.json"), "w") as f:
                json.dump(events, f)

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
            user_dir = os.path.join(tmpdir, "test_user")
            os.makedirs(user_dir)
            reminders = [{
                "id": "r1",
                "fire_at": (datetime.now() - timedelta(minutes=1)).isoformat(),
                "message": "test",
                "label": "Test",
                "recurrence": "none",
                "fired": False,
            }]
            with open(os.path.join(user_dir, "reminders.json"), "w") as f:
                json.dump(reminders, f)

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
            user_dir = os.path.join(tmpdir, "test_user")
            os.makedirs(user_dir)
            reminders = [{
                "id": "r1",
                "fire_at": (datetime.now() - timedelta(minutes=1)).isoformat(),
                "message": "test",
                "label": "Test",
                "recurrence": "none",
                "fired": False,
            }]
            with open(os.path.join(user_dir, "reminders.json"), "w") as f:
                json.dump(reminders, f)

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
        # Should not raise
        await scheduler._check_all()
