"""Tests for pawlia.tools."""

import json
import os
import tempfile
from datetime import datetime, timedelta

from pawlia.tools.base import Tool, ToolRegistry
from pawlia.tools.bash import BashTool
from pawlia.tools.reminder import ReminderTool


class TestBashTool:
    def test_echo(self):
        tool = BashTool()
        result = tool.execute({"command": "echo hello"})
        assert "hello" in result

    def test_empty_command(self):
        tool = BashTool()
        result = tool.execute({"command": ""})
        assert "Error" in result

    def test_spec(self):
        tool = BashTool()
        spec = tool.as_openai_spec()
        assert spec["function"]["name"] == "bash"
        assert "command" in spec["function"]["parameters"]["properties"]

    def test_cwd(self):
        tool = BashTool()
        with tempfile.TemporaryDirectory() as tmpdir:
            result = tool.execute({"command": "pwd"}, context={"cwd": tmpdir})
            # Git Bash on Windows returns POSIX paths; compare directory name only
            assert os.path.basename(tmpdir) in result

    def test_timeout(self):
        tool = BashTool()
        result = tool.execute(
            {"command": "sleep 10"},
            context={"timeout": 1},
        )
        assert "timed out" in result

    def test_nonzero_exit(self):
        tool = BashTool()
        result = tool.execute({"command": "exit 1"})
        assert "Error" in result

    def test_stderr_on_error(self):
        tool = BashTool()
        result = tool.execute({"command": "echo oops >&2; exit 1"})
        assert "oops" in result

    def test_no_output(self):
        tool = BashTool()
        result = tool.execute({"command": "true"})
        assert result == "(no output)"

    def test_no_context_uses_defaults(self):
        tool = BashTool()
        # Should not crash when context is None
        result = tool.execute({"command": "echo ok"})
        assert "ok" in result


class TestReminderTool:
    def test_add_and_list(self):
        tool = ReminderTool()
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = {"user_id": "test_user", "session_dir": tmpdir}

            # Add
            result = tool.execute({
                "action": "add",
                "fire_at": "30m",
                "message": "Test reminder",
                "label": "Test",
            }, context=ctx)
            assert result["success"] is True
            rid = result["reminder_id"]

            # List
            result = tool.execute({"action": "list"}, context=ctx)
            assert result["total"] == 1
            assert result["reminders"][0]["id"] == rid

            # Delete
            result = tool.execute({
                "action": "delete",
                "reminder_id": rid,
            }, context=ctx)
            assert result["success"] is True

            # List again
            result = tool.execute({"action": "list"}, context=ctx)
            assert result["total"] == 0

    def test_no_context(self):
        tool = ReminderTool()
        result = tool.execute({"action": "list"})
        assert result["success"] is False

    def test_parse_minutes(self):
        dt = ReminderTool._parse_fire_at("30m")
        assert dt > datetime.now()
        assert dt < datetime.now() + timedelta(minutes=31)

    def test_parse_min_suffix(self):
        dt = ReminderTool._parse_fire_at("5min")
        assert dt > datetime.now()
        assert dt < datetime.now() + timedelta(minutes=6)

    def test_parse_hours(self):
        dt = ReminderTool._parse_fire_at("2h")
        assert dt > datetime.now() + timedelta(hours=1, minutes=59)

    def test_parse_days(self):
        dt = ReminderTool._parse_fire_at("1d")
        assert dt > datetime.now() + timedelta(hours=23)

    def test_parse_iso(self):
        dt = ReminderTool._parse_fire_at("2026-06-15T14:00:00")
        assert dt == datetime(2026, 6, 15, 14, 0)

    def test_parse_invalid(self):
        try:
            ReminderTool._parse_fire_at("not-a-date")
            assert False, "Should have raised"
        except Exception:
            pass

    def test_add_missing_fire_at(self):
        tool = ReminderTool()
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = {"user_id": "u1", "session_dir": tmpdir}
            result = tool.execute({
                "action": "add",
                "message": "test",
            }, context=ctx)
            assert result["success"] is False
            assert "fire_at" in result["error"]

    def test_add_missing_message(self):
        tool = ReminderTool()
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = {"user_id": "u1", "session_dir": tmpdir}
            result = tool.execute({
                "action": "add",
                "fire_at": "30m",
            }, context=ctx)
            assert result["success"] is False
            assert "message" in result["error"]

    def test_delete_nonexistent(self):
        tool = ReminderTool()
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = {"user_id": "u1", "session_dir": tmpdir}
            result = tool.execute({
                "action": "delete",
                "reminder_id": "fake-id",
            }, context=ctx)
            assert result["success"] is False

    def test_delete_no_id(self):
        tool = ReminderTool()
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = {"user_id": "u1", "session_dir": tmpdir}
            result = tool.execute({"action": "delete"}, context=ctx)
            assert result["success"] is False

    def test_invalid_recurrence_defaults_to_none(self):
        tool = ReminderTool()
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = {"user_id": "u1", "session_dir": tmpdir}
            result = tool.execute({
                "action": "add",
                "fire_at": "30m",
                "message": "test",
                "recurrence": "biweekly",
            }, context=ctx)
            assert result["success"] is True
            # Check stored value
            path = os.path.join(tmpdir, "u1", "reminders.json")
            with open(path) as f:
                data = json.load(f)
            assert data[0]["recurrence"] == "none"

    def test_list_excludes_fired(self):
        tool = ReminderTool()
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = {"user_id": "u1", "session_dir": tmpdir}
            # Add and manually mark as fired
            tool.execute({
                "action": "add",
                "fire_at": "30m",
                "message": "fired one",
            }, context=ctx)
            path = os.path.join(tmpdir, "u1", "reminders.json")
            with open(path) as f:
                data = json.load(f)
            data[0]["fired"] = True
            with open(path, "w") as f:
                json.dump(data, f)

            result = tool.execute({"action": "list"}, context=ctx)
            assert result["total"] == 0


class TestToolRegistry:
    def test_register_and_execute(self):
        registry = ToolRegistry()
        registry.register(BashTool())

        assert "bash" in registry.names()

        result = registry.execute("bash", {"command": "echo registry_test"})
        assert "registry_test" in result

    def test_unknown_tool(self):
        registry = ToolRegistry()
        result = registry.execute("nonexistent", {})
        assert "Error" in result

    def test_string_args_are_normalized_for_single_param_tools(self):
        registry = ToolRegistry()
        registry.register(BashTool())
        result = registry.execute("bash", "echo normalized")
        assert "normalized" in result

    def test_rejects_unexpected_arguments(self):
        registry = ToolRegistry()
        registry.register(BashTool())
        result = registry.execute("bash", {"command": "echo ok", "extra": "nope"})
        assert "Invalid arguments" in result

    def test_rejects_missing_required_arguments(self):
        registry = ToolRegistry()
        registry.register(BashTool())
        result = registry.execute("bash", {})
        assert "Missing required" in result

    def test_get_specs(self):
        registry = ToolRegistry()
        registry.register(BashTool())
        specs = registry.get_specs()
        assert len(specs) == 1
        assert specs[0]["function"]["name"] == "bash"

    def test_fuzzy_resolve_underscore(self):
        registry = ToolRegistry()
        registry.register(ReminderTool())
        result = registry.execute("schedule-reminder", {"action": "list"})
        # Should resolve despite dash vs underscore — but no context so fails
        assert result["success"] is False

    def test_fuzzy_resolve_case(self):
        registry = ToolRegistry()
        registry.register(BashTool())
        result = registry.execute("BASH", {"command": "echo case"})
        assert "case" in result

    def test_multiple_tools(self):
        registry = ToolRegistry()
        registry.register(BashTool())
        registry.register(ReminderTool())
        assert len(registry.names()) == 2
        specs = registry.get_specs()
        assert len(specs) == 2
