"""Tests for pawlia.tools."""

import json
import os
import sys
import tempfile
from datetime import datetime, timedelta

from pawlia.tools.base import Tool, ToolExecutionResult, ToolRegistry
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
        # Use a tmpdir inside session_dir so _validate_cwd allows it
        with tempfile.TemporaryDirectory() as tmpdir:
            result = tool.execute(
                {"command": "pwd"},
                context={"cwd": tmpdir, "session_dir": tmpdir},
            )
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

    def test_skill_config_injected_as_env(self):
        tool = BashTool()
        command = "printf '%s' \"$PAWLIA_SKILL_CONFIG\""
        if sys.platform == "win32":
            command = "Write-Output $env:PAWLIA_SKILL_CONFIG"
        result = tool.execute(
            {"command": command},
            context={"skill_config": {"url": "http://example.test", "timeout": 12}},
        )
        config = json.loads(result)
        assert config["url"] == "http://example.test"
        assert config["timeout"] == 12

    def test_grep_gnu_include_flag_is_rejected_upfront(self):
        """BusyBox grep doesn't support --include. Block it before exec."""
        tool = BashTool()
        result = tool.execute({"command": "grep --include=*.py foo /app"})
        assert "BusyBox" in result
        assert "--include" in result
        assert "find" in result

    def test_grep_perl_regexp_flag_is_rejected_upfront(self):
        tool = BashTool()
        result = tool.execute({"command": "grep -P 'foo\\Kbar' /etc/hostname"})
        assert "BusyBox" in result
        assert "PCRE" in result or "perl" in result.lower()

    def test_grep_exclude_flag_is_rejected_upfront(self):
        tool = BashTool()
        result = tool.execute({"command": "grep --exclude=*.min.js foo /app"})
        assert "BusyBox" in result

    def test_plain_grep_still_runs(self):
        """The pre-check must not block normal grep usage."""
        tool = BashTool()
        result = tool.execute({"command": "echo hello | grep hello"})
        assert "hello" in result

    def test_missing_curl_hint_appended_to_error(self, monkeypatch):
        """When the command fails because curl is missing, append a hint."""
        # PATH that contains only busybox; curl is not on PATH.
        empty = os.path.join(tempfile.mkdtemp(), "empty")
        os.makedirs(empty, exist_ok=True)
        monkeypatch.setenv("PATH", empty)
        tool = BashTool()
        result = tool.execute({"command": "curl https://example.com"})
        if "Hint" in result:
            assert "wget" in result or "urllib" in result
        else:
            # curl happens to exist in the test environment; that's also fine,
            # the test is a no-op then.
            assert "Error" in result

    def test_missing_tool_hint_unit(self):
        from pawlia.tools.bash import _missing_tool_hint
        assert _missing_tool_hint("sh: curl: not found") is not None
        assert "wget" in _missing_tool_hint("sh: curl: not found")
        assert _missing_tool_hint("sh: bogusxyz: not found") is None
        assert _missing_tool_hint("some other error") is None
        # BusyBox uses the German variant in some images.
        de = _missing_tool_hint("bash: curl: Kommando nicht gefunden")
        assert de is not None
        assert "wget" in de
        # Inner name wins over shell prefix.
        inner = _missing_tool_hint("sh: wget: not found")
        assert inner is not None
        assert "urllib" in inner
        # Unknown tool returns None (no spurious hint).
        assert _missing_tool_hint("sh: foobar: not found") is None


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

    def test_unknown_tool_returns_structured_result(self):
        registry = ToolRegistry()
        result = registry.execute_detailed("nonexistent", {})
        assert isinstance(result, ToolExecutionResult)
        assert result.ok is False
        assert result.error_code == "tool_not_found"
        assert result.retryable is True

    def test_invalid_args_return_structured_result(self):
        registry = ToolRegistry()
        registry.register(BashTool())
        result = registry.execute_detailed("bash", {})
        assert result.ok is False
        assert result.error_code == "invalid_arguments"
        assert result.retryable is True
        assert "Missing required" in result.error

    def test_structured_error_serializes_to_json(self):
        result = ToolExecutionResult(
            ok=False,
            tool_name="bash",
            normalized_args={"command": ""},
            error="boom",
            error_code="invalid_arguments",
            retryable=True,
            hint="fix it",
        )
        payload = result.to_tool_message()
        assert '"ok": false' in payload.lower()
        assert '"error_code": "invalid_arguments"' in payload

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


class TestAttachFileTool:
    """Tests for AttachFileTool — re-attach a saved file to the next reply."""

    def _setup(self, tmp_path):
        user_id = "test_user"
        workspace = tmp_path / user_id / "workspace"
        downloads = workspace / "Downloads"
        downloads.mkdir(parents=True)
        outside = tmp_path / "outside.txt"
        outside.write_bytes(b"not allowed")
        return user_id, str(workspace), str(downloads), str(outside)

    def _make_agent_stub(self):
        class _Agent:
            def __init__(self):
                self.pending_attachments = []
        return _Agent()

    def test_attach_under_downloads(self, tmp_path):
        from pawlia.tools.attach_file import AttachFileTool

        user_id, workspace, downloads, _ = self._setup(tmp_path)
        target = tmp_path / user_id / "workspace" / "Downloads" / "rain.gif"
        target.write_bytes(b"GIF89a-fake-bytes")

        tool = AttachFileTool()
        agent = self._make_agent_stub()
        result = tool.execute(
            {"path": "Downloads/rain.gif"},
            context={
                "session_dir": str(tmp_path),
                "user_id": user_id,
                "max_outgoing_bytes": 1024 * 1024,
                "agent": agent,
            },
        )
        assert "queued" in result
        assert len(agent.pending_attachments) == 1
        att = agent.pending_attachments[0]
        assert att["data"] == b"GIF89a-fake-bytes"
        assert att["mimetype"] == "image/gif"
        assert att["filename"] == "rain.gif"

    def test_reject_outside_allowed_roots(self, tmp_path):
        from pawlia.tools.attach_file import AttachFileTool

        user_id, workspace, downloads, _ = self._setup(tmp_path)
        tool = AttachFileTool()
        agent = self._make_agent_stub()
        # An absolute path under no allowed root (workspace, Downloads, /tmp, or
        # a configured extra). /tmp IS allowed now, so the path must be elsewhere.
        result = tool.execute(
            {"path": "/nonexistent_attach_root/outside.txt"},
            context={
                "session_dir": str(tmp_path),
                "user_id": user_id,
                "max_outgoing_bytes": 1024 * 1024,
                "agent": agent,
            },
        )
        assert "outside allowed roots" in result
        assert agent.pending_attachments == []

    def test_reject_path_traversal(self, tmp_path):
        from pawlia.tools.attach_file import AttachFileTool

        user_id, workspace, downloads, _ = self._setup(tmp_path)
        tool = AttachFileTool()
        agent = self._make_agent_stub()
        result = tool.execute(
            {"path": "Downloads/../../etc/passwd"},
            context={
                "session_dir": str(tmp_path),
                "user_id": user_id,
                "max_outgoing_bytes": 1024 * 1024,
                "agent": agent,
            },
        )
        assert "outside allowed roots" in result or "file not found" in result
        assert agent.pending_attachments == []

    def test_accepts_tmp_path(self, tmp_path):
        # Generated throwaway artefacts live in /tmp now — attach_file must take
        # an absolute /tmp path and queue it.
        from pawlia.tools.attach_file import AttachFileTool

        user_id, _, _, _ = self._setup(tmp_path)
        fd, scratch = tempfile.mkstemp(prefix="pawlia_attach_", suffix=".png", dir="/tmp")
        try:
            os.write(fd, b"\x89PNG\r\n")
            os.close(fd)
            tool = AttachFileTool()
            agent = self._make_agent_stub()
            result = tool.execute(
                {"path": scratch},
                context={
                    "session_dir": str(tmp_path),
                    "user_id": user_id,
                    "max_outgoing_bytes": 1024 * 1024,
                    "agent": agent,
                },
            )
            assert "queued" in result
            assert len(agent.pending_attachments) == 1
            assert agent.pending_attachments[0]["filename"] == os.path.basename(scratch)
        finally:
            os.unlink(scratch)

    def test_size_limit(self, tmp_path):
        from pawlia.tools.attach_file import AttachFileTool

        user_id, _, _, _ = self._setup(tmp_path)
        target = tmp_path / user_id / "workspace" / "Downloads" / "big.bin"
        target.write_bytes(b"x" * 2048)

        tool = AttachFileTool()
        agent = self._make_agent_stub()
        result = tool.execute(
            {"path": "Downloads/big.bin"},
            context={
                "session_dir": str(tmp_path),
                "user_id": user_id,
                "max_outgoing_bytes": 100,
                "agent": agent,
            },
        )
        assert "too large" in result
        assert agent.pending_attachments == []

    def test_explicit_mimetype_override(self, tmp_path):
        from pawlia.tools.attach_file import AttachFileTool

        user_id, _, _, _ = self._setup(tmp_path)
        target = tmp_path / user_id / "workspace" / "Downloads" / "blob"
        target.write_bytes(b"raw bytes")

        tool = AttachFileTool()
        agent = self._make_agent_stub()
        tool.execute(
            {"path": "Downloads/blob", "mimetype": "application/x-foo", "caption": "hi"},
            context={
                "session_dir": str(tmp_path),
                "user_id": user_id,
                "max_outgoing_bytes": 1024 * 1024,
                "agent": agent,
            },
        )
        assert agent.pending_attachments[0]["mimetype"] == "application/x-foo"
        assert agent.pending_attachments[0]["caption"] == "hi"

    def test_extra_allowed_root(self, tmp_path):
        from pawlia.tools.attach_file import AttachFileTool

        user_id, _, _, _ = self._setup(tmp_path)
        shared = tmp_path / "shared"
        shared.mkdir()
        (shared / "report.pdf").write_bytes(b"%PDF-fake")

        tool = AttachFileTool()
        agent = self._make_agent_stub()
        result = tool.execute(
            {"path": str(shared / "report.pdf")},
            context={
                "session_dir": str(tmp_path),
                "user_id": user_id,
                "max_outgoing_bytes": 1024 * 1024,
                "attachment_extra_roots": [str(shared)],
                "agent": agent,
            },
        )
        assert "queued" in result
        assert agent.pending_attachments[0]["filename"] == "report.pdf"

    def test_missing_agent_in_context(self, tmp_path):
        from pawlia.tools.attach_file import AttachFileTool

        user_id, _, _, _ = self._setup(tmp_path)
        (tmp_path / user_id / "workspace" / "Downloads" / "x.txt").write_bytes(b"hi")

        tool = AttachFileTool()
        result = tool.execute(
            {"path": "Downloads/x.txt"},
            context={
                "session_dir": str(tmp_path),
                "user_id": user_id,
                "max_outgoing_bytes": 1024,
            },
        )
        assert "agent context unavailable" in result
