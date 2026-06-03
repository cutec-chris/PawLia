"""Integration tests for workflow.yaml building blocks.

Tests that workflow commands resolve correct paths and that the underlying
scripts accept the documented arguments. Catches issues like:
- Wrong {scripts_dir}/{skill_dir} resolution
- Invalid argument choices (e.g. --recurrence rejecting RRULE strings)
- Missing script files
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
SKILLS_DIR = REPO / "skills"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_workflow(skill_name: str) -> dict:
    """Load and parse a skill's workflow.yaml."""
    path = SKILLS_DIR / skill_name / "workflow.yaml"
    assert path.exists(), f"workflow.yaml not found for skill '{skill_name}'"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_command(command: str, skill_dir: str, scripts_dir: str) -> str:
    """Resolve {skill_dir} and {scripts_dir} placeholders in a command."""
    return command.replace("{skill_dir}", skill_dir).replace("{scripts_dir}", scripts_dir)


def _run_command(command: str, env: dict = None, timeout: int = 30) -> subprocess.CompletedProcess:
    """Run a shell command and return the result."""
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    return subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        env=merged_env,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# Test: All workflow.yaml files parse correctly
# ---------------------------------------------------------------------------

class TestWorkflowParsing:
    """Verify that all workflow.yaml files are valid YAML and have required fields."""

    @pytest.fixture(params=["automation", "organizer"])
    def workflow(self, request):
        return _load_workflow(request.param)

    def test_workflow_has_skill_field(self, workflow):
        assert "skill" in workflow

    def test_workflow_has_workflows_list(self, workflow):
        assert "workflows" in workflow
        assert isinstance(workflow["workflows"], list)
        assert len(workflow["workflows"]) > 0

    def test_workflow_has_building_blocks(self, workflow):
        for wf in workflow["workflows"]:
            assert "building_blocks" in wf
            assert isinstance(wf["building_blocks"], list)
            assert len(wf["building_blocks"]) > 0

    def test_building_blocks_have_required_fields(self, workflow):
        for wf in workflow["workflows"]:
            for block in wf["building_blocks"]:
                assert "id" in block, f"Building block missing 'id'"
                assert "command" in block, f"Building block '{block.get('id')}' missing 'command'"
                assert "description" in block, f"Building block '{block.get('id')}' missing 'description'"


# ---------------------------------------------------------------------------
# Test: Automation workflow commands resolve to existing paths
# ---------------------------------------------------------------------------

class TestAutomationPaths:
    """Verify that automation workflow commands resolve to valid paths."""

    def _get_automation_blocks(self) -> list:
        wf = _load_workflow("automation")
        return wf["workflows"][0]["building_blocks"]

    def test_add_job_command_resolves_to_existing_script(self):
        blocks = self._get_automation_blocks()
        add_job = next(b for b in blocks if b["id"] == "add-job")

        skill_dir = str(SKILLS_DIR / "automation")
        resolved = _resolve_command(add_job["command"], skill_dir, skill_dir)

        # Extract the python script path from the command
        # Command format: python <path> add-job ...
        parts = resolved.split()
        script_path = Path(parts[1])

        assert script_path.exists(), f"Script not found: {script_path}"
        assert script_path.name == "organizer.py"

    def test_all_automation_commands_resolve_to_existing_script(self):
        blocks = self._get_automation_blocks()
        skill_dir = str(SKILLS_DIR / "automation")

        for block in blocks:
            resolved = _resolve_command(block["command"], skill_dir, skill_dir)
            parts = resolved.split()
            script_path = Path(parts[1])

            assert script_path.exists(), \
                f"Block '{block['id']}': script not found: {script_path}"


# ---------------------------------------------------------------------------
# Test: Organizer workflow commands resolve to existing paths
# ---------------------------------------------------------------------------

class TestOrganizerPaths:
    """Verify that organizer workflow commands resolve to valid paths."""

    def _get_organizer_blocks(self) -> list:
        wf = _load_workflow("organizer")
        return wf["workflows"][0]["building_blocks"]

    def test_all_organizer_commands_resolve_to_existing_script(self):
        blocks = self._get_organizer_blocks()
        scripts_dir = str(SKILLS_DIR / "organizer" / "scripts")

        for block in blocks:
            resolved = _resolve_command(block["command"], scripts_dir, scripts_dir)
            parts = resolved.split()
            script_path = Path(parts[1])

            assert script_path.exists(), \
                f"Block '{block['id']}': script not found: {script_path}"


# ---------------------------------------------------------------------------
# Test: Organizer script accepts documented arguments
# ---------------------------------------------------------------------------

class TestOrganizerScriptArguments:
    """Verify that organizer.py accepts the arguments documented in workflow.yaml."""

    def _run_organizer(self, *args, env_vars: dict = None) -> subprocess.CompletedProcess:
        script = SKILLS_DIR / "organizer" / "scripts" / "organizer.py"
        env = os.environ.copy()
        env["PAWLIA_USER_ID"] = "test_user"
        if env_vars:
            env.update(env_vars)
        return subprocess.run(
            [sys.executable, str(script), *args],
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
        )

    def test_add_reminder_accepts_none_recurrence(self, tmp_path):
        env = {"PAWLIA_SESSION_DIR": str(tmp_path)}
        result = self._run_organizer(
            "add-reminder",
            "--fire-at", "10m",
            "--message", "Test reminder",
            "--recurrence", "none",
            env_vars=env,
        )
        assert result.returncode == 0, f"Failed: {result.stderr}"
        data = json.loads(result.stdout)
        assert data["success"] is True

    def test_add_reminder_accepts_daily_recurrence(self, tmp_path):
        env = {"PAWLIA_SESSION_DIR": str(tmp_path)}
        result = self._run_organizer(
            "add-reminder",
            "--fire-at", "10m",
            "--message", "Daily reminder",
            "--recurrence", "daily",
            env_vars=env,
        )
        assert result.returncode == 0, f"Failed: {result.stderr}"
        data = json.loads(result.stdout)
        assert data["success"] is True

    def test_add_reminder_accepts_weekly_recurrence(self, tmp_path):
        env = {"PAWLIA_SESSION_DIR": str(tmp_path)}
        result = self._run_organizer(
            "add-reminder",
            "--fire-at", "10m",
            "--message", "Weekly reminder",
            "--recurrence", "weekly",
            env_vars=env,
        )
        assert result.returncode == 0, f"Failed: {result.stderr}"
        data = json.loads(result.stdout)
        assert data["success"] is True

    def test_add_reminder_accepts_rrule_recurrence(self, tmp_path):
        """Test that RRULE strings like FREQ=WEEKLY;BYDAY=MO,WE are accepted."""
        env = {"PAWLIA_SESSION_DIR": str(tmp_path)}
        result = self._run_organizer(
            "add-reminder",
            "--fire-at", "10m",
            "--message", "RRULE reminder",
            "--recurrence", "FREQ=WEEKLY;BYDAY=MO,WE",
            env_vars=env,
        )
        assert result.returncode == 0, f"Failed: {result.stderr}"
        data = json.loads(result.stdout)
        assert data["success"] is True

    def test_add_reminder_accepts_monthly_recurrence(self, tmp_path):
        env = {"PAWLIA_SESSION_DIR": str(tmp_path)}
        result = self._run_organizer(
            "add-reminder",
            "--fire-at", "10m",
            "--message", "Monthly reminder",
            "--recurrence", "monthly",
            env_vars=env,
        )
        assert result.returncode == 0, f"Failed: {result.stderr}"
        data = json.loads(result.stdout)
        assert data["success"] is True

    def test_add_job_accepts_valid_schedule(self, tmp_path):
        env = {"PAWLIA_SESSION_DIR": str(tmp_path)}
        result = self._run_organizer(
            "add-job",
            "--name", "Test Job",
            "--instruction", "Show tasks",
            "--schedule", "16:00",
            env_vars=env,
        )
        assert result.returncode == 0, f"Failed: {result.stderr}"
        data = json.loads(result.stdout)
        assert data["success"] is True

    def test_add_job_accepts_interval_schedule(self, tmp_path):
        env = {"PAWLIA_SESSION_DIR": str(tmp_path)}
        result = self._run_organizer(
            "add-job",
            "--name", "Interval Job",
            "--instruction", "Check weather",
            "--schedule", "interval:1h",
            env_vars=env,
        )
        assert result.returncode == 0, f"Failed: {result.stderr}"
        data = json.loads(result.stdout)
        assert data["success"] is True

    def test_add_job_accepts_weekly_schedule(self, tmp_path):
        env = {"PAWLIA_SESSION_DIR": str(tmp_path)}
        result = self._run_organizer(
            "add-job",
            "--name", "Weekly Job",
            "--instruction", "Weekly report",
            "--schedule", "weekly:0:09:00",
            env_vars=env,
        )
        assert result.returncode == 0, f"Failed: {result.stderr}"
        data = json.loads(result.stdout)
        assert data["success"] is True

    def test_list_jobs_works(self, tmp_path):
        env = {"PAWLIA_SESSION_DIR": str(tmp_path)}
        result = self._run_organizer(
            "list-jobs",
            env_vars=env,
        )
        assert result.returncode == 0, f"Failed: {result.stderr}"
        data = json.loads(result.stdout)
        assert "jobs" in data or data.get("success") is True

    def test_list_reminders_works(self, tmp_path):
        env = {"PAWLIA_SESSION_DIR": str(tmp_path)}
        result = self._run_organizer(
            "list-reminders",
            env_vars=env,
        )
        assert result.returncode == 0, f"Failed: {result.stderr}"
        data = json.loads(result.stdout)
        assert "reminders" in data or data.get("success") is True


# ---------------------------------------------------------------------------
# Test: Full workflow execution simulation
# ---------------------------------------------------------------------------

class TestWorkflowExecution:
    """Simulate executing workflow building blocks end-to-end."""

    def test_automation_add_job_workflow(self, tmp_path):
        """Simulate the automation add-job building block."""
        wf = _load_workflow("automation")
        blocks = wf["workflows"][0]["building_blocks"]
        add_job = next(b for b in blocks if b["id"] == "add-job")

        skill_dir = str(SKILLS_DIR / "automation")
        command = _resolve_command(add_job["command"], skill_dir, skill_dir)

        # Replace placeholders with test values
        command = command.replace("{name}", "Test Automation")
        command = command.replace("{schedule}", "16:00")
        command = command.replace("{instruction}", "Show open tasks")

        env = {
            "PAWLIA_USER_ID": "test_user",
            "PAWLIA_SESSION_DIR": str(tmp_path),
        }

        result = _run_command(command, env=env)
        assert result.returncode == 0, f"Command failed: {result.stderr}"
        data = json.loads(result.stdout)
        assert data["success"] is True

    def test_organizer_add_reminder_workflow(self, tmp_path):
        """Simulate the organizer add-reminder building block."""
        wf = _load_workflow("organizer")
        blocks = wf["workflows"][0]["building_blocks"]
        add_reminder = next(b for b in blocks if b["id"] == "add-reminder")

        scripts_dir = str(SKILLS_DIR / "organizer" / "scripts")
        command = _resolve_command(add_reminder["command"], scripts_dir, scripts_dir)

        # Replace placeholders with test values
        command = command.replace("{fire_at}", "10m")
        command = command.replace("{message}", "Test reminder")
        command = command.replace("{label}", "Test")
        command = command.replace("{recurrence}", "none")

        env = {
            "PAWLIA_USER_ID": "test_user",
            "PAWLIA_SESSION_DIR": str(tmp_path),
        }

        result = _run_command(command, env=env)
        assert result.returncode == 0, f"Command failed: {result.stderr}"
        data = json.loads(result.stdout)
        assert data["success"] is True

    def test_organizer_add_reminder_with_rrule_workflow(self, tmp_path):
        """Simulate add-reminder with RRULE recurrence."""
        wf = _load_workflow("organizer")
        blocks = wf["workflows"][0]["building_blocks"]
        add_reminder = next(b for b in blocks if b["id"] == "add-reminder")

        scripts_dir = str(SKILLS_DIR / "organizer" / "scripts")
        command = _resolve_command(add_reminder["command"], scripts_dir, scripts_dir)

        # Replace placeholders with test values including RRULE
        command = command.replace("{fire_at}", "10m")
        command = command.replace("{message}", "RRULE reminder")
        command = command.replace("{label}", "Test")
        command = command.replace("{recurrence}", "FREQ=WEEKLY;BYDAY=MO,WE")

        env = {
            "PAWLIA_USER_ID": "test_user",
            "PAWLIA_SESSION_DIR": str(tmp_path),
        }

        result = _run_command(command, env=env)
        assert result.returncode == 0, f"Command failed: {result.stderr}"
        data = json.loads(result.stdout)
        assert data["success"] is True
