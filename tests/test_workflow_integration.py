"""Integration tests for workflow.yaml building blocks.

Tests that workflow commands resolve correct paths and that the underlying
scripts accept the documented arguments. Catches issues like:
- Wrong {scripts_dir}/{skill_dir} resolution
- Invalid argument choices (e.g. --recurrence rejecting RRULE strings)
- Missing script files

Uses the real WorkflowExecutor._substitute to resolve paths, not a manual
reimplementation of the substitution logic.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from pawlia.skills.executor import WorkflowExecutor
from pawlia.skills.workflow_schema import BuildingBlock, CompiledWorkflow, Workflow

REPO = Path(__file__).resolve().parents[1]
SKILLS_DIR = REPO / "skills"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_compiled(skill_name: str) -> CompiledWorkflow:
    """Load and parse a skill's workflow.yaml into a CompiledWorkflow model."""
    path = SKILLS_DIR / skill_name / "workflow.yaml"
    assert path.exists(), f"workflow.yaml not found for skill '{skill_name}'"
    with open(path, encoding="utf-8") as f:
        return CompiledWorkflow.model_validate(yaml.safe_load(f))


def _make_executor(skill_name: str) -> WorkflowExecutor:
    """Create a WorkflowExecutor with minimal context for path resolution."""
    skill_dir = str(SKILLS_DIR / skill_name)
    return WorkflowExecutor(
        tool_registry=None,
        context={"cwd": skill_dir},
        llm=None,
    )


def _find_blocks(compiled: CompiledWorkflow) -> list[BuildingBlock]:
    """Return all building blocks across all workflows."""
    return [b for w in compiled.workflows for b in w.building_blocks]


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
# Test: All workflow.yaml files parse correctly via CompiledWorkflow schema
# ---------------------------------------------------------------------------

class TestWorkflowParsing:
    """Verify that all workflow.yaml files are valid against the schema."""

    @pytest.fixture(params=["automation", "organizer"])
    def compiled(self, request):
        return _load_compiled(request.param)

    def test_parses_as_valid_compiled_workflow(self, compiled):
        assert isinstance(compiled, CompiledWorkflow)
        assert compiled.skill
        assert compiled.version

    def test_has_workflows(self, compiled):
        assert len(compiled.workflows) > 0
        for wf in compiled.workflows:
            assert isinstance(wf, Workflow)

    def test_has_building_blocks(self, compiled):
        for wf in compiled.workflows:
            assert len(wf.building_blocks) > 0
            for block in wf.building_blocks:
                assert isinstance(block, BuildingBlock)
                assert block.id
                assert block.command
                assert block.description


# ---------------------------------------------------------------------------
# Test: Workflow commands resolve to existing script paths using real executor
# ---------------------------------------------------------------------------

class TestWorkflowPaths:
    """Use the real WorkflowExecutor._substitute to validate paths."""

    def _check_blocks_resolve(self, skill_name: str):
        compiled = _load_compiled(skill_name)
        executor = _make_executor(skill_name)
        for block in _find_blocks(compiled):
            resolved = executor._substitute(block.command, {})
            parts = resolved.split()
            script_path = Path(parts[1])
            assert script_path.exists(), \
                f"Skill '{skill_name}' block '{block.id}': script not found: {script_path}"

    def test_automation_commands_resolve(self):
        self._check_blocks_resolve("automation")

    def test_organizer_commands_resolve(self):
        self._check_blocks_resolve("organizer")


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
# Test: Full workflow execution using the real executor's _substitute
# ---------------------------------------------------------------------------

class TestWorkflowExecution:
    """Simulate executing workflow building blocks end-to-end."""

    def test_automation_add_job_workflow(self, tmp_path):
        """Simulate the automation add-job building block via executor."""
        compiled = _load_compiled("automation")
        block = next(b for w in compiled.workflows for b in w.building_blocks if b.id == "add-job")

        executor = _make_executor("automation")
        params = {"name": "Test Automation", "schedule": "16:00", "instruction": "Show open tasks"}
        command = executor._substitute(block.command, params)

        env = {
            "PAWLIA_USER_ID": "test_user",
            "PAWLIA_SESSION_DIR": str(tmp_path),
        }
        result = _run_command(command, env=env)
        assert result.returncode == 0, f"Command failed: {result.stderr}"
        data = json.loads(result.stdout)
        assert data["success"] is True

    def test_organizer_add_reminder_workflow(self, tmp_path):
        """Simulate the organizer add-reminder building block via executor."""
        compiled = _load_compiled("organizer")
        block = next(b for w in compiled.workflows for b in w.building_blocks if b.id == "add-reminder")

        executor = _make_executor("organizer")
        params = {"fire_at": "10m", "message": "Test reminder", "label": "Test", "recurrence": "none"}
        command = executor._substitute(block.command, params)

        env = {
            "PAWLIA_USER_ID": "test_user",
            "PAWLIA_SESSION_DIR": str(tmp_path),
        }
        result = _run_command(command, env=env)
        assert result.returncode == 0, f"Command failed: {result.stderr}"
        data = json.loads(result.stdout)
        assert data["success"] is True

    def test_organizer_add_reminder_with_rrule_workflow(self, tmp_path):
        """Simulate add-reminder with RRULE recurrence via executor."""
        compiled = _load_compiled("organizer")
        block = next(b for w in compiled.workflows for b in w.building_blocks if b.id == "add-reminder")

        executor = _make_executor("organizer")
        params = {
            "fire_at": "10m",
            "message": "RRULE reminder",
            "label": "Test",
            "recurrence": "FREQ=WEEKLY;BYDAY=MO,WE",
        }
        command = executor._substitute(block.command, params)

        env = {
            "PAWLIA_USER_ID": "test_user",
            "PAWLIA_SESSION_DIR": str(tmp_path),
        }
        result = _run_command(command, env=env)
        assert result.returncode == 0, f"Command failed: {result.stderr}"
        data = json.loads(result.stdout)
        assert data["success"] is True
