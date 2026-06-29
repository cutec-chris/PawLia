import json
from pathlib import Path
from unittest.mock import patch

import pytest

from pawlia.coding import _build_task_prompt, run_fix, run_implement
from pawlia.coding.coding import _extract_and_write_files


# ── _build_task_prompt ──────────────────────────────────────────────────


def test_build_task_prompt_implement_contains_task():
    prompt = _build_task_prompt(
        skill_md_body="Do something useful.",
        task="Create a new skill",
        existing_files={},
        references={},
        mode="implement",
    )
    assert "Create a new skill" in prompt
    assert "Do something useful." in prompt


def test_build_task_prompt_fix_includes_error():
    prompt = _build_task_prompt(
        skill_md_body="",
        task="Fix the bug",
        existing_files={},
        references={},
        mode="fix",
        error_output="NameError: foo not defined",
        failing_command="python scripts/run.py",
    )
    assert "NameError: foo not defined" in prompt
    assert "python scripts/run.py" in prompt


def test_build_task_prompt_includes_existing_files():
    prompt = _build_task_prompt(
        skill_md_body="",
        task="Update",
        existing_files={"scripts/main.py": "print('hello')"},
        references={},
        mode="implement",
    )
    assert "scripts/main.py" in prompt
    assert "print('hello')" in prompt


def test_build_task_prompt_references_truncated_at_2000():
    long_ref = "x" * 5000
    prompt = _build_task_prompt(
        skill_md_body="",
        task="Task",
        existing_files={},
        references={"ref.md": long_ref},
        mode="implement",
    )
    assert long_ref[:2000] in prompt
    assert long_ref[2000:] not in prompt


def test_build_task_prompt_implement_has_rules():
    prompt = _build_task_prompt(
        skill_md_body="",
        task="Task",
        existing_files={},
        references={},
        mode="implement",
    )
    assert "JSON" in prompt
    assert "success" in prompt


def test_build_task_prompt_fix_has_rules():
    prompt = _build_task_prompt(
        skill_md_body="",
        task="Task",
        existing_files={},
        references={},
        mode="fix",
    )
    assert "root cause" in prompt


# ── _extract_and_write_files ───────────────────────────────────────────


def _skill(tmp_path: Path) -> Path:
    """Create a minimal skill directory with scripts/."""
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "scripts").mkdir()
    return skill


def test_extract_writes_named_file_into_scripts(tmp_path):
    skill = _skill(tmp_path)
    content = "```scripts/main.py\nprint('hi')\n```"

    written = _extract_and_write_files(content, skill)

    assert written == ["scripts/main.py"]
    assert (skill / "scripts" / "main.py").read_text().strip() == "print('hi')"


def test_extract_writes_subpath_file(tmp_path):
    skill = _skill(tmp_path)
    content = "```scripts/lib/util.py\nx = 1\n```"

    written = _extract_and_write_files(content, skill)

    assert written == ["scripts/lib/util.py"]
    assert (skill / "scripts" / "lib" / "util.py").read_text().strip() == "x = 1"


def test_extract_ignores_bare_language_tags(tmp_path):
    skill = _skill(tmp_path)
    content = "```python\nprint('not a filename')\n```"

    written = _extract_and_write_files(content, skill)

    assert written == []
    assert list((skill / "scripts").iterdir()) == []


def test_extract_writes_multiple_files_in_order(tmp_path):
    skill = _skill(tmp_path)
    content = (
        "```scripts/a.py\na\n```\n"
        "```scripts/b.py\nb\n```\n"
        "```python\nignored\n```\n"
        "```scripts/c.py\nc\n```\n"
    )

    written = _extract_and_write_files(content, skill)

    assert written == ["scripts/a.py", "scripts/b.py", "scripts/c.py"]


def test_extract_returns_empty_when_no_fences(tmp_path):
    skill = _skill(tmp_path)
    # Real-world failure mode: model returns a full replacement script
    # without per-file fencing. Must be reported as zero files written
    # so the caller can surface a meaningful error.
    content = "Here is the fixed script:\n\nimport os\nprint(os.getcwd())\n"

    written = _extract_and_write_files(content, skill)

    assert written == []


# ── run_implement / run_fix → _run_llm (sole path) ─────────────────────


class _StubResponse:
    def __init__(self, content: str):
        self.content = content


class _StubLLM:
    def __init__(self, content: str):
        self._content = content
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return _StubResponse(self._content)


def test_run_implement_writes_files_via_llm(tmp_path):
    skill = _skill(tmp_path)
    (skill / "SKILL.md").write_text("---\nname: test\n---\n# Test\n")
    config = {"models": {"coder": {"model": "stub", "provider": "stub"}}}
    stub = _StubLLM(
        "```scripts/main.py\nprint('generated')\n```"
    )

    with patch("pawlia.llm.LLMFactory") as factory:
        factory.return_value.get.return_value = stub
        result = run_implement(skill, "Write main.py", config)

    assert result["ok"] is True
    assert result["backend"] == "llm"
    assert result["files_written"] == ["scripts/main.py"]
    assert (skill / "scripts" / "main.py").read_text().strip() == "print('generated')"
    assert stub.calls == 1


def test_run_fix_writes_files_via_llm(tmp_path):
    skill = _skill(tmp_path)
    (skill / "SKILL.md").write_text("---\nname: test\n---\n# Test\n")
    config = {"models": {"coder": {"model": "stub", "provider": "stub"}}}
    stub = _StubLLM(
        "```scripts/main.py\nprint('fixed')\n```"
    )

    with patch("pawlia.llm.LLMFactory") as factory:
        factory.return_value.get.return_value = stub
        result = run_fix(
            skill, error="NameError: x", command="python scripts/main.py", config=config
        )

    assert result["ok"] is True
    assert result["backend"] == "llm"
    assert result["files_written"] == ["scripts/main.py"]
    assert (skill / "scripts" / "main.py").read_text().strip() == "print('fixed')"


def test_run_implement_reports_failure_when_no_files_written(tmp_path):
    skill = _skill(tmp_path)
    (skill / "SKILL.md").write_text("---\nname: test\n---\n# Test\n")
    config = {"models": {"coder": {"model": "stub", "provider": "stub"}}}
    stub = _StubLLM("Here you go, just put this at the top: import os\n")

    with patch("pawlia.llm.LLMFactory") as factory:
        factory.return_value.get.return_value = stub
        result = run_implement(skill, "Add imports", config)

    assert result["ok"] is False
    assert result["backend"] == "llm"
    assert result["files_written"] == []
    assert "import os" in result["output"]


def test_run_implement_catches_llm_exception(tmp_path):
    skill = _skill(tmp_path)
    (skill / "SKILL.md").write_text("---\nname: test\n---\n# Test\n")
    config = {"models": {"coder": {"model": "stub", "provider": "stub"}}}

    class _Boom:
        def invoke(self, messages):
            raise RuntimeError("provider timeout")

    with patch("pawlia.llm.LLMFactory") as factory:
        factory.return_value.get.return_value = _Boom()
        result = run_implement(skill, "task", config)

    assert result["ok"] is False
    assert result["backend"] == "llm"
    assert "provider timeout" in result["error"]
