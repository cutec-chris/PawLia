import pytest

from pawlia.coding import _build_task_prompt, detect_backend


# ── detect_backend ──────────────────────────────────────────────────────

def test_detect_backend_explicit_aider():
    cfg = {"coding": {"backend": "aider"}}
    assert detect_backend(cfg) == "aider"


def test_detect_backend_explicit_opencode():
    cfg = {"coding": {"backend": "opencode"}}
    assert detect_backend(cfg) == "opencode"


def test_detect_backend_explicit_llm():
    cfg = {"coding": {"backend": "llm"}}
    assert detect_backend(cfg) == "llm"


def test_detect_backend_skill_config_overrides_global():
    cfg = {
        "coding": {"backend": "llm"},
        "skill-config": {"skill-creator": {"coding_backend": "aider"}},
    }
    assert detect_backend(cfg) == "aider"


def test_detect_backend_auto_falls_back_to_llm(monkeypatch):
    import shutil
    monkeypatch.setattr(shutil, "which", lambda _: None)
    assert detect_backend({}) == "llm"


def test_detect_backend_auto_picks_aider_when_available(monkeypatch):
    import shutil
    monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/aider" if cmd == "aider" else None)
    assert detect_backend({}) == "aider"


def test_detect_backend_auto_picks_opencode_when_aider_missing(monkeypatch):
    import shutil
    monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/opencode" if cmd == "opencode" else None)
    assert detect_backend({}) == "opencode"


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
