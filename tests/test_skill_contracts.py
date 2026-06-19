"""Skill scripts vs. the contract their SKILL.md advertises — no LLM involved.

These run the skill's own script the way the agent does (subprocess with the
PAWLIA_* environment), and assert the documented behavior: the JSON shape, the
workspace sandbox, pagination, wikilink/section resolution, etc. A failure
means the script no longer honors what its SKILL.md promises.
"""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FILES_SCRIPT = REPO / "skills" / "files" / "scripts" / "files.py"


# ---------------------------------------------------------------------------
# files skill — SKILL.md promises: workspace-relative paths, 200-file
# pagination, wikilink + section resolution, traversal blocked.
# ---------------------------------------------------------------------------
def _files(tmp_path, *args, content=None):
    env = os.environ.copy()
    env["PAWLIA_USER_ID"] = "u1"
    env["PAWLIA_SESSION_DIR"] = str(tmp_path)
    if content is not None:
        env["CONTENT"] = content
    proc = subprocess.run(
        [sys.executable, str(FILES_SCRIPT), *args],
        capture_output=True, text=True, env=env,
    )
    return json.loads(proc.stdout)


def _workspace(tmp_path):
    ws = tmp_path / "u1" / "workspace"
    ws.mkdir(parents=True, exist_ok=True)
    return ws


def test_write_then_read_roundtrips_content(tmp_path):
    _workspace(tmp_path)
    written = _files(tmp_path, "write", "--filename", "notes/today.md",
                     content="Hello workspace")
    assert written["success"] is True

    read = _files(tmp_path, "read", "--filename", "notes/today.md")
    assert read["success"] is True
    assert "Hello workspace" in read["content"]


def test_list_returns_workspace_relative_paths(tmp_path):
    ws = _workspace(tmp_path)
    (ws / "notes").mkdir()
    (ws / "notes" / "a.md").write_text("x", encoding="utf-8")

    result = _files(tmp_path, "list")

    assert result["success"] is True
    names = [f["name"] for f in result["files"]]
    assert "notes/a.md" in names  # recursive, workspace-relative


def test_list_is_paginated_to_200_by_default(tmp_path):
    ws = _workspace(tmp_path)
    for idx in range(205):
        (ws / f"f_{idx:03d}.txt").write_text("x", encoding="utf-8")

    page1 = _files(tmp_path, "list")
    assert page1["count"] == 200
    assert page1["total_count"] == 205
    assert page1["has_more"] is True
    assert page1["next_offset"] == 200

    page2 = _files(tmp_path, "list", "--offset", "200", "--limit", "200")
    assert page2["count"] == 5
    assert page2["has_more"] is False


def test_read_resolves_wikilink_with_section_anchor(tmp_path):
    target = tmp_path / "u1" / "workspace" / "wiki" / "topics" / "topic"
    target.mkdir(parents=True)
    (target / "homelab.md").write_text(
        "# Homelab\n\n## Proxmox\n\n64GB RAM.\n\n## Backup\n\nNightly Borg.\n",
        encoding="utf-8",
    )

    result = _files(tmp_path, "read", "--filename", "[[topic/homelab#Proxmox]]")

    assert result["success"] is True
    assert result["section"] == "Proxmox"
    assert "64GB RAM" in result["content"]
    assert "Nightly Borg" not in result["content"]


def test_read_with_query_returns_only_matching_context(tmp_path):
    ws = _workspace(tmp_path)
    lines = [f"line {i}" for i in range(50)]
    lines[25] = "the secret token is here"
    (ws / "big.md").write_text("\n".join(lines), encoding="utf-8")

    result = _files(tmp_path, "read", "--filename", "big.md", "--query", "secret token")

    assert result["success"] is True
    assert "secret token" in result["content"]
    assert "skipped" in result["content"]  # non-matching stretches collapsed


def test_read_page_fits_skill_runner_result_cap_for_huge_files(tmp_path):
    """Invariant: a single read page must JSON-serialize within the skill
    runner's tool-result cap, for arbitrarily large files AND long lines.

    If a page exceeds the cap, the runner appends a truncation marker and the
    model re-reads the same range in a loop (the 8-turn-for-one-file pathology).
    This guards the scaling case, not just a happy-path small read.
    """
    from pawlia.agents.skill_runner import SkillRunnerAgent

    ws = _workspace(tmp_path)
    # 5000 lines, and every line very long, so neither the line-count limit nor
    # naive truncation alone keeps the page under the cap — only the byte budget.
    big = "\n".join("x" * 300 for _ in range(5000))
    (ws / "huge.txt").write_text(big, encoding="utf-8")

    offset = 0
    pages = 0
    total_lines_seen = 0
    while True:
        result = _files(tmp_path, "read", "--filename", "huge.txt",
                        "--offset", str(offset))
        assert result["success"] is True
        # The exact payload the runner would feed to the model.
        serialized = json.dumps(result)
        assert len(serialized) <= SkillRunnerAgent.MAX_RESULT, (
            f"page at offset {offset} serializes to {len(serialized)} chars, "
            f"exceeding MAX_RESULT={SkillRunnerAgent.MAX_RESULT}"
        )
        assert result["lines_returned"] >= 1  # always make progress
        total_lines_seen += result["lines_returned"]
        pages += 1
        if not result.get("has_more"):
            break
        offset = result["next_offset"]
        assert pages < 5000  # pagination must terminate

    assert total_lines_seen == 5000  # every line is reachable across pages


def test_read_small_file_returns_in_one_page(tmp_path):
    """A normal-sized file must not be split — otherwise the model paginates
    needlessly, the very behavior that inflated latency."""
    ws = _workspace(tmp_path)
    (ws / "small.py").write_text("\n".join(f"line {i}" for i in range(120)),
                                 encoding="utf-8")

    result = _files(tmp_path, "read", "--filename", "small.py")

    assert result["success"] is True
    assert result.get("has_more") is not True
    assert result["lines_returned"] == 120


def test_delete_removes_the_file(tmp_path):
    ws = _workspace(tmp_path)
    (ws / "junk.md").write_text("bye", encoding="utf-8")

    result = _files(tmp_path, "delete", "--filename", "junk.md")

    assert result["success"] is True
    assert not (ws / "junk.md").exists()


def test_path_traversal_outside_the_workspace_is_blocked(tmp_path):
    _workspace(tmp_path)
    secret = tmp_path / "u1" / "secret.md"
    secret.write_text("top secret", encoding="utf-8")

    result = _files(tmp_path, "read", "--filename", "../secret.md")

    assert result["success"] is False
    assert "outside the workspace" in (result.get("error", "")).lower() \
        or "access denied" in (result.get("error", "")).lower()


def test_grep_with_invalid_regex_returns_hint_about_python_re_syntax(tmp_path):
    """The 'Invalid regex' error must explain that --pattern is Python re,
    not a shell glob — otherwise the LLM retries with the same mistake."""
    ws = _workspace(tmp_path)
    (ws / "data.txt").write_text("hello\nworld\n", encoding="utf-8")

    # "*.foo" is a shell glob; the Python re compiler rejects it.
    result = _files(tmp_path, "grep", "--pattern", "*.foo", "--filename", "data.txt")

    assert result["success"] is False
    assert "Invalid regex" in result["error"]
    assert "hint" in result
    assert "Python re" in result["hint"] or "shell glob" in result["hint"]


def test_grep_with_valid_regex_still_works(tmp_path):
    ws = _workspace(tmp_path)
    (ws / "data.txt").write_text("foo\nbar\n", encoding="utf-8")

    result = _files(tmp_path, "grep", "--pattern", "f.o", "--filename", "data.txt")

    assert result["success"] is True
    assert result.get("matches"), f"expected matches, got {result!r}"


# ---------------------------------------------------------------------------
# config skill — promises to discover Piper voices from env / configured dir.
# ---------------------------------------------------------------------------
def _load_config_skill():
    path = REPO / "skills" / "config" / "scripts" / "config.py"
    spec = importlib.util.spec_from_file_location("config_skill_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_config_lists_piper_voices_from_the_env_dir(monkeypatch):
    mod = _load_config_skill()
    monkeypatch.setenv("PAWLIA_PIPER_DIR", "/voices/env")
    monkeypatch.setattr(mod, "_find_config", lambda: None)
    env_dir = mod.os.path.abspath("/voices/env")
    monkeypatch.setattr(mod.os.path, "isdir", lambda p: p == env_dir)

    import glob
    monkeypatch.setattr(glob, "glob", lambda pat: (
        [mod.os.path.join(env_dir, "de_DE-thorsten-low.onnx")]
        if pat == mod.os.path.join(env_dir, "*.onnx") else []
    ))

    assert mod._list_piper_voices() == ["de_DE-thorsten-low"]


def test_config_lists_piper_voices_from_the_configured_model_dir(monkeypatch):
    mod = _load_config_skill()
    monkeypatch.delenv("PAWLIA_PIPER_DIR", raising=False)
    monkeypatch.delenv("PIPER_VOICE_DIR", raising=False)
    monkeypatch.setattr(mod, "_find_config", lambda: "/app/config.yaml")
    monkeypatch.setattr(mod, "_read", lambda p: {
        "tts": {"provider": "piper",
                "piper": {"model": "/voices/config/de_DE-kerstin-low.onnx"}}})
    model_dir = mod.os.path.abspath("/voices/config")
    monkeypatch.setattr(mod.os.path, "isdir", lambda p: p == model_dir)

    import glob
    monkeypatch.setattr(glob, "glob", lambda pat: (
        [mod.os.path.join(model_dir, "de_DE-ramona-low.onnx")]
        if pat == mod.os.path.join(model_dir, "*.onnx") else []
    ))

    assert mod._list_piper_voices() == ["de_DE-ramona-low"]
