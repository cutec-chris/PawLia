"""Direct ChatAgent file tools — the thin wrappers in pawlia.tools.files_tools.

These tools shell out to ``skills/files/scripts/files.py`` (read/list/grep
only). The e2e_* scripts exercise them but are not collected by pytest, so the
CI path never touched them. Here we drive each Tool through ``execute`` against
a real workspace and assert the JSON contract plus the local arg validation
that short-circuits before any subprocess runs.
"""

import json

import pytest

from pawlia.tools.files_tools import GrepFilesTool, ListFilesTool, ReadFileTool


@pytest.fixture
def workspace(tmp_path):
    """A real ``<session_dir>/<user_id>/workspace`` and a matching context."""
    ws = tmp_path / "u1" / "workspace"
    ws.mkdir(parents=True)
    ctx = {"user_id": "u1", "session_dir": str(tmp_path)}
    return ws, ctx


def _result(raw):
    """files.py returns JSON on stdout; the tools pass it through verbatim."""
    return json.loads(raw)


# ---- ReadFileTool ----------------------------------------------------------
def test_read_returns_file_contents(workspace):
    ws, ctx = workspace
    (ws / "notes").mkdir()
    (ws / "notes" / "a.md").write_text("Hello workspace\nsecond line\n", encoding="utf-8")

    out = _result(ReadFileTool().execute({"filename": "notes/a.md"}, ctx))

    assert out["success"] is True
    assert "Hello workspace" in out["content"]


def test_read_honours_offset_and_limit(workspace):
    ws, ctx = workspace
    (ws / "big.md").write_text("\n".join(f"line {i}" for i in range(50)), encoding="utf-8")

    out = _result(ReadFileTool().execute(
        {"filename": "big.md", "offset": 10, "limit": 3}, ctx))

    assert out["success"] is True
    assert "line 10" in out["content"]
    assert "line 0" not in out["content"]


def test_read_with_query_returns_only_matching_context(workspace):
    ws, ctx = workspace
    (ws / "doc.md").write_text(
        "intro\nNEEDLE is here\noutro\n" + "\n".join(f"pad {i}" for i in range(40)),
        encoding="utf-8",
    )

    out = _result(ReadFileTool().execute(
        {"filename": "doc.md", "query": "NEEDLE"}, ctx))

    assert out["success"] is True
    assert "NEEDLE" in out["content"]
    assert "pad 39" not in out["content"]


def test_read_missing_filename_is_rejected_without_subprocess(workspace):
    _, ctx = workspace
    out = _result(ReadFileTool().execute({}, ctx))
    assert out["success"] is False
    assert "filename" in out["error"].lower()


def test_read_nonexistent_file_reports_failure(workspace):
    _, ctx = workspace
    out = _result(ReadFileTool().execute({"filename": "ghost.md"}, ctx))
    assert out["success"] is False


# ---- ListFilesTool ---------------------------------------------------------
def test_list_returns_workspace_relative_paths(workspace):
    ws, ctx = workspace
    (ws / "notes").mkdir()
    (ws / "notes" / "a.md").write_text("x", encoding="utf-8")

    out = _result(ListFilesTool().execute({}, ctx))

    assert out["success"] is True
    assert "notes/a.md" in [f["name"] for f in out["files"]]


def test_list_honours_limit(workspace):
    ws, ctx = workspace
    for i in range(10):
        (ws / f"f{i:02d}.md").write_text("x", encoding="utf-8")

    out = _result(ListFilesTool().execute({"limit": 3}, ctx))

    assert out["success"] is True
    assert len(out["files"]) == 3


# ---- GrepFilesTool ---------------------------------------------------------
def test_grep_finds_pattern_across_files(workspace):
    ws, ctx = workspace
    (ws / "a.md").write_text("the answer is 42\n", encoding="utf-8")
    (ws / "b.md").write_text("nothing here\n", encoding="utf-8")

    out = _result(GrepFilesTool().execute({"pattern": r"answer is \d+"}, ctx))

    assert out["success"] is True
    assert json.dumps(out)  # well-formed; at least one match surfaced
    assert "a.md" in json.dumps(out)


def test_grep_can_restrict_to_a_single_file(workspace):
    ws, ctx = workspace
    (ws / "a.md").write_text("match me\n", encoding="utf-8")
    (ws / "b.md").write_text("match me too\n", encoding="utf-8")

    out = _result(GrepFilesTool().execute(
        {"pattern": "match", "filename": "a.md"}, ctx))

    assert out["success"] is True
    blob = json.dumps(out)
    assert "a.md" in blob
    assert "b.md" not in blob


def test_grep_missing_pattern_is_rejected_without_subprocess(workspace):
    _, ctx = workspace
    out = _result(GrepFilesTool().execute({}, ctx))
    assert out["success"] is False
    assert "pattern" in out["error"].lower()


# ---- context defaulting ----------------------------------------------------
def test_tools_default_user_and_session_when_context_missing(tmp_path, monkeypatch):
    """With no context the tools fall back to user 'default'/cwd '.', so the
    call still runs the script rather than crashing."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "default" / "workspace").mkdir(parents=True)
    out = _result(ListFilesTool().execute({}, None))
    assert out["success"] is True
