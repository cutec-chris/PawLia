"""Tests for pawlia.tools.files_tools."""

import json
import os
import tempfile

from pawlia.tools.files_tools import ReadFileTool, ListFilesTool, GrepFilesTool


class TestReadFileTool:
    def test_read_simple_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            user_id = "testuser"
            workspace = os.path.join(tmpdir, user_id, "workspace")
            os.makedirs(workspace)
            with open(os.path.join(workspace, "hello.md"), "w") as f:
                f.write("# Hello\nThis is a test file.\n")

            tool = ReadFileTool()
            result = tool.execute(
                {"filename": "hello.md"},
                context={"user_id": user_id, "session_dir": tmpdir},
            )
            data = json.loads(result)
            assert data["success"] is True
            assert "Hello" in data["content"]

    def test_read_missing_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            user_id = "testuser"
            workspace = os.path.join(tmpdir, user_id, "workspace")
            os.makedirs(workspace)

            tool = ReadFileTool()
            result = tool.execute(
                {"filename": "missing.md"},
                context={"user_id": user_id, "session_dir": tmpdir},
            )
            data = json.loads(result)
            assert data["success"] is False
            assert "not found" in data["error"].lower()

    def test_spec(self):
        tool = ReadFileTool()
        spec = tool.as_openai_spec()
        assert spec["function"]["name"] == "read_file"
        assert "filename" in spec["function"]["parameters"]["properties"]


class TestListFilesTool:
    def test_list_workspace(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            user_id = "testuser"
            workspace = os.path.join(tmpdir, user_id, "workspace")
            os.makedirs(workspace)
            with open(os.path.join(workspace, "a.md"), "w") as f:
                f.write("A")
            with open(os.path.join(workspace, "b.md"), "w") as f:
                f.write("B")

            tool = ListFilesTool()
            result = tool.execute(
                {},
                context={"user_id": user_id, "session_dir": tmpdir},
            )
            data = json.loads(result)
            assert data["success"] is True
            names = {f["name"] for f in data["files"]}
            assert "a.md" in names
            assert "b.md" in names

    def test_spec(self):
        tool = ListFilesTool()
        spec = tool.as_openai_spec()
        assert spec["function"]["name"] == "list_files"


class TestGrepFilesTool:
    def test_grep_pattern(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            user_id = "testuser"
            workspace = os.path.join(tmpdir, user_id, "workspace")
            os.makedirs(workspace)
            with open(os.path.join(workspace, "notes.md"), "w") as f:
                f.write("Apples are red.\nBananas are yellow.\n")

            tool = GrepFilesTool()
            result = tool.execute(
                {"pattern": "Apples"},
                context={"user_id": user_id, "session_dir": tmpdir},
            )
            data = json.loads(result)
            assert data["success"] is True
            assert len(data["matches"]) >= 1
            assert "Apples" in data["matches"][0]["text"]

    def test_grep_no_match(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            user_id = "testuser"
            workspace = os.path.join(tmpdir, user_id, "workspace")
            os.makedirs(workspace)
            with open(os.path.join(workspace, "notes.md"), "w") as f:
                f.write("Hello world.\n")

            tool = GrepFilesTool()
            result = tool.execute(
                {"pattern": "xyz123"},
                context={"user_id": user_id, "session_dir": tmpdir},
            )
            data = json.loads(result)
            assert data["success"] is True
            assert len(data["matches"]) == 0

    def test_spec(self):
        tool = GrepFilesTool()
        spec = tool.as_openai_spec()
        assert spec["function"]["name"] == "grep_files"
        assert "pattern" in spec["function"]["parameters"]["properties"]
