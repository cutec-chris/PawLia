"""Tests for pawlia.workspace_git."""

import os
import subprocess
import tempfile
import time

import pytest

from pawlia.workspace_git import (
    COMMIT_COOLDOWN,
    auto_commit,
    daily_squash,
    ensure_repo,
    weekly_squash,
)


def _git(cwd, *args):
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True,
    )


def _commit_count(cwd):
    r = _git(cwd, "rev-list", "--count", "HEAD")
    return int(r.stdout.strip()) if r.returncode == 0 else 0


def _write(workspace, name, content="hello"):
    with open(os.path.join(workspace, name), "w") as f:
        f.write(content)


class TestEnsureRepo:
    def test_creates_repo(self):
        with tempfile.TemporaryDirectory() as ws:
            assert ensure_repo(ws)
            assert os.path.isdir(os.path.join(ws, ".git"))

    def test_idempotent(self):
        with tempfile.TemporaryDirectory() as ws:
            ensure_repo(ws)
            ensure_repo(ws)
            assert os.path.isdir(os.path.join(ws, ".git"))


class TestAutoCommit:
    def test_commits_changes(self):
        with tempfile.TemporaryDirectory() as ws:
            ensure_repo(ws)
            # Backdate last commit so cooldown passes
            _git(ws, "commit", "--allow-empty", "--date=2020-01-01T00:00:00", "-m", "old")
            # Use GIT_COMMITTER_DATE to also backdate committer timestamp
            env = os.environ.copy()
            env["GIT_COMMITTER_DATE"] = "2020-01-01T00:00:00"
            subprocess.run(
                ["git", "commit", "--allow-empty", "--date=2020-01-01T00:00:00", "-m", "old2"],
                cwd=ws, capture_output=True, env=env,
            )
            _write(ws, "test.md", "content")
            assert auto_commit(ws)
            assert _commit_count(ws) >= 2

    def test_respects_cooldown(self):
        with tempfile.TemporaryDirectory() as ws:
            ensure_repo(ws)
            _write(ws, "a.md", "a")
            # First commit should work (initial commit is old enough or we just init'd)
            # But auto_commit checks last commit time vs now, and we just committed in ensure_repo
            result = auto_commit(ws)
            # Should be throttled because ensure_repo just committed
            assert result is False

    def test_no_changes_no_commit(self):
        with tempfile.TemporaryDirectory() as ws:
            ensure_repo(ws)
            assert auto_commit(ws) is False


class TestDailySquash:
    def test_squashes_multiple_commits(self):
        with tempfile.TemporaryDirectory() as ws:
            ensure_repo(ws)
            today = time.strftime("%Y-%m-%d")
            # Create multiple commits with today's date
            for i in range(3):
                _write(ws, f"file{i}.md", f"content {i}")
                _git(ws, "add", "-A")
                _git(ws, "commit", "-m", f"{today} {i:02d}:{i:02d}")

            before = _commit_count(ws)
            assert before >= 3
            assert daily_squash(ws)
            after = _commit_count(ws)
            assert after < before

            # Check commit message
            r = _git(ws, "log", "-1", "--format=%s")
            assert r.stdout.strip().startswith("Daily:")

    def test_no_squash_for_single_commit(self):
        with tempfile.TemporaryDirectory() as ws:
            ensure_repo(ws)
            assert daily_squash(ws) is False


class TestWeeklySquash:
    def test_squashes_multiple_commits(self):
        with tempfile.TemporaryDirectory() as ws:
            ensure_repo(ws)
            # Create an old anchor commit (last week)
            _write(ws, "anchor.md", "anchor")
            _git(ws, "add", "-A")
            env = os.environ.copy()
            env["GIT_COMMITTER_DATE"] = "2020-01-01T00:00:00"
            subprocess.run(
                ["git", "commit", "--date=2020-01-01T00:00:00", "-m", "anchor"],
                cwd=ws, capture_output=True, env=env,
            )

            # Create multiple commits this week
            for i in range(3):
                _write(ws, f"week{i}.md", f"week content {i}")
                _git(ws, "add", "-A")
                _git(ws, "commit", "-m", f"day {i}")

            before = _commit_count(ws)
            assert weekly_squash(ws)
            after = _commit_count(ws)
            assert after < before

            r = _git(ws, "log", "-1", "--format=%s")
            assert r.stdout.strip().startswith("Week:")
