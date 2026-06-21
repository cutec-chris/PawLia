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
    pull,
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

    def test_fresh_init_never_commits_skill_cruft(self):
        with tempfile.TemporaryDirectory() as ws:
            os.makedirs(os.path.join(ws, "skills", "bahn", "node_modules", "dep"))
            _write(ws, "skills/bahn/node_modules/dep/a.js", "x")
            _write(ws, "note.md", "keep")
            ensure_repo(ws)
            tracked = _git(ws, "ls-files").stdout
            assert "note.md" in tracked
            assert "node_modules" not in tracked

    def test_untracks_cruft_committed_before_pattern_existed(self):
        with tempfile.TemporaryDirectory() as ws:
            # Skill committed WITH cruft, before our gitignore ran.
            os.makedirs(os.path.join(ws, "skills", "bahn", "node_modules", "dep"))
            os.makedirs(os.path.join(ws, "skills", "bahn", "__pycache__"))
            _write(ws, "skills/bahn/node_modules/dep/a.js", "x")
            _write(ws, "skills/bahn/__pycache__/m.pyc", "x")
            _write(ws, "skills/bahn/SKILL.md", "# bahn")
            _git(ws, "init")
            _git(ws, "config", "user.email", "t@t")
            _git(ws, "config", "user.name", "t")
            _git(ws, "add", "-A")
            _git(ws, "commit", "-m", "with cruft")

            ensure_repo(ws)
            _git(ws, "commit", "-m", "cleanup")

            tracked = _git(ws, "ls-files").stdout
            assert "node_modules" not in tracked
            assert "__pycache__" not in tracked
            assert "skills/bahn/SKILL.md" in tracked
            # Files stay on disk — only untracked, not deleted.
            assert os.path.exists(os.path.join(ws, "skills", "bahn", "node_modules", "dep", "a.js"))


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


def _seed_remote(workspace: str, remote: str) -> str:
    """Init a repo in `workspace`, add `remote` as origin, commit, push.

    Returns the branch name (so tests are independent of the git default branch).
    """
    _git(workspace, "init")
    _git(workspace, "config", "user.name", "test")
    _git(workspace, "config", "user.email", "test@test")
    _write(workspace, "init.md", "init")
    _git(workspace, "add", "-A")
    _git(workspace, "commit", "-m", "init")
    _git(workspace, "remote", "add", "origin", remote)
    branch = _git(workspace, "symbolic-ref", "--short", "HEAD").stdout.strip()
    _git(workspace, "push", "-q", "origin", branch)
    # Establish origin/HEAD so rev-parse origin/HEAD works (clone normally does this)
    _git(workspace, "remote", "set-head", "origin", branch)
    return branch


class TestPull:
    def test_pull_no_remote_returns_false(self):
        with tempfile.TemporaryDirectory() as ws:
            ensure_repo(ws)
            assert pull(ws) is False

    def test_pull_up_to_date(self, monkeypatch):
        monkeypatch.setattr("pawlia.workspace_git.PULL_COOLDOWN", 0)
        with tempfile.TemporaryDirectory() as remote, tempfile.TemporaryDirectory() as ws:
            _git(remote, "init", "--bare")
            _seed_remote(ws, remote)
            # Local HEAD already matches origin/HEAD
            assert pull(ws) is True

    def test_pull_fast_forward(self, monkeypatch):
        """Local behind remote → ff-only advances local HEAD."""
        monkeypatch.setattr("pawlia.workspace_git.PULL_COOLDOWN", 0)
        with tempfile.TemporaryDirectory() as remote, tempfile.TemporaryDirectory() as ws:
            _git(remote, "init", "--bare")
            branch = _seed_remote(ws, remote)

            # Advance the remote from a second clone
            with tempfile.TemporaryDirectory() as other_parent:
                other = os.path.join(other_parent, "other")
                _git(other_parent, "clone", "-q", remote, other)
                _git(other, "config", "user.name", "test")
                _git(other, "config", "user.email", "test@test")
                _write(other, "remote_only.md", "from remote")
                _git(other, "add", "-A")
                _git(other, "commit", "-m", "remote advance")
                _git(other, "push", "-q", "origin", branch)

            # Local ws is behind; pull should fast-forward
            assert pull(ws) is True
            assert os.path.exists(os.path.join(ws, "remote_only.md"))

    def test_pull_divergent_resets_hard(self, monkeypatch):
        """Local diverges from remote → reset --hard, remote wins."""
        monkeypatch.setattr("pawlia.workspace_git.PULL_COOLDOWN", 0)
        with tempfile.TemporaryDirectory() as remote, tempfile.TemporaryDirectory() as ws:
            _git(remote, "init", "--bare")
            branch = _seed_remote(ws, remote)

            # Advance remote from a second clone
            with tempfile.TemporaryDirectory() as other_parent:
                other = os.path.join(other_parent, "other")
                _git(other_parent, "clone", "-q", remote, other)
                _git(other, "config", "user.name", "test")
                _git(other, "config", "user.email", "test@test")
                _write(other, "remote_change.md", "remote wins")
                _git(other, "add", "-A")
                _git(other, "commit", "-m", "remote change")
                _git(other, "push", "-q", "origin", branch)

            # Create a divergent local commit
            _write(ws, "local_change.md", "local loses")
            _git(ws, "add", "-A")
            _git(ws, "commit", "-m", "local change")
            local_head_before = _git(ws, "rev-parse", "HEAD").stdout.strip()

            assert pull(ws) is True

            # Remote state applied, local divergence discarded
            assert os.path.exists(os.path.join(ws, "remote_change.md"))
            assert not os.path.exists(os.path.join(ws, "local_change.md"))
            # HEAD moved away from the divergent local commit
            assert _git(ws, "rev-parse", "HEAD").stdout.strip() != local_head_before

    def test_pull_throttled(self):
        """Pull within cooldown window is skipped."""
        with tempfile.TemporaryDirectory() as remote, tempfile.TemporaryDirectory() as ws:
            _git(remote, "init", "--bare")
            _seed_remote(ws, remote)
            # A recent fetch creates FETCH_HEAD at ~now → cooldown active
            _git(ws, "fetch", "origin")
            assert pull(ws) is False
