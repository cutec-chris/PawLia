"""Workspace Git — auto-commit, daily squash, weekly squash.

Keeps the workspace (Obsidian vault) in a Git repo for syncing.
Internal scheduler state and other non-vault files are excluded via .gitignore.

Commit throttle: at most one commit per COMMIT_COOLDOWN seconds.
Daily squash: all commits from today → one "Daily: YYYY-MM-DD" commit.
Weekly squash: all commits from this week → one "Week: YYYY-Www" commit.
"""

import logging
import os
import subprocess
import time
from datetime import datetime

logger = logging.getLogger("pawlia.workspace_git")

COMMIT_COOLDOWN = 300  # seconds — max 1 commit per 5 minutes
PULL_COOLDOWN = 3600  # seconds — max 1 pull per hour


def _run(cmd: list[str], cwd: str, quiet: bool = False) -> tuple[int, str]:
    """Run a git command, return (returncode, stdout).

    quiet=True: don't log on non-zero rc (used for probe-style calls where
    a non-zero rc is an expected outcome, not an error).
    """
    try:
        r = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=30,
        )
        if r.returncode != 0 and not quiet:
            logger.warning("git %s failed (rc=%d) in %s: %s",
                           " ".join(cmd[1:]), r.returncode, cwd,
                           (r.stderr or r.stdout).strip())
        return r.returncode, r.stdout.strip()
    except Exception as e:
        logger.error("git command failed: %s: %s", cmd, e)
        return 1, ""


def _git(cwd: str, *args: str, quiet: bool = False) -> tuple[int, str]:
    return _run(["git", *args], cwd, quiet=quiet)


def _config_get(workspace: str, key: str) -> str:
    """Quietly read a git config value (empty string if unset — no warning)."""
    try:
        r = subprocess.run(
            ["git", "config", "--get", key],
            cwd=workspace, capture_output=True, text=True, timeout=10,
        )
        return r.stdout.strip()
    except Exception:
        return ""


def _ensure_identity(workspace: str) -> None:
    """Ensure repo-local user.name/user.email are set so commits don't fail."""
    if not _config_get(workspace, "user.name"):
        _git(workspace, "config", "user.name", "pawlia")
    if not _config_get(workspace, "user.email"):
        _git(workspace, "config", "user.email", "pawlia@localhost")


def ensure_repo(workspace: str) -> bool:
    """Initialize a git repo in workspace if not already one. Returns True if repo exists."""
    if os.path.isdir(os.path.join(workspace, ".git")):
        _ensure_identity(workspace)
        return True
    rc, _ = _git(workspace, "init")
    if rc != 0:
        return False
    _ensure_identity(workspace)
    # Write .gitignore for internal files
    gitignore = os.path.join(workspace, ".gitignore")
    if not os.path.exists(gitignore):
        with open(gitignore, "w") as f:
            f.write(".obsidian/workspace.json\n.obsidian/workspace-mobile.json\n")
    _git(workspace, "add", "-A")
    _git(workspace, "commit", "-m", "Initial commit")
    logger.info("Initialized git repo in %s", workspace)
    return True


def auto_commit(workspace: str) -> bool:
    """Commit all changes if there are any and cooldown has passed.

    Returns True if a commit was made.
    """
    if not os.path.isdir(os.path.join(workspace, ".git")):
        return False

    # Check cooldown: look at last commit timestamp (quiet — empty repo returns rc=128)
    rc, ts = _git(workspace, "log", "-1", "--format=%ct", quiet=True)
    if rc == 0 and ts:
        try:
            last_commit = int(ts)
            if time.time() - last_commit < COMMIT_COOLDOWN:
                return False
        except ValueError:
            pass

    # Check for changes
    rc, status = _git(workspace, "status", "--porcelain")
    if rc != 0 or not status:
        return False

    _git(workspace, "add", "-A")
    now = datetime.now()
    msg = now.strftime("%Y-%m-%d %H:%M")
    rc, _ = _git(workspace, "commit", "-m", msg)
    if rc == 0:
        logger.debug("Auto-committed workspace changes: %s", msg)
        return True
    return False


def daily_squash(workspace: str) -> bool:
    """Squash all of today's commits into one 'Daily: YYYY-MM-DD' commit.

    Returns True if squash was performed.
    """
    if not os.path.isdir(os.path.join(workspace, ".git")):
        return False

    today = datetime.now().strftime("%Y-%m-%d")

    # Count today's commits
    rc, out = _git(workspace, "log", "--oneline", f"--since={today} 00:00", f"--until={today} 23:59:59")
    if rc != 0 or not out:
        return False

    commits = [l for l in out.split("\n") if l.strip()]
    if len(commits) <= 1:
        return False

    # Find the parent of the first commit today
    rc, first_hash = _git(workspace, "log", "--reverse", "--format=%H", f"--since={today} 00:00")
    if rc != 0 or not first_hash:
        return False
    first = first_hash.split("\n")[0].strip()

    # Get parent of first today commit
    rc, parent = _git(workspace, "rev-parse", f"{first}~1")
    if rc != 0:
        # First commit ever is from today — reset to root
        rc, parent = _git(workspace, "rev-list", "--max-parents=0", "HEAD")
        if rc != 0 or not parent:
            return False
        parent = parent.split("\n")[0].strip()
        # For root case, we need a different approach
        _git(workspace, "reset", "--soft", parent)
        _git(workspace, "commit", "--amend", "-m", f"Daily: {today}")
        logger.info("Daily squash (from root): %d commits → 1 (%s)", len(commits), today)
        return True

    parent = parent.strip()
    _git(workspace, "reset", "--soft", parent)
    _git(workspace, "commit", "-m", f"Daily: {today}")
    logger.info("Daily squash: %d commits → 1 (%s)", len(commits), today)
    return True


def weekly_squash(workspace: str) -> bool:
    """Squash all of this week's commits into one 'Week: YYYY-Www' commit.

    Returns True if squash was performed.
    """
    if not os.path.isdir(os.path.join(workspace, ".git")):
        return False

    now = datetime.now()
    year, week, _ = now.isocalendar()
    week_label = f"{year}-W{week:02d}"

    # Find Monday of this week
    from datetime import timedelta
    monday = now - timedelta(days=now.weekday())
    since = monday.strftime("%Y-%m-%d 00:00")

    rc, out = _git(workspace, "log", "--oneline", f"--since={since}")
    if rc != 0 or not out:
        return False

    commits = [l for l in out.split("\n") if l.strip()]
    if len(commits) <= 1:
        return False

    # Find parent of first commit this week
    rc, first_hash = _git(workspace, "log", "--reverse", "--format=%H", f"--since={since}")
    if rc != 0 or not first_hash:
        return False
    first = first_hash.split("\n")[0].strip()

    rc, parent = _git(workspace, "rev-parse", f"{first}~1")
    if rc != 0:
        return False

    parent = parent.strip()
    _git(workspace, "reset", "--soft", parent)
    _git(workspace, "commit", "-m", f"Week: {week_label}")
    logger.info("Weekly squash: %d commits → 1 (%s)", len(commits), week_label)
    return True


def pull(workspace: str, remote: str = "origin") -> bool:
    """Fast-forward pull from remote if configured. Throttled to PULL_COOLDOWN.

    Returns True if a pull was attempted and succeeded (including no-op fast-forward).
    Non-fast-forward divergence is logged as a warning; no merge is created.
    """
    if not os.path.isdir(os.path.join(workspace, ".git")):
        return False
    rc, _ = _git(workspace, "remote", "get-url", remote, quiet=True)
    if rc != 0:
        return False

    # Throttle via FETCH_HEAD mtime
    fetch_head = os.path.join(workspace, ".git", "FETCH_HEAD")
    if os.path.exists(fetch_head):
        if time.time() - os.path.getmtime(fetch_head) < PULL_COOLDOWN:
            return False

    rc, _ = _git(workspace, "fetch", remote)
    if rc != 0:
        return False
    rc, _ = _git(workspace, "merge", "--ff-only", f"{remote}/HEAD")
    if rc == 0:
        logger.debug("Pulled workspace from %s (ff-only)", remote)
        return True
    logger.warning("Pull from %s skipped: local diverges from remote (no ff-only possible)", remote)
    return False


def push(workspace: str, remote: str = "origin") -> bool:
    """Push to remote if configured. Returns True on success."""
    if not os.path.isdir(os.path.join(workspace, ".git")):
        return False
    # Check if remote exists (quiet — absent remote is an expected no-op)
    rc, _ = _git(workspace, "remote", "get-url", remote, quiet=True)
    if rc != 0:
        return False
    rc, _ = _git(workspace, "push", "--force-with-lease", remote)
    if rc == 0:
        logger.info("Pushed workspace to %s", remote)
        return True
    logger.warning("Push to %s failed", remote)
    return False
