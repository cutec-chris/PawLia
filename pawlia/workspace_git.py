"""Workspace Git — auto-commit, daily squash, weekly squash.

Keeps the workspace (Obsidian vault) in a Git repo for syncing.
Internal scheduler state and other non-vault files are excluded via .gitignore.

Commit throttle: at most one commit per COMMIT_COOLDOWN seconds.
Daily squash: all commits from today → one "Daily: YYYY-MM-DD" commit.
Weekly squash: all commits from this week → one "Week: YYYY-Www" commit.
"""

import logging
import os
import re
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


_INVALID_PATH_CHARS = re.compile(r'[<>:"|?*]')


def _scan_problematic_paths(workspace: str) -> list[str]:
    """Return all tracked or untracked paths with invalid filename characters."""
    bad: list[str] = []
    rc, out = _git(workspace, "ls-files", "--cached", "--others", "--modified",
                   quiet=True)
    if rc == 0 and out:
        for path in out.splitlines():
            if _INVALID_PATH_CHARS.search(path):
                bad.append(path)
    return bad


def _ensure_identity(workspace: str) -> None:
    """Ensure repo-local user.name/user.email are set so commits don't fail."""
    if not _config_get(workspace, "user.name"):
        _git(workspace, "config", "user.name", "pawlia")
    if not _config_get(workspace, "user.email"):
        _git(workspace, "config", "user.email", "pawlia@localhost")


_GITIGNORE_PATTERNS = [
    ".obsidian/workspace.json",
    ".obsidian/workspace-mobile.json",
    "memory/context_summary.md",
    "memory/private_session",
    "memory/private_thread_*",
    "memory/voice_override.txt",
    "wiki/log.md",
]


def _ensure_gitignore(workspace: str) -> None:
    """Ensure all required patterns are present in the workspace .gitignore."""
    gitignore = os.path.join(workspace, ".gitignore")
    existing = set()
    if os.path.exists(gitignore):
        with open(gitignore, encoding="utf-8") as f:
            existing = {line.rstrip("\n") for line in f}
    missing = [p for p in _GITIGNORE_PATTERNS if p not in existing]
    if missing:
        with open(gitignore, "a", encoding="utf-8") as f:
            for p in missing:
                f.write(p + "\n")
        for p in missing:
            _git(workspace, "rm", "--cached", "--ignore-unmatch", "-q", p)


def _ensure_protect_config(workspace: str) -> None:
    """Enable NTFS-protection and advise about problematic pathnames."""
    _git(workspace, "config", "core.protectNTFS", "true")
    bad = _scan_problematic_paths(workspace)
    if bad:
        logger.warning(
            "Workspace has %d file(s) with characters invalid on Android "
            "(<> : \" | ? *). These will fail on checkout. Renaming suggested: %s",
            len(bad), bad[:5],
        )


def ensure_repo(workspace: str) -> bool:
    """Initialize a git repo in workspace if not already one. Returns True if repo exists."""
    if os.path.isdir(os.path.join(workspace, ".git")):
        _ensure_identity(workspace)
        _ensure_gitignore(workspace)
        _ensure_protect_config(workspace)
        return True
    rc, _ = _git(workspace, "init")
    if rc != 0:
        return False
    _ensure_identity(workspace)
    _ensure_gitignore(workspace)
    _ensure_protect_config(workspace)
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

    # Refuse commit if files with invalid chars are staged
    bad = _scan_problematic_paths(workspace)
    if bad:
        logger.warning(
            "Skipping auto-commit: %d file(s) with invalid characters "
            "(<> : \" | ? *). Rename them first: %s",
            len(bad), bad[:5],
        )
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
    """Pull from remote — the remote always wins on conflict/divergence.

    Throttled to PULL_COOLDOWN. Strategy:
      1. fetch
      2. HEAD == remote/HEAD -> up to date
      3. merge --ff-only -> fast forward (keeps non-conflicting local history)
      4. otherwise -> reset --hard to remote/HEAD (remote wins)

    Callers should commit local changes before calling this so uncommitted
    edits are not destroyed by the hard reset (the scheduler does this).

    Returns True if the local HEAD now matches the remote.
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

    rc, remote_head = _git(workspace, "rev-parse", f"{remote}/HEAD", quiet=True)
    if rc != 0 or not remote_head:
        logger.warning("Pull from %s skipped: %s/HEAD not resolvable", remote, remote)
        return False

    rc, local_head = _git(workspace, "rev-parse", "HEAD", quiet=True)
    if rc == 0 and local_head == remote_head:
        logger.debug("Workspace already up to date with %s", remote)
        return True

    # Try fast-forward first (preserves non-conflicting local history)
    rc, _ = _git(workspace, "merge", "--ff-only", f"{remote}/HEAD", quiet=True)
    if rc == 0:
        logger.debug("Pulled workspace from %s (ff-only)", remote)
        return True

    # Divergent or blocked — remote wins, hard reset
    logger.warning(
        "Workspace diverged from %s — resetting hard to remote (remote wins)", remote,
    )
    rc, _ = _git(workspace, "reset", "--hard", f"{remote}/HEAD")
    if rc == 0:
        logger.info("Reset workspace to %s/HEAD (remote wins on conflict)", remote)
        return True
    logger.error("Hard reset to %s/HEAD failed", remote)
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
