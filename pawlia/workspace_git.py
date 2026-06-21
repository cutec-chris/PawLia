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
PULL_COOLDOWN = 300  # seconds — max 1 pull per 5 minutes


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
_CHAR_TRANS = str.maketrans({'"': "'", ':': '-', '<': '(', '>': ')', '|': '-',
                              '?': '', '*': ''})


def _git_unquote(raw: str) -> str:
    """Decode git's C-string path quoting (applied when paths contain non-ASCII or specials)."""
    if not (raw.startswith('"') and raw.endswith('"')):
        return raw
    raw = raw[1:-1]
    result = bytearray()
    i = 0
    while i < len(raw):
        c = raw[i]
        if c == '\\' and i + 1 < len(raw):
            n = raw[i + 1]
            if n == '\\':
                result.append(ord('\\'))
                i += 2
            elif n == '"':
                result.append(ord('"'))
                i += 2
            elif n == 'n':
                result.append(10)
                i += 2
            elif n == 't':
                result.append(9)
                i += 2
            elif '0' <= n <= '7' and i + 3 < len(raw):
                result.append(int(raw[i + 1:i + 4], 8))
                i += 4
            else:
                result.append(ord(c))
                i += 1
        else:
            result.append(ord(c))
            i += 1
    return result.decode('utf-8', errors='replace')


def _scan_problematic_paths(workspace: str) -> list[str]:
    """Return all tracked or untracked paths with invalid filename characters."""
    bad: list[str] = []
    rc, out = _git(workspace, "ls-files", "--cached", "--others", "--modified",
                   quiet=True)
    if rc == 0 and out:
        for raw in out.splitlines():
            path = _git_unquote(raw)
            if _INVALID_PATH_CHARS.search(path):
                bad.append(path)
    return bad


def _fix_path_component(name: str) -> str:
    fixed = name.translate(_CHAR_TRANS)
    fixed = re.sub(r'-{2,}', '-', fixed)
    return fixed or '_'


def _fix_problematic_paths(workspace: str) -> int:
    """Rename files with invalid characters in-place. Returns count of renamed files."""
    bad = _scan_problematic_paths(workspace)
    if not bad:
        return 0

    renamed = 0
    for rel_path in bad:
        parts = rel_path.replace('\\', '/').split('/')
        fixed_parts = [_fix_path_component(p) for p in parts]
        if fixed_parts == parts:
            continue
        fixed_rel = '/'.join(fixed_parts)

        src = os.path.join(workspace, rel_path)
        dst = os.path.join(workspace, fixed_rel)

        if not os.path.exists(src):
            continue

        os.makedirs(os.path.dirname(dst), exist_ok=True)

        if os.path.exists(dst) and os.path.abspath(src) != os.path.abspath(dst):
            base, ext = os.path.splitext(fixed_rel)
            fixed_rel = f"{base}_1{ext}"
            dst = os.path.join(workspace, fixed_rel)
            if os.path.exists(dst):
                logger.warning("Cannot rename %s: target already exists", rel_path)
                continue

        # git mv for tracked files; fall back to os.rename for untracked
        rc, _ = _git(workspace, "mv", "--", rel_path, fixed_rel, quiet=True)
        if rc == 0:
            logger.info("git mv (invalid chars): %s → %s", rel_path, fixed_rel)
            renamed += 1
        else:
            try:
                os.rename(src, dst)
                logger.info("Renamed (invalid chars): %s → %s", rel_path, fixed_rel)
                renamed += 1
            except OSError as e:
                logger.warning("Failed to rename %s: %s", rel_path, e)

    return renamed


def _ensure_identity(workspace: str) -> None:
    """Ensure repo-local user.name/user.email are set so commits don't fail."""
    if not _config_get(workspace, "user.name"):
        _git(workspace, "config", "user.name", "pawlia")
    if not _config_get(workspace, "user.email"):
        _git(workspace, "config", "user.email", "pawlia@localhost")


# Vault-internal files that must never sync (paths relative to workspace root).
_GITIGNORE_PATTERNS = [
    ".obsidian/workspace.json",
    ".obsidian/workspace-mobile.json",
    "memory/context_summary.md",
    "memory/private_session",
    "memory/private_thread_*",
    "memory/voice_override.txt",
    "wiki/log.md",
]

# Developer/runtime cruft a skill can drop anywhere in the tree (npm packages,
# Python bytecode, virtualenvs, tool caches). Gitignore matches these at any
# depth, so e.g. skills/<name>/node_modules is covered. Kept deliberately
# unambiguous so a real note/folder is never ignored by accident.
_GITIGNORE_CRUFT = [
    "node_modules/",
    "__pycache__/",
    "*.py[cod]",
    ".venv/",
    "venv/",
    "*.egg-info/",
    ".pytest_cache/",
    ".mypy_cache/",
    ".ruff_cache/",
    ".DS_Store",
]


def _ensure_gitignore(workspace: str) -> None:
    """Ensure all required ignore patterns are present, then untrack anything
    that is now ignored.

    The untrack step cleans up cruft a skill committed *before* the pattern
    existed (e.g. a node_modules tree shipped with a third-party skill) — it
    walks the index, not just the freshly added patterns, so nested paths at
    any depth are caught. It is a no-op when nothing ignored is tracked.
    """
    gitignore = os.path.join(workspace, ".gitignore")
    patterns = _GITIGNORE_PATTERNS + _GITIGNORE_CRUFT
    existing: list[str] = []
    if os.path.exists(gitignore):
        with open(gitignore, encoding="utf-8") as f:
            existing = [line.rstrip("\n") for line in f]
    existing_set = set(existing)
    missing = [p for p in patterns if p not in existing_set]
    if missing:
        with open(gitignore, "a", encoding="utf-8") as f:
            if existing and existing[-1] != "":
                f.write("\n")
            for p in missing:
                f.write(p + "\n")

    _untrack_ignored(workspace)


def _untrack_ignored(workspace: str) -> None:
    """Remove from the index every tracked file that current ignore rules match.

    Files stay on disk (``--cached``); they are dropped from version control on
    the next commit. Batched to keep the argument list bounded.
    """
    rc, out = _git(workspace, "ls-files", "-z", "--cached", "--ignored",
                   "--exclude-standard", quiet=True)
    if rc != 0 or not out:
        return
    paths = [p for p in out.split("\0") if p]
    if not paths:
        return
    for i in range(0, len(paths), 100):
        _git(workspace, "rm", "--cached", "--ignore-unmatch", "-q", "--",
             *paths[i:i + 100])
    logger.info("Untracked %d now-ignored file(s) in %s", len(paths), workspace)


def _ensure_protect_config(workspace: str) -> None:
    """Enable NTFS-protection and advise about problematic pathnames."""
    _git(workspace, "config", "core.protectNTFS", "true")
    bad = _scan_problematic_paths(workspace)
    if bad:
        logger.warning(
            "Workspace has %d file(s) with characters invalid on Android "
            "(<> : \" | ? *). Will auto-rename on next commit: %s",
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

    # Auto-rename files with invalid characters, then bail if any remain
    _fix_problematic_paths(workspace)
    bad = _scan_problematic_paths(workspace)
    if bad:
        logger.warning(
            "Skipping auto-commit: %d file(s) with invalid characters still "
            "could not be renamed: %s",
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
