"""Filesystem write-sandbox for skill execution.

Skills run arbitrary scripts through the bash tool. Policy: a skill may only
create or modify files under the current user's session directory
(``session/<user_id>/`` — which holds the workspace, Downloads, credentials and
memory) or under ``/tmp`` for throwaway scratch. Writes anywhere else
(the ``session/`` root, ``/app``, ``$HOME``, ``/`` …) are forbidden.

Enforcement is layered:

* **Runtime** — the bash tool wraps each command in ``bwrap`` (bubblewrap) with
  a read-only root and only the writable roots bind-mounted read-write. The
  kernel then rejects out-of-bounds writes with ``EROFS``/``EACCES``.
* **Test time** — ``skill-creator``'s ``creator.py test`` runs the harness the
  same way and additionally scans for stray files, so a violation fails the
  smoke test at the latest.

Bubblewrap is a tiny, daemon-less helper (the sandbox primitive Flatpak uses)
but it needs *unprivileged user namespaces*, which some container runtimes
disable. :func:`bwrap_available` probes for this once; callers degrade
gracefully (run unsandboxed, log a warning) when it is missing.
"""

import functools
import logging
import os
import subprocess
import sys
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def bwrap_available() -> bool:
    """True if ``bwrap`` exists and can actually create a sandbox here.

    The result is cached for the process lifetime — the answer cannot change
    without a restart.
    """
    if sys.platform != "linux":
        return False
    try:
        proc = subprocess.run(
            ["bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "true"],
            capture_output=True, timeout=10, check=False,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return False
    if proc.returncode != 0:
        logger.info(
            "bubblewrap present but cannot create a sandbox (likely no "
            "unprivileged user namespaces): %s",
            (proc.stderr or b"").decode("utf-8", "replace").strip()[:200],
        )
        return False
    return True


def writable_roots(session_dir: Optional[str], user_id: Optional[str]) -> List[str]:
    """Directories a skill may write to: the per-user session dir + ``/tmp``.

    Falls back to the whole session dir when no user id is known. Returns
    realpaths, de-duplicated, order preserved.
    """
    roots: List[str] = []
    if session_dir:
        base = os.path.join(session_dir, user_id) if user_id else session_dir
        roots.append(os.path.realpath(base))
    roots.append(os.path.realpath("/tmp"))

    seen = set()
    out: List[str] = []
    for r in roots:
        if r and r not in seen:
            seen.add(r)
            out.append(r)
    return out


def wrap_argv(argv: Sequence[str], writable: Sequence[str]) -> List[str]:
    """Prefix ``argv`` with a ``bwrap`` invocation.

    The root filesystem is bind-mounted read-only; each existing path in
    ``writable`` is then bind-mounted read-write on top of it. Network and the
    inherited environment are preserved (no ``--unshare-net``, no env clearing).
    """
    cmd: List[str] = [
        "bwrap",
        "--ro-bind", "/", "/",
        "--dev", "/dev",
        "--proc", "/proc",
    ]
    for path in writable:
        # Skip non-existent sources — bwrap aborts on a missing bind mount.
        if os.path.isdir(path):
            cmd += ["--bind", path, path]
    cmd.append("--")
    cmd.extend(argv)
    return cmd


def snapshot_mtimes(scan_roots: Sequence[str], skip: Sequence[str]) -> Dict[str, float]:
    """Map every regular file under ``scan_roots`` to its mtime.

    Subtrees under any path in ``skip`` (the writable roots) are pruned, so the
    snapshot only covers locations a skill must NOT touch. Used by the test
    harness as a no-dependency backstop when bubblewrap is unavailable.
    """
    skip_real = [os.path.realpath(p) for p in skip]

    def _skipped(path: str) -> bool:
        return any(path == s or path.startswith(s + os.sep) for s in skip_real)

    snap: Dict[str, float] = {}
    for root in scan_roots:
        real_root = os.path.realpath(root)
        if not os.path.isdir(real_root) or _skipped(real_root):
            continue
        for dirpath, dirs, files in os.walk(real_root):
            if _skipped(dirpath):
                dirs[:] = []
                continue
            dirs[:] = [d for d in dirs if not _skipped(os.path.join(dirpath, d))]
            for name in files:
                p = os.path.join(dirpath, name)
                try:
                    snap[p] = os.path.getmtime(p)
                except OSError:
                    pass
    return snap


def diff_stray_writes(
    before: Dict[str, float], scan_roots: Sequence[str], skip: Sequence[str]
) -> List[str]:
    """Return files outside ``skip`` that are new or modified since ``before``."""
    after = snapshot_mtimes(scan_roots, skip)
    stray: List[str] = []
    for path, mtime in after.items():
        prev = before.get(path)
        if prev is None or mtime != prev:
            stray.append(path)
    return sorted(stray)
