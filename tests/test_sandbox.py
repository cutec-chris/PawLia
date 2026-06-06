"""Tests for the filesystem write-sandbox helpers (pawlia.sandbox)."""

import os

from pawlia import sandbox


def test_writable_roots_with_user(tmp_path):
    session = str(tmp_path / "session")
    os.makedirs(os.path.join(session, "alice"))
    roots = sandbox.writable_roots(session, "alice")
    assert os.path.realpath(os.path.join(session, "alice")) in roots
    assert os.path.realpath("/tmp") in roots


def test_writable_roots_without_user(tmp_path):
    session = str(tmp_path / "session")
    os.makedirs(session)
    roots = sandbox.writable_roots(session, None)
    assert os.path.realpath(session) in roots
    assert os.path.realpath("/tmp") in roots


def test_writable_roots_dedup(tmp_path):
    # No session dir → just /tmp, once.
    assert sandbox.writable_roots(None, None) == [os.path.realpath("/tmp")]


def test_wrap_argv_structure(tmp_path):
    writable = str(tmp_path / "w")
    os.makedirs(writable)
    argv = sandbox.wrap_argv(["sh", "-c", "echo hi"], [writable])
    assert argv[0] == "bwrap"
    # Read-only root present.
    assert "--ro-bind" in argv
    # Writable dir bind-mounted read-write.
    assert "--bind" in argv
    assert writable in argv
    # The real command follows the -- separator, untouched.
    sep = argv.index("--")
    assert argv[sep + 1:] == ["sh", "-c", "echo hi"]


def test_wrap_argv_skips_missing_writable(tmp_path):
    missing = str(tmp_path / "does-not-exist")
    argv = sandbox.wrap_argv(["true"], [missing])
    assert missing not in argv


def test_diff_detects_stray_write(tmp_path):
    scan_root = tmp_path / "session"
    writable = scan_root / "alice"
    forbidden = scan_root  # session root, NOT under writable
    writable.mkdir(parents=True)

    before = sandbox.snapshot_mtimes([str(scan_root)], [str(writable)])

    # A write inside the writable root must NOT be flagged.
    (writable / "ok.txt").write_text("fine")
    # A write to the session root (outside writable) MUST be flagged.
    stray_file = forbidden / "radar.png"
    stray_file.write_text("oops")

    stray = sandbox.diff_stray_writes(before, [str(scan_root)], [str(writable)])
    assert str(stray_file) in stray
    assert str(writable / "ok.txt") not in stray


def test_diff_clean_run(tmp_path):
    scan_root = tmp_path / "session"
    writable = scan_root / "alice"
    writable.mkdir(parents=True)
    before = sandbox.snapshot_mtimes([str(scan_root)], [str(writable)])
    (writable / "scratch.txt").write_text("data")
    assert sandbox.diff_stray_writes(before, [str(scan_root)], [str(writable)]) == []
