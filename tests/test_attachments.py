"""Tests for pawlia.attachments — incoming file/image persistence."""

import json
import os


def test_save_incoming_creates_file_and_entry(tmp_path):
    from pawlia import attachments

    user_id = "alice"
    meta = attachments.save_incoming(
        session_dir=str(tmp_path),
        user_id=user_id,
        data=b"hello world",
        filename="note.txt",
        source="matrix",
        mimetype="text/plain",
    )
    assert meta is not None
    assert meta.saved_as == "note.txt"
    assert meta.original_name == "note.txt"
    assert meta.mimetype == "text/plain"
    assert meta.source == "matrix"
    assert meta.size == 11

    # Workspace + Downloads dir should be created on save
    assert os.path.isdir(attachments.workspace_dir(str(tmp_path), user_id))
    saved = os.path.join(attachments.downloads_dir(str(tmp_path), user_id), "note.txt")
    assert os.path.isfile(saved)
    with open(saved, "rb") as f:
        assert f.read() == b"hello world"

    entries = attachments.list_for_user(str(tmp_path), user_id)
    assert len(entries) == 1
    assert entries[0].saved_as == "note.txt"


def test_save_incoming_collision_appends_suffix(tmp_path):
    from pawlia import attachments

    user_id = "bob"
    for i in range(2):
        meta = attachments.save_incoming(
            session_dir=str(tmp_path),
            user_id=user_id,
            data=f"content-{i}".encode(),
            filename="report.pdf",
            source="telegram",
            mimetype="application/pdf",
        )
        assert meta is not None
    ddir = attachments.downloads_dir(str(tmp_path), user_id)
    files = set(os.listdir(ddir))
    assert files == {"report.pdf", "report-1.pdf"}


def test_save_incoming_rejects_empty(tmp_path):
    from pawlia import attachments

    meta = attachments.save_incoming(
        session_dir=str(tmp_path),
        user_id="x",
        data=b"",
        filename="empty",
        source="matrix",
    )
    assert meta is None


def test_save_incoming_rejects_oversize(tmp_path):
    from pawlia import attachments

    meta = attachments.save_incoming(
        session_dir=str(tmp_path),
        user_id="x",
        data=b"x" * 100,
        filename="big.bin",
        source="matrix",
        max_bytes=50,
    )
    assert meta is None


def test_save_incoming_sanitises_path_traversal_in_name(tmp_path):
    from pawlia import attachments

    meta = attachments.save_incoming(
        session_dir=str(tmp_path),
        user_id="x",
        data=b"data",
        filename="../../etc/passwd",
        source="matrix",
    )
    assert meta is not None
    assert ".." not in meta.saved_as
    # The saved file must live inside the Downloads dir, not elsewhere.
    saved = os.path.join(attachments.downloads_dir(str(tmp_path), "x"), meta.saved_as)
    assert os.path.isfile(saved)
    # Defensive: ensure no file was written outside the Downloads dir.
    leaked = tmp_path.parent / "etc" / "passwd"
    assert not leaked.exists()


def test_index_path_lives_outside_workspace(tmp_path):
    from pawlia import attachments

    idx = attachments.index_path(str(tmp_path), "chris")
    # Must be a sibling of workspace/, not inside it.
    workspace = attachments.workspace_dir(str(tmp_path), "chris")
    assert os.path.dirname(idx) != workspace
    assert idx.endswith("downloads_index.json")


def test_index_truncates_at_max(tmp_path):
    from pawlia import attachments

    user_id = "loop"
    for i in range(attachments._INDEX_MAX_ENTRIES + 10):
        attachments.save_incoming(
            session_dir=str(tmp_path),
            user_id=user_id,
            data=str(i).encode(),
            filename=f"f{i}.txt",
            source="matrix",
        )
    entries = attachments.list_for_user(str(tmp_path), user_id)
    assert len(entries) == attachments._INDEX_MAX_ENTRIES
    # Newest entries kept
    assert entries[-1].saved_as == f"f{attachments._INDEX_MAX_ENTRIES + 9}.txt"
