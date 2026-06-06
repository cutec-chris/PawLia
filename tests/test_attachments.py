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
    # Two binaries + their markdown sidecars.
    assert files == {"report.pdf", "report-1.pdf", "report.pdf.md", "report-1.pdf.md"}


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


def test_sidecar_written_with_description(tmp_path):
    from pawlia import attachments

    user_id = "chris"
    meta = attachments.save_incoming(
        session_dir=str(tmp_path),
        user_id=user_id,
        data=b"\x89PNG fake",
        filename="cat.png",
        source="matrix",
        mimetype="image/png",
        description="Eine getigerte Katze auf einem Sofa.",
    )
    assert meta is not None
    assert meta.description == "Eine getigerte Katze auf einem Sofa."

    ddir = attachments.downloads_dir(str(tmp_path), user_id)
    sidecar = os.path.join(ddir, "cat.png.md")
    assert os.path.isfile(sidecar)
    content = open(sidecar, encoding="utf-8").read()
    # Sidecar lives in the workspace (searchable) and carries the description.
    assert "Eine getigerte Katze" in content
    assert 'mimetype: "image/png"' in content

    # Roundtrip: list_for_user reconstructs metadata + description from sidecars.
    entries = attachments.list_for_user(str(tmp_path), user_id)
    assert len(entries) == 1
    assert entries[0].saved_as == "cat.png"
    assert entries[0].description == "Eine getigerte Katze auf einem Sofa."


def test_save_incoming_without_description_uses_placeholder(tmp_path):
    from pawlia import attachments

    attachments.save_incoming(
        session_dir=str(tmp_path),
        user_id="x",
        data=b"data",
        filename="doc.bin",
        source="telegram",
    )
    ddir = attachments.downloads_dir(str(tmp_path), "x")
    content = open(os.path.join(ddir, "doc.bin.md"), encoding="utf-8").read()
    assert "keine Beschreibung" in content
    # Empty description roundtrips to "".
    assert attachments.list_for_user(str(tmp_path), "x")[0].description == ""
