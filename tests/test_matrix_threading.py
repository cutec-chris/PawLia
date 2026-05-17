from pawlia.interfaces.matrix import (
    _decode_matrix_file_text,
    _is_markitdown_matrix_file,
    _is_text_matrix_file,
    _resolve_thread_root,
)


def test_resolve_thread_root_from_m_thread_relation():
    source = {
        "content": {
            "m.relates_to": {
                "rel_type": "m.thread",
                "event_id": "$thread-root",
                "m.in_reply_to": {"event_id": "$last-reply"},
            }
        }
    }

    assert _resolve_thread_root(source, {}) == "$thread-root"


def test_resolve_thread_root_from_known_reply_target():
    source = {
        "content": {
            "m.relates_to": {
                "m.in_reply_to": {"event_id": "$bot-thread-message"},
            }
        }
    }

    known_thread_events = {"$thread-root": "$thread-root", "$bot-thread-message": "$thread-root"}

    assert _resolve_thread_root(source, known_thread_events) == "$thread-root"


def test_resolve_thread_root_returns_none_without_thread_context():
    source = {
        "content": {
            "m.relates_to": {
                "m.in_reply_to": {"event_id": "$plain-reply"},
            }
        }
    }

    assert _resolve_thread_root(source, {}) is None


def test_resolve_thread_root_handles_malformed_payloads():
    assert _resolve_thread_root(None, {}) is None
    assert _resolve_thread_root({"content": None}, {}) is None
    assert _resolve_thread_root({"content": {"m.relates_to": None}}, {}) is None
    assert _resolve_thread_root(
        {"content": {"m.relates_to": {"m.in_reply_to": None}}},
        {},
    ) is None


def test_matrix_file_text_detection_accepts_ics_even_with_generic_mime():
    assert _is_text_matrix_file("termin.ics", "application/octet-stream") is True
    assert _is_text_matrix_file("termin.dat", "text/calendar") is True
    assert _is_text_matrix_file("image.png", "image/png") is False


def test_matrix_file_markitdown_detection_accepts_common_office_formats():
    assert _is_markitdown_matrix_file("angebot.pdf", "application/octet-stream") is True
    assert _is_markitdown_matrix_file("tabelle.xlsx", "application/octet-stream") is True
    assert _is_markitdown_matrix_file(
        "dokument.bin",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ) is True
    assert _is_markitdown_matrix_file("image.png", "image/png") is False


def test_decode_matrix_file_text_handles_utf8_bom():
    text, truncated = _decode_matrix_file_text(
        b"\xef\xbb\xbfBEGIN:VCALENDAR\nSUMMARY:Test\nEND:VCALENDAR\n"
    )

    assert text.startswith("BEGIN:VCALENDAR")
    assert "SUMMARY:Test" in text
    assert truncated is False
