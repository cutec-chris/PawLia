from pawlia.interfaces.matrix import (
    _add_mentions,
    _decode_matrix_file_text,
    _is_markitdown_matrix_file,
    _is_text_matrix_file,
    _make_content,
    _resolve_thread_root,
)


def test_add_mentions_pings_users():
    content = _add_mentions(
        _make_content("🔔 Reminder"),
        [("@chris:example.org", "Chris")],
    )
    assert content["m.mentions"] == {"user_ids": ["@chris:example.org"]}
    assert "https://matrix.to/#/@chris:example.org" in content["formatted_body"]
    assert content["body"].startswith("Chris:")
    assert "🔔 Reminder" in content["body"]


def test_add_mentions_noop_without_members():
    base = _make_content("🔔 Reminder")
    content = _add_mentions(dict(base), [])
    assert "m.mentions" not in content
    assert content["body"] == base["body"]


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


# ── //stop: cancel running turns ────────────────────────────────────────────

import asyncio  # noqa: E402

from pawlia.interfaces.matrix import _cancel_pending_tasks  # noqa: E402


def test_cancel_pending_tasks_cancels_live_and_skips_self_and_done():
    async def _run():
        async def _never():
            await asyncio.Event().wait()

        running = asyncio.create_task(_never())
        another = asyncio.create_task(_never())
        done = asyncio.create_task(asyncio.sleep(0))
        await done  # ensure it has finished
        me = asyncio.current_task()

        n = _cancel_pending_tasks([running, another, done, me], exclude=me)

        assert n == 2                 # running + another
        assert not me.cancelled()     # the //stop task itself survives
        # done task was already finished → not counted, not cancelled
        assert done.done() and not done.cancelled()
        # cancellation resolves once the tasks are awaited
        for t in (running, another):
            try:
                await t
                assert False, "expected CancelledError"
            except asyncio.CancelledError:
                pass
            assert t.cancelled()

    asyncio.run(_run())


def test_cancel_pending_tasks_empty_returns_zero():
    assert _cancel_pending_tasks([], exclude=None) == 0
