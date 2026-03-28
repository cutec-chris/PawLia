from pawlia.interfaces.matrix import _resolve_thread_root


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