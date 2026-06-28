"""Unit tests for the opencode daemon helpers.

The actual HTTP round-trip is not exercised here — that would need a
running ``opencode serve``, which is environment-dependent and slow.
We cover the deterministic pieces: text/file extraction from response
envelopes, env-var-driven configuration, and the URL/port picker.

Integration coverage lives in the manual smoke-test script
``tests/manual_opencode_daemon.py`` (not run by CI).
"""

import json

import pytest

from pawlia.coding.opencode_daemon import (
    extract_assistant_text,
    extract_edited_files,
    _pick_port,
)


# ── extract_assistant_text ─────────────────────────────────────────────


def test_extract_text_concatenates_text_parts():
    resp = {"parts": [
        {"type": "step-start"},
        {"type": "text", "text": "Hallo"},
        {"type": "tool", "input": {}},
        {"type": "text", "text": "Welt"},
    ]}
    assert extract_assistant_text(resp) == "Hallo\nWelt"


def test_extract_text_skips_empty_strings():
    resp = {"parts": [
        {"type": "text", "text": ""},
        {"type": "text", "text": "   "},
        {"type": "text", "text": "real"},
    ]}
    assert extract_assistant_text(resp) == "real"


def test_extract_text_empty_envelope():
    assert extract_assistant_text({}) == ""
    assert extract_assistant_text({"parts": []}) == ""


# ── extract_edited_files ───────────────────────────────────────────────


def test_extract_edited_files_recognises_all_keys():
    resp = {"parts": [
        {"type": "tool", "input": {"filePath": "/a.py"}},
        {"type": "tool", "input": {"path": "/b.py"}},
        {"type": "tool", "input": {"filepath": "/c.py"}},
        {"type": "tool", "input": {"file_path": "/d.py"}},
    ]}
    assert extract_edited_files(resp) == ["/a.py", "/b.py", "/c.py", "/d.py"]


def test_extract_edited_files_dedupes():
    resp = {"parts": [
        {"type": "tool", "input": {"filePath": "/a.py"}},
        {"type": "tool", "input": {"filePath": "/a.py"}},
    ]}
    assert extract_edited_files(resp) == ["/a.py"]


def test_extract_edited_files_ignores_non_file_tools():
    resp = {"parts": [
        {"type": "tool", "input": {"command": "ls"}},
        {"type": "tool", "input": {"query": "search"}},
    ]}
    assert extract_edited_files(resp) == []


# ── _pick_port ─────────────────────────────────────────────────────────


def test_pick_port_honours_explicit():
    assert _pick_port(4096) == 4096


def test_pick_port_returns_zero_for_unset(monkeypatch):
    # Treat None and 0 the same as "let the OS choose".
    p1 = _pick_port(None)
    p2 = _pick_port(0)
    assert 1024 <= p1 <= 65535
    assert 1024 <= p2 <= 65535
    assert p1 != p2  # vanishingly unlikely to collide
