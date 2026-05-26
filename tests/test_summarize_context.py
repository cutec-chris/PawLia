"""Tests for _FallbackLLMWrapper.summarize_context and _summarize_to_fit."""
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from pawlia.llm import _FallbackLLMWrapper


def _make_wrapper():
    llm = MagicMock()
    return _FallbackLLMWrapper(
        llms=[llm],
        labels=["test-model"],
        context_sizes=[4096],
    )


def _msgs(*pairs, system="You are a helper."):
    """Build a message list: system + alternating HumanMessage/AIMessage pairs."""
    result = [SystemMessage(content=system)]
    for human, ai in pairs:
        result.append(HumanMessage(content=human))
        result.append(AIMessage(content=ai))
    return result


# ── summarize_context ───────────────────────────────────────────────────

def test_returns_original_when_short_enough():
    w = _make_wrapper()
    msgs = _msgs(("hi", "hello"))
    # 1 system + 2 messages = 3, threshold = 2 + 1*2 = 4 → pass-through
    result = w.summarize_context(msgs, keep_recent=1)
    assert result == msgs


def test_keeps_system_message():
    w = _make_wrapper()
    msgs = _msgs(("a", "b"), ("c", "d"), ("e", "f"), ("g", "h"))
    result = w.summarize_context(msgs, keep_recent=1)
    assert result[0].content == "You are a helper."


def test_keeps_recent_pairs():
    w = _make_wrapper()
    msgs = _msgs(("a", "b"), ("c", "d"), ("e", "f"), ("g", "h"))
    result = w.summarize_context(msgs, keep_recent=2)
    # Last 2 pairs = 4 messages should appear verbatim
    assert any(m.content == "e" for m in result)
    assert any(m.content == "f" for m in result)
    assert any(m.content == "g" for m in result)
    assert any(m.content == "h" for m in result)


def test_old_messages_replaced_by_summary():
    w = _make_wrapper()
    msgs = _msgs(("first", "resp1"), ("second", "resp2"), ("third", "resp3"))
    result = w.summarize_context(msgs, keep_recent=1)
    contents = [m.content for m in result]
    assert "first" not in contents
    assert "second" not in contents
    assert any("condensed" in c for c in contents)


def test_keep_recent_zero_summarizes_everything():
    """Regression test for keep_recent=0 bug (messages[-0:] == messages)."""
    w = _make_wrapper()
    msgs = _msgs(("a", "b"), ("c", "d"), ("e", "f"))
    result = w.summarize_context(msgs, keep_recent=0)
    # With keep_recent=0 the result must be shorter than the original
    assert len(result) < len(msgs)
    # Should be: system + one summary HumanMessage
    assert len(result) == 2
    assert result[0] == msgs[0]  # system preserved
    assert "condensed" in result[1].content


def test_keep_recent_zero_does_not_return_original():
    """Ensure keep_recent=0 doesn't silently return the unmodified list."""
    w = _make_wrapper()
    msgs = _msgs(("a", "b"), ("c", "d"), ("e", "f"))
    result = w.summarize_context(msgs, keep_recent=0)
    assert result != msgs


def test_orphan_tool_message_stripped_from_tail():
    """A ToolMessage at the start of the tail must be dropped (strict API requirement)."""
    w = _make_wrapper()
    system = SystemMessage(content="sys")
    # Build messages manually: system + old pair + AI-with-tool-call + orphan ToolMessage
    ai_tool = AIMessage(content="", tool_calls=[{"id": "tc1", "name": "foo", "args": {}}])
    tool_msg = ToolMessage(content="result", tool_call_id="tc1")
    human_new = HumanMessage(content="new question")
    ai_new = AIMessage(content="new answer")
    msgs = [system,
            HumanMessage(content="old"), AIMessage(content="old answer"),
            HumanMessage(content="old2"), AIMessage(content="old2 answer"),
            ai_tool, tool_msg, human_new, ai_new]
    result = w.summarize_context(msgs, keep_recent=1)
    # tail starts after summary; ToolMessage should not be first message after summary
    assert not isinstance(result[1], ToolMessage)


def test_human_and_ai_previews_in_summary():
    w = _make_wrapper()
    msgs = _msgs(("Hello world", "Hi there"), ("More stuff", "More reply"), ("recent", "recent"))
    result = w.summarize_context(msgs, keep_recent=1)
    summary = next(m for m in result if "condensed" in m.content)
    assert "[User: Hello world]" in summary.content
    assert "[Assistant: Hi there]" in summary.content


# ── _summarize_to_fit ───────────────────────────────────────────────────

def test_summarize_to_fit_returns_messages_when_already_fit():
    w = _make_wrapper()
    msgs = _msgs(("hi", "hello"))
    # Patch _fits_context to always return True
    w._fits_context = lambda idx, m: True
    result = w._summarize_to_fit(0, msgs)
    assert result == msgs


def test_summarize_to_fit_returns_compressed_list():
    w = _make_wrapper()
    msgs = _msgs(("a", "b"), ("c", "d"), ("e", "f"), ("g", "h"), ("i", "j"))
    # Simulate context that only fits if summarized to <= 5 messages
    def _fits(idx, m):
        return len(m) <= 5
    w._fits_context = _fits
    result = w._summarize_to_fit(0, msgs)
    assert result is not None
    assert len(result) <= 5


def test_summarize_to_fit_returns_none_if_nothing_fits():
    w = _make_wrapper()
    msgs = _msgs(("a", "b"), ("c", "d"), ("e", "f"))
    w._fits_context = lambda idx, m: False
    result = w._summarize_to_fit(0, msgs)
    assert result is None
