"""Context-window summarization in ``pawlia.llm._FallbackLLMWrapper``.

The 6.6. incident: a long tool loop bloated the context until the primary model
was skipped on every call ("context window too small even after progressive
summarization"), and the loop ground on forever. Root cause: the mechanical
condenser kept one truncated line per old message, so its output grew linearly
with the message count and could never be made small enough.

These tests pin the fix: the condensed block is bounded regardless of how many
messages it started from, a real LLM summary collapses the middle to a few
lines, and the overflow flag the skill runner relies on is set/reset correctly.
"""

from typing import Any, List

import pytest

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from pawlia.llm import _FallbackLLMWrapper, _MECHANICAL_SUMMARY_CHAR_BUDGET


class _DummyLLM:
    def __init__(self, model_name: str = "m"):
        self.model_name = model_name
        self.calls = 0

    def invoke(self, messages: List[Any], **kwargs: Any) -> str:
        self.calls += 1
        return f"ok-{self.model_name}"

    async def ainvoke(self, messages: List[Any], **kwargs: Any) -> str:
        return self.invoke(messages, **kwargs)

    def bind_tools(self, *args: Any, **kwargs: Any) -> "_DummyLLM":
        return self


class _FakeSummaryLLM:
    """Stands in for the distillation model: returns a fixed summary text."""

    def __init__(self, text: str):
        self.text = text
        self.calls = 0

    def invoke(self, messages: List[Any], **kwargs: Any) -> AIMessage:
        self.calls += 1
        return AIMessage(content=self.text)

    async def ainvoke(self, messages: List[Any], **kwargs: Any) -> AIMessage:
        return self.invoke(messages, **kwargs)


def _tool_loop_messages(pairs: int, result_chars: int = 500) -> List[BaseMessage]:
    """A system message followed by *pairs* AI(tool_call)/ToolMessage pairs —
    the shape a runaway skill-runner tool loop produces."""
    msgs: List[BaseMessage] = [SystemMessage(content="system prompt")]
    for i in range(pairs):
        ai = AIMessage(content="")
        ai.tool_calls = [{
            "id": f"c{i}", "name": "bash",
            "args": {"command": f"cat /app/skills/weather-radar/SKILL.md  # {i}"},
            "type": "tool_call",
        }]
        msgs.append(ai)
        msgs.append(ToolMessage(content="X" * result_chars, tool_call_id=f"c{i}"))
    return msgs


# ---------------------------------------------------------------------------
# Mechanical condenser is bounded (the actual bug)
# ---------------------------------------------------------------------------
def test_mechanical_summary_is_bounded_not_linear_in_message_count():
    w = _FallbackLLMWrapper([_DummyLLM()], ["m"], [200_000])

    mid = w.summarize_context(_tool_loop_messages(400), keep_recent=1)[1].content
    big = w.summarize_context(_tool_loop_messages(4000), keep_recent=1)[1].content

    # 10x the messages → essentially the same condensed size, both under budget.
    assert len(mid) <= _MECHANICAL_SUMMARY_CHAR_BUDGET + 500
    assert len(big) <= _MECHANICAL_SUMMARY_CHAR_BUDGET + 500
    assert abs(len(big) - len(mid)) <= 500
    # The overflowing cases must note that older lines were dropped.
    assert "dropped" in big and "dropped" in mid


def test_summarize_to_fit_shrinks_long_loop_into_small_window():
    msgs = _tool_loop_messages(125)  # ~35k tokens of raw transcript
    w = _FallbackLLMWrapper([_DummyLLM()], ["m"], [8000])

    assert not w._fits_context(0, msgs)
    fitted = w._summarize_to_fit(0, msgs)

    assert fitted is not None
    assert w._fits_context(0, fitted)


# ---------------------------------------------------------------------------
# Real LLM distillation
# ---------------------------------------------------------------------------
def test_llm_distillation_collapses_middle_to_summary():
    msgs = _tool_loop_messages(125)
    summary = "- Ziel: weather-radar reparieren\n- Befund: Bild wird nicht gespeichert"
    fake = _FakeSummaryLLM(summary)
    w = _FallbackLLMWrapper([_DummyLLM()], ["m"], [8000], summary_llm=fake)

    fitted = w._summarize_to_fit_sync(0, msgs)

    assert fake.calls == 1
    assert fitted is not None and w._fits_context(0, fitted)
    assert fitted[0] is msgs[0]                       # system kept verbatim
    assert fitted[1].content.startswith("[Earlier conversation summarized]")
    assert "weather-radar reparieren" in fitted[1].content
    # The whole middle became one summary message; only the recent tail remains.
    assert len(fitted) <= 2 + w._DISTILL_KEEP_RECENT * 2


def test_llm_distillation_falls_back_to_mechanical_when_summary_empty():
    msgs = _tool_loop_messages(125)
    fake = _FakeSummaryLLM("")  # model returned nothing usable
    w = _FallbackLLMWrapper([_DummyLLM()], ["m"], [8000], summary_llm=fake)

    fitted = w._summarize_to_fit_sync(0, msgs)

    assert fake.calls == 1
    assert fitted is not None and w._fits_context(0, fitted)
    # Mechanical fallback header (distinct from the LLM-summary header).
    assert "messages condensed" in fitted[1].content


# ---------------------------------------------------------------------------
# Overflow flag the skill runner reads
# ---------------------------------------------------------------------------
def test_last_invoke_context_skipped_set_when_nothing_fits():
    # Tiny window + a single un-summarizable message → no fallback can fit.
    w = _FallbackLLMWrapper([_DummyLLM()], ["m"], [10])

    with pytest.raises(Exception):
        w.invoke([HumanMessage(content="x" * 4000)])

    assert w.last_invoke_context_skipped is True


def test_last_invoke_context_skipped_resets_on_fitting_call():
    w = _FallbackLLMWrapper([_DummyLLM()], ["m"], [200_000])

    w.invoke([HumanMessage(content="hi")])

    assert w.last_invoke_context_skipped is False
