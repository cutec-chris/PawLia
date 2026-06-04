"""Pure, deterministic helpers — no LLM, no filesystem, no network.

Consolidates the small algorithmic contracts that used to live in their own
files: thinking-tag stripping, API-error classification, the tool-loop
iteration budget, context summarization, and tool-choice/workflow helpers.
These are spec by design (math and string handling), so they assert the
intended mapping directly.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from pawlia.agents.base import BaseAgent
from pawlia.agents.error_classifier import (
    ErrorCategory,
    classify_error,
    is_retryable,
    should_compact,
)
from pawlia.agents.iteration_budget import IterationBudget
from pawlia.llm import _FallbackLLMWrapper
from pawlia.skills.executor import WorkflowExecutor, _extract_tool_name, _is_tool_choice_error
from pawlia.skills.workflow_schema import BuildingBlock, Workflow


# ---------------------------------------------------------------------------
# Thinking-tag / leaked-token cleanup
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("raw,clean", [
    ("<think>internal</think>Hello", "Hello"),
    ("<thinking>deep thought</thinking>Result", "Result"),
    ("No thinking here", "No thinking here"),
])
def test_strip_thinking_removes_reasoning_blocks(raw, clean):
    assert BaseAgent.strip_thinking(raw) == clean


def test_extract_text_strips_thinking_from_a_message():
    msg = AIMessage(content="<think>hmm</think>The answer is 42")
    assert BaseAgent.extract_text(msg) == "The answer is 42"


# ---------------------------------------------------------------------------
# API-error classification
# ---------------------------------------------------------------------------
class _StatusExc(Exception):
    def __init__(self, msg, status_code):
        super().__init__(msg)
        self.status_code = status_code


@pytest.mark.parametrize("exc,expected", [
    (ValueError("context_length_exceeded"), ErrorCategory.context_overflow),
    (ValueError("messages parameter is illegal '1214'"), ErrorCategory.context_overflow),
    (_StatusExc("payload too large", 413), ErrorCategory.context_overflow),
    (_StatusExc("unauthorized", 401), ErrorCategory.auth_error),
    (_StatusExc("forbidden", 403), ErrorCategory.auth_error),
    (ValueError("invalid api key provided"), ErrorCategory.auth_error),
    (_StatusExc("too many requests", 429), ErrorCategory.rate_limit),
    (ValueError("rate limit exceeded, try again in 10s"), ErrorCategory.rate_limit),
    (_StatusExc("internal server error", 500), ErrorCategory.server_error),
    (_StatusExc("service unavailable", 503), ErrorCategory.server_error),
    (_StatusExc("bad request", 400), ErrorCategory.format_error),
    (RuntimeError("request timed out"), ErrorCategory.timeout),
    (TimeoutError("deadline exceeded"), ErrorCategory.timeout),
    (ValueError("something completely unexpected"), ErrorCategory.unknown),
])
def test_classify_error_maps_to_the_right_category(exc, expected):
    cat, _ = classify_error(exc)
    assert cat == expected


@pytest.mark.parametrize("cat,retryable", [
    (ErrorCategory.rate_limit, True),
    (ErrorCategory.timeout, True),
    (ErrorCategory.server_error, True),
    (ErrorCategory.unknown, True),
    (ErrorCategory.auth_error, False),
    (ErrorCategory.format_error, False),
    (ErrorCategory.context_overflow, False),
])
def test_is_retryable(cat, retryable):
    assert is_retryable(cat) is retryable


def test_only_context_overflow_triggers_compaction():
    assert should_compact(ErrorCategory.context_overflow) is True
    assert should_compact(ErrorCategory.rate_limit) is False
    assert should_compact(ErrorCategory.server_error) is False


# ---------------------------------------------------------------------------
# Iteration budget (tool-loop guard with one grace call)
# ---------------------------------------------------------------------------
def test_budget_allows_exactly_n_calls_plus_one_grace_then_denies():
    b = IterationBudget(2)
    assert b.consume() is True   # 1
    assert b.consume() is True   # 2
    assert b.consume() is True   # grace
    assert b.consume() is False  # denied
    assert b.consume() is False


def test_budget_tracks_used_and_remaining():
    b = IterationBudget(5)
    assert b.remaining == 5 and b.used == 0
    b.consume()
    b.consume()
    assert b.remaining == 3 and b.used == 2


def test_budget_refund_restores_a_slot_but_not_below_zero():
    b = IterationBudget(2)
    b.consume()
    b.consume()
    b.refund()
    assert b.consume() is True  # restored one
    b2 = IterationBudget(2)
    b2.refund()  # no-op
    assert b2.used == 0


def test_budget_remaining_never_negative():
    b = IterationBudget(1)
    b.consume()
    b.consume()  # grace
    b.consume()  # denied
    assert b.remaining == 0


# ---------------------------------------------------------------------------
# Context summarization (_FallbackLLMWrapper)
# ---------------------------------------------------------------------------
def _wrapper():
    from unittest.mock import MagicMock
    return _FallbackLLMWrapper(llms=[MagicMock()], labels=["m"], context_sizes=[4096])


def _msgs(*pairs, system="You are a helper."):
    out = [SystemMessage(content=system)]
    for human, ai in pairs:
        out += [HumanMessage(content=human), AIMessage(content=ai)]
    return out


def test_summarize_passes_short_histories_through_unchanged():
    w = _wrapper()
    msgs = _msgs(("hi", "hello"))
    assert w.summarize_context(msgs, keep_recent=1) == msgs


def test_summarize_keeps_system_and_recent_pairs_condensing_the_rest():
    w = _wrapper()
    msgs = _msgs(("first", "r1"), ("second", "r2"), ("third", "r3"))
    result = w.summarize_context(msgs, keep_recent=1)
    contents = [m.content for m in result]
    assert result[0].content == "You are a helper."          # system kept
    assert "first" not in contents and "second" not in contents  # old condensed
    assert any("condensed" in c for c in contents)
    assert any(m.content == "third" for m in result)          # recent kept


def test_summarize_keep_recent_zero_condenses_everything():
    w = _wrapper()
    msgs = _msgs(("a", "b"), ("c", "d"), ("e", "f"))
    result = w.summarize_context(msgs, keep_recent=0)
    assert result != msgs and len(result) == 2
    assert result[0] == msgs[0]
    assert "condensed" in result[1].content


def test_summarize_drops_orphan_tool_message_from_the_tail():
    w = _wrapper()
    ai_tool = AIMessage(content="", tool_calls=[{"id": "tc1", "name": "foo", "args": {}}])
    msgs = [
        SystemMessage(content="sys"),
        HumanMessage(content="old"), AIMessage(content="old answer"),
        HumanMessage(content="old2"), AIMessage(content="old2 answer"),
        ai_tool, ToolMessage(content="result", tool_call_id="tc1"),
        HumanMessage(content="new question"), AIMessage(content="new answer"),
    ]
    result = w.summarize_context(msgs, keep_recent=1)
    assert not isinstance(result[1], ToolMessage)


def test_summarize_to_fit_compresses_until_it_fits_or_gives_up():
    w = _wrapper()
    msgs = _msgs(("a", "b"), ("c", "d"), ("e", "f"), ("g", "h"), ("i", "j"))
    w._fits_context = lambda idx, m: len(m) <= 5
    fitted = w._summarize_to_fit(0, msgs)
    assert fitted is not None and len(fitted) <= 5

    w._fits_context = lambda idx, m: False
    assert w._summarize_to_fit(0, msgs) is None


# ---------------------------------------------------------------------------
# Tool-choice error detection + workflow config injection
# ---------------------------------------------------------------------------
def test_base_agent_extracts_failed_tool_name_from_api_error():
    failed_gen = '{"name": "searxng", "arguments": {"query": "test"}}'
    err = (
        "Error code: 400 - {'error': {'message': 'Tool choice is none, but model "
        "called a tool', 'code': 'tool_use_failed', 'failed_generation': '"
        + failed_gen + "'}}"
    )
    assert BaseAgent._extract_failed_tool_call(err) == "searxng"
    assert BaseAgent._extract_failed_tool_call("some other error") is None


@pytest.mark.parametrize("exc,is_err", [
    (Exception("tool_use_failed in generation"), True),
    (Exception("Tool choice is none, but model called a tool"), True),
    (Exception("Connection refused"), False),
])
def test_is_tool_choice_error(exc, is_err):
    assert _is_tool_choice_error(exc) is is_err


def test_extract_tool_name_from_failed_generation():
    assert _extract_tool_name('{"name": "files", "arguments": {}}') == "files"
    assert _extract_tool_name("no tool here") == ""


def test_workflow_config_placeholders_are_not_exposed_as_model_parameters():
    executor = WorkflowExecutor(
        tool_registry=None,
        context={"skill_config": {"url": "http://example.test", "timeout": 15}},
        llm=None,
    )
    workflow = Workflow(id="search", trigger="search", max_steps=1, building_blocks=[
        BuildingBlock(
            id="run", description="Run",
            command='python {scripts_dir}/s.py --query "{query}" --url "{url}" --timeout {timeout}',
        )])
    params = executor._blocks_to_tools(workflow)[0]["function"]["parameters"]
    assert params["required"] == ["query"]
    assert "query" in params["properties"]
    assert "url" not in params["properties"] and "timeout" not in params["properties"]


def test_workflow_substitute_fills_config_and_quotes_model_params():
    executor = WorkflowExecutor(
        tool_registry=None,
        context={"cwd": "/tmp/skill", "skill_config": {"url": "http://example.test"}},
        llm=None,
    )
    command = executor._substitute(
        'python {scripts_dir}/s.py --query "{query}" --url "{url}"',
        {"query": "hello"},
    )
    assert "/tmp/skill/scripts/s.py" in command.replace("\\", "/")
    assert "--query hello" in command               # model param, shell-quoted
    assert '--url "http://example.test"' in command  # config value, verbatim


# ---------------------------------------------------------------------------
# ChatAgent max-tool-turns warning — audit problem #17
# ---------------------------------------------------------------------------
from pawlia.agents.chat import ChatAgent


def test_max_tool_turns_warning_summarises_tool_sequence():
    """The budget-exhausted warning must include the tool name counts and
    a tail of recent calls so the cause is obvious in the log, not
    just 'Max chat tool turns reached'."""
    calls = [
        {"name": "list-tasks", "args": {}},
        {"name": "bash", "args": {"command": "ls"}},
        {"name": "list-tasks", "args": {}},
        {"name": "delete-task", "args": {"task-id": "abc"}},
        {"name": "list-tasks", "args": {}},
    ]
    msg = ChatAgent._format_max_tool_turns_warning(20, calls)

    assert "20" in msg
    assert "5 tool calls" in msg
    assert "'list-tasks': 3" in msg
    assert "'bash': 1" in msg
    assert "'delete-task': 1" in msg
    # The last six (in this case all 5) calls appear in tail form.
    assert "list-tasks, bash, list-tasks, delete-task, list-tasks" in msg


def test_max_tool_turns_warning_handles_empty_sequence():
    msg = ChatAgent._format_max_tool_turns_warning(20, [])
    assert "0 tool calls" in msg
    assert "Tail: (none)" in msg


def test_max_tool_turns_warning_truncates_tail_to_last_six():
    """A 50-call loop should still produce a one-line tail."""
    calls = [{"name": f"tool{i}"} for i in range(50)]
    msg = ChatAgent._format_max_tool_turns_warning(20, calls)

    assert "50 tool calls" in msg
    # Tail shows the last 6 names only.
    assert "tool44, tool45, tool46, tool47, tool48, tool49" in msg
