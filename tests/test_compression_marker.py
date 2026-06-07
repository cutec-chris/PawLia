"""Tests for the compression-marker mechanism in ChatAgent.

When ``_prepare_messages_for_context_budget`` compresses older messages,
it injects a ``("__compressed__", summary, None)`` marker into
``session.exchanges`` so that subsequent turns skip the compressed
exchanges on replay instead of re-compressing them from scratch.
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from pawlia.memory import Session


def _tiny_budget_resolver(agent_type, thread_id):
    """Budget just above completion-reserve (2000) so only ~50 tokens are
    available for the prompt — compression triggers on 10 short exchanges."""
    return "2050"





def _count_human(messages):
    return sum(1 for m in messages if isinstance(m, HumanMessage))


def _build_exchange_pairs(session, n):
    """Populate *session.exchanges* with *n* dummy pairs."""
    session.exchanges = []
    for i in range(n):
        session.exchanges.append((f"user query {i}", f"assistant answer {i}", None))


def _messages_from_exchanges(exchanges):
    """Turn exchange tuples into a list of Human→AI messages."""
    msgs = [SystemMessage(content="system prompt")]
    for exc in exchanges:
        if isinstance(exc, (list, tuple)) and len(exc) >= 2:
            msgs.append(HumanMessage(content=exc[0]))
            msgs.append(AIMessage(content=exc[1]))
    return msgs


# ---------------------------------------------------------------------------
# Marker injection
# ---------------------------------------------------------------------------
async def test_marker_injected_when_messages_compressed(make_chat_agent):
    """After Phase-2 compression the session.exchanges list contains a
    ``__compressed__`` marker and the compressed exchanges are removed."""
    agent = make_chat_agent(llm=None)  # type: ignore[arg-type]
    agent._context_window_resolver = _tiny_budget_resolver

    _build_exchange_pairs(agent.session, 10)
    messages = _messages_from_exchanges(agent.session.exchanges)
    messages.append(HumanMessage(content="new user input"))

    n_before = len(agent.session.exchanges)
    messages_compressed = agent._prepare_messages_for_context_budget(
        messages, thread_id=None,
    )

    # Compression should have happened — the compressed list is shorter.
    assert len(messages_compressed) < len(messages)
    assert len(agent.session.exchanges) < n_before
    assert len(agent.session.exchanges) >= 1  # at least the marker

    first = agent.session.exchanges[0]
    assert isinstance(first, (list, tuple))
    assert first[0] == "__compressed__"
    assert isinstance(first[1], str) and len(first[1]) > 0


async def test_no_marker_when_no_compression_needed(make_chat_agent):
    """When messages fit within the budget, no marker is injected."""
    agent = make_chat_agent(llm=None)  # type: ignore[arg-type]
    agent._context_window_resolver = _tiny_budget_resolver

    _build_exchange_pairs(agent.session, 2)
    messages = _messages_from_exchanges(agent.session.exchanges)
    messages.append(HumanMessage(content="hi"))

    before = list(agent.session.exchanges)
    compressed = agent._prepare_messages_for_context_budget(
        messages, thread_id=None,
    )
    assert compressed is not None
    assert agent.session.exchanges == before  # unchanged
    # Verify no marker crept in
    for e in agent.session.exchanges:
        assert e[0] != "__compressed__"


async def test_no_marker_when_persist_compression_is_false(make_chat_agent):
    """Context recovery path does not inject markers."""
    agent = make_chat_agent(llm=None)  # type: ignore[arg-type]
    agent._context_window_resolver = _tiny_budget_resolver

    _build_exchange_pairs(agent.session, 15)
    messages = _messages_from_exchanges(agent.session.exchanges)
    messages.append(HumanMessage(content="new user input"))

    before = list(agent.session.exchanges)
    compressed = agent._prepare_messages_for_context_budget(
        messages, thread_id=None, persist_compression=False,
    )
    assert len(compressed) < len(messages)
    assert agent.session.exchanges == before  # unchanged
    for e in agent.session.exchanges:
        assert e[0] != "__compressed__"


async def test_no_marker_for_threads(make_chat_agent):
    """Thread context does not get markers injected into main session."""
    agent = make_chat_agent(llm=None)  # type: ignore[arg-type]
    agent._context_window_resolver = _tiny_budget_resolver

    _build_exchange_pairs(agent.session, 15)
    messages = _messages_from_exchanges(agent.session.exchanges)
    messages.append(HumanMessage(content="new user input"))

    before = list(agent.session.exchanges)
    compressed = agent._prepare_messages_for_context_budget(
        messages, thread_id="some-thread",
    )
    assert len(compressed) < len(messages)
    assert agent.session.exchanges == before


# ---------------------------------------------------------------------------
# Replay respects markers
# ---------------------------------------------------------------------------
async def test_replay_skips_compressed_exchanges(make_chat_agent):
    """When session.exchanges has a marker, run() replays only exchanges
    after the last marker and injects the summary."""
    from support.llm import ScriptedLLM, Reply

    llm = ScriptedLLM().default(Reply(text="final answer"))
    agent = make_chat_agent(llm=llm)

    # Set up exchanges with a marker (simulating a previous compression):
    # marker + 3 recent exchanges
    agent.session.exchanges = [
        ("__compressed__", "5 old exchanges compressed", None),
        ("recent_q1", "recent_a1", None),
        ("recent_q2", "recent_a2", None),
        ("recent_q3", "recent_a3", None),
    ]

    await agent.run("new user query")

    assert llm.call_count >= 1
    prompt = llm.calls[0]

    # The marker summary should appear as a HumanMessage
    marker_msgs = [m for m in prompt if isinstance(m, HumanMessage)
                   and "5 old exchanges compressed" in str(m.content)]
    assert marker_msgs, "Marker summary text not found in prompt"

    # The old exchanges from before the marker should NOT appear
    prompt_text = " ".join(str(m.content) for m in prompt if isinstance(m, HumanMessage))
    assert "recent_q1" in prompt_text or "recent_q2" in prompt_text or "recent_q3" in prompt_text
    # "user query 0" etc would be the compressed old content — not present
    assert "user query 0" not in prompt_text
    assert "user query 5" not in prompt_text


async def test_multiple_markers_only_last_used(make_chat_agent):
    """When multiple markers exist, only the last marker's summary is used
    and only exchanges after it are replayed."""
    from support.llm import ScriptedLLM, Reply

    llm = ScriptedLLM().default(Reply(text="final answer"))
    agent = make_chat_agent(llm=llm)

    agent.session.exchanges = [
        ("__compressed__", "first compression", None),
        ("mid_q1", "mid_a1", None),
        ("__compressed__", "second compression", None),
        ("final_q1", "final_a1", None),
    ]

    await agent.run("new query")

    assert llm.call_count >= 1
    prompt = llm.calls[0]
    prompt_text = " ".join(str(m.content) for m in prompt)

    # Only the LAST marker's summary should be present
    assert "second compression" in prompt_text
    # "first compression" should NOT be in the prompt (the second marker
    # already covered it)
    assert "first compression" not in prompt_text

    # Only exchanges after the last marker
    assert "final_q1" in prompt_text
    assert "mid_q1" not in prompt_text


async def test_no_marker_replays_all(make_chat_agent):
    """Without any marker, all exchanges are replayed normally."""
    from support.llm import ScriptedLLM, Reply

    llm = ScriptedLLM().default(Reply(text="final answer"))
    agent = make_chat_agent(llm=llm)

    agent.session.exchanges = [
        ("old_q1", "old_a1", None),
        ("old_q2", "old_a2", None),
        ("recent_q1", "recent_a1", None),
    ]

    await agent.run("new query")

    assert llm.call_count >= 1
    prompt = llm.calls[0]
    prompt_text = " ".join(str(m.content) for m in prompt)
    assert "old_q1" in prompt_text
    assert "old_q2" in prompt_text
    assert "recent_q1" in prompt_text


# ---------------------------------------------------------------------------
# Integration: real compression produces a marker
# ---------------------------------------------------------------------------
async def test_run_compresses_and_persists_marker(make_chat_agent):
    """A full run() that triggers context compression produces a
    __compressed__ marker in session.exchanges."""
    from support.llm import ScriptedLLM, Reply

    llm = ScriptedLLM().default(Reply(text="ok"))
    agent = make_chat_agent(llm=llm)
    # Tiny budget so compression triggers immediately
    agent._context_window_resolver = _tiny_budget_resolver

    # Fill many exchanges so the context budget is exceeded
    for i in range(20):
        agent.session.exchanges.append((f"long query {i}" * 10, f"long answer {i}" * 10, None))

    await agent.run("new query")

    # There should now be a marker in the list
    markers = [e for e in agent.session.exchanges
               if isinstance(e, (list, tuple)) and e[0] == "__compressed__"]
    assert markers, "No compression marker found after run()"
