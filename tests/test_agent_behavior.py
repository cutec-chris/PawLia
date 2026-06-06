"""System-level behavior of the chat stack, driven by a deterministic LLM.

These tests describe how the agent *should* behave through its public entry
points (``ChatAgent.run`` and ``RouterAgent.run``) — the model's tool calls and
answers are scripted, everything else (loop, tool dispatch, memory, skill
routing) is the real code. They are contracts, not snapshots of the current
implementation; a failure here means the system diverged from the contract.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from pawlia.agents.router import RouterAgent
from pawlia.memory import MemoryManager
from support.factory import FakeLLMFactory
from support.llm import Reply, ScriptedLLM


# ---------------------------------------------------------------------------
# Direct answers
# ---------------------------------------------------------------------------
async def test_direct_question_returns_text_without_calling_a_skill(make_chat_agent):
    llm = ScriptedLLM().on_text("capital of france", "Paris is the capital.")
    agent = make_chat_agent(llm=llm, skills=[])

    out = await agent.run("What is the capital of France?")

    assert "Paris" in out
    assert llm.call_count == 1  # one round-trip, no tool loop


async def test_thinking_blocks_are_stripped_from_the_final_answer(make_chat_agent):
    llm = ScriptedLLM().on_text("hello", "<think>be friendly</think>Hi there!")
    agent = make_chat_agent(llm=llm, skills=[])

    out = await agent.run("hello")

    assert out == "Hi there!"
    assert "think" not in out.lower()


# ---------------------------------------------------------------------------
# Skill delegation
# ---------------------------------------------------------------------------
async def test_tool_call_delegates_to_named_skill_with_query(make_chat_agent, fake_runner):
    runner = fake_runner(returns="1. realpython.com")
    llm = ScriptedLLM().on(
        "python tutorials",
        ScriptedLLM.tool("searxng", query="python tutorials"),
        Reply(text="Here are some tutorials."),
    )
    agent = make_chat_agent(llm=llm, skills=["searxng"], runner=runner)

    out = await agent.run("find python tutorials")

    assert runner.calls == [("searxng", "python tutorials")]
    assert "tutorials" in out.lower()


async def test_skill_result_is_fed_back_to_the_model(make_chat_agent, fake_runner):
    runner = fake_runner(returns="SUNNY-25-DEGREES")
    llm = ScriptedLLM().on(
        "weather",
        ScriptedLLM.tool("searxng", query="weather rome"),
        Reply(text="It is sunny in Rome."),
    )
    agent = make_chat_agent(llm=llm, skills=["searxng"], runner=runner)

    await agent.run("what is the weather in rome")

    # The second invocation must contain the skill's output so the model can
    # ground its answer in it.
    last_prompt = llm.calls[-1]
    assert any("SUNNY-25-DEGREES" in str(m.content) for m in last_prompt)


async def test_two_parallel_tool_calls_run_both_skills(make_chat_agent, fake_runner):
    runner = fake_runner(per_skill={"searxng": "weatherX", "files": "notesY"})
    llm = ScriptedLLM().on(
        "plan my trip",
        Reply(tool_calls=[
            {"name": "searxng", "args": {"query": "weather rome"}},
            {"name": "files", "args": {"query": "read trip.md"}},
        ]),
        Reply(text="Rome is sunny; your notes say pack light."),
    )
    agent = make_chat_agent(llm=llm, skills=["searxng", "files"], runner=runner)

    out = await agent.run("plan my trip")

    assert {name for name, _ in runner.calls} == {"searxng", "files"}
    assert "Rome" in out


async def test_sequential_skill_rounds_continue_until_the_model_answers(
    make_chat_agent, fake_runner
):
    runner = fake_runner(returns="step-done")
    llm = ScriptedLLM().on(
        "do two steps",
        ScriptedLLM.tool("searxng", query="step one"),
        ScriptedLLM.tool("searxng", query="step two"),
        Reply(text="Both steps done."),
    )
    agent = make_chat_agent(llm=llm, skills=["searxng"], runner=runner)

    out = await agent.run("do two steps")

    assert len(runner.calls) == 2
    assert "Both steps done." in out


# ---------------------------------------------------------------------------
# Robustness to small-model tool-call quirks (intended repair behavior)
# ---------------------------------------------------------------------------
async def test_string_args_are_normalized_to_query(make_chat_agent, fake_runner):
    runner = fake_runner()
    llm = ScriptedLLM().on(
        "search",
        Reply(tool_calls=[{"name": "searxng", "args": "python tutorials"}]),
        Reply(text="done"),
    )
    agent = make_chat_agent(llm=llm, skills=["searxng"], runner=runner)

    await agent.run("search please")

    assert runner.last_query == "python tutorials"


async def test_alias_arg_keys_are_normalized_to_query(make_chat_agent, fake_runner):
    runner = fake_runner()
    llm = ScriptedLLM().on(
        "search",
        Reply(tool_calls=[{"name": "searxng", "args": {"task": "find cats"}}]),
        Reply(text="done"),
    )
    agent = make_chat_agent(llm=llm, skills=["searxng"], runner=runner)

    await agent.run("search please")

    assert runner.last_query == "find cats"


async def test_skill_name_is_fuzzy_resolved_across_dashes_and_case(
    make_chat_agent, fake_runner
):
    runner = fake_runner()
    # Skill registered as "web_search"; model calls "Web-Search".
    llm = ScriptedLLM().on(
        "look it up",
        Reply(tool_calls=[{"name": "Web-Search", "args": {"query": "q"}}]),
        Reply(text="done"),
    )
    agent = make_chat_agent(llm=llm, skills=["web_search"], runner=runner)

    await agent.run("look it up")

    assert runner.calls and runner.calls[0][0] == "web_search"


async def test_unknown_skill_name_is_reported_not_crashed(make_chat_agent, fake_runner):
    runner = fake_runner()
    llm = ScriptedLLM().on(
        "go",
        Reply(tool_calls=[{"name": "does_not_exist", "args": {"query": "x"}}]),
        Reply(text="Sorry, I cannot do that."),
    )
    agent = make_chat_agent(llm=llm, skills=["searxng"], runner=runner)

    out = await agent.run("go")  # must not raise

    assert isinstance(out, str) and out
    assert runner.calls == []  # the missing skill was never dispatched


async def test_text_form_tool_call_is_executed_as_a_real_skill_call(
    make_chat_agent, fake_runner
):
    """A model that writes a tool call as text instead of using the tool API
    should still have it executed."""
    runner = fake_runner(returns="1. result")
    llm = ScriptedLLM().on(
        "search",
        Reply(text='<tool_call>{"name":"searxng","arguments":{"query":"python tutorials"}}</tool_call>'),
        Reply(text="Here are the tutorials."),
    )
    agent = make_chat_agent(llm=llm, skills=["searxng"], runner=runner)

    out = await agent.run("search for tutorials")

    assert runner.last_query == "python tutorials"
    assert "tutorials" in out.lower()


async def test_plain_text_tool_intent_is_nudged_into_a_real_call(
    make_chat_agent, fake_runner
):
    """A model that only *describes* searching should be nudged until it
    actually calls the skill (the nudge appends a follow-up user message)."""
    runner = fake_runner(returns="1. result")
    llm = (
        ScriptedLLM()
        .on("tutorials", Reply(text="I will search the web for that now."))
        # After the nudge, the last user message is the nudge text:
        .on(
            "call it now",
            ScriptedLLM.tool("searxng", query="python tutorials"),
            Reply(text="Here are the tutorials."),
        )
    )
    agent = make_chat_agent(llm=llm, skills=["searxng"], runner=runner)

    out = await agent.run("find python tutorials")

    assert runner.last_query == "python tutorials"
    assert "tutorials" in out.lower()


class _ApiError(Exception):
    """Mimics a provider 400 'tool_use_failed' error."""
    status_code = 400


async def test_tool_use_failed_error_is_recovered_without_failing_the_turn(make_chat_agent):
    """When the provider rejects a turn with tool_use_failed, the agent injects
    a synthetic tool result and retries instead of crashing."""
    err = _ApiError(
        "Error code: 400 - tool_use_failed: model called a tool; "
        'failed_generation: {"name": "searxng", "arguments": {}}'
    )
    llm = ScriptedLLM().on(
        "hello",
        Reply(error=err),
        Reply(text="Recovered and answered."),
    )
    agent = make_chat_agent(llm=llm, skills=["searxng"])

    out = await agent.run("hello there")

    assert "Recovered" in out


# ---------------------------------------------------------------------------
# Interim ("working on it") callbacks
# ---------------------------------------------------------------------------
async def test_first_turn_interim_text_is_suppressed(make_chat_agent, fake_runner):
    """Text emitted alongside the very first tool call is withheld, so intro /
    bootstrap chatter never leaks to the user."""
    seen = []

    async def on_interim(text):
        seen.append(text)

    llm = ScriptedLLM().on(
        "search",
        Reply(text="Let me search for that!", tool_calls=[{"name": "searxng", "args": {"query": "q"}}]),
        Reply(text="Here are the results."),
    )
    agent = make_chat_agent(
        llm=llm, skills=["searxng"], runner=fake_runner(), on_interim=on_interim
    )

    out = await agent.run("search something")

    assert seen == []
    assert "results" in out.lower()


async def test_later_turn_interim_text_is_forwarded(make_chat_agent, fake_runner):
    seen = []

    async def on_interim(text):
        seen.append(text)

    llm = ScriptedLLM().on(
        "search",
        Reply(tool_calls=[{"name": "searxng", "args": {"query": "q1"}}]),
        Reply(text="Let me refine this!", tool_calls=[{"name": "searxng", "args": {"query": "q2"}}]),
        Reply(text="Here are the results."),
    )
    agent = make_chat_agent(
        llm=llm, skills=["searxng"], runner=fake_runner(), on_interim=on_interim
    )

    await agent.run("search something")

    assert len(seen) == 1
    assert "refine" in seen[0].lower()


async def test_direct_answer_emits_no_interim(make_chat_agent):
    seen = []

    async def on_interim(text):
        seen.append(text)

    llm = ScriptedLLM().on_text("hi", "Hello!")
    agent = make_chat_agent(llm=llm, skills=[], on_interim=on_interim)

    await agent.run("hi")

    assert seen == []


# ---------------------------------------------------------------------------
# Memory: persistence and replay
# ---------------------------------------------------------------------------
async def test_exchange_is_persisted_after_a_turn(make_chat_agent, session):
    llm = ScriptedLLM().on_text("ping", "pong")
    agent = make_chat_agent(llm=llm, skills=[])

    await agent.run("ping")

    assert session.exchanges, "the turn should be recorded in session history"
    user_text, bot_text = session.exchanges[-1][0], session.exchanges[-1][1]
    assert user_text == "ping"
    assert bot_text == "pong"


async def test_prior_exchanges_are_replayed_as_human_ai_pairs(make_chat_agent, session):
    session.exchanges.append(("earlier question", "earlier answer", None))
    llm = ScriptedLLM().on_text("follow up", "ok")
    agent = make_chat_agent(llm=llm, skills=[])

    await agent.run("follow up")

    first_prompt = llm.calls[0]
    joined = "\n".join(str(m.content) for m in first_prompt)
    assert "earlier question" in joined
    assert "earlier answer" in joined


# ---------------------------------------------------------------------------
# Vision
# ---------------------------------------------------------------------------
async def test_vision_turn_routes_to_the_vision_model(make_chat_agent):
    text_llm = ScriptedLLM().default(Reply(text="should not be used"))
    vision_llm = ScriptedLLM().on_text("image", "I see a cat.")
    agent = make_chat_agent(llm=text_llm, skills=[], vision_llm=vision_llm)

    tiny_png = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M8AAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
    out = await agent.run("what is in this image?", images=[tiny_png])

    assert "cat" in out.lower()
    assert vision_llm.call_count >= 1
    assert text_llm.call_count == 0


# ---------------------------------------------------------------------------
# RouterAgent: backend dispatch
# ---------------------------------------------------------------------------
def _local_agent_double(answer="local"):
    return SimpleNamespace(
        llm=SimpleNamespace(model_name="m-local", model="m-local", temperature=0.3),
        run=AsyncMock(return_value=answer),
        run_streamed=AsyncMock(return_value=answer),
        on_interim=None,
        on_skill_start=None,
        on_skill_step=None,
        on_skill_done=None,
        on_model_change=None,
    )


def _router(tmp_path, user_id, backend_model):
    mm = MemoryManager(str(tmp_path))
    session = mm.load_session(user_id)
    mm.set_agent_override_value(session, "chat", backend_model)
    local = _local_agent_double()
    router = RouterAgent(
        user_id=user_id,
        llm_factory=FakeLLMFactory(),
        memory=mm,
        session=session,
        skills={"browser": object()},
        local_agent_factory=lambda: local,
        logger=SimpleNamespace(getChild=lambda _name: SimpleNamespace()),
    )
    return router, session, local


async def test_router_uses_local_agent_for_a_pawlia_backend(tmp_path):
    router, _session, local = _router(tmp_path, "bob", "pawlia_model")

    out = await router.run("Hi")

    assert out == "local"
    local.run.assert_awaited_once()
    assert router.list_skills() == ["browser"]


async def test_router_routes_to_hermes_and_persists_the_turn(tmp_path):
    router, session, local = _router(tmp_path, "alice", "hermes")
    hermes = AsyncMock(return_value="Antwort von Hermes")
    router._hermes_client = lambda thread_id=None, agent_type="chat": SimpleNamespace(run=hermes)

    out = await router.run("Hallo")

    assert out == "Antwort von Hermes"
    assert session.exchanges[-1][0] == "Hallo"
    assert session.exchanges[-1][1] == "Antwort von Hermes"
    local.run.assert_not_called()
    assert router.list_skills() == []  # hermes backend exposes no local skills


async def test_router_forwards_allow_skills_flag_to_local_streaming(tmp_path):
    router, _session, local = _router(tmp_path, "bob", "pawlia_model")

    out = await router.run_streamed("Hi", allow_skills=False)

    assert out == "local"
    local.run_streamed.assert_awaited_once_with(
        "Hi",
        system_prompt=None,
        images=None,
        thread_id=None,
        on_sentence=None,
        on_skill_start=None,
        on_skill_step=None,
        on_skill_done=None,
        allow_skills=False,
    )


# ---------------------------------------------------------------------------
# RouterAgent exposes the inner ChatAgent's attachment queue
# ---------------------------------------------------------------------------
def test_router_proxies_pending_attachments_to_local_agent():
    """attach_file queues onto the local ChatAgent; interfaces drain the queue
    off the RouterAgent. The proxy must expose the *same* live list, or
    attachments are silently dropped."""
    import logging

    class _StubChat:
        def __init__(self):
            self.pending_attachments = []
            self.on_interim = None
            self.on_skill_start = None
            self.on_skill_step = None
            self.on_skill_done = None
            self.on_model_change = None
            self._on_fallback = None

    stub = _StubChat()
    router = RouterAgent(
        user_id="u",
        llm_factory=None,
        memory=None,
        session=None,
        skills={},
        local_agent_factory=lambda: stub,
        logger=logging.getLogger("test"),
    )

    # No local agent materialized yet → empty, no crash.
    assert router.pending_attachments == []

    router._ensure_local_agent()
    stub.pending_attachments.append({"filename": "x.jpg"})

    # The router exposes the inner agent's queue (same live list).
    assert router.pending_attachments == [{"filename": "x.jpg"}]
    assert router.pending_attachments is stub.pending_attachments
