"""Behavior of the streaming entry point ``ChatAgent.run_streamed``.

Streaming has its own contract: text is delivered sentence-by-sentence via an
``on_sentence`` callback (for incremental TTS), tool calls fall back to
non-streamed execution, and ``allow_skills=False`` must never run a tool. The
``astream`` path does not go through ``_invoke``/``_sanitize`` like ``run``
does, so it is covered separately here.
"""

from support.llm import Reply, ScriptedLLM


def _collector():
    out = []

    async def on_sentence(s):
        out.append(s)

    return out, on_sentence


async def test_streamed_answer_is_emitted_sentence_by_sentence(make_chat_agent):
    sentences, on_sentence = _collector()
    llm = ScriptedLLM().on_text("tell me", "First sentence. Second one! Third?")
    agent = make_chat_agent(llm=llm, skills=[])

    result = await agent.run_streamed("tell me a few things", on_sentence=on_sentence)

    assert result == "First sentence. Second one! Third?"
    # Delivered as discrete, trimmed sentences in order (ready for TTS).
    assert [s.strip() for s in sentences] == [
        "First sentence.",
        "Second one!",
        "Third?",
    ]


async def test_no_sentence_is_emitted_while_inside_a_think_block(make_chat_agent):
    sentences, on_sentence = _collector()
    llm = ScriptedLLM().on_text(
        "ponder", "<think>let me reason about this</think>The answer is 42."
    )
    agent = make_chat_agent(llm=llm, skills=[])

    result = await agent.run_streamed("ponder this", on_sentence=on_sentence)

    assert result == "The answer is 42."
    assert all("reason" not in s for s in sentences)
    assert "".join(sentences).strip() == "The answer is 42."


async def test_streamed_direct_answer_is_persisted(make_chat_agent, session):
    _, on_sentence = _collector()
    llm = ScriptedLLM().on_text("ping", "pong.")
    agent = make_chat_agent(llm=llm, skills=[])

    await agent.run_streamed("ping", on_sentence=on_sentence)

    assert session.exchanges[-1][0] == "ping"
    assert session.exchanges[-1][1] == "pong."


async def test_streamed_tool_call_falls_back_then_streams_the_final_answer(
    make_chat_agent, fake_runner
):
    sentences, on_sentence = _collector()
    runner = fake_runner(returns="raw data")
    llm = ScriptedLLM().on(
        "weather",
        ScriptedLLM.tool("searxng", query="weather rome"),
        Reply(text="It is sunny in Rome today."),
    )
    agent = make_chat_agent(llm=llm, skills=["searxng"], runner=runner)

    result = await agent.run_streamed("what is the weather in rome", on_sentence=on_sentence)

    assert runner.last_query == "weather rome"
    assert result == "It is sunny in Rome today."
    # The final answer (not the tool turn) is what gets streamed to the user.
    assert "".join(sentences).strip() == result.strip()


async def test_allow_skills_false_returns_text_and_never_runs_a_tool(
    make_chat_agent, fake_runner
):
    runner = fake_runner()
    llm = ScriptedLLM().on(
        "search",
        Reply(
            text="I cannot use tools right now.",
            tool_calls=[{"name": "searxng", "args": {"query": "q"}}],
        ),
    )
    agent = make_chat_agent(llm=llm, skills=["searxng"], runner=runner)

    result = await agent.run_streamed("search now", allow_skills=False)

    assert runner.calls == []
    assert result == "I cannot use tools right now."


async def test_streamed_bracket_form_tool_call_does_not_leak_to_tts(
    make_chat_agent, fake_runner
):
    """A model that writes a skill call inline as ``[tool call: skill(args)]``
    must not have the marker (or its arguments) read out loud during the
    streamed turn. The fake-call retry path should produce a real
    structured tool call instead.

    Regression for: log/chris_…-20260629-144428/container.log line 5863
    where the bracketed marker was streamed as TTS and the actual call
    was never executed.
    """
    sentences, on_sentence = _collector()
    runner = fake_runner(returns="weather data")
    llm = (
        ScriptedLLM()
        # 1st turn: bracket-form fake call (the bug).
        .on("weather", Reply(text='[tool call: searxng(query="weather rome")]'))
        # 2nd turn: after the fake-call nudge, a real structured call.
        # 3rd turn: after the tool result, the final answer.
        .on(
            "real tool call now",
            ScriptedLLM.tool("searxng", query="weather rome"),
            Reply(text="It is sunny in Rome today."),
        )
    )
    agent = make_chat_agent(llm=llm, skills=["searxng"], runner=runner)

    result = await agent.run_streamed(
        "what is the weather in rome", on_sentence=on_sentence
    )

    assert runner.last_query == "weather rome"
    assert result == "It is sunny in Rome today."
    spoken = " ".join(sentences)
    # Neither the marker nor any of its arguments may be spoken aloud.
    assert "[tool call" not in spoken
    assert "tool call" not in spoken
    assert "weather rome" not in spoken


async def test_streamed_bracket_form_with_preamble_keeps_preamble_but_drops_marker(
    make_chat_agent, fake_runner
):
    """A model that emits a legitimate preamble BEFORE the bracket-form
    marker must still have the preamble spoken; only the marker and what
    follows it must be suppressed."""
    sentences, on_sentence = _collector()
    runner = fake_runner(returns="weather data")
    llm = (
        ScriptedLLM()
        .on(
            "weather",
            Reply(
                text='Ok, ich bin dran! [tool call: searxng(query="weather rome")]'
            ),
        )
        .on(
            "real tool call now",
            ScriptedLLM.tool("searxng", query="weather rome"),
            Reply(text="It is sunny in Rome today."),
        )
    )
    agent = make_chat_agent(llm=llm, skills=["searxng"], runner=runner)

    result = await agent.run_streamed(
        "what is the weather in rome", on_sentence=on_sentence
    )

    assert runner.last_query == "weather rome"
    assert result == "It is sunny in Rome today."
    spoken = " ".join(sentences)
    assert "Ok, ich bin dran!" in spoken
    assert "[tool call" not in spoken
    assert "weather rome" not in spoken
