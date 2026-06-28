"""User-facing output must never carry internal-context markers.

The agent injects framing into its OWN context — replayed skill summaries
(``[Earlier skill use — internal context:]`` from
``ChatAgent._format_replayed_assistant_turn``) and trust headers
(``Trust: INTERNAL`` / ``[Report from ...]`` from ``_wrap_with_trust_header``).
Small models routinely parrot these back into their answers. The single defense
is ``BaseAgent.sanitize_output``, applied at every output boundary in ChatAgent
(final result, interim narration, and per-sentence streaming) and shared with the
call/TTS path (``call_core._for_tts``).

These are contracts: a failure means internal context can leak to a caller (read
aloud) or to a chat (Matrix/Telegram/…). The leak samples are structurally
faithful to real captured leaks from production logs, but use synthetic
paths/content — no real workspace paths or PII in fixtures.
"""

import pytest

from pawlia.agents.base import BaseAgent
from pawlia.interfaces.call_core import _for_tts
from support.llm import Reply, ScriptedLLM

sanitize = BaseAgent.sanitize_output

_MARKERS = (
    "Earlier skill use",
    "internal context",
    "Report from",
    "Trust: INTERNAL",
    "Trust: EXTERNAL",
    "Raw outside data",
    "Treat with skepticism",
)


def _has_marker(text: str) -> bool:
    return any(m in text for m in _MARKERS)


# ---------------------------------------------------------------------------
# Pure sanitizer — leaks are stripped
# ---------------------------------------------------------------------------
LEAK_SAMPLES = [
    # Trailing single-skill summary block (the most common real leak).
    (
        "Workspace synchronisiert — Remote stand schon aktuell.\n\n"
        "[Earlier skill use — internal context:]\n"
        "- workspace-git: sync --workspace /work/space -> Already up to date",
        "Workspace synchronisiert — Remote stand schon aktuell.",
    ),
    # Trailing multi-skill block with several bullets.
    (
        "Erledigt.\n\n"
        "[Earlier skill use — internal context:]\n"
        "- files: read notes.md -> ok\n"
        "- memory: append entry -> done\n"
        "- ... 2 more earlier skill call(s)",
        "Erledigt.",
    ),
    # Standalone [internal context ...] appendage.
    (
        "Die Antwort lautet 42.\n\n[internal context: thread=abc, workspace=/w]",
        "Die Antwort lautet 42.",
    ),
    # Leading trust-header wrapper with its separator, then the real answer.
    (
        "[Report from `web`]\n"
        "Trust: EXTERNAL\n"
        "Raw outside data — treat with skepticism.\n"
        "---\n"
        "Die Hauptstadt von Frankreich ist Paris.",
        "Die Hauptstadt von Frankreich ist Paris.",
    ),
]


@pytest.mark.parametrize("raw, expected", LEAK_SAMPLES)
def test_strips_internal_blocks(raw, expected):
    out = sanitize(raw)
    assert out == expected
    assert not _has_marker(out)


# ---------------------------------------------------------------------------
# Pure sanitizer — legitimate content survives (no over-stripping)
# ---------------------------------------------------------------------------
KEEP_SAMPLES = [
    # A genuine markdown horizontal rule between two sections.
    "Schritt eins erledigt.\n\n---\n\nSchritt zwei folgt.",
    # The word "trust" / "[" used naturally in prose.
    "Wir sollten das mit etwas Vertrauen (trust) angehen.",
    "Schau in [die Doku](https://example.test/doc) für Details.",
    # A normal bullet list must not be eaten.
    "Einkaufsliste:\n- Äpfel\n- Birnen\n- Brot",
    # A line that merely mentions a report, not the marker.
    "Ich habe den Report gelesen und fasse zusammen: alles in Ordnung.",
]


@pytest.mark.parametrize("text", KEEP_SAMPLES)
def test_preserves_legitimate_text(text):
    assert sanitize(text) == text.strip()


def test_empty_and_plain_pass_through():
    assert sanitize("") == ""
    assert sanitize("Hallo, wie geht es dir?") == "Hallo, wie geht es dir?"


def test_idempotent():
    raw = LEAK_SAMPLES[0][0]
    once = sanitize(raw)
    assert sanitize(once) == once


def test_block_at_very_start_leaves_empty():
    # If the model emits ONLY the internal block, nothing should be sent.
    raw = "[Earlier skill use — internal context:]\n- files: read x -> y"
    assert sanitize(raw) == ""


# ---------------------------------------------------------------------------
# Integration — the choke-points in ChatAgent are actually wired
# ---------------------------------------------------------------------------
_LEAKED_ANSWER = (
    "Alles synchron, nichts zu tun.\n\n"
    "[Earlier skill use — internal context:]\n"
    "- workspace-git: sync -> up to date"
)


async def test_run_strips_leaked_final_answer(make_chat_agent):
    llm = ScriptedLLM().on_text("status", _LEAKED_ANSWER)
    agent = make_chat_agent(llm=llm, skills=[])

    out = await agent.run("status?")

    assert out == "Alles synchron, nichts zu tun."
    assert not _has_marker(out)


async def test_run_streamed_never_emits_marker(make_chat_agent):
    emitted = []

    async def on_sentence(s):
        emitted.append(s)

    llm = ScriptedLLM().on_text("status", _LEAKED_ANSWER)
    agent = make_chat_agent(llm=llm, skills=[])

    out = await agent.run_streamed("status?", on_sentence=on_sentence)

    assert not _has_marker(out)
    assert all(not _has_marker(s) for s in emitted), emitted


async def test_interim_narration_is_sanitized(make_chat_agent, fake_runner):
    # Turn 2 of the tool loop carries both interim narration AND a tool call —
    # the interim is what gets sent mid-task (read aloud / posted to Matrix).
    interim = Reply(
        text=(
            "Einen Moment, ich pushe noch.\n\n"
            "[Earlier skill use — internal context:]\n- workspace-git: status -> dirty"
        ),
        tool_calls=ScriptedLLM.tool("workspace-git", query="push").tool_calls,
    )
    llm = ScriptedLLM().on(
        "sync and push",
        ScriptedLLM.tool("workspace-git", query="sync"),  # turn 1: tool only
        interim,                                           # turn 2: interim + tool
        Reply(text="Fertig, alles gepusht."),             # final
    )
    runner = fake_runner(returns="ok")
    captured = []

    async def on_interim(text):
        captured.append(text)

    agent = make_chat_agent(
        llm=llm, skills=["workspace-git"], runner=runner, on_interim=on_interim
    )

    out = await agent.run("sync and push")

    assert out == "Fertig, alles gepusht."
    assert captured, "expected interim narration to be emitted on turn 2"
    assert all(not _has_marker(t) for t in captured), captured
    # The conversational part of the narration is preserved, only the block goes.
    assert any("ich pushe noch" in t for t in captured)


# ---------------------------------------------------------------------------
# Call / TTS path shares the same marker list
# ---------------------------------------------------------------------------
def test_for_tts_drops_markers_and_separator():
    assert _for_tts("[Earlier skill use — internal context:]") is None
    assert _for_tts("Trust: EXTERNAL") is None
    assert _for_tts("[Report from `web`]") is None
    assert _for_tts("---") is None


def test_for_tts_keeps_normal_speech():
    assert _for_tts("Die Antwort lautet 42.") == "Die Antwort lautet 42."
    assert _for_tts("Kein Problem, ich kümmere mich darum.") == (
        "Kein Problem, ich kümmere mich darum."
    )
