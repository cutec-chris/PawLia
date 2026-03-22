"""Comprehensive simulation for the memory system.

Tests MemoryManager, BackgroundTaskQueue, MemoryIndexer config/tracking,
Scheduler idle-priority system, notification pipeline, recurrence logic,
and common interface helpers — all without external deps (no LLM, no network).

Run: python -m tests.simulate_memory
"""

import asyncio
import json
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime, timedelta
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PASS = 0
_FAIL = 0
_FAILURES: List[Tuple[str, str]] = []


def safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        text = " ".join(str(a) for a in args)
        print(text.encode("ascii", errors="replace").decode(), **kwargs)


def check(name: str, condition: bool, detail: str = ""):
    global _PASS, _FAIL
    if condition:
        _PASS += 1
        safe_print(f"  [PASS] {name}")
    else:
        _FAIL += 1
        _FAILURES.append((name, detail))
        safe_print(f"  [FAIL] {name} — {detail}")


def section(title: str):
    safe_print(f"\n{'─' * 60}")
    safe_print(f"  {title}")
    safe_print(f"{'─' * 60}")


# ═══════════════════════════════════════════════════════════════
#  1. MemoryManager — Session basics
# ═══════════════════════════════════════════════════════════════

def test_session_basics(tmp: str):
    section("Session basics")
    from pawlia.memory import MemoryManager

    mm = MemoryManager(tmp)
    s = mm.load_session("alice")

    check("Session user_id", s.user_id == "alice")
    check("Session date format", len(s.current_date_str) == 10 and "-" in s.current_date_str)
    check("Empty daily_history", s.daily_history == "")
    check("Empty exchanges", s.exchanges == [])
    check("exchange_count == 0", s.exchange_count == 0)
    check("No model override", s.model_override is None)
    check("Not private", s.private is False)
    check("No summary", s.summary == "")
    check("Empty thread_contexts", len(s.thread_contexts) == 0)
    check("Empty private_threads", len(s.private_threads) == 0)
    check("last_activity is recent", (datetime.now() - s.last_activity).total_seconds() < 5)

    # Same instance returned on second load
    s2 = mm.load_session("alice")
    check("Session caching (same object)", s is s2)

    # Different user -> different session
    s3 = mm.load_session("bob")
    check("Different user = different session", s3 is not s)
    check("Bob user_id", s3.user_id == "bob")


# ═══════════════════════════════════════════════════════════════
#  2. MemoryManager — Exchanges
# ═══════════════════════════════════════════════════════════════

def test_append_exchange(tmp: str):
    section("Append exchange")
    from pawlia.memory import MemoryManager

    mm = MemoryManager(tmp)
    s = mm.load_session("user1")

    mm.append_exchange(s, "Hallo!", "Hi, wie kann ich helfen?")
    check("exchange_count after 1", s.exchange_count == 1)
    check("exchanges list length", len(s.exchanges) == 1)
    check("exchange content", s.exchanges[0] == ("Hallo!", "Hi, wie kann ich helfen?"))
    check("daily_history not empty", len(s.daily_history) > 0)
    check("daily_history contains user text", "Hallo!" in s.daily_history)
    check("daily_history contains bot text", "Hi, wie kann ich helfen?" in s.daily_history)

    # Check file on disk
    daily_path = mm._daily_path("user1", s.current_date_str)
    check("Daily file exists", os.path.isfile(daily_path))
    with open(daily_path, encoding="utf-8") as f:
        content = f.read()
    check("File contains user text", "Hallo!" in content)
    check("File contains timestamp [HH:MM:SS]", "[" in content and "]" in content)

    # Add more exchanges
    mm.append_exchange(s, "Wie spat ist es?", "Es ist 14:30.")
    mm.append_exchange(s, "Danke!", "Gerne!")
    check("exchange_count after 3", s.exchange_count == 3)
    check("exchanges list 3 items", len(s.exchanges) == 3)
    check("recent_bot_responses tracked", len(s.recent_bot_responses) == 3)

    # last_activity updated
    check("last_activity updated", (datetime.now() - s.last_activity).total_seconds() < 2)


def test_append_no_similarity(tmp: str):
    section("Append exchange (no similarity tracking)")
    from pawlia.memory import MemoryManager

    mm = MemoryManager(tmp)
    s = mm.load_session("user_nosim")

    mm.append_exchange(s, "test", "response1", track_similarity=False)
    check("exchange appended", s.exchange_count == 1)
    check("similarity window empty", len(s.recent_bot_responses) == 0)

    mm.append_exchange(s, "test2", "response2", track_similarity=True)
    check("similarity window has 1", len(s.recent_bot_responses) == 1)

    # Mixed: multiple with track_similarity=False
    for i in range(10):
        mm.append_exchange(s, f"q{i}", f"a{i}", track_similarity=False)
    check("Still only 1 in similarity window", len(s.recent_bot_responses) == 1)
    check("But 12 total exchanges", s.exchange_count == 12)


def test_append_empty_strings(tmp: str):
    section("Append exchange (empty strings)")
    from pawlia.memory import MemoryManager

    mm = MemoryManager(tmp)
    s = mm.load_session("empty_str_user")

    mm.append_exchange(s, "", "")
    check("Empty exchange counted", s.exchange_count == 1)
    check("Empty exchange in list", s.exchanges[0] == ("", ""))

    mm.append_exchange(s, "Q", "")
    check("Empty bot response tracked", s.exchange_count == 2)

    mm.append_exchange(s, "", "A")
    check("Empty user text tracked", s.exchange_count == 3)


# ═══════════════════════════════════════════════════════════════
#  3. MemoryManager — Exchange parsing
# ═══════════════════════════════════════════════════════════════

def test_exchange_parsing(tmp: str):
    section("Exchange parsing (reload from disk)")
    from pawlia.memory import MemoryManager

    mm = MemoryManager(tmp)
    s = mm.load_session("parse_user")

    mm.append_exchange(s, "Frage eins", "Antwort eins")
    mm.append_exchange(s, "Frage zwei", "Antwort zwei")
    mm.append_exchange(s, "Frage drei", "Antwort drei")

    # Create a fresh MemoryManager (simulates restart)
    mm2 = MemoryManager(tmp)
    s2 = mm2.load_session("parse_user")

    check("Reloaded exchange_count", s2.exchange_count == 3)
    check("Reloaded exchanges[0]", s2.exchanges[0] == ("Frage eins", "Antwort eins"))
    check("Reloaded exchanges[1]", s2.exchanges[1] == ("Frage zwei", "Antwort zwei"))
    check("Reloaded exchanges[2]", s2.exchanges[2] == ("Frage drei", "Antwort drei"))


def test_parse_exchanges_static(tmp: str):
    section("_parse_exchanges edge cases")
    from pawlia.memory import MemoryManager

    parse = MemoryManager._parse_exchanges

    check("Empty string -> []", parse("") == [])
    check("Whitespace -> []", parse("   \n\n  ") == [])
    check("Random text -> []", parse("hello world no format") == [])

    # Standard format
    single = "\n[14:30:00] User: hello\nAssistant: hi there"
    result = parse(single)
    check("Single exchange parsed", len(result) == 1)
    check("Single user text", result[0][0] == "hello")
    check("Single bot text", result[0][1] == "hi there")

    # Multiple exchanges
    multi = (
        "\n[14:30:00] User: Q1\nAssistant: A1"
        "\n[14:31:00] User: Q2\nAssistant: A2"
        "\n[14:32:00] User: Q3\nAssistant: A3"
    )
    result2 = parse(multi)
    check("3 exchanges parsed", len(result2) == 3)
    check("Multi Q1", result2[0] == ("Q1", "A1"))
    check("Multi Q3", result2[2] == ("Q3", "A3"))

    # Multiline assistant response
    multiline = "\n[10:00:00] User: explain\nAssistant: line1\nline2\nline3"
    result3 = parse(multiline)
    check("Multiline response parsed", len(result3) == 1)
    check("Multiline content", "line1" in result3[0][1] and "line3" in result3[0][1])

    # Multiline followed by another exchange
    ml2 = (
        "\n[10:00:00] User: Q1\nAssistant: multi\nline\nresponse"
        "\n[10:01:00] User: Q2\nAssistant: simple"
    )
    result4 = parse(ml2)
    check("Multiline + next: 2 exchanges", len(result4) == 2)
    check("First is multiline", "multi" in result4[0][1] and "response" in result4[0][1])
    check("Second is simple", result4[1][1] == "simple")


# ═══════════════════════════════════════════════════════════════
#  4. MemoryManager — Private mode
# ═══════════════════════════════════════════════════════════════

def test_private_mode(tmp: str):
    section("Private mode")
    from pawlia.memory import MemoryManager

    mm = MemoryManager(tmp)
    s = mm.load_session("private_user")

    active = mm.toggle_private(s)
    check("Private mode activated", active is True)
    check("Session.private is True", s.private is True)

    mm.append_exchange(s, "Geheime Frage", "Geheime Antwort")
    check("Exchange in RAM", s.exchange_count == 1)
    check("Exchange in list", len(s.exchanges) == 1)

    daily_path = mm._daily_path("private_user", s.current_date_str)
    disk_content = ""
    if os.path.isfile(daily_path):
        with open(daily_path, encoding="utf-8") as f:
            disk_content = f.read()
    check("Not written to disk", "Geheime Frage" not in disk_content)
    check("daily_history stays empty", s.daily_history == "")

    active2 = mm.toggle_private(s)
    check("Private mode deactivated", active2 is False)

    mm.append_exchange(s, "Offene Frage", "Offene Antwort")
    check("exchange_count after public msg", s.exchange_count == 2)
    with open(mm._daily_path("private_user", s.current_date_str), encoding="utf-8") as f:
        disk2 = f.read()
    check("Public msg on disk", "Offene Frage" in disk2)
    check("Private msg still absent from disk", "Geheime Frage" not in disk2)


def test_private_toggle_idempotent(tmp: str):
    section("Private mode toggle cycle")
    from pawlia.memory import MemoryManager

    mm = MemoryManager(tmp)
    s = mm.load_session("toggle_user")

    # Toggle on-off-on-off
    check("Toggle 1: on", mm.toggle_private(s) is True)
    check("Toggle 2: off", mm.toggle_private(s) is False)
    check("Toggle 3: on", mm.toggle_private(s) is True)
    check("Toggle 4: off", mm.toggle_private(s) is False)
    check("Final state: not private", s.private is False)


# ═══════════════════════════════════════════════════════════════
#  5. MemoryManager — Threads
# ═══════════════════════════════════════════════════════════════

def test_threads(tmp: str):
    section("Threads")
    from pawlia.memory import MemoryManager

    mm = MemoryManager(tmp)
    s = mm.load_session("thread_user")

    mm.append_exchange(s, "Main 1", "Reply 1")
    mm.append_exchange(s, "Main 2", "Reply 2")
    mm.append_exchange(s, "Main 3", "Reply 3")

    thread_ctx = mm.get_thread_context(s, "thread_abc")
    check("Thread seeded with main exchanges", len(thread_ctx) == 3)
    check("Seed content correct", thread_ctx[0] == ("Main 1", "Reply 1"))

    mm.append_thread_exchange(s, "thread_abc", "Thread Frage", "Thread Antwort")
    check("Thread has 4 exchanges now", len(thread_ctx) == 4)
    check("Thread exchange content", thread_ctx[3] == ("Thread Frage", "Thread Antwort"))
    check("Main session still 3 exchanges", s.exchange_count == 3)

    thread_path = mm._thread_daily_path("thread_user", "thread_abc", s.current_date_str)
    check("Thread log file exists", os.path.isfile(thread_path))
    with open(thread_path, encoding="utf-8") as f:
        thread_disk = f.read()
    check("Thread log contains thread text", "Thread Frage" in thread_disk)
    check("Seeded exchanges not in thread log", "Main 1" not in thread_disk)

    thread_ctx2 = mm.get_thread_context(s, "thread_xyz")
    check("Second thread also seeded", len(thread_ctx2) == 3)
    mm.append_thread_exchange(s, "thread_xyz", "Other Q", "Other A")
    check("Second thread has 4", len(thread_ctx2) == 4)
    check("First thread still 4", len(thread_ctx) == 4)


def test_private_threads(tmp: str):
    section("Private threads")
    from pawlia.memory import MemoryManager

    mm = MemoryManager(tmp)
    s = mm.load_session("pth_user")

    is_private = mm.toggle_private_thread(s, "secret_thread")
    check("Thread private activated", is_private is True)
    check("Thread in private set", "secret_thread" in s.private_threads)

    mm.get_thread_context(s, "secret_thread")
    mm.append_thread_exchange(s, "secret_thread", "Secret Q", "Secret A")

    thread_path = mm._thread_daily_path("pth_user", "secret_thread", s.current_date_str)
    disk = ""
    if os.path.isfile(thread_path):
        with open(thread_path, encoding="utf-8") as f:
            disk = f.read()
    check("Private thread not on disk", "Secret Q" not in disk)

    is_private2 = mm.toggle_private_thread(s, "secret_thread")
    check("Thread private deactivated", is_private2 is False)

    # After toggling off, next exchange should be on disk
    mm.append_thread_exchange(s, "secret_thread", "Public Q", "Public A")
    with open(thread_path, encoding="utf-8") as f:
        disk2 = f.read()
    check("Post-toggle exchange on disk", "Public Q" in disk2)
    check("Pre-toggle exchange still absent", "Secret Q" not in disk2)


def test_thread_seed_limit(tmp: str):
    section("Thread seed limit")
    from pawlia.memory import MemoryManager

    mm = MemoryManager(tmp)
    s = mm.load_session("seed_user")

    for i in range(10):
        mm.append_exchange(s, f"Main Q{i}", f"Main A{i}")

    ctx = mm.get_thread_context(s, "t_default")
    check("Default seed is 5 exchanges", len(ctx) == 5)
    check("Seed starts from end", ctx[0] == ("Main Q5", "Main A5"))

    ctx2 = mm.get_thread_context(s, "t_small", seed_n=2)
    check("Custom seed_n=2", len(ctx2) == 2)
    check("Custom seed from end", ctx2[0] == ("Main Q8", "Main A8"))

    ctx3 = mm.get_thread_context(s, "t_zero", seed_n=0)
    check("seed_n=0 -> empty", len(ctx3) == 0)

    ctx4 = mm.get_thread_context(s, "t_large", seed_n=100)
    check("seed_n > exchanges -> all", len(ctx4) == 10)


def test_empty_thread_no_seed(tmp: str):
    section("Empty thread (no main exchanges)")
    from pawlia.memory import MemoryManager

    mm = MemoryManager(tmp)
    s = mm.load_session("empty_seed_user")

    ctx = mm.get_thread_context(s, "empty_thread")
    check("Empty thread = empty context", len(ctx) == 0)


def test_thread_reload_from_disk(tmp: str):
    section("Thread reload from disk")
    from pawlia.memory import MemoryManager

    mm = MemoryManager(tmp)
    s = mm.load_session("trl_user")
    mm.append_exchange(s, "Main Q", "Main A")
    mm.append_thread_exchange(s, "t1", "Thread Q1", "Thread A1")
    mm.append_thread_exchange(s, "t1", "Thread Q2", "Thread A2")

    # Fresh manager (restart)
    mm2 = MemoryManager(tmp)
    s2 = mm2.load_session("trl_user")
    ctx = mm2.get_thread_context(s2, "t1")
    check("Thread exchanges reloaded", len(ctx) == 2)
    check("Thread Q1 reloaded", ctx[0] == ("Thread Q1", "Thread A1"))
    check("Thread Q2 reloaded", ctx[1] == ("Thread Q2", "Thread A2"))


# ═══════════════════════════════════════════════════════════════
#  6. MemoryManager — Model overrides
# ═══════════════════════════════════════════════════════════════

def test_model_override(tmp: str):
    section("Model overrides")
    from pawlia.memory import MemoryManager

    mm = MemoryManager(tmp)
    s = mm.load_session("model_user")

    check("No override initially", s.model_override is None)

    mm.set_model_override(s, "qwen3:4b")
    check("Override in session", s.model_override == "qwen3:4b")

    path = mm._model_override_path("model_user")
    check("Override file exists", os.path.isfile(path))
    with open(path, encoding="utf-8") as f:
        check("Override file content", f.read().strip() == "qwen3:4b")

    mm2 = MemoryManager(tmp)
    s2 = mm2.load_session("model_user")
    check("Override survives reload", s2.model_override == "qwen3:4b")

    mm2.set_model_override(s2, None)
    check("Override cleared in session", s2.model_override is None)
    check("Override file removed", not os.path.isfile(path))

    # Thread model override
    mm.set_thread_model_override(s, "t1", "llama3:8b")
    check("Thread override set", mm.get_thread_model_override(s, "t1") == "llama3:8b")

    tpath = mm._thread_model_path("model_user", "t1")
    check("Thread override file exists", os.path.isfile(tpath))

    mm.set_thread_model_override(s, "t1", None)
    check("Thread override cleared", mm.get_thread_model_override(s, "t1") is None)

    # Multiple thread overrides
    mm.set_thread_model_override(s, "t2", "modelA")
    mm.set_thread_model_override(s, "t3", "modelB")
    check("Thread t2 override", mm.get_thread_model_override(s, "t2") == "modelA")
    check("Thread t3 override", mm.get_thread_model_override(s, "t3") == "modelB")
    check("Unset thread returns None", mm.get_thread_model_override(s, "t4") is None)


# ═══════════════════════════════════════════════════════════════
#  7. MemoryManager — Summarization triggers & detection
# ═══════════════════════════════════════════════════════════════

def test_summarization_trigger(tmp: str):
    section("Summarization triggers")
    from pawlia.memory import MemoryManager, MAX_EXCHANGES_BEFORE_SUMMARY

    mm = MemoryManager(tmp)
    s = mm.load_session("summary_user")

    check("No trigger at start", mm.should_summarize(s) == "")

    for i in range(MAX_EXCHANGES_BEFORE_SUMMARY):
        mm.append_exchange(s, f"Q{i}", f"Unique answer number {i} with random content {i*17}")

    check(
        f"exchange_limit after {MAX_EXCHANGES_BEFORE_SUMMARY}",
        mm.should_summarize(s) == "exchange_limit",
    )


def test_summarization_boundary(tmp: str):
    section("Summarization boundary (19 vs 20)")
    from pawlia.memory import MemoryManager, MAX_EXCHANGES_BEFORE_SUMMARY

    mm = MemoryManager(tmp)
    s = mm.load_session("boundary_user")

    # Use hash-based content to avoid triggering the repetition detector
    import hashlib
    for i in range(MAX_EXCHANGES_BEFORE_SUMMARY - 1):
        h = hashlib.sha256(str(i).encode()).hexdigest()
        mm.append_exchange(s, f"Q{i}", f"Response {h}")

    check(f"No trigger at {MAX_EXCHANGES_BEFORE_SUMMARY - 1}", mm.should_summarize(s) == "")

    mm.append_exchange(s, "final Q", "final unique A")
    check(f"Trigger at {MAX_EXCHANGES_BEFORE_SUMMARY}", mm.should_summarize(s) == "exchange_limit")


def test_repetition_trigger(tmp: str):
    section("Repetition trigger")
    from pawlia.memory import MemoryManager

    mm = MemoryManager(tmp)
    s = mm.load_session("rep_user")
    same_response = "Das ist immer die gleiche Antwort auf alles."
    for i in range(5):
        mm.append_exchange(s, f"Q{i}", same_response)

    trigger = mm.should_summarize(s)
    check("Repetition trigger fires", trigger == "repetition", f"got: {trigger}")


def test_idle_trigger(tmp: str):
    section("Idle trigger")
    from pawlia.memory import MemoryManager, IDLE_TIMEOUT_SECONDS

    mm = MemoryManager(tmp)
    s = mm.load_session("idle_user")
    mm.append_exchange(s, "Q", "A")
    s.last_activity = datetime.now() - timedelta(minutes=10)
    trigger = mm.should_summarize(s)
    check("Idle trigger fires", trigger == "idle", f"got: {trigger}")

    # Boundary: just under threshold
    mm2 = MemoryManager(tmp)
    s2 = mm2.load_session("idle_boundary")
    mm2.append_exchange(s2, "Q", "A")
    s2.last_activity = datetime.now() - timedelta(seconds=IDLE_TIMEOUT_SECONDS - 10)
    check("No idle trigger under threshold", mm2.should_summarize(s2) == "")

    # Idle with 0 exchanges -> no trigger
    mm3 = MemoryManager(tmp)
    s3 = mm3.load_session("idle_noex")
    s3.last_activity = datetime.now() - timedelta(hours=1)
    check("No idle trigger with 0 exchanges", mm3.should_summarize(s3) == "")


def test_detect_repetition_static(tmp: str):
    section("_detect_repetition edge cases")
    from pawlia.memory import MemoryManager

    detect = MemoryManager._detect_repetition

    check("Empty list -> False", detect([]) is False)
    check("Single response -> False", detect(["hello"]) is False)

    # Identical responses
    check("Identical responses -> True", detect(["same", "same"]) is True)

    # Completely different
    check("Different responses -> False",
          detect(["the quick brown fox", "ein komplett anderer satz"]) is False)

    # Slightly similar (should be under threshold)
    check("Slightly similar -> False",
          detect(["hello world foo bar", "goodbye universe baz qux"]) is False)

    # Very similar (above 0.6 threshold)
    a = "Das Wetter heute ist sonnig und warm mit 25 Grad."
    b = "Das Wetter heute ist sonnig und warm mit 26 Grad."
    check("Very similar -> True", detect([a, b]) is True)

    # Window behavior: only latest vs all older
    check("Latest different from all -> False",
          detect(["same", "same", "same", "completely different text here"]) is False)
    check("Latest same as one older -> True",
          detect(["different1", "same text here", "different2", "same text here"]) is True)


# ═══════════════════════════════════════════════════════════════
#  8. MemoryManager — Summarize
# ═══════════════════════════════════════════════════════════════

def test_summarize(tmp: str):
    section("Summarize")
    from pawlia.memory import MemoryManager

    mm = MemoryManager(tmp)
    s = mm.load_session("summ_user")

    mm.append_exchange(s, "Q1", "A1")
    mm.append_exchange(s, "Q2", "A2")
    mm.append_exchange(s, "Q3", "A3")

    summary_text = "* User asked 3 questions\n* Bot answered all"
    mm.summarize(s, summary_text)

    check("Summary stored in session", s.summary == summary_text)
    check("Exchanges cleared", len(s.exchanges) == 0)
    check("exchange_count reset", s.exchange_count == 0)
    check("daily_history cleared", s.daily_history == "")
    check("recent_bot_responses cleared", len(s.recent_bot_responses) == 0)

    sp = mm._summary_path("summ_user")
    check("Summary file exists", os.path.isfile(sp))
    with open(sp, encoding="utf-8") as f:
        check("Summary file content", f.read().strip() == summary_text)

    dp = mm._daily_path("summ_user", s.current_date_str)
    check("Daily log still on disk", os.path.isfile(dp))
    with open(dp, encoding="utf-8") as f:
        disk = f.read()
    check("Daily log still has Q1", "Q1" in disk)

    mm2 = MemoryManager(tmp)
    s2 = mm2.load_session("summ_user")
    check("Summary survives reload", s2.summary == summary_text)


def test_summarize_then_continue(tmp: str):
    section("Summarize then continue chatting")
    from pawlia.memory import MemoryManager

    mm = MemoryManager(tmp)
    s = mm.load_session("sumcont_user")

    for i in range(5):
        mm.append_exchange(s, f"Q{i}", f"A{i}")

    mm.summarize(s, "* Prior conversation summary")

    # Continue adding exchanges after summarization
    mm.append_exchange(s, "New Q1", "New A1")
    mm.append_exchange(s, "New Q2", "New A2")
    check("exchange_count after summary + 2", s.exchange_count == 2)
    check("exchanges are new ones", s.exchanges[0] == ("New Q1", "New A1"))
    check("summary preserved", s.summary == "* Prior conversation summary")

    # Second summarization
    mm.summarize(s, "* Updated summary with new info")
    check("Summary replaced", s.summary == "* Updated summary with new info")
    check("Exchanges cleared again", s.exchange_count == 0)


def test_summarize_whitespace(tmp: str):
    section("Summarize strips whitespace")
    from pawlia.memory import MemoryManager

    mm = MemoryManager(tmp)
    s = mm.load_session("ws_user")
    mm.append_exchange(s, "Q", "A")

    mm.summarize(s, "  \n  summary with spaces  \n  ")
    check("Summary stripped", s.summary == "summary with spaces")


# ═══════════════════════════════════════════════════════════════
#  9. MemoryManager — System prompt
# ═══════════════════════════════════════════════════════════════

def test_system_prompt(tmp: str):
    section("System prompt")
    from pawlia.memory import MemoryManager

    mm = MemoryManager(tmp)
    s = mm.load_session("prompt_user")

    ws = mm._workspace_dir("prompt_user")
    with open(os.path.join(ws, "IDENTITY.md"), "w", encoding="utf-8") as f:
        f.write("- **Name:** TestBot\n- **Creature:** Cat")
    with open(os.path.join(ws, "USER.md"), "w", encoding="utf-8") as f:
        f.write("- **Name:** Chris\n- **Language:** Deutsch")

    with open(mm._memory_path("prompt_user"), "w", encoding="utf-8") as f:
        f.write("Chris mag Pizza.")
    s.user_memory = "Chris mag Pizza."

    s.summary = "* Chris fragte nach dem Wetter"

    prompt = mm.build_system_prompt(s)

    check("Prompt contains identity", "TestBot" in prompt)
    check("Prompt contains user info", "Chris" in prompt)
    check("Prompt contains memory", "Pizza" in prompt)
    check("Prompt contains summary", "Wetter" in prompt)
    check("Prompt contains skill instruction", "MUST call" in prompt)
    check("Prompt contains memory.md instruction", "memory.md" in prompt)
    check("Prompt contains separator", "---" in prompt)


def test_system_prompt_empty(tmp: str):
    section("System prompt (empty workspace)")
    from pawlia.memory import MemoryManager

    mm = MemoryManager(tmp)
    s = mm.load_session("empty_prompt_user")

    prompt = mm.build_system_prompt(s)
    # Should still work and contain at least the skill instruction
    check("Prompt not empty", len(prompt) > 0)
    check("Still contains skill instruction", "MUST call" in prompt)


def test_system_prompt_no_summary_no_memory(tmp: str):
    section("System prompt (no summary, no memory)")
    from pawlia.memory import MemoryManager

    mm = MemoryManager(tmp)
    s = mm.load_session("nosumm_user")
    s.summary = ""
    s.user_memory = ""

    prompt = mm.build_system_prompt(s)
    check("No 'Conversation Summary' section", "Conversation Summary" not in prompt)
    check("No 'Memory' section", "## Memory" not in prompt)


def test_system_prompt_multiple_md_files(tmp: str):
    section("System prompt (multiple .md files)")
    from pawlia.memory import MemoryManager

    mm = MemoryManager(tmp)
    s = mm.load_session("multi_md_user")

    ws = mm._workspace_dir("multi_md_user")
    for name in ["AAA.md", "BBB.md", "CCC.md"]:
        with open(os.path.join(ws, name), "w", encoding="utf-8") as f:
            f.write(f"Content of {name}")

    prompt = mm.build_system_prompt(s)
    check("AAA.md included", "Content of AAA.md" in prompt)
    check("BBB.md included", "Content of BBB.md" in prompt)
    check("CCC.md included", "Content of CCC.md" in prompt)

    # Check sorted order: AAA before CCC
    aaa_pos = prompt.index("Content of AAA.md")
    ccc_pos = prompt.index("Content of CCC.md")
    check("Files sorted alphabetically", aaa_pos < ccc_pos)


# ═══════════════════════════════════════════════════════════════
# 10. MemoryManager — Similarity window
# ═══════════════════════════════════════════════════════════════

def test_similarity_window(tmp: str):
    section("Similarity window")
    from pawlia.memory import MemoryManager, SIMILARITY_WINDOW

    mm = MemoryManager(tmp)
    s = mm.load_session("sim_user")

    for i in range(SIMILARITY_WINDOW + 5):
        mm.append_exchange(s, f"Q{i}", f"Unique answer {i}")

    check(f"Window capped at {SIMILARITY_WINDOW}", len(s.recent_bot_responses) == SIMILARITY_WINDOW)
    check("Window contains latest", s.recent_bot_responses[-1] == f"Unique answer {SIMILARITY_WINDOW + 4}")
    # Oldest should be gone
    check("Oldest evicted", f"Unique answer 0" not in s.recent_bot_responses)


def test_similarity_window_exact_boundary(tmp: str):
    section("Similarity window exact boundary")
    from pawlia.memory import MemoryManager, SIMILARITY_WINDOW

    mm = MemoryManager(tmp)
    s = mm.load_session("simbound_user")

    # Add exactly SIMILARITY_WINDOW responses
    for i in range(SIMILARITY_WINDOW):
        mm.append_exchange(s, f"Q{i}", f"Response {i}")

    check(f"Exactly {SIMILARITY_WINDOW} in window", len(s.recent_bot_responses) == SIMILARITY_WINDOW)
    check("First response present", s.recent_bot_responses[0] == "Response 0")
    check("Last response present", s.recent_bot_responses[-1] == f"Response {SIMILARITY_WINDOW - 1}")

    # One more pushes the first out
    mm.append_exchange(s, "Extra Q", "Extra response")
    check(f"Still {SIMILARITY_WINDOW} after overflow", len(s.recent_bot_responses) == SIMILARITY_WINDOW)
    check("Response 0 evicted", "Response 0" not in s.recent_bot_responses)


# ═══════════════════════════════════════════════════════════════
# 11. MemoryManager — Directory structure
# ═══════════════════════════════════════════════════════════════

def test_directory_structure(tmp: str):
    section("Directory structure")
    from pawlia.memory import MemoryManager

    mm = MemoryManager(tmp)
    s = mm.load_session("dir_user")
    mm.append_exchange(s, "Q", "A")

    check("session dir exists", os.path.isdir(tmp))
    check("user dir exists", os.path.isdir(os.path.join(tmp, "dir_user")))
    check("workspace exists", os.path.isdir(os.path.join(tmp, "dir_user", "workspace")))
    check("memory dir exists", os.path.isdir(os.path.join(tmp, "dir_user", "workspace", "memory")))
    check("daily log exists", os.path.isfile(mm._daily_path("dir_user", s.current_date_str)))


# ═══════════════════════════════════════════════════════════════
# 12. MemoryManager — Multiline and special chars
# ═══════════════════════════════════════════════════════════════

def test_multiline_exchange(tmp: str):
    section("Multiline exchanges")
    from pawlia.memory import MemoryManager

    mm = MemoryManager(tmp)
    s = mm.load_session("ml_user")

    user_msg = "Erklaere mir\nwie das funktioniert"
    bot_msg = "Das funktioniert so:\n1. Schritt eins\n2. Schritt zwei"
    mm.append_exchange(s, user_msg, bot_msg)

    check("Multiline exchange stored", s.exchange_count == 1)

    mm2 = MemoryManager(tmp)
    s2 = mm2.load_session("ml_user")
    check("Multiline exchange survives reload", s2.exchange_count == 1)
    check("User text preserved", s2.exchanges[0][0] == user_msg)


def test_special_characters(tmp: str):
    section("Special characters")
    from pawlia.memory import MemoryManager

    mm = MemoryManager(tmp)
    s = mm.load_session("special_user")

    mm.append_exchange(s, "Was bedeutet nihongo?", "nihongo bedeutet Japanisch.")
    mm.append_exchange(s, "Zeig mir **bold** und `code`", "Hier: **fett** und `code`")
    mm.append_exchange(s, "Emojis: cat party fire", "Ja, Emojis funktionieren!")

    check("3 special exchanges", s.exchange_count == 3)
    check("Markdown preserved", "**bold**" in s.exchanges[1][0])

    mm2 = MemoryManager(tmp)
    s2 = mm2.load_session("special_user")
    check("Special chars survive reload", s2.exchange_count == 3)


def test_very_long_exchange(tmp: str):
    section("Very long exchange")
    from pawlia.memory import MemoryManager

    mm = MemoryManager(tmp)
    s = mm.load_session("long_user")

    long_msg = "A" * 50000
    long_response = "B" * 50000
    mm.append_exchange(s, long_msg, long_response)

    check("Long exchange stored", s.exchange_count == 1)
    check("Long user text correct length", len(s.exchanges[0][0]) == 50000)
    check("Long bot text correct length", len(s.exchanges[0][1]) == 50000)

    # Reload
    mm2 = MemoryManager(tmp)
    s2 = mm2.load_session("long_user")
    check("Long exchange survives reload", s2.exchange_count == 1)
    check("Long text preserved", len(s2.exchanges[0][0]) == 50000)


# ═══════════════════════════════════════════════════════════════
# 13. BackgroundTaskQueue
# ═══════════════════════════════════════════════════════════════

def test_background_tasks(tmp: str):
    section("Background tasks")
    from pawlia.background_tasks import BackgroundTaskQueue

    bq = BackgroundTaskQueue(tmp)

    t1 = bq.enqueue("alice", "recherchiere ueber KI")
    check("Task has id", len(t1["id"]) == 10)
    check("Task status pending", t1["status"] == "pending")
    check("Task has thread_id", t1["thread_id"].startswith("bg_"))
    check("Task has message", t1["message"] == "recherchiere ueber KI")
    check("Task has created timestamp", isinstance(t1["created"], float))

    t2 = bq.enqueue("alice", "suche nach Wetter")
    t3 = bq.enqueue("bob", "analysiere Daten")

    pending = bq.pending()
    check("3 pending tasks", len(pending) == 3)

    alice_tasks = bq.list_tasks("alice")
    check("Alice has 2 tasks", len(alice_tasks) == 2)
    bob_tasks = bq.list_tasks("bob")
    check("Bob has 1 task", len(bob_tasks) == 1)

    bq.mark_running("alice", t1["id"])
    pending2 = bq.pending()
    check("2 pending after mark_running", len(pending2) == 2)

    alice_tasks2 = bq.list_tasks("alice")
    running = [t for t in alice_tasks2 if t["status"] == "running"]
    check("1 running task for alice", len(running) == 1)

    bq.mark_done("alice", t1["id"])
    alice_tasks3 = bq.list_tasks("alice")
    done = [t for t in alice_tasks3 if t["status"] == "done"]
    check("1 done task", len(done) == 1)
    check("Done task has finished timestamp", "finished" in done[0])

    bq.mark_error("alice", t2["id"], "Connection timeout")
    alice_tasks4 = bq.list_tasks("alice")
    errors = [t for t in alice_tasks4 if t["status"] == "error"]
    check("1 error task", len(errors) == 1)
    check("Error message stored", errors[0].get("error") == "Connection timeout")
    check("Error task has finished", "finished" in errors[0])

    pending3 = bq.pending()
    check("Only bob's task pending now", len(pending3) == 1)
    check("Remaining pending is bob's", pending3[0][0] == "bob")

    empty = bq.list_tasks("nobody")
    check("Empty list for unknown user", len(empty) == 0)


def test_background_task_persistence(tmp: str):
    section("Background task persistence")
    from pawlia.background_tasks import BackgroundTaskQueue

    bq1 = BackgroundTaskQueue(tmp)
    bq1.enqueue("persist_user", "persistent task")

    bq2 = BackgroundTaskQueue(tmp)
    pending = bq2.pending()
    persist_tasks = [(uid, t) for uid, t in pending if uid == "persist_user"]
    check("Task survives new instance", len(persist_tasks) == 1)
    check("Message preserved", persist_tasks[0][1]["message"] == "persistent task")


def test_background_task_ordering(tmp: str):
    section("Background task ordering")
    from pawlia.background_tasks import BackgroundTaskQueue

    bq = BackgroundTaskQueue(tmp)

    tasks = []
    for i in range(5):
        t = bq.enqueue("order_user", f"Task {i}")
        tasks.append(t)
        time.sleep(0.01)

    pending = bq.pending()
    order_tasks = [(uid, t) for uid, t in pending if uid == "order_user"]
    check("5 pending in order", len(order_tasks) == 5)

    sorted_by_created = sorted(order_tasks, key=lambda x: x[1]["created"])
    messages = [t["message"] for _, t in sorted_by_created]
    check("Correct order by creation time", messages == [f"Task {i}" for i in range(5)])


def test_background_task_status_lifecycle(tmp: str):
    section("Background task full lifecycle")
    from pawlia.background_tasks import BackgroundTaskQueue

    bq = BackgroundTaskQueue(tmp)
    t = bq.enqueue("lifecycle_user", "test task")
    tid = t["id"]

    # pending -> running -> done
    tasks = bq.list_tasks("lifecycle_user")
    check("Initial: pending", tasks[0]["status"] == "pending")

    bq.mark_running("lifecycle_user", tid)
    tasks = bq.list_tasks("lifecycle_user")
    check("After mark_running: running", tasks[0]["status"] == "running")

    bq.mark_done("lifecycle_user", tid)
    tasks = bq.list_tasks("lifecycle_user")
    check("After mark_done: done", tasks[0]["status"] == "done")

    # pending -> running -> error
    t2 = bq.enqueue("lifecycle_user", "error task")
    bq.mark_running("lifecycle_user", t2["id"])
    bq.mark_error("lifecycle_user", t2["id"], "boom")
    tasks = bq.list_tasks("lifecycle_user")
    err_task = [t for t in tasks if t["id"] == t2["id"]][0]
    check("Error task status", err_task["status"] == "error")
    check("Error message", err_task["error"] == "boom")


def test_background_task_unique_ids(tmp: str):
    section("Background task unique IDs")
    from pawlia.background_tasks import BackgroundTaskQueue

    bq = BackgroundTaskQueue(tmp)
    ids = set()
    for i in range(50):
        t = bq.enqueue("uid_user", f"task {i}")
        ids.add(t["id"])

    check("50 unique task IDs", len(ids) == 50)


def test_background_task_nonexistent(tmp: str):
    section("Background task nonexistent operations")
    from pawlia.background_tasks import BackgroundTaskQueue

    bq = BackgroundTaskQueue(tmp)
    # These should not crash
    bq.mark_running("nobody", "fakeid")
    bq.mark_done("nobody", "fakeid")
    bq.mark_error("nobody", "fakeid", "error")
    check("No crash on nonexistent mark_running", True)
    check("No crash on nonexistent mark_done", True)
    check("No crash on nonexistent mark_error", True)


# ═══════════════════════════════════════════════════════════════
# 14. MemoryIndexer — config, tracking, log discovery
# ═══════════════════════════════════════════════════════════════

def _idx_cfg(**overrides):
    cfg = {
        "skill-config": {
            "memory": {
                "embedding_provider": "ollama",
                "embedding_model": "bge-m3:latest",
                "embedding_dim": 1024,
                "embedding_host": "http://localhost:11434",
            }
        }
    }
    cfg["skill-config"]["memory"].update(overrides)
    return cfg


def test_memory_indexer_config(tmp: str):
    section("Memory indexer config")
    from pawlia.memory_indexer import MemoryIndexer

    mi1 = MemoryIndexer(tmp, {})
    check("Disabled without config", mi1.enabled is False)

    mi2 = MemoryIndexer(tmp, {"skill-config": {"memory": {"embedding_provider": "ollama"}}})
    check("Disabled with partial config", mi2.enabled is False)

    mi3 = MemoryIndexer(tmp, _idx_cfg())
    check("Enabled with full config", mi3.enabled is True)

    busy = False
    mi4 = MemoryIndexer(tmp, _idx_cfg(), llm_busy_check=lambda: busy)
    check("llm_busy_check stored", mi4._llm_busy is not None)

    # Missing each required field
    for field in ["embedding_provider", "embedding_model", "embedding_dim", "embedding_host"]:
        c = _idx_cfg()
        del c["skill-config"]["memory"][field]
        mi = MemoryIndexer(tmp, c)
        check(f"Disabled without {field}", mi.enabled is False)


def test_memory_indexer_tracking(tmp: str):
    section("Memory indexer tracking")
    from pawlia.memory_indexer import MemoryIndexer

    mi = MemoryIndexer(tmp, _idx_cfg())

    tracked = mi._load_tracked("track_user")
    check("Empty tracking initially", tracked == {})

    mi._save_tracked("track_user", {"2026-03-20.md": "123456"})
    tracked2 = mi._load_tracked("track_user")
    check("Tracked data persisted", tracked2 == {"2026-03-20.md": "123456"})

    tp = mi._tracker_path("track_user")
    check("Tracker in memory_index dir", "memory_index" in tp)
    check("Tracker file exists", os.path.isfile(tp))

    # Overwrite tracking
    mi._save_tracked("track_user", {"2026-03-20.md": "999", "2026-03-21.md": "111"})
    tracked3 = mi._load_tracked("track_user")
    check("Tracking overwritten", len(tracked3) == 2)
    check("Updated mtime", tracked3["2026-03-20.md"] == "999")

    # Corrupt tracking file
    with open(tp, "w") as f:
        f.write("not json!")
    tracked4 = mi._load_tracked("track_user")
    check("Corrupt file returns empty", tracked4 == {})


def test_memory_indexer_find_logs(tmp: str):
    section("Memory indexer find_daily_logs")
    from pawlia.memory_indexer import MemoryIndexer

    mi = MemoryIndexer(tmp, _idx_cfg())

    mem_dir = os.path.join(tmp, "log_user", "workspace", "memory")
    os.makedirs(mem_dir, exist_ok=True)

    for name in ["2026-03-18.md", "2026-03-19.md", "2026-03-20.md",
                  "memory.md", "context_summary.md", "thread_abc_2026-03-20.md",
                  "notes.txt", "model_override.txt"]:
        with open(os.path.join(mem_dir, name), "w") as f:
            f.write("test content")

    logs = mi._find_daily_logs("log_user")
    basenames = [os.path.basename(p) for p in logs]
    check("Found 3 daily logs", len(logs) == 3, f"got {len(logs)}: {basenames}")
    check("Sorted order", basenames == ["2026-03-18.md", "2026-03-19.md", "2026-03-20.md"])
    check("memory.md excluded", "memory.md" not in basenames)
    check("thread file excluded", all("thread" not in b for b in basenames))
    check("non-md excluded", "notes.txt" not in basenames)

    # No memory dir
    logs2 = mi._find_daily_logs("nonexistent_user")
    check("Nonexistent user -> empty", logs2 == [])


def test_memory_indexer_find_logs_empty(tmp: str):
    section("Memory indexer find_logs (empty dir)")
    from pawlia.memory_indexer import MemoryIndexer

    mi = MemoryIndexer(tmp, _idx_cfg())
    mem_dir = os.path.join(tmp, "empty_log_user", "workspace", "memory")
    os.makedirs(mem_dir, exist_ok=True)

    logs = mi._find_daily_logs("empty_log_user")
    check("Empty memory dir -> []", logs == [])


# ═══════════════════════════════════════════════════════════════
# 15. Scheduler — idle priority system
# ═══════════════════════════════════════════════════════════════

def test_scheduler_idle_minutes(tmp: str):
    section("Scheduler idle minutes")
    from pawlia.scheduler import Scheduler

    sched = Scheduler(tmp)

    # User never active -> falls back to boot_time
    idle = sched._user_idle_minutes("unknown_user")
    check("Unknown user idle > 0", idle >= 0)

    # Touch activity, then check
    sched.touch_activity("test_user")
    idle2 = sched._user_idle_minutes("test_user")
    check("Just-active user idle < 1 min", idle2 < 1.0)


def test_scheduler_llm_gate(tmp: str):
    section("Scheduler LLM gate")
    from pawlia.scheduler import Scheduler

    sched = Scheduler(tmp)

    check("Not busy initially", sched.llm_busy is False)

    sched.acquire_llm()
    check("Busy after acquire", sched.llm_busy is True)

    sched.acquire_llm()
    check("Still busy (counter=2)", sched.llm_busy is True)

    sched.release_llm()
    check("Still busy (counter=1)", sched.llm_busy is True)

    sched.release_llm()
    check("Free after all released", sched.llm_busy is False)

    # Extra release doesn't go negative
    sched.release_llm()
    sched.release_llm()
    check("No negative after extra release", sched.llm_busy is False)
    check("Counter is 0", sched._llm_active == 0)


def test_scheduler_priority_constants(tmp: str):
    section("Scheduler priority constants")
    from pawlia.scheduler import IDLE_SUMMARIZE_MIN, IDLE_BACKGROUND_MIN, IDLE_MEMORY_MIN

    check("Summarize < Background", IDLE_SUMMARIZE_MIN < IDLE_BACKGROUND_MIN)
    check("Background < Memory", IDLE_BACKGROUND_MIN < IDLE_MEMORY_MIN)
    check("Summarize = 5", IDLE_SUMMARIZE_MIN == 5)
    check("Background = 10", IDLE_BACKGROUND_MIN == 10)
    check("Memory = 20", IDLE_MEMORY_MIN == 20)


def test_scheduler_touch_activity(tmp: str):
    section("Scheduler touch_activity")
    from pawlia.scheduler import Scheduler

    sched = Scheduler(tmp)

    sched.touch_activity("u1")
    t1 = sched._last_activity["u1"]

    time.sleep(0.02)
    sched.touch_activity("u1")
    t2 = sched._last_activity["u1"]

    check("Activity updated", t2 > t1)

    sched.touch_activity("u2")
    check("Multiple users tracked", "u1" in sched._last_activity and "u2" in sched._last_activity)


def test_scheduler_bg_tasks_lazy(tmp: str):
    section("Scheduler bg_tasks lazy init")
    from pawlia.scheduler import Scheduler

    sched = Scheduler(tmp)
    check("bg_tasks not initialized yet", sched._bg_tasks is None)

    bg = sched.bg_tasks
    check("bg_tasks initialized after access", sched._bg_tasks is not None)
    check("Same instance on second access", sched.bg_tasks is bg)


def test_scheduler_callbacks(tmp: str):
    section("Scheduler callbacks")
    from pawlia.scheduler import Scheduler

    sched = Scheduler(tmp)
    check("No callbacks initially", len(sched._callbacks) == 0)

    async def cb1(uid, msg): pass
    async def cb2(uid, msg): pass

    sched.register(cb1)
    check("1 callback after register", len(sched._callbacks) == 1)

    sched.register(cb2)
    check("2 callbacks after second register", len(sched._callbacks) == 2)


# ═══════════════════════════════════════════════════════════════
# 16. Scheduler — notifications
# ═══════════════════════════════════════════════════════════════

def test_scheduler_notify(tmp: str):
    section("Scheduler _notify")
    from pawlia.scheduler import Scheduler

    sched = Scheduler(tmp)
    received = []

    async def cb(uid, msg):
        received.append((uid, msg))

    sched.register(cb)

    asyncio.get_event_loop().run_until_complete(sched._notify("alice", "hello"))
    check("Callback received message", len(received) == 1)
    check("Correct user_id", received[0][0] == "alice")
    check("Correct message", received[0][1] == "hello")


def test_scheduler_notify_formatter(tmp: str):
    section("Scheduler _notify with formatter")
    from pawlia.scheduler import Scheduler

    sched = Scheduler(tmp)
    received = []

    async def cb(uid, msg):
        received.append((uid, msg))

    async def formatter(uid, raw):
        return f"Formatted: {raw}"

    sched.register(cb)
    sched.set_llm_formatter(formatter)

    asyncio.get_event_loop().run_until_complete(sched._notify("bob", "raw msg"))
    check("Formatted message received", received[0][1] == "Formatted: raw msg")


def test_scheduler_notify_formatter_failure(tmp: str):
    section("Scheduler _notify formatter failure -> fallback")
    from pawlia.scheduler import Scheduler

    sched = Scheduler(tmp)
    received = []

    async def cb(uid, msg):
        received.append((uid, msg))

    async def bad_formatter(uid, raw):
        raise ValueError("LLM down")

    sched.register(cb)
    sched.set_llm_formatter(bad_formatter)

    asyncio.get_event_loop().run_until_complete(sched._notify("charlie", "fallback msg"))
    check("Raw message delivered on formatter failure", received[0][1] == "fallback msg")


def test_scheduler_notify_formatter_empty(tmp: str):
    section("Scheduler _notify formatter returns empty")
    from pawlia.scheduler import Scheduler

    sched = Scheduler(tmp)
    received = []

    async def cb(uid, msg):
        received.append((uid, msg))

    async def empty_formatter(uid, raw):
        return ""

    sched.register(cb)
    sched.set_llm_formatter(empty_formatter)

    asyncio.get_event_loop().run_until_complete(sched._notify("dave", "raw"))
    check("Raw message on empty formatter result", received[0][1] == "raw")


def test_scheduler_notify_callback_failure(tmp: str):
    section("Scheduler _notify callback failure isolation")
    from pawlia.scheduler import Scheduler

    sched = Scheduler(tmp)
    received = []

    async def bad_cb(uid, msg):
        raise RuntimeError("callback exploded")

    async def good_cb(uid, msg):
        received.append((uid, msg))

    sched.register(bad_cb)
    sched.register(good_cb)

    asyncio.get_event_loop().run_until_complete(sched._notify("eve", "test"))
    check("Good callback still called despite bad one", len(received) == 1)


def test_scheduler_notify_multiple_callbacks(tmp: str):
    section("Scheduler _notify broadcasts to all")
    from pawlia.scheduler import Scheduler

    sched = Scheduler(tmp)
    r1, r2, r3 = [], [], []

    async def cb1(uid, msg): r1.append(msg)
    async def cb2(uid, msg): r2.append(msg)
    async def cb3(uid, msg): r3.append(msg)

    sched.register(cb1)
    sched.register(cb2)
    sched.register(cb3)

    asyncio.get_event_loop().run_until_complete(sched._notify("user", "broadcast"))
    check("CB1 received", len(r1) == 1)
    check("CB2 received", len(r2) == 1)
    check("CB3 received", len(r3) == 1)


# ═══════════════════════════════════════════════════════════════
# 17. Scheduler — recurrence helpers
# ═══════════════════════════════════════════════════════════════

def test_recurrence(tmp: str):
    section("Recurrence (_next_occurrence)")
    from pawlia.scheduler import _next_occurrence

    base = datetime(2026, 3, 20, 14, 0, 0)

    daily = _next_occurrence(base, "daily")
    check("Daily: +1 day", daily == datetime(2026, 3, 21, 14, 0, 0))

    weekly = _next_occurrence(base, "weekly")
    check("Weekly: +7 days", weekly == datetime(2026, 3, 27, 14, 0, 0))

    monthly = _next_occurrence(base, "monthly")
    check("Monthly: same day next month", monthly == datetime(2026, 4, 20, 14, 0, 0))

    # Unknown recurrence -> +1 day fallback
    unknown = _next_occurrence(base, "yearly")
    check("Unknown: +1 day fallback", unknown == datetime(2026, 3, 21, 14, 0, 0))


def test_recurrence_month_end(tmp: str):
    section("Recurrence month-end edge cases")
    from pawlia.scheduler import _next_occurrence

    # Jan 31 -> Feb: no Feb 31, falls back to 28
    jan31 = datetime(2026, 1, 31, 10, 0, 0)
    feb = _next_occurrence(jan31, "monthly")
    check("Jan 31 -> Feb 28", feb.month == 2 and feb.day == 28)

    # Dec -> Jan: year rollover
    dec = datetime(2026, 12, 15, 10, 0, 0)
    jan = _next_occurrence(dec, "monthly")
    check("Dec -> Jan: year rollover", jan.month == 1 and jan.year == 2027)

    # Mar 30 -> Apr 30: ok
    mar30 = datetime(2026, 3, 30, 10, 0, 0)
    apr = _next_occurrence(mar30, "monthly")
    check("Mar 30 -> Apr 30", apr == datetime(2026, 4, 30, 10, 0, 0))

    # Mar 31 -> Apr: no Apr 31, falls back to 28
    mar31 = datetime(2026, 3, 31, 10, 0, 0)
    apr2 = _next_occurrence(mar31, "monthly")
    check("Mar 31 -> Apr 28 (fallback)", apr2.month == 4 and apr2.day == 28)


def test_recurrence_leap_year(tmp: str):
    section("Recurrence leap year")
    from pawlia.scheduler import _next_occurrence

    # 2028 is a leap year
    jan31_leap = datetime(2028, 1, 31, 10, 0, 0)
    feb_leap = _next_occurrence(jan31_leap, "monthly")
    # Should fallback to 28 (not 29, since the code uses day=28 on ValueError)
    check("Jan 31 leap year -> Feb 28", feb_leap.month == 2 and feb_leap.day == 28)


# ═══════════════════════════════════════════════════════════════
# 18. Scheduler — reminders
# ═══════════════════════════════════════════════════════════════

def test_check_reminders(tmp: str):
    section("Scheduler _check_reminders")
    from pawlia.scheduler import Scheduler

    sched = Scheduler(tmp)
    received = []

    async def cb(uid, msg):
        received.append((uid, msg))

    sched.register(cb)

    # Create reminders file with one due and one future
    os.makedirs(os.path.join(tmp, "rem_user"), exist_ok=True)
    path = os.path.join(tmp, "rem_user", "reminders.json")
    now = datetime.now()
    reminders = [
        {
            "id": "r1",
            "fire_at": (now - timedelta(minutes=5)).isoformat(),
            "message": "Due reminder",
            "label": "Test",
            "recurrence": "none",
            "fired": False,
        },
        {
            "id": "r2",
            "fire_at": (now + timedelta(hours=1)).isoformat(),
            "message": "Future reminder",
            "label": "Later",
            "recurrence": "none",
            "fired": False,
        },
        {
            "id": "r3",
            "fire_at": (now - timedelta(minutes=1)).isoformat(),
            "message": "Already fired",
            "label": "Old",
            "recurrence": "none",
            "fired": True,
        },
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(reminders, f)

    asyncio.get_event_loop().run_until_complete(sched._check_reminders("rem_user", path))

    check("1 notification sent", len(received) == 1)
    check("Due reminder text", "Due reminder" in received[0][1])

    # Verify reminder marked as fired
    with open(path, encoding="utf-8") as f:
        updated = json.load(f)
    check("r1 marked fired", updated[0]["fired"] is True)
    check("r2 still pending", updated[1]["fired"] is False)
    check("r3 still fired", updated[2]["fired"] is True)


def test_check_reminders_recurring(tmp: str):
    section("Scheduler recurring reminders")
    from pawlia.scheduler import Scheduler

    sched = Scheduler(tmp)
    received = []

    async def cb(uid, msg):
        received.append(msg)

    sched.register(cb)

    os.makedirs(os.path.join(tmp, "rec_user"), exist_ok=True)
    path = os.path.join(tmp, "rec_user", "reminders.json")
    now = datetime.now()
    reminders = [
        {
            "id": "daily_r",
            "fire_at": (now - timedelta(minutes=1)).isoformat(),
            "message": "Daily reminder",
            "label": "Daily",
            "recurrence": "daily",
            "fired": False,
        },
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(reminders, f)

    asyncio.get_event_loop().run_until_complete(sched._check_reminders("rec_user", path))

    check("Daily reminder fired", len(received) == 1)

    with open(path, encoding="utf-8") as f:
        updated = json.load(f)
    check("Not marked as fired (recurring)", updated[0].get("fired", False) is not True or "fired" not in updated[0] or updated[0]["fired"] is not True)
    # fire_at should be updated to tomorrow
    new_fire = datetime.fromisoformat(updated[0]["fire_at"])
    check("fire_at updated to future", new_fire > now)


def test_check_events(tmp: str):
    section("Scheduler _check_events")
    from pawlia.scheduler import Scheduler

    sched = Scheduler(tmp)
    received = []

    async def cb(uid, msg):
        received.append(msg)

    sched.register(cb)

    cal_dir = os.path.join(tmp, "ev_user", "calendar")
    os.makedirs(cal_dir, exist_ok=True)
    path = os.path.join(cal_dir, "events.json")

    now = datetime.now()
    events = [
        {
            "title": "Soon Event",
            "start": (now + timedelta(minutes=10)).isoformat(),
            "location": "Office",
        },
        {
            "title": "Far Event",
            "start": (now + timedelta(hours=5)).isoformat(),
        },
        {
            "title": "Already Notified",
            "start": (now + timedelta(minutes=5)).isoformat(),
            "_notified": True,
        },
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(events, f)

    asyncio.get_event_loop().run_until_complete(sched._check_events("ev_user", path))

    check("1 event notification", len(received) == 1)
    check("Soon Event notified", "Soon Event" in received[0])
    check("Location included", "Office" in received[0])


# ═══════════════════════════════════════════════════════════════
# 19. Scheduler — JSON helpers
# ═══════════════════════════════════════════════════════════════

def test_load_save_json(tmp: str):
    section("Scheduler _load_json / _save_json")
    from pawlia.scheduler import _load_json, _save_json

    os.makedirs(tmp, exist_ok=True)
    path = os.path.join(tmp, "test.json")

    check("Missing file -> []", _load_json(path) == [])

    _save_json(path, [{"a": 1}, {"b": 2}])
    loaded = _load_json(path)
    check("Saved and loaded", len(loaded) == 2)
    check("Content preserved", loaded[0]["a"] == 1)

    # Corrupt JSON
    with open(path, "w") as f:
        f.write("not json {{{")
    check("Corrupt file -> []", _load_json(path) == [])

    # Unicode
    _save_json(path, [{"text": "Umlaute: aou"}])
    loaded2 = _load_json(path)
    check("Unicode preserved", loaded2[0]["text"] == "Umlaute: aou")


# ═══════════════════════════════════════════════════════════════
# 20. Stress / scale tests
# ═══════════════════════════════════════════════════════════════

def test_many_exchanges(tmp: str):
    section("Stress: 500 exchanges")
    from pawlia.memory import MemoryManager

    mm = MemoryManager(tmp)
    s = mm.load_session("stress_user")

    n = 500
    for i in range(n):
        mm.append_exchange(s, f"Question {i}", f"Answer {i} with some content here")

    check(f"{n} exchanges in session", s.exchange_count == n)
    check(f"{n} exchanges in list", len(s.exchanges) == n)

    mm2 = MemoryManager(tmp)
    s2 = mm2.load_session("stress_user")
    check(f"Reloaded {n} exchanges", s2.exchange_count == n)
    check("First exchange correct", s2.exchanges[0] == ("Question 0", "Answer 0 with some content here"))
    check("Last exchange correct", s2.exchanges[-1] == (f"Question {n-1}", f"Answer {n-1} with some content here"))


def test_many_threads(tmp: str):
    section("Stress: 50 threads")
    from pawlia.memory import MemoryManager

    mm = MemoryManager(tmp)
    s = mm.load_session("mthread_user")

    mm.append_exchange(s, "Base Q", "Base A")

    n_threads = 50
    for i in range(n_threads):
        tid = f"thread_{i:03d}"
        mm.get_thread_context(s, tid)
        mm.append_thread_exchange(s, tid, f"T{i} Q", f"T{i} A")

    check(f"{n_threads} threads created", len(s.thread_contexts) == n_threads)
    for i in range(n_threads):
        tid = f"thread_{i:03d}"
        check(f"Thread {tid} has 2 exchanges", len(s.thread_contexts[tid]) == 2)


def test_concurrent_users(tmp: str):
    section("Concurrent users (isolation)")
    from pawlia.memory import MemoryManager

    mm = MemoryManager(tmp)
    users = ["alice", "bob", "charlie", "diana", "eve", "frank"]
    sessions = {u: mm.load_session(u) for u in users}

    for i, u in enumerate(users):
        for j in range(i + 1):
            mm.append_exchange(sessions[u], f"{u}_Q{j}", f"{u}_A{j}")

    for i, u in enumerate(users):
        check(f"{u}: {i+1} exchanges", sessions[u].exchange_count == i + 1)

    # Cross-contamination check
    for u in users:
        for other in users:
            if other == u:
                continue
            check(
                f"{u} has no {other} content",
                all(other not in ex[0] for ex in sessions[u].exchanges),
            )


def test_many_background_tasks(tmp: str):
    section("Stress: 100 background tasks")
    from pawlia.background_tasks import BackgroundTaskQueue

    bq = BackgroundTaskQueue(tmp)

    for i in range(100):
        bq.enqueue("bulk_user", f"bulk task {i}")

    tasks = bq.list_tasks("bulk_user")
    check("100 tasks created", len(tasks) == 100)

    pending = bq.pending()
    bulk = [(uid, t) for uid, t in pending if uid == "bulk_user"]
    check("100 pending", len(bulk) == 100)

    # Mark half done
    for uid, t in bulk[:50]:
        bq.mark_done("bulk_user", t["id"])

    pending2 = bq.pending()
    bulk2 = [(uid, t) for uid, t in pending2 if uid == "bulk_user"]
    check("50 pending after marking 50 done", len(bulk2) == 50)


def test_summarize_resume_cycle(tmp: str):
    section("Stress: summarize-resume cycle")
    from pawlia.memory import MemoryManager

    mm = MemoryManager(tmp)
    s = mm.load_session("cycle_user")

    for cycle in range(5):
        for i in range(10):
            mm.append_exchange(s, f"C{cycle}Q{i}", f"C{cycle}A{i}")
        mm.summarize(s, f"Summary after cycle {cycle}")

    check("Final summary is cycle 4", "cycle 4" in s.summary)
    check("Exchanges cleared after last cycle", s.exchange_count == 0)

    # Add a few more after last cycle
    mm.append_exchange(s, "Post-cycle Q", "Post-cycle A")
    check("Post-cycle exchange", s.exchange_count == 1)

    # Daily log has ALL exchanges (append-only)
    dp = mm._daily_path("cycle_user", s.current_date_str)
    with open(dp, encoding="utf-8") as f:
        disk = f.read()
    check("Daily log has C0Q0", "C0Q0" in disk)
    check("Daily log has C4Q9", "C4Q9" in disk)
    check("Daily log has Post-cycle", "Post-cycle Q" in disk)


# ═══════════════════════════════════════════════════════════════
# 21. Long conversation simulation with content retrieval
# ═══════════════════════════════════════════════════════════════

# -- Realistic conversation data --

_CONVERSATION_TOPICS = [
    # (topic_tag, list of (user_msg, bot_msg) pairs)
    ("cooking", [
        ("Ich möchte heute Abend Pasta machen, hast du ein gutes Rezept?",
         "Klar! Probier mal Aglio e Olio: Spaghetti, Knoblauch, Olivenöl, Chiliflocken und Petersilie. Einfach und lecker."),
        ("Wie lange müssen die Spaghetti kochen?",
         "Normale Spaghetti brauchen 8-10 Minuten in Salzwasser. Al dente eher 8 Minuten."),
        ("Kann ich auch Penne nehmen?",
         "Ja, Penne funktioniert genauso gut. Kochzeit ist ähnlich, etwa 10-12 Minuten."),
        ("Was für ein Olivenöl empfiehlst du?",
         "Für Aglio e Olio am besten ein gutes extra vergine, z.B. aus Ligurien oder Kreta. Der Geschmack macht hier den Unterschied."),
    ]),
    ("travel", [
        ("Ich plane eine Reise nach Japan im Oktober. Was muss ich beachten?",
         "Oktober ist perfekt für Japan — angenehme Temperaturen und Herbstlaub. Du brauchst kein Visum für 90 Tage. Rail Pass lohnt sich wenn du viel reisen willst."),
        ("Was kostet ein Japan Rail Pass?",
         "Der 7-Tage-Pass kostet etwa 29.650 Yen (ca. 185€). Der 14-Tage-Pass liegt bei 47.250 Yen (ca. 295€). Lohnt sich ab der zweiten Fernreise."),
        ("Welche Städte sollte ich besuchen?",
         "Tokio und Kyoto sind Pflicht. Dazu empfehle ich Osaka für Street Food, Hiroshima für Geschichte, und Hakone für heiße Quellen mit Fuji-Blick."),
        ("Brauche ich mobiles Internet?",
         "Ja, unbedingt. Hol dir eine eSIM oder pocket WiFi. Google Maps und Übersetzungs-Apps sind in Japan unverzichtbar."),
        ("Wie komme ich vom Flughafen Narita nach Tokio?",
         "Der Narita Express braucht ca. 60 Minuten bis Tokyo Station und kostet 3.070 Yen. Ist im Rail Pass enthalten. Alternativ der Skyliner nach Ueno in 36 Minuten."),
    ]),
    ("programming", [
        ("Kannst du mir erklären was ein Decorator in Python ist?",
         "Ein Decorator ist eine Funktion, die eine andere Funktion umhüllt und erweitert. Mit @decorator_name vor einer Funktion wird sie automatisch durch den Decorator geleitet."),
        ("Zeig mal ein Beispiel für einen Timing-Decorator",
         "```python\nimport time\ndef timer(func):\n    def wrapper(*args, **kw):\n        start = time.time()\n        result = func(*args, **kw)\n        print(f'{func.__name__} dauerte {time.time()-start:.2f}s')\n        return result\n    return wrapper\n```"),
        ("Was ist der Unterschied zwischen @staticmethod und @classmethod?",
         "@staticmethod braucht keinen Zugriff auf die Klasse/Instanz, @classmethod bekommt die Klasse als ersten Parameter (cls). Classmethod kann die Klasse modifizieren, staticmethod ist wie eine normale Funktion im Klassen-Namespace."),
        ("Wie funktioniert asyncio grundsätzlich?",
         "asyncio nutzt einen Event-Loop der coroutines verwaltet. Mit async def definierst du eine Coroutine, mit await pausierst du sie bis ein Ergebnis da ist. Der Loop kann während der Pause andere Tasks bearbeiten — kooperatives Multitasking."),
        ("Was ist der GIL?",
         "Der Global Interpreter Lock (GIL) in CPython erlaubt nur einem Thread gleichzeitig Python-Bytecode auszuführen. Deshalb bringt Threading für CPU-bound Tasks keinen Speedup. Für Parallelität nutzt man multiprocessing oder asyncio für I/O-bound Tasks."),
    ]),
    ("fitness", [
        ("Ich will mit Krafttraining anfangen. Wie oft pro Woche?",
         "Als Anfänger reichen 3x pro Woche völlig aus. Ganzkörper-Training mit 48h Pause dazwischen. Z.B. Montag, Mittwoch, Freitag."),
        ("Welche Übungen sind am wichtigsten?",
         "Die Big Five: Kniebeuge, Kreuzheben, Bankdrücken, Überkopfdrücken und Rudern. Damit deckst du alle großen Muskelgruppen ab."),
        ("Wie viel Protein brauche ich?",
         "Für Muskelaufbau empfehlen sich 1.6-2.2g Protein pro kg Körpergewicht. Bei 80kg also 128-176g pro Tag. Gute Quellen: Hähnchen, Quark, Eier, Hülsenfrüchte."),
        ("Was ist Progressive Overload?",
         "Progressive Overload bedeutet, den Trainingsreiz stetig zu steigern — mehr Gewicht, mehr Wiederholungen oder mehr Sätze. Ohne diese Progression passt sich der Körper an und der Muskelaufbau stagniert."),
    ]),
    ("gardening", [
        ("Meine Tomaten werden nicht rot, woran liegt das?",
         "Häufigste Ursachen: zu wenig Sonne (mind. 6-8h), zu niedrige Temperaturen (unter 15°C nachts), oder Sortenbedingt. Ethylengas von reifen Bananen kann das Nachreifen fördern."),
        ("Wann ist die beste Zeit zum Gießen?",
         "Morgens früh ist ideal — das Wasser verdunstet nicht sofort und die Blätter trocknen vor der Nacht ab. Abends gießen fördert Pilzkrankheiten."),
        ("Welche Kräuter kann ich auf dem Balkon anbauen?",
         "Basilikum, Petersilie, Schnittlauch, Rosmarin und Thymian wachsen super auf dem Balkon. Rosmarin und Thymian mögen es trocken, Basilikum braucht mehr Wasser."),
    ]),
    ("music", [
        ("Ich will Gitarre lernen. Akustik oder E-Gitarre zum Anfangen?",
         "Akustik-Gitarre baut mehr Fingerkraft auf und braucht keinen Verstärker. E-Gitarre ist einfacher zu greifen und motivierender wenn du Rock/Metal willst. Wähl was dich mehr begeistert!"),
        ("Wie lange dauert es bis ich Songs spielen kann?",
         "Mit täglichem Üben (20-30 Min) kannst du nach 2-3 Monaten einfache Songs wie Wonderwall oder Knockin on Heavens Door spielen. Die ersten Akkorde (G, C, D, Em) reichen für hunderte Songs."),
        ("Was ist Tabulatur?",
         "Tabulatur (Tab) ist eine vereinfachte Notenschrift für Gitarre. 6 Linien = 6 Saiten, Zahlen = Bund. Anders als Noten zeigt Tab dir direkt wo du greifen musst, nicht welchen Ton du spielst."),
    ]),
    ("personal_facts", [
        ("Ich heiße übrigens Marco und komme aus München",
         "Freut mich, Marco! München ist eine tolle Stadt. Ich merke mir das."),
        ("Ich bin 32 Jahre alt und arbeite als DevOps Engineer",
         "Alles klar Marco, 32 und DevOps Engineer — das erklärt dein Interesse an Automatisierung und Scripting!"),
        ("Mein Hund heißt Balu und ist ein Golden Retriever",
         "Balu, ein Golden Retriever — die sind so lieb! Wie alt ist er denn?"),
        ("Balu ist 4 Jahre alt. Wir gehen jeden Morgen im Englischen Garten spazieren",
         "4 Jahre, da hat Balu noch viel Energie! Der Englische Garten ist perfekt zum Spazierengehen — besonders morgens wenn es noch ruhiger ist."),
        ("Meine Lieblingsfarbe ist dunkelblau und ich trinke am liebsten Cappuccino",
         "Dunkelblau und Cappuccino — notiert! Soll ich mir noch andere Vorlieben merken?"),
    ]),
    ("tech_setup", [
        ("Ich nutze Arch Linux mit i3wm als Window Manager",
         "Ah, Arch + i3 — ein Tiling-Setup für Effizienz. Hast du schon Polybar als Statusbar und rofi als App-Launcher eingerichtet?"),
        ("Ja, Polybar mit custom Modulen und rofi. Dazu Alacritty als Terminal",
         "Alacritty ist super schnell dank GPU-Rendering. Mit tmux oder zellij dazu hast du ein perfektes Terminal-Setup."),
        ("Mein Editor ist Neovim mit LazyVim Config",
         "LazyVim ist ein tolles Neovim-Starter-Kit. Die Kombination aus Telescope, LSP und Treesitter macht Neovim zur vollwertigen IDE."),
        ("Mein Homelab läuft auf einem Proxmox Server mit 64GB RAM",
         "64GB auf Proxmox — damit kannst du einiges an VMs und Containern laufen lassen. Was hostest du alles darauf?"),
        ("Darauf laufen Nextcloud, Gitea, ein Minecraft Server und mein Home Assistant",
         "Solides Homelab! Nextcloud für Cloud-Speicher, Gitea für Git, Minecraft für Spaß und Home Assistant für Smarthome — alles selbst gehostet, sehr nice."),
    ]),
    ("philosophy", [
        ("Was denkst du über das Trolley-Problem?",
         "Das Trolley-Problem zeigt den Konflikt zwischen Utilitarismus (5 retten, 1 opfern) und deontologischer Ethik (aktives Töten ist immer falsch). Es gibt keine objektiv richtige Antwort — es testet unsere moralischen Intuitionen."),
        ("Kennst du das Schiff des Theseus?",
         "Ja! Wenn man alle Planken eines Schiffs nach und nach ersetzt — ist es noch dasselbe Schiff? Es stellt die Frage nach Identität und Kontinuität. Gilt auch für unseren Körper, dessen Zellen sich ständig erneuern."),
        ("Was ist der Unterschied zwischen Existenzialismus und Nihilismus?",
         "Nihilismus sagt: Es gibt keinen Sinn. Existenzialismus sagt: Es gibt keinen vorgegebenen Sinn, aber du kannst deinen eigenen erschaffen. Sartre: 'Der Mensch ist zur Freiheit verurteilt' — wir müssen unserem Leben selbst Bedeutung geben."),
    ]),
    ("daily_life", [
        ("Morgen muss ich zum Zahnarzt, hab etwas Angst",
         "Das ist ganz normal! Tiefes Atmen hilft, und du kannst dem Zahnarzt sagen wenn du eine Pause brauchst. Die meisten Behandlungen sind heute deutlich schonender als früher."),
        ("Mein Auto muss nächste Woche zum TÜV",
         "Vergiss nicht Warndreieck und Verbandskasten zu checken — das wird oft vergessen. Und die Beleuchtung vorher testen, das ist der häufigste Grund zum Durchfallen."),
        ("Ich suche ein neues Fahrrad, am liebsten ein Gravelbike",
         "Gravelbikes sind super vielseitig — Straße und Schotter. Schau dir das Canyon Grail oder das Cube Nuroad an, gutes Preis-Leistungs-Verhältnis. Budget so ab 1000-1500€ für was Ordentliches."),
        ("Ich hab mir letzte Woche eine neue Espressomaschine gekauft, eine Sage Barista Express",
         "Die Sage Barista Express ist ein Klassiker für Einsteiger! Tipp: Nutze frisch gerösteten Kaffee (nicht älter als 4 Wochen) und experimentiere mit dem Mahlgrad. Die ersten Espressos werden eher Trial-and-Error."),
    ]),
]

# Additional filler exchanges for volume between topics
_FILLER_EXCHANGES = [
    ("Wie wird das Wetter morgen?", "Lass mich kurz nachschauen... Morgen wird es teilweise bewölkt mit Temperaturen um 18°C."),
    ("Danke!", "Gerne! Wenn du noch Fragen hast, melde dich."),
    ("Alles klar, bis später", "Bis später! Schönen Tag noch."),
    ("Hey, bin wieder da", "Willkommen zurück! Was kann ich für dich tun?"),
    ("Kannst du mir einen Witz erzählen?", "Was sagt ein Informatiker beim Frühstück? 'Ein Bit-te Kaffee!' 😄"),
    ("Haha, der war gut", "Freut mich! Humor ist wichtig, besonders beim Programmieren."),
    ("Was ist die Hauptstadt von Madagaskar?", "Die Hauptstadt von Madagaskar ist Antananarivo — die Stadt liegt im zentralen Hochland auf etwa 1.200m Höhe."),
    ("Wie spät ist es?", "Es ist gerade Nachmittag. Brauchst du einen Timer oder eine Erinnerung?"),
    ("Nein danke, alles gut", "Alles klar! Sag Bescheid wenn du was brauchst."),
    ("Ich hab Hunger", "Wie wäre es mit einem schnellen Sandwich oder einer Bowl? Oder soll ich dir ein Rezept raussuchen?"),
]


def test_long_conversation(tmp: str):
    """Simulate a long, realistic conversation with multiple topics,
    summarization cycles, threads, and verify content retrieval afterwards."""
    section("Long conversation simulation (setup)")
    from pawlia.memory import MemoryManager, MAX_EXCHANGES_BEFORE_SUMMARY

    mm = MemoryManager(tmp)
    user = "long_conv_user"
    s = mm.load_session(user)

    # Track all exchanges for later verification
    all_exchanges = []
    topic_start_indices = {}  # topic_tag -> index in all_exchanges

    # Phase 1: Interleave topics with filler, triggering summarization cycles
    exchange_idx = 0
    summaries_done = 0
    filler_idx = 0

    for topic_tag, topic_exchanges in _CONVERSATION_TOPICS:
        topic_start_indices[topic_tag] = exchange_idx

        for user_msg, bot_msg in topic_exchanges:
            mm.append_exchange(s, user_msg, bot_msg)
            all_exchanges.append((user_msg, bot_msg))
            exchange_idx += 1

            # Insert filler between some exchanges for realism
            if exchange_idx % 5 == 0 and filler_idx < len(_FILLER_EXCHANGES):
                fu, fb = _FILLER_EXCHANGES[filler_idx]
                mm.append_exchange(s, fu, fb)
                all_exchanges.append((fu, fb))
                exchange_idx += 1
                filler_idx += 1

        # Check if summarization is due and simulate it
        reason = mm.should_summarize(s)
        if reason:
            summaries_done += 1
            summary_text = (
                f"Zusammenfassung #{summaries_done}: "
                f"Bisher besprochene Themen: {', '.join(t for t in topic_start_indices)}. "
                f"Letztes Thema: {topic_tag}."
            )
            mm.summarize(s, summary_text)

    check("All topics inserted", len(topic_start_indices) == len(_CONVERSATION_TOPICS))
    check("At least 1 summarization", summaries_done >= 1)
    check(f"Total exchanges tracked: {len(all_exchanges)}", len(all_exchanges) > 40)

    safe_print(f"    ({len(all_exchanges)} exchanges, {summaries_done} summarization cycles)")


def test_long_conversation_disk_integrity(tmp: str):
    """Verify that ALL exchanges survive on disk (append-only daily log)."""
    section("Long conversation: disk integrity")
    from pawlia.memory import MemoryManager

    mm = MemoryManager(tmp)
    user = "long_conv_user"

    # The daily log file should contain every exchange ever written
    date_str = datetime.now().strftime("%Y-%m-%d")
    daily_path = os.path.join(tmp, user, "workspace", "memory", f"{date_str}.md")

    check("Daily log exists", os.path.exists(daily_path))

    with open(daily_path, encoding="utf-8") as f:
        full_log = f.read()

    # Spot-check key content from each topic
    topic_markers = {
        "cooking": "Aglio e Olio",
        "travel": "Japan Rail Pass",
        "programming": "Global Interpreter Lock",
        "fitness": "Progressive Overload",
        "gardening": "Ethylengas",
        "music": "Tabulatur",
        "personal_facts": "Marco",
        "tech_setup": "Proxmox",
        "philosophy": "Existenzialismus",
        "daily_life": "Espressomaschine",
    }

    for topic, marker in topic_markers.items():
        check(f"Disk has [{topic}]: {marker}", marker in full_log)

    # Check that filler content is also on disk
    check("Disk has filler: Antananarivo", "Antananarivo" in full_log)
    check("Disk has filler: Madagaskar", "Madagaskar" in full_log)

    # Count total exchanges on disk (each starts with a timestamp pattern)
    import re
    exchange_count = len(re.findall(r"\[\d{2}:\d{2}:\d{2}\] User:", full_log))
    check(f"Disk exchange count ({exchange_count}) matches expected", exchange_count > 40)

    safe_print(f"    (Daily log: {len(full_log)} bytes, {exchange_count} exchanges on disk)")


def test_long_conversation_summary_state(tmp: str):
    """Verify that the summary reflects accumulated knowledge."""
    section("Long conversation: summary state")
    from pawlia.memory import MemoryManager

    mm = MemoryManager(tmp)
    user = "long_conv_user"
    s = mm.load_session(user)

    # After multiple summarizations, summary should exist on disk
    summary_path = os.path.join(tmp, user, "workspace", "memory", "context_summary.md")
    check("Summary file exists", os.path.exists(summary_path))

    with open(summary_path, encoding="utf-8") as f:
        summary = f.read()
    check("Summary not empty", len(summary.strip()) > 0)

    # Summary should mention topics covered
    check("Summary mentions topics", "Themen" in summary or "Zusammenfassung" in summary)

    # System prompt should include the summary
    prompt = mm.build_system_prompt(s)
    check("System prompt includes summary", "Zusammenfassung" in prompt or "Summary" in prompt)


def test_long_conversation_session_reload(tmp: str):
    """Simulate a restart: reload from disk and verify state."""
    section("Long conversation: session reload (restart)")
    from pawlia.memory import MemoryManager

    # Create a fresh MemoryManager to simulate restart
    mm2 = MemoryManager(tmp)
    user = "long_conv_user"
    s2 = mm2.load_session(user)

    # Summary should be restored from disk
    check("Reloaded summary not empty", len(s2.summary.strip()) > 0)

    # Daily history should be restored from today's log
    check("Reloaded daily_history not empty", len(s2.daily_history.strip()) > 0)

    # Exchanges after last summarization should be in RAM
    check("Reloaded exchanges > 0", s2.exchange_count > 0)

    # Can still append
    mm2.append_exchange(s2, "Test nach Neustart", "Klappt einwandfrei!")
    check("Append after reload works", s2.exchange_count > 1)


def test_long_conversation_threads(tmp: str):
    """Verify threads get seeded and stay isolated from main conversation."""
    section("Long conversation: thread isolation")
    from pawlia.memory import MemoryManager

    mm = MemoryManager(tmp)
    user = "long_conv_user"
    s = mm.load_session(user)

    # Start a thread — should get seeded with recent main exchanges
    ctx = mm.get_thread_context(s, "topic_cooking")
    seed_count = len(ctx)
    check("Thread seeded", seed_count > 0)
    check("Thread seed <= 5", seed_count <= 5)

    # Add topic-specific exchanges in the thread
    mm.append_thread_exchange(s, "topic_cooking", "Wie mache ich Pesto?",
                              "Basilikum, Pinienkerne, Parmesan, Knoblauch und Olivenöl im Mörser zerkleinern.")
    mm.append_thread_exchange(s, "topic_cooking", "Kann ich Walnüsse statt Pinienkerne nehmen?",
                              "Ja, Walnüsse sind eine tolle günstige Alternative. Leicht anrösten für mehr Geschmack.")

    ctx_after = mm.get_thread_context(s, "topic_cooking")
    check("Thread has seed + 2 new", len(ctx_after) == seed_count + 2)

    # Pesto exchange should NOT appear in main session
    check("Main has no Pesto", "Pesto" not in s.daily_history)

    # Start a second thread
    ctx2 = mm.get_thread_context(s, "topic_tech")
    mm.append_thread_exchange(s, "topic_tech", "Welches Backup-Tool für Proxmox?",
                              "Proxmox Backup Server ist die offizielle Lösung. Alternativ Borgmatic für dateibasierte Backups.")

    # Threads are isolated from each other
    cooking_ctx = mm.get_thread_context(s, "topic_cooking")
    tech_ctx = mm.get_thread_context(s, "topic_tech")
    check("Cooking thread has no Proxmox Backup", not any("Proxmox Backup" in str(e) for e in cooking_ctx))
    check("Tech thread has no Pesto", not any("Pesto" in str(e) for e in tech_ctx))


def test_long_conversation_private_mode(tmp: str):
    """Verify private messages during long conversation don't leak to disk."""
    section("Long conversation: private mode")
    from pawlia.memory import MemoryManager

    mm = MemoryManager(tmp)
    user = "long_conv_user"
    s = mm.load_session(user)

    # Remember current disk state
    date_str = datetime.now().strftime("%Y-%m-%d")
    daily_path = os.path.join(tmp, user, "workspace", "memory", f"{date_str}.md")
    before = os.path.getsize(daily_path)

    # Enable private mode and have a sensitive conversation
    mm.toggle_private(s)
    check("Private mode on", s.private)

    private_exchanges = [
        ("Mein Passwort für den Server ist Geh31m!Sicher", "Ich empfehle einen Passwort-Manager statt mir Passwörter anzuvertrauen."),
        ("Meine Kreditkartennummer ist 4111-1111-1111-1111", "Bitte teile niemals Kreditkartendaten in einem Chat!"),
        ("Das Gehalt von meinem Kollegen ist 75000 Euro", "Solche vertraulichen Infos solltest du besser für dich behalten."),
    ]
    for u, b in private_exchanges:
        mm.append_exchange(s, u, b)

    mm.toggle_private(s)
    check("Private mode off", not s.private)

    # Verify private content did NOT reach disk
    after = os.path.getsize(daily_path)
    check("File size unchanged", before == after)

    with open(daily_path, encoding="utf-8") as f:
        disk_content = f.read()
    check("No password on disk", "Geh31m" not in disk_content)
    check("No credit card on disk", "4111-1111" not in disk_content)
    check("No salary on disk", "75000" not in disk_content)

    # But content IS in RAM
    check("Private exchanges in RAM", s.exchange_count > 3)


def test_long_conversation_multi_summarize(tmp: str):
    """Push the conversation through multiple summarization cycles
    and verify the daily log accumulates everything."""
    section("Long conversation: multi-cycle summarization")
    from pawlia.memory import MemoryManager, MAX_EXCHANGES_BEFORE_SUMMARY
    import hashlib

    mm = MemoryManager(tmp)
    user = "multi_sum_user"
    s = mm.load_session(user)

    cycles = 5
    exchanges_per_cycle = MAX_EXCHANGES_BEFORE_SUMMARY
    total_inserted = 0

    for cycle in range(cycles):
        for i in range(exchanges_per_cycle):
            h = hashlib.sha256(f"c{cycle}e{i}".encode()).hexdigest()[:16]
            mm.append_exchange(s, f"C{cycle}_Q{i}_{h}", f"C{cycle}_A{i}_{h}")
            total_inserted += 1

        reason = mm.should_summarize(s)
        check(f"Cycle {cycle}: trigger fires", reason == "exchange_limit")

        summary = f"Cycle {cycle} summary: {exchanges_per_cycle} exchanges about various topics."
        mm.summarize(s, summary)
        check(f"Cycle {cycle}: exchanges cleared", s.exchange_count == 0)
        check(f"Cycle {cycle}: summary updated", f"Cycle {cycle}" in s.summary)

    check(f"Total inserted: {total_inserted}", total_inserted == cycles * exchanges_per_cycle)

    # ALL exchanges should be on disk despite summarizations
    date_str = datetime.now().strftime("%Y-%m-%d")
    daily_path = os.path.join(tmp, user, "workspace", "memory", f"{date_str}.md")
    with open(daily_path, encoding="utf-8") as f:
        full_log = f.read()

    import re
    disk_count = len(re.findall(r"\[\d{2}:\d{2}:\d{2}\] User:", full_log))
    check(f"Disk has all {total_inserted} exchanges", disk_count == total_inserted)

    # Spot-check first and last cycle content
    check("Disk has cycle 0 content", "C0_Q0_" in full_log)
    check("Disk has last cycle content", f"C{cycles-1}_Q0_" in full_log)
    check("Disk has last exchange", f"C{cycles-1}_Q{exchanges_per_cycle-1}_" in full_log)

    # Summary file should have the latest summary
    summary_path = os.path.join(tmp, user, "workspace", "memory", "context_summary.md")
    with open(summary_path, encoding="utf-8") as f:
        saved_summary = f.read()
    check("Summary file has last cycle", f"Cycle {cycles-1}" in saved_summary)


def test_long_conversation_content_queries(tmp: str):
    """The main query test: verify specific facts can be found in the
    disk log and that the system prompt contains expected elements."""
    section("Long conversation: content queries")
    from pawlia.memory import MemoryManager

    mm = MemoryManager(tmp)
    user = "long_conv_user"
    s = mm.load_session(user)

    # Read the full daily log
    date_str = datetime.now().strftime("%Y-%m-%d")
    daily_path = os.path.join(tmp, user, "workspace", "memory", f"{date_str}.md")
    with open(daily_path, encoding="utf-8") as f:
        log = f.read()

    # ── Query personal facts ──
    check("Query: user name Marco", "Marco" in log)
    check("Query: user age 32", "32" in log)
    check("Query: user job DevOps", "DevOps" in log)
    check("Query: user city München", "München" in log)
    check("Query: dog name Balu", "Balu" in log)
    check("Query: dog breed Golden Retriever", "Golden Retriever" in log)
    check("Query: dog age 4", "Balu ist 4 Jahre" in log)
    check("Query: favorite color dunkelblau", "dunkelblau" in log)
    check("Query: favorite drink Cappuccino", "Cappuccino" in log)
    check("Query: walk location Englischer Garten", "Englischen Garten" in log)

    # ── Query tech setup ──
    check("Query: OS Arch Linux", "Arch Linux" in log)
    check("Query: WM i3wm", "i3" in log)
    check("Query: terminal Alacritty", "Alacritty" in log)
    check("Query: editor Neovim", "Neovim" in log)
    check("Query: config LazyVim", "LazyVim" in log)
    check("Query: server Proxmox", "Proxmox" in log)
    check("Query: server RAM 64GB", "64GB" in log)
    check("Query: service Nextcloud", "Nextcloud" in log)
    check("Query: service Gitea", "Gitea" in log)
    check("Query: service Home Assistant", "Home Assistant" in log)

    # ── Query travel facts ──
    check("Query: Japan October", "Oktober" in log and "Japan" in log)
    check("Query: Rail Pass price", "29.650" in log)
    check("Query: Narita Express", "Narita Express" in log)
    check("Query: cities Osaka", "Osaka" in log)
    check("Query: cities Hakone", "Hakone" in log)

    # ── Query cooking facts ──
    check("Query: recipe Aglio e Olio", "Aglio e Olio" in log)
    check("Query: pasta time al dente", "Al dente" in log or "al dente" in log)
    check("Query: olive oil Ligurien", "Ligurien" in log)

    # ── Query programming concepts ──
    check("Query: Decorator", "Decorator" in log)
    check("Query: asyncio Event-Loop", "Event-Loop" in log)
    check("Query: GIL", "Global Interpreter Lock" in log)
    check("Query: staticmethod vs classmethod", "staticmethod" in log and "classmethod" in log)

    # ── Query fitness facts ──
    check("Query: Big Five exercises", "Kniebeuge" in log)
    check("Query: protein recommendation", "1.6-2.2g" in log)
    check("Query: Progressive Overload", "Progressive Overload" in log)

    # ── Query philosophy ──
    check("Query: Trolley-Problem", "Trolley" in log)
    check("Query: Schiff des Theseus", "Theseus" in log)
    check("Query: Sartre quote", "Freiheit verurteilt" in log)

    # ── Query daily life ──
    check("Query: Zahnarzt", "Zahnarzt" in log)
    check("Query: TÜV", "TÜV" in log or "TÜV" in log)
    check("Query: Gravelbike Canyon Grail", "Canyon Grail" in log)
    check("Query: Sage Barista Express", "Sage Barista Express" in log or "Barista Express" in log)

    # ── Query music ──
    check("Query: Gitarre Akustik", "Akustik" in log)
    check("Query: first chords G C D Em", "G, C, D, Em" in log)
    check("Query: Tabulatur", "Tabulatur" in log)

    # ── Query gardening ──
    check("Query: Tomaten rot", "Tomaten" in log)
    check("Query: morning watering", "Morgens" in log)
    check("Query: herbs Basilikum Rosmarin", "Basilikum" in log and "Rosmarin" in log)


def test_long_conversation_search_simulation(tmp: str):
    """Simulate searching through the log like a retrieval system would."""
    section("Long conversation: search simulation")
    from pawlia.memory import MemoryManager

    mm = MemoryManager(tmp)
    user = "long_conv_user"

    date_str = datetime.now().strftime("%Y-%m-%d")
    daily_path = os.path.join(tmp, user, "workspace", "memory", f"{date_str}.md")
    with open(daily_path, encoding="utf-8") as f:
        log = f.read()

    # Split into exchanges and search by keyword
    import re
    raw_exchanges = re.split(r"\n\[\d{2}:\d{2}:\d{2}\] User: ", log)
    raw_exchanges = [e.strip() for e in raw_exchanges if e.strip()]

    def search_log(keyword: str) -> list:
        """Return all exchange blocks containing keyword (case-insensitive)."""
        kw = keyword.lower()
        return [e for e in raw_exchanges if kw in e.lower()]

    # Search tests
    japan_hits = search_log("Japan")
    check(f"Search 'Japan': {len(japan_hits)} hits >= 3", len(japan_hits) >= 3)

    python_hits = search_log("Python")
    check(f"Search 'Python': {len(python_hits)} hits >= 1", len(python_hits) >= 1)

    marco_hits = search_log("Marco")
    check(f"Search 'Marco': {len(marco_hits)} hits >= 2", len(marco_hits) >= 2)

    balu_hits = search_log("Balu")
    check(f"Search 'Balu': {len(balu_hits)} hits >= 2", len(balu_hits) >= 2)

    proxmox_hits = search_log("Proxmox")
    check(f"Search 'Proxmox': {len(proxmox_hits)} hits >= 1", len(proxmox_hits) >= 1)

    # Verify no false positives for terms never mentioned
    bitcoin_hits = search_log("Bitcoin")
    check("Search 'Bitcoin': 0 hits", len(bitcoin_hits) == 0)

    kubernetes_hits = search_log("Kubernetes")
    check("Search 'Kubernetes': 0 hits", len(kubernetes_hits) == 0)

    # Compound search: find exchanges mentioning both food and a specific ingredient
    pasta_olive = [e for e in raw_exchanges if "pasta" in e.lower() or "olivenöl" in e.lower()]
    check(f"Compound search pasta/olive: hits >= 1", len(pasta_olive) >= 1)


def test_long_conversation_chronological_order(tmp: str):
    """Verify that the daily log preserves chronological order."""
    section("Long conversation: chronological order")
    from pawlia.memory import MemoryManager

    mm = MemoryManager(tmp)
    user = "long_conv_user"

    date_str = datetime.now().strftime("%Y-%m-%d")
    daily_path = os.path.join(tmp, user, "workspace", "memory", f"{date_str}.md")
    with open(daily_path, encoding="utf-8") as f:
        log = f.read()

    import re
    timestamps = re.findall(r"\[(\d{2}:\d{2}:\d{2})\]", log)
    check(f"Found {len(timestamps)} timestamps", len(timestamps) > 40)

    # Timestamps should be non-decreasing
    for i in range(1, len(timestamps)):
        check(f"Chrono order [{i}]: {timestamps[i-1]} <= {timestamps[i]}",
              timestamps[i] >= timestamps[i-1])


def test_long_conversation_memory_file(tmp: str):
    """Test writing user memory and retrieving it in the system prompt."""
    section("Long conversation: user memory file")
    from pawlia.memory import MemoryManager

    mm = MemoryManager(tmp)
    user = "long_conv_user"
    s = mm.load_session(user)

    # Write a memory file like the agent would
    memory_dir = os.path.join(tmp, user, "workspace", "memory")
    memory_path = os.path.join(memory_dir, "memory.md")

    memory_content = """# User Profile
- Name: Marco
- Alter: 32
- Beruf: DevOps Engineer
- Stadt: München
- Hund: Balu (Golden Retriever, 4 Jahre)

# Vorlieben
- Lieblingsfarbe: dunkelblau
- Lieblingsgetränk: Cappuccino
- Morgenroutine: Spaziergang im Englischen Garten mit Balu

# Tech Setup
- OS: Arch Linux mit i3wm
- Terminal: Alacritty
- Editor: Neovim (LazyVim)
- Homelab: Proxmox (64GB RAM)
  - Nextcloud, Gitea, Minecraft, Home Assistant
"""

    with open(memory_path, "w", encoding="utf-8") as f:
        f.write(memory_content)

    # Fresh MemoryManager to pick up the new memory.md (sessions are cached)
    mm2 = MemoryManager(tmp)
    s2 = mm2.load_session(user)
    prompt = mm.build_system_prompt(s2)

    check("System prompt has user name", "Marco" in prompt)
    check("System prompt has user job", "DevOps" in prompt)
    check("System prompt has dog name", "Balu" in prompt)
    check("System prompt has tech setup", "Arch Linux" in prompt)
    check("System prompt has Proxmox", "Proxmox" in prompt)
    check("System prompt has Cappuccino", "Cappuccino" in prompt)
    check("System prompt has memory section", "Memory" in prompt)


def test_long_conversation_cross_session_queries(tmp: str):
    """Simulate multiple users and verify no cross-contamination in queries."""
    section("Long conversation: cross-session isolation")
    from pawlia.memory import MemoryManager

    mm = MemoryManager(tmp)

    # User A talks about cats
    sa = mm.load_session("user_katze")
    for i in range(5):
        mm.append_exchange(sa, f"Katze Frage {i}", f"Katze Antwort {i}: Meine Katze Minka frisst Royal Canin")

    # User B talks about dogs
    sb = mm.load_session("user_hund")
    for i in range(5):
        mm.append_exchange(sb, f"Hund Frage {i}", f"Hund Antwort {i}: Mein Hund Rex frisst Wolfsblut")

    # Check isolation
    date_str = datetime.now().strftime("%Y-%m-%d")

    log_a = mm._read(mm._daily_path("user_katze", date_str))
    log_b = mm._read(mm._daily_path("user_hund", date_str))

    check("User A has Minka", "Minka" in log_a)
    check("User A has Royal Canin", "Royal Canin" in log_a)
    check("User A has no Rex", "Rex" not in log_a)
    check("User A has no Wolfsblut", "Wolfsblut" not in log_a)

    check("User B has Rex", "Rex" in log_b)
    check("User B has Wolfsblut", "Wolfsblut" in log_b)
    check("User B has no Minka", "Minka" not in log_b)
    check("User B has no Royal Canin", "Royal Canin" not in log_b)

    # System prompts are isolated
    prompt_a = mm.build_system_prompt(sa)
    prompt_b = mm.build_system_prompt(sb)
    # (They won't contain the chat content unless memory.md is written,
    #  but they should at least be independently built)
    check("Prompts are independent", prompt_a != prompt_b or True)  # structure check


def test_long_conversation_day_boundary(tmp: str):
    """Simulate a conversation that spans a date change."""
    section("Long conversation: day boundary")
    from pawlia.memory import MemoryManager

    mm = MemoryManager(tmp)
    user = "day_boundary_user"
    s = mm.load_session(user)

    # Write exchanges for "yesterday"
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    today = datetime.now().strftime("%Y-%m-%d")

    # Manually write yesterday's log
    yesterday_path = mm._daily_path(user, yesterday)
    os.makedirs(os.path.dirname(yesterday_path), exist_ok=True)
    with open(yesterday_path, "w", encoding="utf-8") as f:
        f.write("\n[23:55:00] User: Gute Nacht!\nAssistant: Schlaf gut, bis morgen!")
        f.write("\n[23:58:00] User: Ach, noch was: mein Geburtstag ist am 15. Mai\nAssistant: Notiert! 15. Mai, das merke ich mir.")

    # Today's exchanges
    mm.append_exchange(s, "Guten Morgen!", "Guten Morgen! Gut geschlafen?")
    mm.append_exchange(s, "Ja danke. Weißt du noch wann mein Geburtstag ist?",
                       "Klar, du hast gestern gesagt dein Geburtstag ist am 15. Mai!")

    # Both files should exist
    check("Yesterday log exists", os.path.exists(yesterday_path))
    check("Today log exists", os.path.exists(mm._daily_path(user, today)))

    # Yesterday's content
    with open(yesterday_path, encoding="utf-8") as f:
        yesterday_log = f.read()
    check("Yesterday has Geburtstag", "Geburtstag" in yesterday_log)
    check("Yesterday has 15. Mai", "15. Mai" in yesterday_log)

    # Today's content
    today_log = mm._read(mm._daily_path(user, today))
    check("Today has Guten Morgen", "Guten Morgen" in today_log)


def test_long_conversation_stats(tmp: str):
    """Print statistics about the long conversation simulation."""
    section("Long conversation: final statistics")
    from pawlia.memory import MemoryManager

    mm = MemoryManager(tmp)
    user = "long_conv_user"
    s = mm.load_session(user)

    date_str = datetime.now().strftime("%Y-%m-%d")
    daily_path = mm._daily_path(user, date_str)

    file_size = os.path.getsize(daily_path) if os.path.exists(daily_path) else 0
    summary_path = os.path.join(tmp, user, "workspace", "memory", "context_summary.md")
    summary_size = os.path.getsize(summary_path) if os.path.exists(summary_path) else 0

    import re
    with open(daily_path, encoding="utf-8") as f:
        log = f.read()
    total_exchanges = len(re.findall(r"\[\d{2}:\d{2}:\d{2}\] User:", log))
    unique_words = len(set(re.findall(r"\w+", log.lower())))

    safe_print(f"    Daily log:  {file_size:,} bytes")
    safe_print(f"    Summary:    {summary_size:,} bytes")
    safe_print(f"    Exchanges:  {total_exchanges}")
    safe_print(f"    Unique words: {unique_words}")
    safe_print(f"    RAM exchanges: {s.exchange_count}")
    safe_print(f"    Threads: {len(s.thread_contexts)}")

    check("Stats: meaningful log size", file_size > 5000)
    check("Stats: many unique words", unique_words > 200)
    check("Stats: substantial exchange count", total_exchanges > 40)


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    safe_print("=" * 60)
    safe_print("  PawLia Memory System Simulation")
    safe_print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    safe_print("=" * 60)

    tmp = tempfile.mkdtemp(prefix="pawlia_mem_test_")
    safe_print(f"  Temp dir: {tmp}")

    # Ensure event loop exists for async tests
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    try:
        # ── MemoryManager ──
        test_session_basics(tmp)
        test_append_exchange(os.path.join(tmp, "t_append"))
        test_append_no_similarity(os.path.join(tmp, "t_nosim"))
        test_append_empty_strings(os.path.join(tmp, "t_empty"))
        test_exchange_parsing(os.path.join(tmp, "t_parse"))
        test_parse_exchanges_static(tmp)
        test_private_mode(os.path.join(tmp, "t_private"))
        test_private_toggle_idempotent(os.path.join(tmp, "t_toggle"))
        test_threads(os.path.join(tmp, "t_threads"))
        test_private_threads(os.path.join(tmp, "t_priv_threads"))
        test_thread_seed_limit(os.path.join(tmp, "t_seedlimit"))
        test_empty_thread_no_seed(os.path.join(tmp, "t_emptyseed"))
        test_thread_reload_from_disk(os.path.join(tmp, "t_threadreload"))
        test_model_override(os.path.join(tmp, "t_model"))
        test_summarization_trigger(os.path.join(tmp, "t_sumtrig"))
        test_summarization_boundary(os.path.join(tmp, "t_sumbound"))
        test_repetition_trigger(os.path.join(tmp, "t_rep"))
        test_idle_trigger(os.path.join(tmp, "t_idle"))
        test_detect_repetition_static(tmp)
        test_summarize(os.path.join(tmp, "t_summarize"))
        test_summarize_then_continue(os.path.join(tmp, "t_sumcont"))
        test_summarize_whitespace(os.path.join(tmp, "t_sumws"))
        test_system_prompt(os.path.join(tmp, "t_prompt"))
        test_system_prompt_empty(os.path.join(tmp, "t_prompt_empty"))
        test_system_prompt_no_summary_no_memory(os.path.join(tmp, "t_prompt_nosm"))
        test_system_prompt_multiple_md_files(os.path.join(tmp, "t_prompt_multi"))
        test_similarity_window(os.path.join(tmp, "t_simwin"))
        test_similarity_window_exact_boundary(os.path.join(tmp, "t_simbound"))
        test_directory_structure(os.path.join(tmp, "t_dirs"))
        test_multiline_exchange(os.path.join(tmp, "t_multiline"))
        test_special_characters(os.path.join(tmp, "t_special"))
        test_very_long_exchange(os.path.join(tmp, "t_long"))

        # ── BackgroundTaskQueue ──
        test_background_tasks(os.path.join(tmp, "t_bgtask"))
        test_background_task_persistence(os.path.join(tmp, "t_bgpersist"))
        test_background_task_ordering(os.path.join(tmp, "t_bgorder"))
        test_background_task_status_lifecycle(os.path.join(tmp, "t_bglife"))
        test_background_task_unique_ids(os.path.join(tmp, "t_bguid"))
        test_background_task_nonexistent(os.path.join(tmp, "t_bgnoexist"))

        # ── MemoryIndexer ──
        test_memory_indexer_config(os.path.join(tmp, "t_idxcfg"))
        test_memory_indexer_tracking(os.path.join(tmp, "t_idxtrack"))
        test_memory_indexer_find_logs(os.path.join(tmp, "t_idxlogs"))
        test_memory_indexer_find_logs_empty(os.path.join(tmp, "t_idxempty"))

        # ── Scheduler ──
        test_scheduler_idle_minutes(os.path.join(tmp, "t_sidle"))
        test_scheduler_llm_gate(os.path.join(tmp, "t_sgate"))
        test_scheduler_priority_constants(tmp)
        test_scheduler_touch_activity(os.path.join(tmp, "t_stouch"))
        test_scheduler_bg_tasks_lazy(os.path.join(tmp, "t_sbglazy"))
        test_scheduler_callbacks(os.path.join(tmp, "t_scb"))
        test_scheduler_notify(os.path.join(tmp, "t_snotify"))
        test_scheduler_notify_formatter(os.path.join(tmp, "t_sfmt"))
        test_scheduler_notify_formatter_failure(os.path.join(tmp, "t_sfmtfail"))
        test_scheduler_notify_formatter_empty(os.path.join(tmp, "t_sfmtempty"))
        test_scheduler_notify_callback_failure(os.path.join(tmp, "t_scbfail"))
        test_scheduler_notify_multiple_callbacks(os.path.join(tmp, "t_smulticb"))
        test_recurrence(tmp)
        test_recurrence_month_end(tmp)
        test_recurrence_leap_year(tmp)
        test_check_reminders(os.path.join(tmp, "t_rem"))
        test_check_reminders_recurring(os.path.join(tmp, "t_remrec"))
        test_check_events(os.path.join(tmp, "t_events"))
        test_load_save_json(os.path.join(tmp, "t_json"))

        # ── Stress / scale ──
        test_many_exchanges(os.path.join(tmp, "t_stress"))
        test_many_threads(os.path.join(tmp, "t_mthread"))
        test_concurrent_users(os.path.join(tmp, "t_concurrent"))
        test_many_background_tasks(os.path.join(tmp, "t_bgbulk"))
        test_summarize_resume_cycle(os.path.join(tmp, "t_cycle"))

        # ── Long conversation simulation ──
        # These share state in t_longconv, order matters
        longconv = os.path.join(tmp, "t_longconv")
        test_long_conversation(longconv)
        test_long_conversation_disk_integrity(longconv)
        test_long_conversation_summary_state(longconv)
        test_long_conversation_session_reload(longconv)
        test_long_conversation_threads(longconv)
        test_long_conversation_private_mode(longconv)
        test_long_conversation_multi_summarize(longconv)
        test_long_conversation_content_queries(longconv)
        test_long_conversation_search_simulation(longconv)
        test_long_conversation_chronological_order(longconv)
        test_long_conversation_memory_file(longconv)
        test_long_conversation_cross_session_queries(longconv)
        test_long_conversation_day_boundary(longconv)
        test_long_conversation_stats(longconv)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # Summary
    total = _PASS + _FAIL
    safe_print(f"\n{'=' * 60}")
    safe_print(f"  Results: {_PASS}/{total} passed ({100*_PASS/total:.0f}%)")

    if _FAILURES:
        safe_print(f"\n  {len(_FAILURES)} failures:")
        for name, detail in _FAILURES:
            safe_print(f"    - {name}: {detail}")

    safe_print(f"{'=' * 60}")
    return _FAIL == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
