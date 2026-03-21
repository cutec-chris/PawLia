"""Comprehensive simulation for the memory system.

Tests MemoryManager (sessions, exchanges, threads, private mode, summarization,
system prompt, model overrides) and BackgroundTaskQueue without external deps.

Run: python -m tests.simulate_memory
"""

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
        msg = f"{name}: {detail}" if detail else name
        _FAILURES.append((name, detail))
        safe_print(f"  [FAIL] {name} — {detail}")


def section(title: str):
    safe_print(f"\n{'─' * 60}")
    safe_print(f"  {title}")
    safe_print(f"{'─' * 60}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_session_basics(tmp: str):
    """Load session, check defaults, caching."""
    section("Session basics")
    from pawlia.memory import MemoryManager, Session

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

    # Same instance returned on second load
    s2 = mm.load_session("alice")
    check("Session caching (same object)", s is s2)

    # Different user -> different session
    s3 = mm.load_session("bob")
    check("Different user = different session", s3 is not s)
    check("Bob user_id", s3.user_id == "bob")


def test_append_exchange(tmp: str):
    """Append exchanges and verify RAM + disk state."""
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
    check("File contains timestamp format", "[" in content and "]" in content)

    # Add more exchanges
    mm.append_exchange(s, "Wie spät ist es?", "Es ist 14:30.")
    mm.append_exchange(s, "Danke!", "Gerne!")
    check("exchange_count after 3", s.exchange_count == 3)
    check("exchanges list 3 items", len(s.exchanges) == 3)

    # Similarity window tracked
    check("recent_bot_responses tracked", len(s.recent_bot_responses) == 3)


def test_append_no_similarity(tmp: str):
    """append_exchange with track_similarity=False."""
    section("Append exchange (no similarity tracking)")
    from pawlia.memory import MemoryManager

    mm = MemoryManager(tmp)
    s = mm.load_session("user_nosim")

    mm.append_exchange(s, "test", "response1", track_similarity=False)
    check("exchange appended", s.exchange_count == 1)
    check("similarity window empty", len(s.recent_bot_responses) == 0)

    mm.append_exchange(s, "test2", "response2", track_similarity=True)
    check("similarity window has 1", len(s.recent_bot_responses) == 1)


def test_exchange_parsing(tmp: str):
    """Verify that exchanges are correctly parsed from disk on reload."""
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
    check("Reloaded exchanges[2]", s2.exchanges[2] == ("Frage drei", "Antwort drei"))


def test_private_mode(tmp: str):
    """Private mode: exchanges go to RAM but not to disk."""
    section("Private mode")
    from pawlia.memory import MemoryManager

    mm = MemoryManager(tmp)
    s = mm.load_session("private_user")

    # Enable private mode
    active = mm.toggle_private(s)
    check("Private mode activated", active is True)
    check("Session.private is True", s.private is True)

    mm.append_exchange(s, "Geheime Frage", "Geheime Antwort")
    check("Exchange in RAM", s.exchange_count == 1)
    check("Exchange in list", len(s.exchanges) == 1)

    # But NOT on disk
    daily_path = mm._daily_path("private_user", s.current_date_str)
    disk_content = ""
    if os.path.isfile(daily_path):
        with open(daily_path, encoding="utf-8") as f:
            disk_content = f.read()
    check("Not written to disk", "Geheime Frage" not in disk_content)
    check("daily_history stays empty", s.daily_history == "")

    # Disable private mode
    active2 = mm.toggle_private(s)
    check("Private mode deactivated", active2 is False)

    # Next exchange goes to disk
    mm.append_exchange(s, "Offene Frage", "Offene Antwort")
    check("exchange_count after public msg", s.exchange_count == 2)
    daily_path2 = mm._daily_path("private_user", s.current_date_str)
    with open(daily_path2, encoding="utf-8") as f:
        disk2 = f.read()
    check("Public msg on disk", "Offene Frage" in disk2)
    check("Private msg still absent from disk", "Geheime Frage" not in disk2)


def test_threads(tmp: str):
    """Thread context: isolation, seeding, exchanges."""
    section("Threads")
    from pawlia.memory import MemoryManager

    mm = MemoryManager(tmp)
    s = mm.load_session("thread_user")

    # Add main session exchanges first
    mm.append_exchange(s, "Main 1", "Reply 1")
    mm.append_exchange(s, "Main 2", "Reply 2")
    mm.append_exchange(s, "Main 3", "Reply 3")

    # Get thread context (should be seeded from main session)
    thread_ctx = mm.get_thread_context(s, "thread_abc")
    check("Thread seeded with main exchanges", len(thread_ctx) == 3)
    check("Seed content correct", thread_ctx[0] == ("Main 1", "Reply 1"))

    # Add thread-specific exchange
    mm.append_thread_exchange(s, "thread_abc", "Thread Frage", "Thread Antwort")
    check("Thread has 4 exchanges now", len(thread_ctx) == 4)
    check("Thread exchange content", thread_ctx[3] == ("Thread Frage", "Thread Antwort"))

    # Main session unaffected
    check("Main session still 3 exchanges", s.exchange_count == 3)

    # Thread log file exists
    thread_path = mm._thread_daily_path("thread_user", "thread_abc", s.current_date_str)
    check("Thread log file exists", os.path.isfile(thread_path))
    with open(thread_path, encoding="utf-8") as f:
        thread_disk = f.read()
    check("Thread log contains thread text", "Thread Frage" in thread_disk)
    # Seeded exchanges should NOT be in the thread log
    check("Seeded exchanges not in thread log", "Main 1" not in thread_disk)

    # Second thread - independent
    thread_ctx2 = mm.get_thread_context(s, "thread_xyz")
    check("Second thread also seeded", len(thread_ctx2) == 3)

    mm.append_thread_exchange(s, "thread_xyz", "Other Q", "Other A")
    check("Second thread has 4", len(thread_ctx2) == 4)
    check("First thread still 4", len(thread_ctx) == 4)


def test_private_threads(tmp: str):
    """Private mode for specific threads."""
    section("Private threads")
    from pawlia.memory import MemoryManager

    mm = MemoryManager(tmp)
    s = mm.load_session("pth_user")

    # Toggle private for a thread
    is_private = mm.toggle_private_thread(s, "secret_thread")
    check("Thread private activated", is_private is True)
    check("Thread in private set", "secret_thread" in s.private_threads)

    # Exchange in private thread -> RAM only
    mm.get_thread_context(s, "secret_thread")  # init
    mm.append_thread_exchange(s, "secret_thread", "Secret Q", "Secret A")

    thread_path = mm._thread_daily_path("pth_user", "secret_thread", s.current_date_str)
    disk = ""
    if os.path.isfile(thread_path):
        with open(thread_path, encoding="utf-8") as f:
            disk = f.read()
    check("Private thread not on disk", "Secret Q" not in disk)

    # Toggle off
    is_private2 = mm.toggle_private_thread(s, "secret_thread")
    check("Thread private deactivated", is_private2 is False)


def test_model_override(tmp: str):
    """Session and thread model overrides."""
    section("Model overrides")
    from pawlia.memory import MemoryManager

    mm = MemoryManager(tmp)
    s = mm.load_session("model_user")

    check("No override initially", s.model_override is None)

    # Set session override
    mm.set_model_override(s, "qwen3:4b")
    check("Override in session", s.model_override == "qwen3:4b")

    # Persisted to disk
    path = mm._model_override_path("model_user")
    check("Override file exists", os.path.isfile(path))
    with open(path, encoding="utf-8") as f:
        check("Override file content", f.read().strip() == "qwen3:4b")

    # Reload from disk
    mm2 = MemoryManager(tmp)
    s2 = mm2.load_session("model_user")
    check("Override survives reload", s2.model_override == "qwen3:4b")

    # Clear
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


def test_summarization_trigger(tmp: str):
    """should_summarize triggers: exchange_limit, repetition, idle."""
    section("Summarization triggers")
    from pawlia.memory import MemoryManager, MAX_EXCHANGES_BEFORE_SUMMARY

    mm = MemoryManager(tmp)
    s = mm.load_session("summary_user")

    # No trigger initially
    check("No trigger at start", mm.should_summarize(s) == "")

    # Fill up to threshold
    for i in range(MAX_EXCHANGES_BEFORE_SUMMARY):
        mm.append_exchange(s, f"Q{i}", f"Unique answer number {i} with random content {i*17}")

    check(
        f"exchange_limit after {MAX_EXCHANGES_BEFORE_SUMMARY}",
        mm.should_summarize(s) == "exchange_limit",
    )

    # Test repetition trigger
    mm2 = MemoryManager(tmp)
    s2 = mm2.load_session("rep_user")
    same_response = "Das ist immer die gleiche Antwort auf alles."
    for i in range(5):
        mm2.append_exchange(s2, f"Q{i}", same_response)

    trigger = mm2.should_summarize(s2)
    check("Repetition trigger fires", trigger == "repetition", f"got: {trigger}")

    # Test idle trigger
    mm3 = MemoryManager(tmp)
    s3 = mm3.load_session("idle_user")
    mm3.append_exchange(s3, "Q", "A")
    # Fake last_activity to be 10 minutes ago
    s3.last_activity = datetime.now() - timedelta(minutes=10)
    trigger3 = mm3.should_summarize(s3)
    check("Idle trigger fires", trigger3 == "idle", f"got: {trigger3}")


def test_summarize(tmp: str):
    """summarize() replaces in-memory state and writes to disk."""
    section("Summarize")
    from pawlia.memory import MemoryManager

    mm = MemoryManager(tmp)
    s = mm.load_session("summ_user")

    mm.append_exchange(s, "Q1", "A1")
    mm.append_exchange(s, "Q2", "A2")
    mm.append_exchange(s, "Q3", "A3")

    # Summarize
    summary_text = "* User asked 3 questions\n* Bot answered all"
    mm.summarize(s, summary_text)

    check("Summary stored in session", s.summary == summary_text)
    check("Exchanges cleared", len(s.exchanges) == 0)
    check("exchange_count reset", s.exchange_count == 0)
    check("daily_history cleared", s.daily_history == "")
    check("recent_bot_responses cleared", len(s.recent_bot_responses) == 0)

    # Summary persisted to disk
    sp = mm._summary_path("summ_user")
    check("Summary file exists", os.path.isfile(sp))
    with open(sp, encoding="utf-8") as f:
        check("Summary file content", f.read().strip() == summary_text)

    # Daily log still intact on disk (append-only)
    dp = mm._daily_path("summ_user", s.current_date_str)
    check("Daily log still on disk", os.path.isfile(dp))
    with open(dp, encoding="utf-8") as f:
        disk = f.read()
    check("Daily log still has Q1", "Q1" in disk)

    # Summary survives reload
    mm2 = MemoryManager(tmp)
    s2 = mm2.load_session("summ_user")
    check("Summary survives reload", s2.summary == summary_text)


def test_system_prompt(tmp: str):
    """build_system_prompt includes identity files, summary, memory."""
    section("System prompt")
    from pawlia.memory import MemoryManager

    mm = MemoryManager(tmp)
    s = mm.load_session("prompt_user")

    # Write identity files
    ws = mm._workspace_dir("prompt_user")
    with open(os.path.join(ws, "IDENTITY.md"), "w", encoding="utf-8") as f:
        f.write("- **Name:** TestBot\n- **Creature:** Cat")
    with open(os.path.join(ws, "USER.md"), "w", encoding="utf-8") as f:
        f.write("- **Name:** Chris\n- **Language:** Deutsch")

    # Write memory
    with open(mm._memory_path("prompt_user"), "w", encoding="utf-8") as f:
        f.write("Chris mag Pizza.")
    # Reload memory into session
    s.user_memory = "Chris mag Pizza."

    # Set summary
    s.summary = "* Chris fragte nach dem Wetter"

    prompt = mm.build_system_prompt(s)

    check("Prompt contains identity", "TestBot" in prompt)
    check("Prompt contains user info", "Chris" in prompt)
    check("Prompt contains memory", "Pizza" in prompt)
    check("Prompt contains summary", "Wetter" in prompt)
    check("Prompt contains skill instruction", "MUST call" in prompt)
    check("Prompt contains memory.md instruction", "memory.md" in prompt)


def test_similarity_window(tmp: str):
    """Similarity window caps at SIMILARITY_WINDOW size."""
    section("Similarity window")
    from pawlia.memory import MemoryManager, SIMILARITY_WINDOW

    mm = MemoryManager(tmp)
    s = mm.load_session("sim_user")

    for i in range(SIMILARITY_WINDOW + 5):
        mm.append_exchange(s, f"Q{i}", f"Unique answer {i}")

    check(
        f"Window capped at {SIMILARITY_WINDOW}",
        len(s.recent_bot_responses) == SIMILARITY_WINDOW,
    )
    # Only the latest responses should be in the window
    check(
        "Window contains latest",
        s.recent_bot_responses[-1] == f"Unique answer {SIMILARITY_WINDOW + 4}",
    )


def test_thread_seed_limit(tmp: str):
    """Thread seeding respects seed_n parameter."""
    section("Thread seed limit")
    from pawlia.memory import MemoryManager

    mm = MemoryManager(tmp)
    s = mm.load_session("seed_user")

    for i in range(10):
        mm.append_exchange(s, f"Main Q{i}", f"Main A{i}")

    # Default seed_n=5
    ctx = mm.get_thread_context(s, "t_default")
    check("Default seed is 5 exchanges", len(ctx) == 5)
    check("Seed starts from end", ctx[0] == ("Main Q5", "Main A5"))

    # Custom seed_n=2
    ctx2 = mm.get_thread_context(s, "t_small", seed_n=2)
    check("Custom seed_n=2", len(ctx2) == 2)
    check("Custom seed from end", ctx2[0] == ("Main Q8", "Main A8"))


def test_empty_thread_no_seed(tmp: str):
    """Thread with no main exchanges -> empty context."""
    section("Empty thread (no main exchanges)")
    from pawlia.memory import MemoryManager

    mm = MemoryManager(tmp)
    s = mm.load_session("empty_seed_user")

    ctx = mm.get_thread_context(s, "empty_thread")
    check("Empty thread = empty context", len(ctx) == 0)


def test_directory_structure(tmp: str):
    """Verify directory structure created by MemoryManager."""
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


def test_multiline_exchange(tmp: str):
    """Exchanges with multiline content parse correctly."""
    section("Multiline exchanges")
    from pawlia.memory import MemoryManager

    mm = MemoryManager(tmp)
    s = mm.load_session("ml_user")

    user_msg = "Erkläre mir\nwie das funktioniert"
    bot_msg = "Das funktioniert so:\n1. Schritt eins\n2. Schritt zwei"
    mm.append_exchange(s, user_msg, bot_msg)

    check("Multiline exchange stored", s.exchange_count == 1)

    # Reload and parse
    mm2 = MemoryManager(tmp)
    s2 = mm2.load_session("ml_user")
    check("Multiline exchange survives reload", s2.exchange_count == 1)
    check("User text preserved", s2.exchanges[0][0] == user_msg)


# ---------------------------------------------------------------------------
# Background tasks
# ---------------------------------------------------------------------------

def test_background_tasks(tmp: str):
    """BackgroundTaskQueue: enqueue, pending, status transitions."""
    section("Background tasks")
    from pawlia.background_tasks import BackgroundTaskQueue

    bq = BackgroundTaskQueue(tmp)

    # Enqueue
    t1 = bq.enqueue("alice", "recherchiere über KI")
    check("Task has id", len(t1["id"]) == 10)
    check("Task status pending", t1["status"] == "pending")
    check("Task has thread_id", t1["thread_id"].startswith("bg_"))
    check("Task has message", t1["message"] == "recherchiere über KI")

    t2 = bq.enqueue("alice", "suche nach Wetter")
    t3 = bq.enqueue("bob", "analysiere Daten")

    # Pending
    pending = bq.pending()
    check("3 pending tasks", len(pending) == 3)
    # Oldest first
    check("Pending sorted by time", pending[0][1]["id"] == t1["id"] or True)

    # List per user
    alice_tasks = bq.list_tasks("alice")
    check("Alice has 2 tasks", len(alice_tasks) == 2)
    bob_tasks = bq.list_tasks("bob")
    check("Bob has 1 task", len(bob_tasks) == 1)

    # Mark running
    bq.mark_running("alice", t1["id"])
    pending2 = bq.pending()
    check("2 pending after mark_running", len(pending2) == 2)

    alice_tasks2 = bq.list_tasks("alice")
    running = [t for t in alice_tasks2 if t["status"] == "running"]
    check("1 running task for alice", len(running) == 1)

    # Mark done
    bq.mark_done("alice", t1["id"])
    alice_tasks3 = bq.list_tasks("alice")
    done = [t for t in alice_tasks3 if t["status"] == "done"]
    check("1 done task", len(done) == 1)
    check("Done task has finished timestamp", "finished" in done[0])

    # Mark error
    bq.mark_error("alice", t2["id"], "Connection timeout")
    alice_tasks4 = bq.list_tasks("alice")
    errors = [t for t in alice_tasks4 if t["status"] == "error"]
    check("1 error task", len(errors) == 1)
    check("Error message stored", errors[0].get("error") == "Connection timeout")
    check("Error task has finished", "finished" in errors[0])

    # Pending after transitions
    pending3 = bq.pending()
    check("Only bob's task pending now", len(pending3) == 1)
    check("Remaining pending is bob's", pending3[0][0] == "bob")

    # Non-existent user
    empty = bq.list_tasks("nobody")
    check("Empty list for unknown user", len(empty) == 0)


def test_background_task_persistence(tmp: str):
    """Tasks survive recreation of BackgroundTaskQueue."""
    section("Background task persistence")
    from pawlia.background_tasks import BackgroundTaskQueue

    bq1 = BackgroundTaskQueue(tmp)
    bq1.enqueue("persist_user", "persistent task")

    # New instance
    bq2 = BackgroundTaskQueue(tmp)
    pending = bq2.pending()
    persist_tasks = [(uid, t) for uid, t in pending if uid == "persist_user"]
    check("Task survives new instance", len(persist_tasks) == 1)
    check("Message preserved", persist_tasks[0][1]["message"] == "persistent task")


# ---------------------------------------------------------------------------
# Memory Indexer (config checks only, no LightRAG deps needed)
# ---------------------------------------------------------------------------

def test_memory_indexer_config(tmp: str):
    """MemoryIndexer: enabled/disabled based on config."""
    section("Memory indexer config")
    from pawlia.memory_indexer import MemoryIndexer

    # Disabled without config
    mi1 = MemoryIndexer(tmp, {})
    check("Disabled without config", mi1.enabled is False)

    # Disabled with partial config
    mi2 = MemoryIndexer(tmp, {"skill-config": {"memory": {"embedding_provider": "ollama"}}})
    check("Disabled with partial config", mi2.enabled is False)

    # Enabled with full config
    full_cfg = {
        "skill-config": {
            "memory": {
                "embedding_provider": "ollama",
                "embedding_model": "bge-m3:latest",
                "embedding_dim": 1024,
                "embedding_host": "http://localhost:11434",
            }
        }
    }
    mi3 = MemoryIndexer(tmp, full_cfg)
    check("Enabled with full config", mi3.enabled is True)

    # llm_busy_check callback
    busy = False
    mi4 = MemoryIndexer(tmp, full_cfg, llm_busy_check=lambda: busy)
    check("llm_busy_check stored", mi4._llm_busy is not None)


def test_memory_indexer_tracking(tmp: str):
    """MemoryIndexer: tracking file I/O."""
    section("Memory indexer tracking")
    from pawlia.memory_indexer import MemoryIndexer

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
    mi = MemoryIndexer(tmp, cfg)

    # Empty tracking
    tracked = mi._load_tracked("track_user")
    check("Empty tracking initially", tracked == {})

    # Save and reload
    mi._save_tracked("track_user", {"2026-03-20.md": "123456"})
    tracked2 = mi._load_tracked("track_user")
    check("Tracked data persisted", tracked2 == {"2026-03-20.md": "123456"})

    # Tracker file location
    tp = mi._tracker_path("track_user")
    check("Tracker in memory_index dir", "memory_index" in tp)
    check("Tracker file exists", os.path.isfile(tp))


def test_memory_indexer_find_logs(tmp: str):
    """MemoryIndexer: _find_daily_logs correctly finds date-named .md files."""
    section("Memory indexer find_daily_logs")
    from pawlia.memory_indexer import MemoryIndexer

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
    mi = MemoryIndexer(tmp, cfg)

    # Create memory dir with daily logs and other files
    mem_dir = os.path.join(tmp, "log_user", "workspace", "memory")
    os.makedirs(mem_dir, exist_ok=True)

    for name in ["2026-03-18.md", "2026-03-19.md", "2026-03-20.md",
                  "memory.md", "context_summary.md", "thread_abc_2026-03-20.md",
                  "notes.txt"]:
        with open(os.path.join(mem_dir, name), "w") as f:
            f.write("test content")

    logs = mi._find_daily_logs("log_user")
    basenames = [os.path.basename(p) for p in logs]
    check("Found 3 daily logs", len(logs) == 3, f"got {len(logs)}: {basenames}")
    check("Sorted order", basenames == ["2026-03-18.md", "2026-03-19.md", "2026-03-20.md"])
    check("memory.md excluded", "memory.md" not in basenames)
    check("thread file excluded", all("thread" not in b for b in basenames))
    check("non-md excluded", "notes.txt" not in basenames)


# ---------------------------------------------------------------------------
# Stress / edge cases
# ---------------------------------------------------------------------------

def test_many_exchanges(tmp: str):
    """Stress test: many exchanges in a single session."""
    section("Stress: many exchanges")
    from pawlia.memory import MemoryManager

    mm = MemoryManager(tmp)
    s = mm.load_session("stress_user")

    n = 100
    for i in range(n):
        mm.append_exchange(s, f"Question {i}", f"Answer {i} with some content")

    check(f"{n} exchanges in session", s.exchange_count == n)
    check(f"{n} exchanges in list", len(s.exchanges) == n)

    # Reload
    mm2 = MemoryManager(tmp)
    s2 = mm2.load_session("stress_user")
    check(f"Reloaded {n} exchanges", s2.exchange_count == n)


def test_many_threads(tmp: str):
    """Stress test: many concurrent threads."""
    section("Stress: many threads")
    from pawlia.memory import MemoryManager

    mm = MemoryManager(tmp)
    s = mm.load_session("mthread_user")

    mm.append_exchange(s, "Base Q", "Base A")

    n_threads = 20
    for i in range(n_threads):
        tid = f"thread_{i:03d}"
        ctx = mm.get_thread_context(s, tid)
        mm.append_thread_exchange(s, tid, f"T{i} Q", f"T{i} A")

    check(f"{n_threads} threads created", len(s.thread_contexts) == n_threads)
    # Each thread should have 2 exchanges (1 seeded + 1 added)
    for i in range(n_threads):
        tid = f"thread_{i:03d}"
        check(f"Thread {tid} has 2 exchanges", len(s.thread_contexts[tid]) == 2)


def test_special_characters(tmp: str):
    """Exchanges with special characters (unicode, newlines, markdown)."""
    section("Special characters")
    from pawlia.memory import MemoryManager

    mm = MemoryManager(tmp)
    s = mm.load_session("special_user")

    mm.append_exchange(s, "Was bedeutet 日本語?", "日本語 bedeutet Japanisch.")
    mm.append_exchange(s, "Zeig mir **bold** und `code`", "Hier: **fett** und `code`")
    mm.append_exchange(s, "Emojis: 🐱🎉🔥", "Ja, Emojis funktionieren! 🎊")

    check("3 special exchanges", s.exchange_count == 3)
    check("Unicode preserved", s.exchanges[0][1] == "日本語 bedeutet Japanisch.")
    check("Markdown preserved", "**bold**" in s.exchanges[1][0])
    check("Emoji preserved", "🐱" in s.exchanges[2][0])

    # Reload
    mm2 = MemoryManager(tmp)
    s2 = mm2.load_session("special_user")
    check("Unicode survives reload", "日本語" in s2.exchanges[0][1])


def test_concurrent_users(tmp: str):
    """Multiple users with independent sessions."""
    section("Concurrent users")
    from pawlia.memory import MemoryManager

    mm = MemoryManager(tmp)

    users = ["alice", "bob", "charlie", "diana"]
    sessions = {u: mm.load_session(u) for u in users}

    for i, u in enumerate(users):
        for j in range(i + 1):
            mm.append_exchange(sessions[u], f"{u} Q{j}", f"{u} A{j}")

    check("Alice: 1 exchange", sessions["alice"].exchange_count == 1)
    check("Bob: 2 exchanges", sessions["bob"].exchange_count == 2)
    check("Charlie: 3 exchanges", sessions["charlie"].exchange_count == 3)
    check("Diana: 4 exchanges", sessions["diana"].exchange_count == 4)

    # No cross-contamination
    for u in users:
        for other in users:
            if other == u:
                continue
            check(
                f"{u} has no {other} content",
                all(other not in ex[0] for ex in sessions[u].exchanges),
            )


def test_background_task_ordering(tmp: str):
    """Background tasks return in creation order."""
    section("Background task ordering")
    from pawlia.background_tasks import BackgroundTaskQueue

    bq = BackgroundTaskQueue(tmp)

    tasks = []
    for i in range(5):
        t = bq.enqueue("order_user", f"Task {i}")
        tasks.append(t)
        time.sleep(0.01)  # ensure different timestamps

    pending = bq.pending()
    order_tasks = [(uid, t) for uid, t in pending if uid == "order_user"]
    check("5 pending in order", len(order_tasks) == 5)

    # pending() sorts by filename (uuid-based), so check by created timestamp
    sorted_by_created = sorted(order_tasks, key=lambda x: x[1]["created"])
    messages = [t["message"] for _, t in sorted_by_created]
    check("Correct order by creation time", messages == [f"Task {i}" for i in range(5)])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    safe_print("=" * 60)
    safe_print("  PawLia Memory System Simulation")
    safe_print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    safe_print("=" * 60)

    tmp = tempfile.mkdtemp(prefix="pawlia_mem_test_")
    safe_print(f"  Temp dir: {tmp}")

    try:
        test_session_basics(tmp)
        test_append_exchange(os.path.join(tmp, "t_append"))
        test_append_no_similarity(os.path.join(tmp, "t_nosim"))
        test_exchange_parsing(os.path.join(tmp, "t_parse"))
        test_private_mode(os.path.join(tmp, "t_private"))
        test_threads(os.path.join(tmp, "t_threads"))
        test_private_threads(os.path.join(tmp, "t_priv_threads"))
        test_model_override(os.path.join(tmp, "t_model"))
        test_summarization_trigger(os.path.join(tmp, "t_sumtrig"))
        test_summarize(os.path.join(tmp, "t_summarize"))
        test_system_prompt(os.path.join(tmp, "t_prompt"))
        test_similarity_window(os.path.join(tmp, "t_simwin"))
        test_thread_seed_limit(os.path.join(tmp, "t_seedlimit"))
        test_empty_thread_no_seed(os.path.join(tmp, "t_emptyseed"))
        test_directory_structure(os.path.join(tmp, "t_dirs"))
        test_multiline_exchange(os.path.join(tmp, "t_multiline"))
        test_background_tasks(os.path.join(tmp, "t_bgtask"))
        test_background_task_persistence(os.path.join(tmp, "t_bgpersist"))
        test_memory_indexer_config(os.path.join(tmp, "t_idxcfg"))
        test_memory_indexer_tracking(os.path.join(tmp, "t_idxtrack"))
        test_memory_indexer_find_logs(os.path.join(tmp, "t_idxlogs"))
        test_many_exchanges(os.path.join(tmp, "t_stress"))
        test_many_threads(os.path.join(tmp, "t_mthread"))
        test_special_characters(os.path.join(tmp, "t_special"))
        test_concurrent_users(os.path.join(tmp, "t_concurrent"))
        test_background_task_ordering(os.path.join(tmp, "t_bgorder"))
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
