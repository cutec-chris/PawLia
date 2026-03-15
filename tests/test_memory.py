"""Tests for pawlia.memory."""

import os
import tempfile
from datetime import datetime, timedelta

from pawlia.memory import (
    IDLE_TIMEOUT_SECONDS,
    MAX_EXCHANGES_BEFORE_SUMMARY,
    MemoryManager,
    Session,
)


class TestSession:
    def test_init_defaults(self):
        s = Session("user1")
        assert s.user_id == "user1"
        assert s.daily_history == ""
        assert s.user_memory == ""
        assert s.exchanges == []
        assert s.exchange_count == 0
        assert s.recent_bot_responses == []
        assert s.summary == ""
        assert s.current_date_str == datetime.now().strftime("%Y-%m-%d")


class TestParseExchanges:
    def test_single_exchange(self):
        history = "\n[12:30:45] User: hello\nAssistant: hi there"
        result = MemoryManager._parse_exchanges(history)
        assert result == [("hello", "hi there")]

    def test_multiple_exchanges(self):
        history = (
            "\n[10:00:00] User: first\nAssistant: response1"
            "\n[10:01:00] User: second\nAssistant: response2"
        )
        result = MemoryManager._parse_exchanges(history)
        assert len(result) == 2
        assert result[0] == ("first", "response1")
        assert result[1] == ("second", "response2")

    def test_multiline_response(self):
        history = (
            "\n[10:00:00] User: tell me something\n"
            "Assistant: line one\nline two\nline three"
        )
        result = MemoryManager._parse_exchanges(history)
        assert len(result) == 1
        assert "line one" in result[0][1]
        assert "line three" in result[0][1]

    def test_empty_history(self):
        assert MemoryManager._parse_exchanges("") == []

    def test_no_matches(self):
        assert MemoryManager._parse_exchanges("random text") == []


class TestMemoryManager:
    def _make_mm(self, tmpdir):
        return MemoryManager(tmpdir)

    def test_load_session_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mm = self._make_mm(tmpdir)
            session = mm.load_session("new_user")
            assert session.user_id == "new_user"
            assert session.daily_history == ""
            assert session.exchanges == []
            assert session.exchange_count == 0

    def test_dirs_created(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mm = self._make_mm(tmpdir)
            mm.load_session("u1")
            assert os.path.isdir(os.path.join(tmpdir, "u1", "workspace", "memory"))

    def test_append_exchange(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mm = self._make_mm(tmpdir)
            session = mm.load_session("u1")

            mm.append_exchange(session, "hi", "hello")

            assert session.exchange_count == 1
            assert len(session.exchanges) == 1
            assert session.exchanges[0] == ("hi", "hello")
            assert "hi" in session.daily_history
            assert "hello" in session.daily_history
            assert len(session.recent_bot_responses) == 1

    def test_append_exchange_no_similarity_tracking(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mm = self._make_mm(tmpdir)
            session = mm.load_session("u1")

            mm.append_exchange(session, "hi", "hello", track_similarity=False)

            assert session.exchange_count == 1
            assert len(session.recent_bot_responses) == 0

    def test_append_persists_to_disk(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mm = self._make_mm(tmpdir)
            session = mm.load_session("u1")

            mm.append_exchange(session, "q1", "a1")
            mm.append_exchange(session, "q2", "a2")

            # Load fresh session from disk
            session2 = mm.load_session("u1")
            assert session2.exchange_count == 2
            assert session2.exchanges[0] == ("q1", "a1")
            assert session2.exchanges[1] == ("q2", "a2")

    def test_similarity_window_limit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mm = self._make_mm(tmpdir)
            session = mm.load_session("u1")

            for i in range(10):
                mm.append_exchange(session, f"q{i}", f"unique response {i}")

            # Window is 4
            assert len(session.recent_bot_responses) == 4

    def test_summarize_clears_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mm = self._make_mm(tmpdir)
            session = mm.load_session("u1")

            mm.append_exchange(session, "q1", "a1")
            mm.append_exchange(session, "q2", "a2")

            mm.summarize(session, "- User asked two questions")

            assert session.summary == "- User asked two questions"
            assert session.daily_history == ""
            assert session.exchanges == []
            assert session.exchange_count == 0
            assert session.recent_bot_responses == []

    def test_summarize_persists_to_disk(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mm = self._make_mm(tmpdir)
            session = mm.load_session("u1")
            mm.append_exchange(session, "q1", "a1")

            mm.summarize(session, "bullet point summary")

            # Load fresh
            session2 = mm.load_session("u1")
            assert session2.summary == "bullet point summary"

    def test_summarize_replaces_not_appends(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mm = self._make_mm(tmpdir)
            session = mm.load_session("u1")

            mm.summarize(session, "first summary")
            mm.append_exchange(session, "q", "a")
            mm.summarize(session, "second summary")

            assert session.summary == "second summary"
            assert "first" not in session.summary


class TestShouldSummarize:
    def test_no_trigger(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mm = MemoryManager(tmpdir)
            session = mm.load_session("u1")
            mm.append_exchange(session, "hi", "hello")
            assert mm.should_summarize(session) == ""

    def test_exchange_limit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mm = MemoryManager(tmpdir)
            session = mm.load_session("u1")
            for i in range(MAX_EXCHANGES_BEFORE_SUMMARY):
                mm.append_exchange(session, f"q{i}", f"unique answer number {i}")
            assert mm.should_summarize(session) == "exchange_limit"

    def test_repetition_trigger(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mm = MemoryManager(tmpdir)
            session = mm.load_session("u1")
            # Add very similar responses
            mm.append_exchange(session, "q1", "The answer is 42 and that is final")
            mm.append_exchange(session, "q2", "The answer is 42 and that is final!")
            assert mm.should_summarize(session) == "repetition"

    def test_idle_trigger(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mm = MemoryManager(tmpdir)
            session = mm.load_session("u1")
            mm.append_exchange(session, "q1", "a1")
            # Fake old last_activity
            session.last_activity = datetime.now() - timedelta(seconds=IDLE_TIMEOUT_SECONDS + 10)
            assert mm.should_summarize(session) == "idle"

    def test_idle_needs_exchanges(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mm = MemoryManager(tmpdir)
            session = mm.load_session("u1")
            session.last_activity = datetime.now() - timedelta(seconds=IDLE_TIMEOUT_SECONDS + 10)
            # No exchanges -> no idle trigger
            assert mm.should_summarize(session) == ""


class TestDetectRepetition:
    def test_no_repetition(self):
        assert MemoryManager._detect_repetition(["apple", "banana", "cherry"]) is False

    def test_identical(self):
        assert MemoryManager._detect_repetition(["same thing", "same thing"]) is True

    def test_single_response(self):
        assert MemoryManager._detect_repetition(["only one"]) is False

    def test_empty(self):
        assert MemoryManager._detect_repetition([]) is False

    def test_similar_but_not_identical(self):
        # Very similar strings should trigger
        assert MemoryManager._detect_repetition([
            "The weather today is sunny and warm",
            "The weather today is sunny and warm!",
        ]) is True

    def test_different_enough(self):
        assert MemoryManager._detect_repetition([
            "Python is great for data science",
            "JavaScript is the language of the web",
        ]) is False


class TestBuildSystemPrompt:
    def test_includes_skill_instruction(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mm = MemoryManager(tmpdir)
            session = mm.load_session("u1")
            prompt = mm.build_system_prompt(session)
            assert "MUST call the matching skill" in prompt

    def test_includes_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mm = MemoryManager(tmpdir)
            session = mm.load_session("u1")
            session.summary = "User likes Python"
            prompt = mm.build_system_prompt(session)
            assert "User likes Python" in prompt
            assert "Conversation Summary" in prompt

    def test_includes_memory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mm = MemoryManager(tmpdir)
            session = mm.load_session("u1")
            # Write memory file
            mem_path = mm._memory_path("u1")
            with open(mem_path, "w") as f:
                f.write("- Name: Chris\n- Likes: Coffee")
            session.user_memory = "- Name: Chris\n- Likes: Coffee"
            prompt = mm.build_system_prompt(session)
            assert "Chris" in prompt
            assert "## Memory" in prompt
