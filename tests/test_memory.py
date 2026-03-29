"""Tests for pawlia.memory."""

import os
import tempfile
from datetime import datetime, timedelta

from pawlia.memory import (
    FORCE_SUMMARY_EXCHANGES,
    KEEP_RECENT_EXCHANGES,
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
        # New format: (user_text, bot_text, tool_calls_info)
        assert result == [("hello", "hi there", None)]

    def test_multiple_exchanges(self):
        history = (
            "\n[10:00:00] User: first\nAssistant: response1"
            "\n[10:01:00] User: second\nAssistant: response2"
        )
        result = MemoryManager._parse_exchanges(history)
        assert len(result) == 2
        assert result[0] == ("first", "response1", None)
        assert result[1] == ("second", "response2", None)

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
            assert session.exchanges[0] == ("hi", "hello", None)
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
            assert session2.exchanges[0] == ("q1", "a1", None)
            assert session2.exchanges[1] == ("q2", "a2", None)

    def test_similarity_window_limit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mm = self._make_mm(tmpdir)
            session = mm.load_session("u1")

            for i in range(10):
                mm.append_exchange(session, f"q{i}", f"unique response {i}")

            # Window is 4
            assert len(session.recent_bot_responses) == 4

    def test_summarize_keeps_recent_exchanges(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mm = self._make_mm(tmpdir)
            session = mm.load_session("u1")

            # Add more than KEEP_RECENT_EXCHANGES exchanges
            for i in range(8):
                mm.append_exchange(session, f"q{i}", f"a{i}")

            mm.summarize(session, "- User asked eight questions")

            assert session.summary == "- User asked eight questions"
            assert session.daily_history == ""
            assert len(session.exchanges) == KEEP_RECENT_EXCHANGES
            assert session.exchange_count == KEEP_RECENT_EXCHANGES
            # Should keep the LAST 5
            assert session.exchanges[0] == ("q3", "a3", None)
            assert session.exchanges[-1] == ("q7", "a7", None)
            assert session.recent_bot_responses == []

    def test_summarize_keeps_all_when_fewer_than_limit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mm = self._make_mm(tmpdir)
            session = mm.load_session("u1")

            mm.append_exchange(session, "q1", "a1")
            mm.append_exchange(session, "q2", "a2")

            mm.summarize(session, "- User asked two questions")

            assert session.summary == "- User asked two questions"
            assert len(session.exchanges) == 2
            assert session.exchange_count == 2
            assert session.exchanges[0] == ("q1", "a1", None)

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

    def test_no_trigger_below_limit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mm = MemoryManager(tmpdir)
            session = mm.load_session("u1")
            # Use very different answers to avoid repetition trigger
            answers = [
                "Python is great", "The sky is blue", "42 is the answer",
                "Cats are fluffy", "Berlin is a city", "Coffee is good",
                "Rain tomorrow", "Books are nice", "Music rocks",
            ]
            for i in range(MAX_EXCHANGES_BEFORE_SUMMARY - 1):
                mm.append_exchange(session, f"q{i}", answers[i])
            assert mm.should_summarize(session) == ""

    def test_force_trigger(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mm = MemoryManager(tmpdir)
            session = mm.load_session("u1")
            for i in range(FORCE_SUMMARY_EXCHANGES):
                mm.append_exchange(session, f"q{i}", f"unique answer {i}")
            assert mm.should_summarize(session) == "force"


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

    def test_call_mode_adds_phone_instructions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mm = MemoryManager(tmpdir)
            session = mm.load_session("u_call")

            prompt = mm.build_system_prompt(session, mode="call")

            assert "Conversation Mode: Phone Call" in prompt
            assert "Keep answers compact" in prompt

    def test_default_mode_does_not_add_phone_instructions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mm = MemoryManager(tmpdir)
            session = mm.load_session("u_chat")

            prompt = mm.build_system_prompt(session)

            assert "Conversation Mode: Phone Call" not in prompt


class TestSystemPromptIdentityFiles:
    """Verify soul.md, identity.md, and user.md are all present in the prompt."""

    def test_contains_soul_md(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mm = MemoryManager(tmpdir)
            session = mm.load_session("u_soul")
            # _ensure_identity_files copies the template; check its content appears
            prompt = mm.build_system_prompt(session)
            assert "SOUL.md" in prompt or "Core Truths" in prompt, \
                "soul.md content should be in the system prompt"

    def test_contains_identity_md(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mm = MemoryManager(tmpdir)
            session = mm.load_session("u_id")
            ws = mm._workspace_dir("u_id")
            with open(os.path.join(ws, "identity.md"), "w") as f:
                f.write("# identity.md\n- **Name:** TestBot\n- **Creature:** Cat")
            prompt = mm.build_system_prompt(session)
            assert "TestBot" in prompt
            assert "identity.md" in prompt

    def test_contains_user_md(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mm = MemoryManager(tmpdir)
            session = mm.load_session("u_usr")
            ws = mm._workspace_dir("u_usr")
            with open(os.path.join(ws, "user.md"), "w") as f:
                f.write("# user.md\n- **Name:** Chris\n- **Language:** Deutsch")
            prompt = mm.build_system_prompt(session)
            assert "Chris" in prompt
            assert "user.md" in prompt

    def test_all_three_identity_files_present(self):
        """All three identity files must appear in a single prompt."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mm = MemoryManager(tmpdir)
            session = mm.load_session("u_all3")
            ws = mm._workspace_dir("u_all3")
            with open(os.path.join(ws, "soul.md"), "w") as f:
                f.write("# SOUL\nI am kind and helpful.")
            with open(os.path.join(ws, "identity.md"), "w") as f:
                f.write("# IDENTITY\n- **Name:** PawLia")
            with open(os.path.join(ws, "user.md"), "w") as f:
                f.write("# USER\n- **Name:** Chris")
            prompt = mm.build_system_prompt(session)
            assert "SOUL" in prompt, "soul.md missing from prompt"
            assert "PawLia" in prompt, "identity.md missing from prompt"
            assert "Chris" in prompt, "user.md missing from prompt"

    def test_identity_files_order(self):
        """Identity files appear in defined order: identity, user, soul."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mm = MemoryManager(tmpdir)
            session = mm.load_session("u_order")
            ws = mm._workspace_dir("u_order")
            with open(os.path.join(ws, "soul.md"), "w") as f:
                f.write("MARKER_SOUL")
            with open(os.path.join(ws, "identity.md"), "w") as f:
                f.write("MARKER_IDENTITY")
            with open(os.path.join(ws, "user.md"), "w") as f:
                f.write("MARKER_USER")
            prompt = mm.build_system_prompt(session)
            pos_id = prompt.index("MARKER_IDENTITY")
            pos_usr = prompt.index("MARKER_USER")
            pos_soul = prompt.index("MARKER_SOUL")
            assert pos_id < pos_usr < pos_soul, \
                f"Expected identity < user < soul, got {pos_id}, {pos_usr}, {pos_soul}"


class TestExchangesInMessageHistory:
    """Verify that recent exchanges end up as HumanMessage/AIMessage pairs."""

    def test_exchanges_replayed_in_messages(self):
        """Simulate what ChatAgent.run does: build messages from session.exchanges."""
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        with tempfile.TemporaryDirectory() as tmpdir:
            mm = MemoryManager(tmpdir)
            session = mm.load_session("u_hist")
            # Add 5 exchanges
            for i in range(1, 6):
                session.exchanges.append((f"user_msg_{i}", f"bot_msg_{i}"))

            # Replicate the message-building logic from ChatAgent.run
            prompt = mm.build_system_prompt(session)
            messages = [SystemMessage(content=prompt)]
            for user_text, bot_text in session.exchanges:
                messages.append(HumanMessage(content=user_text))
                messages.append(AIMessage(content=bot_text))

            # 1 system + 5*2 exchange messages = 11
            assert len(messages) == 11
            assert isinstance(messages[0], SystemMessage)
            # Check all exchanges are present in order
            for i in range(1, 6):
                human = messages[2 * i - 1]
                ai = messages[2 * i]
                assert isinstance(human, HumanMessage)
                assert isinstance(ai, AIMessage)
                assert human.content == f"user_msg_{i}"
                assert ai.content == f"bot_msg_{i}"

    def test_exchanges_not_in_system_prompt(self):
        """Exchanges must NOT appear in the system prompt text itself."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mm = MemoryManager(tmpdir)
            session = mm.load_session("u_noexch")
            session.exchanges.append(("hello user", "hello bot"))
            prompt = mm.build_system_prompt(session)
            assert "hello user" not in prompt
            assert "hello bot" not in prompt

    def test_new_thread_starts_empty(self):
        """A new thread should start empty and stay isolated from the main session."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mm = MemoryManager(tmpdir)
            session = mm.load_session("u_thread")
            # Add 7 exchanges to main session
            for i in range(1, 8):
                session.exchanges.append((f"msg_{i}", f"reply_{i}"))

            thread_ctx = mm.get_thread_context(session, "new_thread")
            assert thread_ctx == []


class TestSummaryFromExchanges:
    """Verify that summary replaces exchanges and appears in the prompt."""

    def test_summarize_keeps_recent_exchanges(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mm = MemoryManager(tmpdir)
            session = mm.load_session("u_summ")
            for i in range(10):
                session.exchanges.append((f"q{i}", f"a{i}"))
            session.exchange_count = 10

            mm.summarize(session, "User asked 10 questions about Python.")

            assert session.summary == "User asked 10 questions about Python."
            assert len(session.exchanges) == KEEP_RECENT_EXCHANGES
            assert session.exchange_count == KEEP_RECENT_EXCHANGES
            assert session.exchanges[0] == ("q5", "a5")
            assert session.exchanges[-1] == ("q9", "a9")

    def test_summary_in_system_prompt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mm = MemoryManager(tmpdir)
            session = mm.load_session("u_summ2")
            session.summary = "User discussed weather and pizza."
            prompt = mm.build_system_prompt(session)
            assert "## Conversation Summary" in prompt
            assert "weather and pizza" in prompt

    def test_no_summary_section_when_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mm = MemoryManager(tmpdir)
            session = mm.load_session("u_nosumm")
            session.summary = ""
            prompt = mm.build_system_prompt(session)
            assert "Conversation Summary" not in prompt
