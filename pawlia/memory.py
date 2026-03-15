"""Session and MemoryManager for PawLia.

Directory layout:

    session/{user_id}/
        workspace/
            memory/
                {YYYY-MM-DD}.md       daily chat log
                memory.md             persistent user facts
            ...                       skill working files
"""

import logging
import os
import re
import shutil
from datetime import datetime
from difflib import SequenceMatcher
from typing import List, Optional, Tuple

# Summarization trigger thresholds
MAX_EXCHANGES_BEFORE_SUMMARY = 20
SIMILARITY_THRESHOLD = 0.6  # 0-1, how similar two bot responses must be
SIMILARITY_WINDOW = 4  # compare last N bot responses
IDLE_TIMEOUT_SECONDS = 300  # 5 minutes


class Session:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.current_date_str = datetime.now().strftime("%Y-%m-%d")

        # In-memory cache
        self.daily_history: str = ""
        self.user_memory: str = ""

        # Structured exchange pairs for LLM message building
        self.exchanges: List[Tuple[str, str]] = []  # (user_text, bot_text)

        # Summarization state
        self.exchange_count: int = 0
        self.recent_bot_responses: List[str] = []
        self.last_activity: datetime = datetime.now()
        self.summary: str = ""  # accumulated summary from prior rounds


class MemoryManager:
    def __init__(self, session_dir: str, logger: Optional[logging.Logger] = None):
        self.session_dir = session_dir
        self.logger = logger or logging.getLogger("pawlia.memory")
        os.makedirs(session_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _workspace_dir(self, user_id: str) -> str:
        path = os.path.join(self.session_dir, user_id, "workspace")
        os.makedirs(path, exist_ok=True)
        return path

    def _memory_dir(self, user_id: str) -> str:
        path = os.path.join(self._workspace_dir(user_id), "memory")
        os.makedirs(path, exist_ok=True)
        return path

    def _daily_path(self, user_id: str, date_str: str) -> str:
        return os.path.join(self._memory_dir(user_id), f"{date_str}.md")

    def _memory_path(self, user_id: str) -> str:
        return os.path.join(self._memory_dir(user_id), "memory.md")

    def _prompts_dir(self) -> str:
        return os.path.join(os.path.dirname(__file__), "prompts")

    def _ensure_identity_files(self, workspace: str) -> None:
        """Copy missing identity templates + bootstrap.md into workspace.
        When all three are filled (differ from their templates), delete bootstrap.md.
        """
        identity_map = {
            "soul.md": "soul.md",
            "IDENTITY.md": "identity.md",
            "USER.md": "user.md",
        }
        prompts_dir = self._prompts_dir()

        missing = [ws for ws in identity_map if not os.path.exists(os.path.join(workspace, ws))]

        if missing:
            bootstrap_dst = os.path.join(workspace, "bootstrap.md")
            if not os.path.exists(bootstrap_dst):
                bootstrap_src = os.path.join(prompts_dir, "bootstrap.md")
                if os.path.exists(bootstrap_src):
                    shutil.copy2(bootstrap_src, bootstrap_dst)

            for ws_name in missing:
                dst = os.path.join(workspace, ws_name)
                src = os.path.join(prompts_dir, identity_map[ws_name])
                if os.path.exists(src):
                    shutil.copy2(src, dst)

    # ------------------------------------------------------------------
    # Load / save
    # ------------------------------------------------------------------

    @staticmethod
    def _read(path: str) -> str:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        return ""

    def _summary_path(self, user_id: str) -> str:
        return os.path.join(self._memory_dir(user_id), "context_summary.md")

    @staticmethod
    def _parse_exchanges(history: str) -> List[Tuple[str, str]]:
        """Parse flat history text into (user, assistant) pairs."""
        # Format: \n[HH:MM:SS] User: ...\nAssistant: ...
        pattern = re.compile(
            r"\[[\d:]+\]\s*User:\s*(.*?)\nAssistant:\s*(.*?)(?=\n\[[\d:]+\]\s*User:|\Z)",
            re.DOTALL,
        )
        return [(m.group(1).strip(), m.group(2).strip()) for m in pattern.finditer(history)]

    def load_session(self, user_id: str) -> Session:
        """Load session data from disk and return a Session object."""
        self._memory_dir(user_id)  # ensure dirs exist

        session = Session(user_id)
        session.daily_history = self._read(self._daily_path(user_id, session.current_date_str))
        session.user_memory = self._read(self._memory_path(user_id))
        session.summary = self._read(self._summary_path(user_id))
        session.exchanges = self._parse_exchanges(session.daily_history)
        session.exchange_count = len(session.exchanges)
        return session

    def build_system_prompt(self, session: Session) -> str:
        """Build the system prompt from workspace identity files + memory."""
        workspace = self._workspace_dir(session.user_id)
        self._ensure_identity_files(workspace)
        parts: list[str] = []

        # Include all .md files at workspace root (non-recursive), sorted
        try:
            ws_files = sorted(
                f for f in os.listdir(workspace)
                if f.lower().endswith(".md") and os.path.isfile(os.path.join(workspace, f))
            )
        except OSError:
            ws_files = []

        for filename in ws_files:
            content = self._read(os.path.join(workspace, filename))
            if content.strip():
                parts.append(content.strip())

        if session.summary.strip():
            parts.append(
                f"## Conversation Summary\n{session.summary.strip()}"
            )

        # Recent exchanges are passed as structured HumanMessage/AIMessage
        # pairs (see ChatAgent.run), NOT included here as flat text.

        if session.user_memory.strip():
            parts.append(f"## Memory\n{session.user_memory.strip()}")

        parts.append(
            "IMPORTANT: You have skills (tools) available. "
            "When a user asks for information that a skill can provide "
            "(routes, train connections, searches, file operations, etc.), "
            "you MUST call the matching skill. NEVER guess or make up answers — "
            "always use the skill to get real data.\n"
            "Only answer directly for simple conversation (greetings, opinions, "
            "general knowledge).\n"
            "When you learn a persistent fact or preference about the user "
            "(name, language, habits, preferences, etc.), "
            "use the files skill to append it to memory/memory.md."
        )

        return "\n\n---\n\n".join(parts)

    def append_exchange(
        self,
        session: Session,
        user_text: str,
        bot_text: str,
        *,
        track_similarity: bool = True,
    ) -> None:
        """Append a user/assistant exchange to the daily log (RAM + disk).

        When ``track_similarity`` is False the response is NOT added to
        the similarity window.  Use this for skill-backed responses whose
        content is inherently repetitive (e.g. file listings).
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = f"\n[{timestamp}] User: {user_text}\nAssistant: {bot_text}"

        session.daily_history += entry
        session.exchanges.append((user_text, bot_text))
        session.exchange_count += 1
        session.last_activity = datetime.now()

        if track_similarity:
            session.recent_bot_responses.append(bot_text)
            if len(session.recent_bot_responses) > SIMILARITY_WINDOW:
                session.recent_bot_responses.pop(0)

        daily_path = self._daily_path(session.user_id, session.current_date_str)
        with open(daily_path, "a", encoding="utf-8") as f:
            f.write(entry)

    # ------------------------------------------------------------------
    # Summarization
    # ------------------------------------------------------------------

    def should_summarize(self, session: Session) -> str:
        """Check whether conversation should be summarized.

        Returns the trigger reason (empty string = no summary needed).
        """
        if session.exchange_count >= MAX_EXCHANGES_BEFORE_SUMMARY:
            return "exchange_limit"

        if self._detect_repetition(session.recent_bot_responses):
            return "repetition"

        idle = (datetime.now() - session.last_activity).total_seconds()
        if session.exchange_count > 0 and idle >= IDLE_TIMEOUT_SECONDS:
            return "idle"

        return ""

    @staticmethod
    def _detect_repetition(responses: List[str]) -> bool:
        """Return True if recent bot responses are too similar to each other."""
        if len(responses) < 2:
            return False

        latest = responses[-1]
        for older in responses[:-1]:
            ratio = SequenceMatcher(None, latest, older).ratio()
            if ratio >= SIMILARITY_THRESHOLD:
                return True
        return False

    def summarize(self, session: Session, summary_text: str) -> None:
        """Replace detailed history with a summary.

        ``summary_text`` is the LLM-generated summary of the conversation.
        The raw history is kept on disk (append-only daily log) but the
        in-memory history is replaced so the system prompt stays compact.
        """
        session.summary = summary_text.strip()
        session.daily_history = ""
        session.exchanges.clear()
        session.exchange_count = 0
        session.recent_bot_responses.clear()

        # Persist summary to disk alongside the daily log
        summary_path = self._summary_path(session.user_id)
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(session.summary)

        self.logger.info("Conversation summarized for %s", session.user_id)
