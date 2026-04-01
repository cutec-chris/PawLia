"""Session and MemoryManager for PawLia.

Directory layout:

    session/{user_id}/
        workspace/
            memory/
                {YYYY-MM-DD}.md       daily chat log
                memory.md             persistent user facts
            ...                       skill working files
"""

import json
import logging
import os
import re
import shutil
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Set, Tuple

from pawlia.prompt_utils import load_system_prompt
from pawlia.utils import ensure_dir

# Summarization trigger thresholds
MAX_EXCHANGES_BEFORE_SUMMARY = 10
FORCE_SUMMARY_EXCHANGES = 30  # force summarize even if user is active
KEEP_RECENT_EXCHANGES = 5  # exchanges to keep intact after summarization
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
        # (user_text, bot_text, tool_calls_info) where tool_calls_info is a list of
        # dicts with 'name', 'args', and 'result' keys, or None if no tool calls
        self.exchanges: List[Tuple[str, str, Optional[List[Dict[str, Any]]]]] = []  # type: ignore

        # Summarization state
        self.exchange_count: int = 0
        self.recent_bot_responses: List[str] = []
        self.last_activity: datetime = datetime.now()
        self.summary: str = ""  # accumulated summary from prior rounds

        # Optional model override (e.g. set via /model command)
        self.model_override: Optional[str] = None

        # Per-thread exchange lists (loaded/seeded lazily by get_thread_context)
        self.thread_contexts: Dict[str, List[Tuple[str, str]]] = {}

        # Per-thread model overrides (loaded lazily by get_thread_model_override)
        self.thread_model_overrides: Dict[str, Optional[str]] = {}

        # Private mode: exchanges are kept in RAM but not written to disk.
        # Resets on restart (intentional).
        self.private: bool = False            # CLI / session-level
        self.private_threads: Set[str] = set()  # per-thread


class MemoryManager:
    def __init__(self, session_dir: str, logger: Optional[logging.Logger] = None):
        self.session_dir = session_dir
        self.logger = logger or logging.getLogger("pawlia.memory")
        ensure_dir(session_dir)
        self._sessions: Dict[str, Session] = {}  # cached session instances

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _workspace_dir(self, user_id: str) -> str:
        path = os.path.join(self.session_dir, user_id, "workspace")
        return ensure_dir(path)

    def _memory_dir(self, user_id: str) -> str:
        path = os.path.join(self._workspace_dir(user_id), "memory")
        return ensure_dir(path)

    def _daily_path(self, user_id: str, date_str: str) -> str:
        return os.path.join(self._memory_dir(user_id), f"{date_str}.md")

    def _memory_path(self, user_id: str) -> str:
        return os.path.join(self._memory_dir(user_id), "memory.md")

    def _prompts_dir(self) -> str:
        return os.path.join(os.path.dirname(__file__), "prompts")

    def _ensure_identity_files(self, workspace: str) -> None:
        """Copy missing identity templates + bootstrap.md into workspace.

        Once all three identity files have been customized (differ from
        their templates), bootstrap.md is deleted automatically.
        """
        identity_map = {
            "soul.md": "soul.md",
            "identity.md": "identity.md",
            "user.md": "user.md",
        }
        prompts_dir = self._prompts_dir()
        bootstrap_dst = os.path.join(workspace, "bootstrap.md")

        missing = [ws for ws in identity_map if not os.path.exists(os.path.join(workspace, ws))]

        if missing:
            if not os.path.exists(bootstrap_dst):
                bootstrap_src = os.path.join(prompts_dir, "bootstrap.md")
                if os.path.exists(bootstrap_src):
                    shutil.copy2(bootstrap_src, bootstrap_dst)

            for ws_name in missing:
                dst = os.path.join(workspace, ws_name)
                src = os.path.join(prompts_dir, identity_map[ws_name])
                if os.path.exists(src):
                    shutil.copy2(src, dst)
        elif os.path.exists(bootstrap_dst):
            # All identity files exist — check if they've been customized
            all_customized = True
            for ws_name, tmpl_name in identity_map.items():
                tmpl = os.path.join(prompts_dir, tmpl_name)
                ws_file = os.path.join(workspace, ws_name)
                if os.path.exists(tmpl) and self._read(ws_file) == self._read(tmpl):
                    all_customized = False
                    break
            if all_customized:
                os.remove(bootstrap_dst)
                self.logger.info("Bootstrap complete — removed bootstrap.md")

    @staticmethod
    def _strip_frontmatter(text: str) -> str:
        """Remove YAML frontmatter (--- ... ---) from markdown content."""
        stripped = text.lstrip()
        if stripped.startswith("---"):
            parts = stripped.split("---", 2)
            if len(parts) >= 3:
                return parts[2]
        return text

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

    def _model_override_path(self, user_id: str) -> str:
        return os.path.join(self._memory_dir(user_id), "model_override.txt")

    def _private_session_path(self, user_id: str) -> str:
        return os.path.join(self._memory_dir(user_id), "private_session")

    def _private_thread_path(self, user_id: str, thread_id: str) -> str:
        return os.path.join(self._memory_dir(user_id), f"private_thread_{thread_id}")

    def _thread_daily_path(self, user_id: str, thread_id: str, date_str: str) -> str:
        return os.path.join(self._memory_dir(user_id), f"thread_{thread_id}_{date_str}.md")

    def _thread_model_path(self, user_id: str, thread_id: str) -> str:
        return os.path.join(self._memory_dir(user_id), f"thread_{thread_id}_model.txt")

    @staticmethod
    def _parse_exchanges(history: str) -> List[Tuple[str, str, Optional[List[Dict[str, Any]]]]]:
        """Parse flat history text into (user, assistant, tool_calls_info) pairs.

        Format:
            [HH:MM:SS] User: ...
            Assistant: ...
            <!-- TOOL_CALL: {"name": "...", "args": {...}, "result": "..."} -->
        """
        pattern = re.compile(
            r"\[[\d:]+\]\s*User:\s*(.*?)\nAssistant:\s*(.*?)(?=\n\[[\d:]+\]\s*User:|\Z)",
            re.DOTALL,
        )

        exchanges: List[Tuple[str, str, Optional[List[Dict[str, Any]]]]] = []
        for m in pattern.finditer(history):
            user_text = m.group(1).strip()
            bot_text = m.group(2).strip()

            # Parse tool call comments from bot_text
            tool_calls_info = None
            tool_pattern = re.compile(r'<!--\s*TOOL_CALL:\s*(\{.*?\})\s*-->', re.DOTALL)
            tool_matches = tool_pattern.findall(bot_text)

            if tool_matches:
                tool_calls_info = []
                # Remove tool comments from visible bot_text
                visible_bot_text = bot_text
                for match in tool_matches:
                    try:
                        tool_calls_info.append(json.loads(match))
                        # Remove the comment from visible text
                        visible_bot_text = visible_bot_text.replace(
                            f'<!-- TOOL_CALL: {match} -->', ''
                        ).strip()
                    except json.JSONDecodeError:
                        pass
                bot_text = visible_bot_text
            else:
                tool_calls_info = None

            exchanges.append((user_text, bot_text, tool_calls_info))
        return exchanges

    def load_session(self, user_id: str) -> Session:
        """Load or return cached session for a user.

        Returns the same Session instance for the same user_id, so all
        callers (agent, command handlers, etc.) share one object.
        """
        if user_id in self._sessions:
            return self._sessions[user_id]

        self._memory_dir(user_id)  # ensure dirs exist
        self._ensure_identity_files(self._workspace_dir(user_id))

        session = Session(user_id)
        session.daily_history = self._read(self._daily_path(user_id, session.current_date_str))
        session.user_memory = self._read(self._memory_path(user_id))
        session.summary = self._read(self._summary_path(user_id))
        session.exchanges = self._parse_exchanges(session.daily_history)
        session.exchange_count = len(session.exchanges)
        override = self._read(self._model_override_path(user_id)).strip()
        session.model_override = override or None
        session.private = os.path.isfile(self._private_session_path(user_id))

        self._sessions[user_id] = session
        return session

    def set_model_override(self, session: Session, model: Optional[str]) -> None:
        """Persist a model override for this session.  Pass None to clear."""
        session.model_override = model
        path = self._model_override_path(session.user_id)
        if model:
            with open(path, "w", encoding="utf-8") as f:
                f.write(model)
        elif os.path.exists(path):
            os.remove(path)

    def get_thread_context(
        self, session: Session, thread_id: str,
    ) -> List[Tuple[str, str]]:
        """Return the exchange list for a thread, loading from disk on first access.

        New threads start empty. Only exchanges from that thread are replayed
        into the model context.
        """
        if thread_id not in session.thread_contexts:
            path = self._thread_daily_path(
                session.user_id, thread_id, session.current_date_str
            )
            exchanges = self._parse_exchanges(self._read(path))
            session.thread_contexts[thread_id] = exchanges
        return session.thread_contexts[thread_id]

    def get_thread_model_override(self, session: Session, thread_id: str) -> Optional[str]:
        """Return the model override for a thread, loading from disk on first access."""
        if thread_id not in session.thread_model_overrides:
            val = self._read(self._thread_model_path(session.user_id, thread_id)).strip()
            session.thread_model_overrides[thread_id] = val or None
        return session.thread_model_overrides[thread_id]

    def set_thread_model_override(
        self, session: Session, thread_id: str, model: Optional[str]
    ) -> None:
        """Persist a model override for a specific thread.  Pass None to clear."""
        session.thread_model_overrides[thread_id] = model
        path = self._thread_model_path(session.user_id, thread_id)
        if model:
            with open(path, "w", encoding="utf-8") as f:
                f.write(model)
        elif os.path.exists(path):
            os.remove(path)

    def toggle_private_thread(self, session: Session, thread_id: str) -> bool:
        """Toggle private mode for a thread. Returns the new state."""
        path = self._private_thread_path(session.user_id, thread_id)
        if thread_id in session.private_threads:
            session.private_threads.discard(thread_id)
            if os.path.isfile(path):
                os.remove(path)
            return False
        session.private_threads.add(thread_id)
        with open(path, "w") as f:
            f.write("")
        return True

    def toggle_private(self, session: Session) -> bool:
        """Toggle session-level private mode. Returns the new state."""
        session.private = not session.private
        path = self._private_session_path(session.user_id)
        if session.private:
            with open(path, "w") as f:
                f.write("")
        elif os.path.isfile(path):
            os.remove(path)
        return session.private

    def append_thread_exchange(
        self,
        session: Session,
        thread_id: str,
        user_text: str,
        bot_text: str,
        tool_calls_info: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Append an exchange to a thread's log (RAM; disk skipped if private)."""
        exchanges = self.get_thread_context(session, thread_id)
        exchanges.append((user_text, bot_text, tool_calls_info))
        if thread_id in session.private_threads:
            return
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = f"\n[{timestamp}] User: {user_text}\nAssistant: {bot_text}"

        # Append tool call information as HTML comments (hidden from display)
        if tool_calls_info:
            for tc in tool_calls_info:
                tool_json = json.dumps(tc, ensure_ascii=False)
                entry += f"\n<!-- TOOL_CALL: {tool_json} -->"

        path = self._thread_daily_path(
            session.user_id, thread_id, session.current_date_str
        )
        with open(path, "a", encoding="utf-8") as f:
            f.write(entry)

    def build_system_prompt(
        self,
        session: Session,
        skills: Optional[Dict[str, Any]] = None,
        mode: str = "chat",
    ) -> str:
        """Build the system prompt from workspace identity files + memory.

        ``skills`` maps skill name → AgentSkill so the prompt can list
        each skill with its description.

        ``mode`` can add context-specific instructions, e.g. for live calls.
        """
        workspace = self._workspace_dir(session.user_id)
        self._ensure_identity_files(workspace)
        parts: list[str] = []

        # Only include the identity files that define the assistant's persona
        _IDENTITY_FILES = ("bootstrap.md", "identity.md", "user.md", "soul.md", "memory.md")
        ws_files = [f for f in _IDENTITY_FILES
                    if os.path.isfile(os.path.join(workspace, f))]

        for filename in ws_files:
            content = self._strip_frontmatter(
                self._read(os.path.join(workspace, filename))
            )
            if content.strip():
                parts.append(f"[Source: {filename}]\n{content.strip()}")

        if session.summary.strip():
            parts.append(
                f"## Conversation Summary\n{session.summary.strip()}"
            )

        # Recent exchanges are passed as structured HumanMessage/AIMessage
        # pairs (see ChatAgent.run), NOT included here as flat text.

        if session.user_memory.strip():
            parts.append(f"## Memory\n{session.user_memory.strip()}")

        parts.append(
            f"Current date and time: {datetime.now().strftime('%A, %d. %B %Y %H:%M')}"
        )

        mode_block = self._build_mode_instructions(mode)
        if mode_block:
            parts.append(mode_block)

        # Skill instructions
        skill_block = self._build_skill_instructions(skills or {})
        parts.append(skill_block)

        return "\n\n════════════════════\n\n".join(parts)

    @staticmethod
    def _build_mode_instructions(mode: str) -> str:
        """Build additional instructions for special conversation modes."""
        if mode == "call":
            return load_system_prompt("calls/live_call.md")
        return ""

    @staticmethod
    def _build_skill_instructions(skills: Dict[str, Any]) -> str:
        """Build explicit skill usage instructions for the system prompt."""
        lines = load_system_prompt("chat/skill_capabilities_intro.md").splitlines()
        for name, skill in skills.items():
            desc = getattr(skill, "description", "")
            if desc:
                lines.append(f"- **{name}**: {desc}")
            else:
                lines.append(f"- {name}")

        has_search = any(
            s in skills for s in ("perplexica", "searxng", "researcher")
        )

        has_memory = "memory" in skills

        lines.append("")
        lines.extend(load_system_prompt("chat/skill_rules.md").splitlines())
        if has_memory:
            lines.append(load_system_prompt("chat/memory_rule.md"))

        return "\n".join(lines)

    def append_exchange(
        self,
        session: Session,
        user_text: str,
        bot_text: str,
        *,
        track_similarity: bool = True,
        tool_calls_info: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Append a user/assistant exchange to the daily log (RAM + disk).

        When ``track_similarity`` is False the response is NOT added to
        the similarity window.  Use this for skill-backed responses whose
        content is inherently repetitive (e.g. file listings).

        ``tool_calls_info`` is a list of dicts with 'name', 'args', and 'result'
        keys representing tool calls made during this exchange.
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = f"\n[{timestamp}] User: {user_text}\nAssistant: {bot_text}"

        # Append tool call information as HTML comments (hidden from display)
        if tool_calls_info:
            for tc in tool_calls_info:
                tool_json = json.dumps(tc, ensure_ascii=False)
                entry += f"\n<!-- TOOL_CALL: {tool_json} -->"

        session.exchanges.append((user_text, bot_text, tool_calls_info))
        session.exchange_count += 1
        session.last_activity = datetime.now()

        if track_similarity:
            session.recent_bot_responses.append(bot_text)
            if len(session.recent_bot_responses) > SIMILARITY_WINDOW:
                session.recent_bot_responses.pop(0)

        if session.private:
            return

        session.daily_history += entry
        daily_path = self._daily_path(session.user_id, session.current_date_str)
        with open(daily_path, "a", encoding="utf-8") as f:
            f.write(entry)

    # ------------------------------------------------------------------
    # Summarization
    # ------------------------------------------------------------------

    def should_summarize(self, session: Session) -> str:
        """Check whether conversation should be summarized.

        Returns the trigger reason (empty string = no summary needed).
        The scheduler gates most triggers behind its own idle check
        (IDLE_SUMMARIZE_MIN); only "force" bypasses that gate.
        """
        if session.exchange_count >= FORCE_SUMMARY_EXCHANGES:
            return "force"

        if session.exchange_count >= MAX_EXCHANGES_BEFORE_SUMMARY:
            return "exchange_limit"

        if self._detect_repetition(session.recent_bot_responses):
            return "repetition"

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
        The last KEEP_RECENT_EXCHANGES exchanges are kept intact so the
        LLM always has immediate conversational context.
        """
        session.summary = summary_text.strip()
        session.daily_history = ""
        kept = session.exchanges[-KEEP_RECENT_EXCHANGES:]
        session.exchanges.clear()
        session.exchanges.extend(kept)
        session.exchange_count = len(kept)
        session.recent_bot_responses.clear()

        # Persist summary to disk alongside the daily log
        summary_path = self._summary_path(session.user_id)
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(session.summary)

        self.logger.info("Conversation summarized for %s", session.user_id)
