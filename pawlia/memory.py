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
import yaml

try:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
except ImportError:  # pragma: no cover — Python <3.9
    ZoneInfo = None  # type: ignore
    ZoneInfoNotFoundError = Exception  # type: ignore


def _local_now(tz_name: Optional[str]) -> datetime:
    """Return now() in the given IANA timezone, falling back to server local."""
    if tz_name and ZoneInfo is not None:
        try:
            return datetime.now(ZoneInfo(tz_name))
        except ZoneInfoNotFoundError:
            pass
        except Exception:
            pass
    return datetime.now()

from pawlia.prompt_utils import load_system_prompt
from pawlia.utils import ensure_dir

# Summarization trigger thresholds
MAX_EXCHANGES_BEFORE_SUMMARY = 20
FORCE_SUMMARY_EXCHANGES = 30  # force summarize even if user is active
KEEP_RECENT_EXCHANGES = 5  # exchanges to keep intact after summarization
SIMILARITY_THRESHOLD = 0.6  # 0-1, how similar two bot responses must be
SIMILARITY_WINDOW = 4  # compare last N bot responses
IDLE_TIMEOUT_SECONDS = 300  # 5 minutes
SESSION_FORMAT_VERSION = 2

# Cheap token estimate — 1 token ≈ 4 chars for mixed German/English text.
# Avoids loading tiktoken on every scheduler tick.
_CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Approximate token count for *text* using a char-based heuristic.

    Coarse but fast — good enough for trigger thresholds. Real tokenizers
    are 10-20× slower and would dominate the scheduler tick.
    """
    if not text:
        return 0
    return (len(text) + _CHARS_PER_TOKEN - 1) // _CHARS_PER_TOKEN


def estimate_session_tokens(session: "Session") -> int:
    """Rough token footprint of the model context built from a Session."""
    return (
        estimate_tokens(session.summary)
        + estimate_tokens(session.daily_history)
        + estimate_tokens(session.user_memory)
    )

_EXCHANGE_PATTERN = re.compile(
    r"\[[\d:]+\]\s*User:\s*(.*?)\nAssistant:\s*(.*?)(?=\n\[[\d:]+\]\s*User:|\Z)",
    re.DOTALL,
)
_TOOL_CALL_PATTERN = re.compile(r'<!--\s*TOOL_CALL:\s*(\{.*?\})\s*-->', re.DOTALL)


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
        # Partial override of config.yaml -> agents:
        self.agent_overrides: Dict[str, Any] = {}

        # Optional TTS voice override (piper voice name without .onnx)
        self.voice_override: Optional[str] = None

        # IANA timezone name (e.g. "Europe/Berlin") for time strings shown to
        # the model. None = fall back to the server's local time, which is
        # almost always wrong in containerized deployments (UTC).
        self.timezone: Optional[str] = None

        # Skills disabled for this session
        self.disabled_skills: List[str] = []

        # Per-thread exchange lists (loaded/seeded lazily by get_thread_context)
        self.thread_contexts: Dict[str, List[Tuple[str, str]]] = {}

        # Private mode: exchanges are kept in RAM but not written to disk.
        # Resets on restart (intentional).
        self.private: bool = False            # CLI / session-level
        self.private_threads: Set[str] = set()  # per-thread

        # Workspace context search: None = not yet run (triggers on first substantive turn),
        # [] = ran but found nothing, [...] = cached hits for this session.
        # Cleared and re-run when a topic shift is detected.
        self.workspace_refs: Optional[list] = None

        # Set by ChatAgent when a topic shift is detected; consumed by _persist()
        # to prepend a section heading to the daily log entry.
        self.pending_topic_heading: Optional[str] = None


_DEFAULT_READ_QUERY = "<keywords>"


def _format_workspace_refs(hits: list, user_query: str = "") -> str:
    """Format workspace hits as minimal pointers — model must call `files read`.

    Only the wikilink and section heading are shown; snippet content is
    intentionally NOT included. The model is expected to load the actual file
    via `files read` before answering, so claims trace back to the file rather
    than to a paraphrased preview that can drift.

    research/*/README.md hits are deduped against their project's content files
    so each project appears once with one ready-to-use read call.
    """
    read_query = _read_query_for(user_query)
    lines = [
        "## Workspace Notes Available",
        "These wiki files were keyword-matched against the user's recent "
        "message. They are *suggestions*, not proof of relevance: the match "
        "may be coincidental (shared words, similar topic name) without "
        "actually answering the user's question.",
        "",
        "**Rules:**",
        "- First decide whether the user is actually asking about one of these "
        "topics. If the conversation is heading somewhere else (small talk, "
        "follow-up, clarification, correction of an earlier mistake), ignore "
        "these refs entirely.",
        "- Only call `files read` if a heading clearly matches the user's "
        "current question. When in doubt, ask the user before reading.",
        "- Never weave content from these files into a reply unless you "
        "actually read the file *and* it answers what was asked. Do not "
        "invent a topic the user did not bring up.",
        "- Never invent a file path or heading that isn't listed here.",
        "",
    ]

    research_readmes: dict = {}   # project_dir -> (ref, description)
    research_content: dict = {}   # project_dir -> first content-file hit
    other_hits: list = []
    seen_content: set = set()

    for hit in hits:
        path = hit.path
        ref = hit.wikilink_ref
        parts = path.split("/")
        if len(parts) >= 3 and parts[0] == "research":
            project_dir = parts[1]
            if parts[-1] == "README.md":
                research_readmes[project_dir] = (ref, hit.snippet or "")
            else:
                if project_dir not in research_content and path not in seen_content:
                    research_content[project_dir] = hit
                    seen_content.add(path)
        else:
            if path not in seen_content:
                seen_content.add(path)
                other_hits.append(hit)

    rendered_projects: set = set()

    for project_dir, (readme_ref, description) in research_readmes.items():
        content_hit = research_content.get(project_dir)
        content_ref = content_hit.wikilink_ref if content_hit else readme_ref
        entry = f"- **{readme_ref}** — {description or project_dir}"
        if content_hit and content_hit.heading:
            entry += f" *(matched section: {content_hit.heading})*"
        entry += f"\n  → `files read --query \"{read_query}\" {content_ref}`"
        lines.append(entry)
        rendered_projects.add(project_dir)

    for project_dir, hit in research_content.items():
        if project_dir in rendered_projects:
            continue
        ref = hit.wikilink_ref
        entry = f"- **{ref}** *(research/{project_dir})*"
        if hit.heading:
            entry += f" *(section: {hit.heading})*"
        entry += f"\n  → `files read --query \"{read_query}\" {ref}`"
        lines.append(entry)

    for hit in other_hits:
        ref = hit.wikilink_ref
        entry = f"- **{ref}**"
        if hit.heading:
            entry += f" *(section: {hit.heading})*"
        entry += f"\n  → `files read --query \"{read_query}\" {ref}`"
        lines.append(entry)

    return "\n".join(lines)


def _read_query_for(user_query: str) -> str:
    """Distill the user message into a 3–6 word `files read` --query value.

    Falls back to a placeholder if the message is too short to extract content
    words. Keeps the call site simple — caller passes the raw user input.
    """
    text = (user_query or "").strip()
    if not text:
        return _DEFAULT_READ_QUERY
    tokens = [t for t in re.findall(r"\w+", text.lower()) if len(t) >= 4]
    seen: set = set()
    distilled: list = []
    for t in tokens:
        if t in seen:
            continue
        seen.add(t)
        distilled.append(t)
        if len(distilled) >= 6:
            break
    return " ".join(distilled) if distilled else _DEFAULT_READ_QUERY


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

    def _session_version_path(self, user_id: str) -> str:
        return os.path.join(self.session_dir, user_id, "session_version.txt")

    def _session_config_path(self, user_id: str) -> str:
        return os.path.join(self.session_dir, user_id, "config.yaml")

    def _agent_overrides_path(self, user_id: str) -> str:
        return os.path.join(self._memory_dir(user_id), "agent_overrides.yaml")

    def _voice_override_path(self, user_id: str) -> str:
        return os.path.join(self._memory_dir(user_id), "voice_override.txt")

    def _read_session_config(self, user_id: str) -> Dict[str, Any]:
        return self._read_yaml(self._session_config_path(user_id))

    def _write_session_config(self, user_id: str, data: Dict[str, Any]) -> None:
        path = self._session_config_path(user_id)
        if data:
            self._write_yaml(path, data)
        elif os.path.exists(path):
            os.remove(path)

    def _update_session_config(self, user_id: str, key: str, value: Any) -> None:
        data = self._read_session_config(user_id)
        if value is None or value == {} or value == []:
            data.pop(key, None)
        else:
            data[key] = value
        self._write_session_config(user_id, data)

    def _private_session_path(self, user_id: str) -> str:
        return os.path.join(self._memory_dir(user_id), "private_session")

    def _private_thread_path(self, user_id: str, thread_id: str) -> str:
        return os.path.join(self._memory_dir(user_id), f"private_thread_{thread_id}")

    def _thread_daily_path(self, user_id: str, thread_id: str, date_str: str) -> str:
        return os.path.join(self._memory_dir(user_id), f"thread_{thread_id}_{date_str}.md")

    def _thread_agent_overrides_path(self, user_id: str, thread_id: str) -> str:
        return os.path.join(self._memory_dir(user_id), f"thread_{thread_id}_agents.yaml")

    @staticmethod
    def _read_yaml(path: str) -> Dict[str, Any]:
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _write_yaml(path: str, data: Dict[str, Any]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)

    @staticmethod
    def _thread_marker_start(thread_id: str) -> str:
        return "<!-- PAWLIA_THREAD_SECTION -->"

    @staticmethod
    def _thread_marker_end(thread_id: str) -> str:
        return "<!-- /PAWLIA_THREAD_SECTION -->"

    @classmethod
    def _thread_section_pattern(cls, thread_id: Optional[str] = None) -> re.Pattern[str]:
        if thread_id is None:
            return re.compile(
                r"\n*## Thread [^\n]+\n<!-- PAWLIA_THREAD_SECTION -->\n?(.*?)\n<!-- /PAWLIA_THREAD_SECTION -->\n*",
                re.DOTALL,
            )
        escaped = re.escape(thread_id)
        return re.compile(
            rf"\n*## Thread {escaped}\n<!-- PAWLIA_THREAD_SECTION -->\n?(.*?)\n<!-- /PAWLIA_THREAD_SECTION -->\n*",
            re.DOTALL,
        )

    @staticmethod
    def _clean_agent_overrides(data: Any) -> Dict[str, Any]:
        if not isinstance(data, dict):
            return {}

        cleaned: Dict[str, Any] = {}
        for key in ("default", "defaults", "chat", "skill_runner", "vision", "compiler"):
            value = data.get(key)
            if isinstance(value, (str, dict)) and value:
                cleaned[key] = value

        skills = data.get("skills")
        if isinstance(skills, dict):
            cleaned_skills = {
                str(name): value
                for name, value in skills.items()
                if isinstance(value, (str, dict)) and value
            }
            if cleaned_skills:
                cleaned["skills"] = cleaned_skills

        return cleaned

    @staticmethod
    def _deep_merge_agent_overrides(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(base)
        for key, value in extra.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = MemoryManager._deep_merge_agent_overrides(merged[key], value)
            else:
                merged[key] = value
        return merged

    @staticmethod
    def _delete_nested_path(data: Dict[str, Any], parts: List[str]) -> None:
        current = data
        parents: List[Tuple[Dict[str, Any], str]] = []
        for part in parts[:-1]:
            child = current.get(part)
            if not isinstance(child, dict):
                return
            parents.append((current, part))
            current = child
        current.pop(parts[-1], None)
        if not current:
            for parent, key in reversed(parents):
                child = parent.get(key)
                if isinstance(child, dict) and not child:
                    parent.pop(key, None)
                else:
                    break

    def _sync_legacy_model_fields(self, session: Session) -> None:
        session.model_override = self.get_agent_override_value(session, "chat")

    def _read_session_version(self, user_id: str) -> int:
        raw = self._read(self._session_version_path(user_id)).strip()
        try:
            return int(raw)
        except (TypeError, ValueError):
            return 1

    def _write_session_version(self, user_id: str, version: int) -> None:
        with open(self._session_version_path(user_id), "w", encoding="utf-8") as f:
            f.write(str(version))

    @staticmethod
    def _get_nested_override(data: Dict[str, Any], path: str) -> Optional[str]:
        current: Any = data
        for part in path.split("."):
            if not isinstance(current, dict):
                return None
            current = current.get(part)
        return current if isinstance(current, str) and current else None

    def _daily_log_paths(self, user_id: str) -> List[str]:
        memory_dir = self._memory_dir(user_id)
        if not os.path.isdir(memory_dir):
            return []
        names = [
            name for name in os.listdir(memory_dir)
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}\.md", name)
        ]
        return [os.path.join(memory_dir, name) for name in sorted(names)]

    @classmethod
    def _extract_main_history(cls, daily_text: str) -> str:
        return cls._thread_section_pattern().sub("\n", daily_text).rstrip()

    @classmethod
    def _extract_thread_history(cls, daily_text: str, thread_id: str) -> str:
        match = cls._thread_section_pattern(thread_id).search(daily_text)
        return match.group(1).strip() if match else ""

    @staticmethod
    def _format_exchange_entry(
        user_text: str,
        bot_text: str,
        *,
        tool_calls_info: Optional[List[Dict[str, Any]]] = None,
        timestamp: Optional[str] = None,
        topic_heading: Optional[str] = None,
        tz_name: Optional[str] = None,
    ) -> str:
        stamp = timestamp or _local_now(tz_name).strftime("%H:%M:%S")
        entry = f"[{stamp}] User: {user_text}\nAssistant: {bot_text}"
        if tool_calls_info:
            for tc in tool_calls_info:
                tool_json = json.dumps(tc, ensure_ascii=False)
                entry += f"\n<!-- TOOL_CALL: {tool_json} -->"
        if topic_heading:
            entry = f"\n## {topic_heading}\n\n{entry}"
        return entry

    @classmethod
    def _render_thread_section(cls, thread_id: str, body: str) -> str:
        return (
            f"## Thread {thread_id}\n"
            f"{cls._thread_marker_start(thread_id)}\n"
            f"{body.strip()}\n"
            f"{cls._thread_marker_end(thread_id)}"
        )

    @classmethod
    def _upsert_thread_section(cls, daily_text: str, thread_id: str, block: str) -> str:
        pattern = cls._thread_section_pattern(thread_id)
        match = pattern.search(daily_text)
        cleaned_block = block.strip()
        if match:
            current = match.group(1).strip()
            body = f"{current}\n{cleaned_block}" if current else cleaned_block
            replacement = cls._render_thread_section(thread_id, body)
            return f"{daily_text[:match.start()]}{replacement}{daily_text[match.end():]}".rstrip() + "\n"

        section = cls._render_thread_section(thread_id, cleaned_block)
        if daily_text.strip():
            return daily_text.rstrip() + "\n\n" + section + "\n"
        return section + "\n"

    @classmethod
    def _append_main_entry_to_daily_text(cls, daily_text: str, entry: str) -> str:
        thread_match = re.search(r"^## Thread ", daily_text, re.MULTILINE)
        cleaned_entry = entry.strip()
        if thread_match:
            before = daily_text[:thread_match.start()].rstrip()
            after = daily_text[thread_match.start():].lstrip("\n")
            if before:
                return before + "\n" + cleaned_entry + "\n\n" + after
            return cleaned_entry + "\n\n" + after
        if daily_text.strip():
            return daily_text.rstrip() + "\n" + cleaned_entry + "\n"
        return cleaned_entry + "\n"

    def _write_daily_text(self, user_id: str, date_str: str, text: str) -> None:
        with open(self._daily_path(user_id, date_str), "w", encoding="utf-8") as f:
            f.write(text)

    def _append_main_entry_to_daily(self, user_id: str, date_str: str, entry: str) -> None:
        path = self._daily_path(user_id, date_str)
        current = self._read(path)
        self._write_daily_text(user_id, date_str, self._append_main_entry_to_daily_text(current, entry))

    def _append_thread_block_to_daily(self, user_id: str, date_str: str, thread_id: str, block: str) -> None:
        path = self._daily_path(user_id, date_str)
        current = self._read(path)
        self._write_daily_text(user_id, date_str, self._upsert_thread_section(current, thread_id, block))

    def migrate_session(self, user_id: str) -> int:
        """Migrate a session directory to the current on-disk log format."""
        if self._read_session_version(user_id) >= SESSION_FORMAT_VERSION:
            return 0

        migrated = 0
        memory_dir = self._memory_dir(user_id)
        for name in sorted(os.listdir(memory_dir)):
            match = re.fullmatch(r"thread_(.+)_(\d{4}-\d{2}-\d{2})\.md", name)
            if not match:
                continue
            thread_id, date_str = match.groups()
            legacy_path = os.path.join(memory_dir, name)
            legacy_body = self._read(legacy_path).strip()
            if legacy_body:
                self._append_thread_block_to_daily(user_id, date_str, thread_id, legacy_body)
                migrated += 1
            os.remove(legacy_path)

        self._write_session_version(user_id, SESSION_FORMAT_VERSION)
        return migrated

    @staticmethod
    def _parse_exchanges(history: str) -> List[Tuple[str, str, Optional[List[Dict[str, Any]]]]]:
        """Parse flat history text into (user, assistant, tool_calls_info) pairs.

        Format:
            [HH:MM:SS] User: ...
            Assistant: ...
            <!-- TOOL_CALL: {"name": "...", "args": {...}, "result": "..."} -->
        """
        exchanges: List[Tuple[str, str, Optional[List[Dict[str, Any]]]]] = []
        for m in _EXCHANGE_PATTERN.finditer(history):
            user_text = m.group(1).strip()
            bot_text = m.group(2).strip()

            tool_calls_info = None
            tool_matches = _TOOL_CALL_PATTERN.findall(bot_text)

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

    def _load_session_config_with_migration(self, user_id: str) -> Dict[str, Any]:
        """Read session config, migrating legacy files on first access."""
        data = self._read_session_config(user_id)
        changed = False

        legacy_agents = self._agent_overrides_path(user_id)
        if "agents" not in data and os.path.isfile(legacy_agents):
            agents = self._clean_agent_overrides(self._read_yaml(legacy_agents))
            if agents:
                data["agents"] = agents
            os.remove(legacy_agents)
            self.logger.info("Migrated agent_overrides.yaml → session config for '%s'", user_id)
            changed = True

        legacy_voice = self._voice_override_path(user_id)
        if "tts" not in data and os.path.isfile(legacy_voice):
            voice = self._read(legacy_voice).strip()
            if voice:
                data["tts"] = {"voice": voice}
            os.remove(legacy_voice)
            self.logger.info("Migrated voice_override.txt → session config for '%s'", user_id)
            changed = True

        if changed:
            self._write_session_config(user_id, data)

        return data

    def load_session(self, user_id: str) -> Session:
        """Load or return cached session for a user.

        Returns the same Session instance for the same user_id, so all
        callers (agent, command handlers, etc.) share one object.
        """
        if user_id in self._sessions:
            return self._sessions[user_id]

        self._memory_dir(user_id)  # ensure dirs exist
        self.migrate_session(user_id)
        self._ensure_identity_files(self._workspace_dir(user_id))

        session = Session(user_id)
        today_text = self._read(self._daily_path(user_id, session.current_date_str))
        session.daily_history = self._extract_main_history(today_text)
        session.user_memory = self._read(self._memory_path(user_id))
        session.summary = self._read(self._summary_path(user_id))
        session.exchanges = self._parse_exchanges(session.daily_history)
        session.exchange_count = len(session.exchanges)
        session_cfg = self._load_session_config_with_migration(user_id)
        session.agent_overrides = self._clean_agent_overrides(
            session_cfg.get("agents") or {}
        )
        self._sync_legacy_model_fields(session)
        session.voice_override = (session_cfg.get("tts") or {}).get("voice") or None
        session.disabled_skills = [
            str(s) for s in (session_cfg.get("disabled_skills") or []) if s
        ]
        session.timezone = (session_cfg.get("user") or {}).get("timezone") or None
        session.private = os.path.isfile(self._private_session_path(user_id))

        self._sessions[user_id] = session
        return session

    def get_agent_overrides(self, session: Session) -> Dict[str, Any]:
        return dict(session.agent_overrides)

    def get_thread_agent_overrides(self, session: Session, thread_id: str) -> Dict[str, Any]:
        return self.get_agent_overrides(session)

    def effective_agent_overrides(
        self,
        session: Session,
        thread_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self._clean_agent_overrides(dict(session.agent_overrides))

    def get_agent_override_value(
        self,
        session: Session,
        path: str,
        thread_id: Optional[str] = None,
    ) -> Optional[str]:
        return self._get_nested_override(session.agent_overrides, path)

    def set_agent_overrides(
        self,
        session: Session,
        overrides: Optional[Dict[str, Any]],
    ) -> None:
        cleaned = self._clean_agent_overrides(overrides or {})
        session.agent_overrides = cleaned
        self._update_session_config(session.user_id, "agents", cleaned or None)
        self._sync_legacy_model_fields(session)

    def set_agent_override_value(
        self,
        session: Session,
        path: str,
        value: Optional[str],
    ) -> None:
        target = self.get_agent_overrides(session)
        parts = [part for part in path.split(".") if part]
        if not parts:
            return

        if value:
            current = target
            for part in parts[:-1]:
                child = current.get(part)
                if not isinstance(child, dict):
                    child = {}
                    current[part] = child
                current = child
            current[parts[-1]] = value
        else:
            self._delete_nested_path(target, parts)

        self.set_agent_overrides(session, target)

    def set_model_override(self, session: Session, model: Optional[str]) -> None:
        """Persist a model override for this session.  Pass None to clear."""
        self.set_agent_override_value(session, "chat", model)

    def set_voice_override(self, session: Session, voice: Optional[str]) -> None:
        """Persist a TTS voice override for this session.  Pass None to clear."""
        session.voice_override = voice
        cfg = self._read_session_config(session.user_id)
        tts = dict(cfg.get("tts") or {})
        if voice:
            tts["voice"] = voice
        else:
            tts.pop("voice", None)
        cfg["tts"] = tts if tts else None
        if not cfg.get("tts"):
            cfg.pop("tts", None)
        self._write_session_config(session.user_id, cfg)

    def set_disabled_skills(self, session: Session, skills: List[str]) -> None:
        """Persist the disabled_skills list for this session."""
        cleaned = [str(s) for s in skills if s]
        session.disabled_skills = cleaned
        self._update_session_config(session.user_id, "disabled_skills", cleaned or None)

    def add_disabled_skill(self, session: Session, skill: str) -> None:
        """Add a skill to the disabled list (idempotent)."""
        if skill not in session.disabled_skills:
            self.set_disabled_skills(session, session.disabled_skills + [skill])

    def remove_disabled_skill(self, session: Session, skill: str) -> None:
        """Remove a skill from the disabled list."""
        self.set_disabled_skills(session, [s for s in session.disabled_skills if s != skill])

    def get_thread_context(
        self, session: Session, thread_id: str,
    ) -> List[Tuple[str, str]]:
        """Return the exchange list for a thread, loading from disk on first access.

        New threads start empty. Only exchanges from that thread are replayed
        into the model context.
        """
        if thread_id not in session.thread_contexts:
            exchanges: List[Tuple[str, str, Optional[List[Dict[str, Any]]]]] = []
            for path in self._daily_log_paths(session.user_id):
                thread_history = self._extract_thread_history(self._read(path), thread_id)
                if thread_history:
                    exchanges.extend(self._parse_exchanges(thread_history))
            session.thread_contexts[thread_id] = exchanges
        return session.thread_contexts[thread_id]

    def get_thread_model_override(self, session: Session, thread_id: str) -> Optional[str]:
        """Return the session-wide chat model override.

        Thread-specific model overrides no longer exist; threads inherit the
        same per-session agent selection.
        """
        return session.model_override

    def set_thread_model_override(
        self, session: Session, thread_id: str, model: Optional[str]
    ) -> None:
        """Persist the session-wide chat model override.

        Thread-specific model overrides no longer exist; threads inherit the
        same per-session agent selection.
        """
        self.set_model_override(session, model)

    def toggle_private_thread(self, session: Session, thread_id: str) -> bool:
        """Toggle private mode for a thread. Returns the new state."""
        path = self._private_thread_path(session.user_id, thread_id)
        if thread_id in session.private_threads:
            session.private_threads.discard(thread_id)
            if os.path.isfile(path):
                os.remove(path)
            return False
        session.private_threads.add(thread_id)
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return True

    def toggle_private(self, session: Session) -> bool:
        """Toggle session-level private mode. Returns the new state."""
        session.private = not session.private
        path = self._private_session_path(session.user_id)
        if session.private:
            with open(path, "w", encoding="utf-8") as f:
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
        """Append an exchange to a thread section in the daily log."""
        exchanges = self.get_thread_context(session, thread_id)
        exchanges.append((user_text, bot_text, tool_calls_info))
        if thread_id in session.private_threads:
            return
        entry = self._format_exchange_entry(
            user_text,
            bot_text,
            tool_calls_info=tool_calls_info,
            tz_name=session.timezone,
        )
        self._append_thread_block_to_daily(session.user_id, session.current_date_str, thread_id, entry)

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
        bootstrap_active = "bootstrap.md" in ws_files

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

        # workspace_refs are injected into the user message (see chat.py
        # _augment_with_workspace_refs), not the system prompt — that's what
        # actually triggers the model to call the files skill.

        if session.timezone:
            now_str = _local_now(session.timezone).strftime("%A, %d. %B %Y %H:%M")
            parts.append(
                f"Current date and time: {now_str} ({session.timezone}). "
                "This is the user's local time — always use it directly; "
                "do not convert or apply offsets."
            )
        else:
            now_str = datetime.now().strftime("%A, %d. %B %Y %H:%M")
            parts.append(
                f"Current date and time: {now_str} (server local time — user "
                "has not configured a timezone in session config; ask the user "
                "or set it via the `config` skill if precise local time matters)."
            )

        mode_block = self._build_mode_instructions(mode)
        if mode_block:
            parts.append(mode_block)

        # Skill instructions
        skill_block = self._build_skill_instructions(
            skills or {}, bootstrap_active=bootstrap_active,
        )
        parts.append(skill_block)

        return "\n\n════════════════════\n\n".join(parts)

    @staticmethod
    def _build_mode_instructions(mode: str) -> str:
        """Build additional instructions for special conversation modes."""
        if mode == "call":
            return load_system_prompt("calls/live_call.md")
        return ""

    @staticmethod
    def _build_skill_instructions(
        skills: Dict[str, Any], *, bootstrap_active: bool = False,
    ) -> str:
        """Build explicit skill usage instructions for the system prompt.

        During bootstrap, the skill rules ("only answer directly for
        greetings") conflict with the bootstrap script ("first message must
        be …"). We trim down to just the capability list so the bootstrap
        instructions at the top of the prompt win.
        """
        lines = load_system_prompt("chat/skill_capabilities_intro.md").splitlines()
        for name, skill in skills.items():
            desc = getattr(skill, "description", "")
            if desc:
                lines.append(f"- **{name}**: {desc}")
            else:
                lines.append(f"- {name}")

        if bootstrap_active:
            lines.append("")
            lines.append(
                "Bootstrap is active — follow the script at the top of this "
                "prompt before doing anything else."
            )
            return "\n".join(lines)

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
        topic_heading: Optional[str] = None,
    ) -> None:
        """Append a user/assistant exchange to the daily log (RAM + disk).

        When ``track_similarity`` is False the response is NOT added to
        the similarity window.  Use this for skill-backed responses whose
        content is inherently repetitive (e.g. file listings).

        ``tool_calls_info`` is a list of dicts with 'name', 'args', and 'result'
        keys representing tool calls made during this exchange.

        ``topic_heading`` inserts a markdown section heading before the entry
        in the daily log, marking a topic shift for Dream Wiki segmentation.
        """
        entry = self._format_exchange_entry(
            user_text,
            bot_text,
            tool_calls_info=tool_calls_info,
            topic_heading=topic_heading,
            tz_name=session.timezone,
        )

        session.exchanges.append((user_text, bot_text, tool_calls_info))
        session.exchange_count += 1
        session.last_activity = datetime.now()

        if track_similarity:
            session.recent_bot_responses.append(bot_text)
            if len(session.recent_bot_responses) > SIMILARITY_WINDOW:
                session.recent_bot_responses.pop(0)

        if session.private:
            return

        session.daily_history = (
            session.daily_history.rstrip() + "\n" + entry
            if session.daily_history.strip() else entry
        )
        self._append_main_entry_to_daily(session.user_id, session.current_date_str, entry)

    # ------------------------------------------------------------------
    # Summarization
    # ------------------------------------------------------------------

    def should_summarize(
        self,
        session: Session,
        summary_threshold_tokens: Optional[int] = None,
    ) -> str:
        """Check whether conversation should be summarized.

        Returns the trigger reason (empty string = no summary needed).
        The scheduler gates most "soft" triggers behind its own idle check
        (IDLE_SUMMARIZE_MIN); "force" / "tokens_force" bypass that gate.

        ``summary_threshold_tokens`` (if provided) enables token-based
        triggering against the model's context window:
        - reaching the threshold returns "tokens"
        - reaching 1.5× the threshold returns "tokens_force" (bypasses idle)
        """
        if session.exchange_count >= FORCE_SUMMARY_EXCHANGES:
            return "force"

        if summary_threshold_tokens and summary_threshold_tokens > 0:
            tokens = estimate_session_tokens(session)
            if tokens >= summary_threshold_tokens * 3 // 2:
                return "tokens_force"
            if tokens >= summary_threshold_tokens:
                return "tokens"

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
