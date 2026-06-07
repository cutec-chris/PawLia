"""ChatAgent - conversational dispatcher that delegates work to SkillRunners.

The ChatAgent has NO tools of its own. It only knows about available skills
(via their OpenAI function specs). When the LLM decides a skill is needed,
the ChatAgent spawns a SkillRunnerAgent to do the actual work, then
incorporates the result into its final response.
"""

import asyncio
import json
import logging
import os
import re
import uuid
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, List, Optional, Tuple

# Callback types
InterimCallback = Callable[[str], Awaitable[None]]
SkillStartCallback = Callable[[str, str], Awaitable[None]]  # (skill_name, query)
SkillDoneCallback = Callable[[str, str], Awaitable[None]]  # (skill_name, result)

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI

from pawlia.agents.base import BaseAgent, log_prompt
from pawlia.agents.iteration_budget import IterationBudget
from pawlia.prompt_utils import load_system_prompt
from pawlia.skills.loader import AgentSkill

if TYPE_CHECKING:
    from pawlia.memory import MemoryManager, Session

_SENTENCE_RE = re.compile(r'[.!?…]\s')
_RE_CODE_BLOCK = re.compile(r'```[^\n]*\n(.*?)(?:```|$)', re.DOTALL)
_RE_TOOL_CALL_TAG = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.DOTALL)
# Small local models like to print file content in markdown blocks with the
# filename as a heading instead of calling the files skill. Match the first
# heading like "# identity.md - …" and recover it as a files write call.
_RE_FILENAME_HEADING = re.compile(
    r"^\s*#+\s+(?P<name>[\w./\-]+\.(?:md|txt|json|yaml|yml))\b",
    re.IGNORECASE,
)
# Same models also write bare "delete bootstrap.md" commands in code blocks.
_RE_DELETE_COMMAND = re.compile(
    r"^\s*(?:delete|rm)\s+(?P<name>[\w./\-]+\.(?:md|txt|json|yaml|yml))\s*$",
    re.IGNORECASE,
)

_FAKE_TOOL_CALL_NUDGE = (
    "You wrote a tool call as plain text or a code block instead of using the "
    "actual function-call mechanism. Do NOT write commands as text. "
    "Use a real tool call now."
)
_DISABLED_SKILL_NUDGE = (
    "That skill is not available in this session. "
    "Do NOT attempt to call it again. Answer directly from what you already know."
)
_MAX_FAKE_TOOL_RETRIES = 5
_EMPTY_TURN2_NUDGE = (
    "You have reached the maximum number of tool calls. "
    "Do NOT call any more tools. "
    "Summarize what you found so far and give the user a direct text response."
)
_CHAT_CONTINUE_NUDGE = (
    "The task is not complete yet. If you need a skill, call it now instead of describing "
    "what you plan to do. Only answer the user once the requested work is actually done."
)
_WORKSPACE_GROUNDING_NUDGE = (
    "Relevant workspace sections are listed above. Before answering factual questions about "
    "those notes, read the most relevant section with the files skill now. Prefer `files read-section` "
    "or reading the exact section-ref instead of answering from memory."
)
_MAX_CHAT_TOOL_TURNS = 16
_MAX_CHAT_NUDGES = 3
_REPLAY_TOOL_RESULT_LIMIT = 240
_REPLAY_TOOL_CALLS_LIMIT = 3
_LIVE_TOOL_RESULT_LIMIT = 12_000
_PERSIST_TOOL_RESULT_LIMIT = 2_000
_CONTEXT_COMPLETION_RESERVE_TOKENS = 2_000
_CONTEXT_MIN_NON_SYSTEM_KEEP = 6
_CONTEXT_TRIMMED_TEXT_LIMIT = 1_500
_CONTEXT_TRIMMED_TOOL_LIMIT = 4_000
# Preview length when condensing dropped messages in Phase 2 of the context-budget
# pass. Kept in line with llm.py `_summarize_to_fit` (commit 1cf68fc) so the two
# compression paths lose roughly the same amount of context.
_SUMMARY_PREVIEW_CHARS = 500
_MAX_CONTEXT_RECOVERY_RETRIES = 3
_DEFERRED_TOOL_INTENT_RE = re.compile(
    r"(?:\b(?:let me|i(?:'ll| will| am going to)|first\s*,?\s*i(?:'ll| will)|now\s+i(?:'ll| will)|"
    r"ich\s+(?:werde|schaue|suche|pruefe|prüfe|checke|oeffne|öffne)|lass\s+mich)\b.{0,140}"
    r"\b(?:search|look\s+up|check|browse|inspect|open|use|call|run|internet|web|online|browser|tool|skill|script|"
    r"recherch(?:e|ieren)?|suche|schaue|prüfe|pruefe|checke|öffne|oeffne|internet|online|tool|skill|skript)\b)",
    re.IGNORECASE | re.DOTALL,
)
_DEFERRED_PLACEHOLDER_RE = re.compile(
    r"(?:\b(?:please\s+wait|one\s+moment|just\s+a\s+moment|"
    r"working\s+on\s+it|processing(?:\s+your)?\s+(?:request|message|audio|recording)|"
    r"bitte\s+warten|warte(?:\s+kurz)?|einen\s+moment|einen\s+augenblick|"
    r"ich\s+(?:verarbeite|bearbeite|prüfe|pruefe)\s+(?:die|deine|ihre)?\s*"
    r"(?:aufnahme|anfrage|nachricht)?))\b",
    re.IGNORECASE,
)


def _augment_with_workspace_refs(user_input: str, session: Any) -> tuple[str | None, str]:
    """Return (context_block, clean_user_input).

    When workspace refs are available, *context_block* is a self-contained
    preamble referencing matched wiki sections.  It is returned separately so
    callers can inject it as a *distinct* HumanMessage — the model sees it as
    auxiliary context rather than part of the user's current utterance.

    When no refs exist *context_block* is ``None``.
    """
    if not session or not session.workspace_refs:
        return None, user_input
    try:
        from pawlia.memory import _format_workspace_refs
        block = _format_workspace_refs(session.workspace_refs, user_query=user_input)
        context = (
            "[Workspace context — optional reference material]\n"
            "The sections below were keyword-matched from the user's workspace.\n"
            "They *may* be relevant to the current question, or they may be a\n"
            "coincidental match.  Only read and use them if they clearly answer\n"
            "what the user just asked.\n\n"
            + block
        )
        return context, user_input
    except Exception:
        return None, user_input


def _remove_workspace_refs_from_messages(
    messages: List[BaseMessage],
    original_user_input: str,
) -> List[BaseMessage]:
    """Drop the workspace-context preamble after it served its routing purpose.

    The context block is a separate HumanMessage prepended by
    ``_augment_with_workspace_refs``.  It is removed entirely (not replaced with
    a stub) so tool-loop turns don't keep re-reading the same suggestions.
    The actual user message (the next HumanMessage) is left untouched.
    """
    cleaned = list(messages)
    for idx, msg in enumerate(cleaned):
        if (
            isinstance(msg, HumanMessage)
            and isinstance(msg.content, str)
            and msg.content.startswith("[Workspace context")
        ):
            cleaned.pop(idx)
            break
    return cleaned


def _split_sentences(text: str) -> Tuple[List[str], str]:
    """Split *text* into complete sentences and a remainder.

    A sentence boundary is punctuation (. ! ? …) followed by whitespace.
    Returns ``(complete_sentences, remaining_text)``.
    """
    sentences: List[str] = []
    while True:
        m = _SENTENCE_RE.search(text)
        if not m:
            break
        end = m.start() + 1  # include the punctuation char
        sentences.append(text[:end].strip())
        text = text[end:].lstrip()
    return sentences, text


DEFAULT_SYSTEM_PROMPT = load_system_prompt("chat/default.md")


class ChatAgent(BaseAgent):
    """Dispatcher agent - no tools, only skill descriptions.

    For every task that requires tools, it delegates to a SkillRunnerAgent
    via the ``skill_runner_factory`` callback.
    """


    def __init__(
        self,
        llm: ChatOpenAI,
        skills: Dict[str, AgentSkill],
        skill_runner_factory: Callable[[AgentSkill, Optional[str]], Any],
        logger: Optional[logging.Logger] = None,
        memory: Optional["MemoryManager"] = None,
        session: Optional["Session"] = None,
        on_interim: Optional[InterimCallback] = None,
        vision_llm: Optional[ChatOpenAI] = None,
        workspace_search_cfg: Optional[Dict[str, Any]] = None,
        max_tool_turns: Optional[int] = None,
        direct_tools: Optional[Dict[str, "Tool"]] = None,
        attachment_cfg: Optional[Dict[str, Any]] = None,
        llm_factory: Optional[Any] = None,
    ):
        super().__init__(llm, logger)
        self._all_skills = dict(skills)  # unfiltered — kept for re-enable support
        self.skills = skills
        self.skill_runner_factory = skill_runner_factory
        self.memory = memory
        self.session = session
        self.on_interim = on_interim
        self._workspace_search_cfg: Dict[str, Any] = workspace_search_cfg or {}
        self._attachment_cfg: Dict[str, Any] = dict(attachment_cfg or {})
        self.on_skill_start: Optional[SkillStartCallback] = None  # (skill_name, query)
        self.on_skill_step: Optional[InterimCallback] = None      # (step_description)
        self.on_skill_done: Optional[SkillDoneCallback] = None    # (skill_name, result)
        self.on_model_change: Optional[Callable[[str], None]] = None  # (new_model)
        self._on_fallback: Optional[Callable[[str, str], None]] = None  # (from_model, to_model)
        self.max_tool_turns = max_tool_turns if (isinstance(max_tool_turns, int) and max_tool_turns > 0) else _MAX_CHAT_TOOL_TURNS

        # Direct tools (e.g. read_file, list_files) — executed inline without SkillRunner
        self.direct_tools: Dict[str, "Tool"] = dict(direct_tools) if direct_tools else {}

        # Files queued by direct tools (e.g. attach_file) to be sent as part
        # of the next reply. Drained by the chat interface after run() returns.
        # Each entry: {"data": bytes, "mimetype": str, "filename": str,
        #              "caption": Optional[str], "size": int}.
        self.pending_attachments: List[Dict[str, Any]] = []

        # Bind all tool specs (skills + direct tools) so the LLM can call them
        self._bind_all_tools(llm, vision_llm)

        # Resolver for session/thread-specific agent selection at run() time.
        # Set by App.make_agent after construction.
        self._agent_llm_resolver: Optional[Callable[[str, Optional[str]], Any]] = None
        # Resolves config keys (e.g. "fast") to actual model names (e.g. "qwen3.5:4b").
        self._model_name_resolver: Optional[Callable[[str], str]] = None

        # Callback to re-discover workspace skills (set by App.make_agent).
        # Called after each skill returns so that skills created at runtime
        # (e.g. by skill-creator) become available immediately.
        self._skills_refresher: Optional[Callable[[], None]] = None
        # Optional callback returning the active context window for an agent type.
        self._context_window_resolver: Optional[Callable[[str, Optional[str]], int]] = None

        # LLM factory + per-thread agent overrides — used to probe whether the
        # image-handling model can actually see, and (when it can't) to borrow
        # a vision-capable model from the fallback chain to describe images.
        self._llm_factory: Optional[Any] = llm_factory
        self._overrides_resolver: Optional[Callable[[Optional[str]], Dict[str, Any]]] = None

    def _bind_all_tools(self, llm=None, vision_llm=None) -> None:
        """Bind skill specs + direct tool specs to the LLM."""
        base_llm = llm or self.llm
        vision_llm = vision_llm or getattr(self, "vision_llm", None)
        self._skill_specs = [s.as_openai_spec() for s in self.skills.values()]
        self._direct_tool_specs = [t.as_openai_spec() for t in self.direct_tools.values()]
        all_specs = self._skill_specs + self._direct_tool_specs
        if all_specs:
            self.bound_llm = base_llm.bind_tools(all_specs, tool_choice="auto")
            self.vision_bound_llm = (vision_llm or base_llm).bind_tools(
                all_specs, tool_choice="auto"
            )
        else:
            self.bound_llm = base_llm
            self.vision_bound_llm = vision_llm or base_llm

    def _apply_disabled_skills(self) -> None:
        """Filter self.skills against session.disabled_skills and rebind LLM tools."""
        if not (self.memory and self.session):
            return
        disabled = set(self.session.disabled_skills or [])
        self.skills = {k: v for k, v in self._all_skills.items() if k not in disabled}
        self._bind_all_tools()
        self.logger.info(
            "Skills rebound: %d active, %d disabled",
            len(self.skills), len(disabled),
        )

    def _run_workspace_search(self, query: str) -> None:
        """Run BM25 search over workspace and cache results on session."""
        if not (self.memory and self.session):
            return
        try:
            from pawlia.workspace_search import WorkspaceSearch
            workspace = self.memory._workspace_dir(self.session.user_id)
            # Bootstrap mode: workspace only contains identity templates, which
            # the system prompt already includes verbatim. Searching them
            # injects duplicate snippets plus instructions ("reply from the
            # snippet") that conflict with the bootstrap script.
            if os.path.isfile(os.path.join(workspace, "bootstrap.md")):
                self.session.workspace_refs = []
                return
            hits = WorkspaceSearch(workspace, config=self._workspace_search_cfg).search(query)
            self.session.workspace_refs = hits
            if hits:
                self.logger.debug(
                    "Workspace search: %d hit(s) for %r", len(hits), query[:60]
                )
        except Exception as exc:
            self.logger.warning("Workspace search failed: %s", exc)
            self.session.workspace_refs = []  # prevent retry on next turn

    def _handle_workspace_context(self, user_input: str, *, allow_workspace_search: bool = False) -> None:
        """Re-run workspace search on every substantive message.

        BM25 over the workspace is cheap, and caching hits from an earlier
        question lets follow-ups about a different aspect of the same topic
        run on stale snippets — which leads to hallucinations when the cached
        hits don't cover the new sub-question. Always re-searching keeps the
        injected context aligned with what the user is actually asking right now.

        - Skips short/small-talk messages ("hi", "ok", "danke", …).
        - Marks a topic heading in the daily log on significant shifts.
        - Disabled by default; enable per session via ``workspace-search.enabled: true``
          in the session config.
        """
        if not (self.memory and self.session):
            return
        if not allow_workspace_search and not self._workspace_search_cfg.get("enabled", False):
            return
        try:                                                               
            from pawlia.workspace_search import WorkspaceSearch
        except ImportError:
            return

        if not WorkspaceSearch.is_substantive(user_input):
            return  # small talk — skip entirely

        if (
            self.session.workspace_refs is not None
            and WorkspaceSearch.is_topic_shift(user_input, self.session.exchanges)
        ):
            self.session.pending_topic_heading = WorkspaceSearch.make_topic_heading(user_input)

        self.session.workspace_refs = None
        self._run_workspace_search(user_input)

    def build_system_prompt(
        self,
        *,
        mode: str = "chat",
        system_prompt: Optional[str] = None,
        thread_id: Optional[str] = None,
        extra_context: Optional[str] = None,
    ) -> str:
        """Resolve the system prompt for a chat or call context."""
        if system_prompt:
            return system_prompt
        if self.memory and self.session:
            return self.memory.build_system_prompt(
                self.session,
                skills=self.skills,
                mode=mode,
                extra_context=extra_context,
            )
        return DEFAULT_SYSTEM_PROMPT

    def _refresh_and_rebind_skills(self) -> bool:
        """Re-discover workspace skills and rebind LLM tools if new skills appeared.

        Called after each skill returns.  Returns True if tools were rebound
        (caller must update its local ``active_llm`` reference).
        """
        if not self._skills_refresher:
            return False

        prev_count = len(self._skill_specs)
        self._skills_refresher()  # updates self.skills with ALL skills (no filter)

        # Capture any new workspace skills into _all_skills, then re-apply disabled filter.
        self._all_skills.update(self.skills)
        self._apply_disabled_skills()

        if len(self.skills) == prev_count:
            return False

        return True

    async def _execute_tool_calls(
        self,
        tool_calls: List[Dict],
        thread_id: Optional[str],
        on_skill_start: Optional[SkillStartCallback],
        on_skill_step: Optional[InterimCallback],
        on_skill_done: Optional[SkillDoneCallback],
    ) -> Tuple[List[ToolMessage], List[Dict[str, Any]], bool]:
        """Execute a batch of LLM tool calls and return results.

        Skills are delegated to a SkillRunnerAgent; direct tools are executed
        inline without spawning a sub-agent.

        Returns (tool_messages, tool_calls_info_entries, skill_creator_called).
        """
        tool_messages: List[ToolMessage] = []
        tool_calls_info: List[Dict[str, Any]] = []
        skill_creator_called = False

        for tool_call in tool_calls:
            tool_name = str(tool_call.get("name", "") or "").strip()
            resolved_name = self._resolve_skill_name(tool_name) or tool_name

            # ---- Direct tool path (no SkillRunner overhead) ----
            direct_tool = self.direct_tools.get(resolved_name)
            if direct_tool:
                raw_args = tool_call.get("args", {}) or {}
                normalized_args = dict(raw_args) if isinstance(raw_args, dict) else {}
                query = json.dumps(normalized_args, ensure_ascii=False)
                self.logger.info("Direct tool '%s': %s", resolved_name, query[:80])
                if on_skill_start:
                    try:
                        await on_skill_start(resolved_name, query)
                    except Exception as exc:
                        self.logger.debug("on_skill_start error: %s", exc)
                context = {"user_id": "", "session_dir": ""}
                if self.session:
                    context["user_id"] = getattr(self.session, "user_id", "")
                if self.memory:
                    context["session_dir"] = self.memory.session_dir
                # Direct tools (e.g. attach_file) use this to push files onto
                # the agent's outgoing attachment queue.
                context["agent"] = self
                # Attachment policy from the [attachments] config section.
                context["max_outgoing_bytes"] = int(
                    self._attachment_cfg.get("max_outgoing_bytes") or 26214400
                )
                context["attachment_extra_roots"] = list(
                    self._attachment_cfg.get("extra_allowed_roots") or []
                )
                try:
                    result = direct_tool.execute(normalized_args, context=context)
                except Exception as exc:
                    result = f"Error: {exc}"
                raw_result = result
                result = self._wrap_with_trust_header(result, direct_tool, query)
                if on_skill_done:
                    try:
                        await on_skill_done(resolved_name, result)
                    except Exception as exc:
                        self.logger.debug("on_skill_done error: %s", exc)
                tool_calls_info.append({
                    "name": resolved_name,
                    "args": normalized_args,
                    "result": self._limit_tool_result(raw_result, limit=_PERSIST_TOOL_RESULT_LIMIT),
                })
                tool_messages.append(ToolMessage(
                    content=self._limit_tool_result(result),
                    tool_call_id=tool_call.get("id", ""),
                ))
                continue

            # ---- Skill path (delegate to SkillRunner) ----
            skill_name, normalized_args, error = self._decode_skill_call(tool_call)
            query = normalized_args.get("query", "")
            skill = self.skills.get(skill_name)

            if error:
                self.logger.warning("Skill call rejected: %s", error)
                result = error
                raw_result = result
            elif skill:
                self.logger.info("Delegating to skill '%s': %s", skill_name, query[:80])
                if on_skill_start:
                    try:
                        await on_skill_start(skill_name, query)
                    except Exception as exc:
                        self.logger.debug("on_skill_start error: %s", exc)
                runner = self.skill_runner_factory(skill, thread_id)
                runner.on_step = on_skill_step
                # Serialise skills that must not run concurrently per user.
                # skill-creator writes to the workspace and hammers the same
                # rate-limited LLM; two simultaneous instances interfere.
                _SERIALIZED_SKILLS = {"skill-creator"}
                if skill_name in _SERIALIZED_SKILLS and self.session:
                    _lock = self.session.get_skill_lock(skill_name)
                    if _lock.locked():
                        self.logger.info(
                            "Queuing '%s': another instance is already running for this user",
                            skill_name,
                        )
                    async with _lock:
                        result = await runner.run(query=query)
                else:
                    result = await runner.run(query=query)
                result = self._process_directives(result, thread_id)
                raw_result = result
                result = self._wrap_with_trust_header(result, skill, query)
                if skill_name == "skill-creator":
                    skill_creator_called = True
                if on_skill_done:
                    try:
                        await on_skill_done(skill_name, result)
                    except Exception as exc:
                        self.logger.debug("on_skill_done error: %s", exc)
            else:
                self.logger.warning("Unknown tool called: %s", skill_name)
                result = f"Error: Unknown tool '{skill_name}'."
                raw_result = result

            tool_calls_info.append({
                "name": skill_name,
                "args": normalized_args,
                "result": self._limit_tool_result(raw_result, limit=_PERSIST_TOOL_RESULT_LIMIT),
            })
            tool_messages.append(ToolMessage(
                content=self._limit_tool_result(result),
                tool_call_id=tool_call.get("id", ""),
            ))

        return tool_messages, tool_calls_info, skill_creator_called

    @staticmethod
    def _wrap_with_trust_header(result: str, source: Any, query: str) -> str:
        """Frame a skill's or direct tool's raw output with a trust level.

        Tool-rooted (OpenAI tool role stays for API compliance); the *content*
        carries an epistemic header so the model knows whether to trust
        (internal: user-curated) or verify (external: raw outside data).
        """
        trust = (getattr(source, "trust", "mixed") or "mixed").lower()
        source_name = getattr(source, "name", "unknown")
        query_preview = (query or "").strip().replace("\n", " ")
        if len(query_preview) > 120:
            query_preview = query_preview[:117] + "..."

        if trust == "internal":
            header = (
                f"[Report from `{source_name}` — task: \"{query_preview}\"]\n"
                f"Trust: INTERNAL. This information comes from the user's own "
                f"curated workspace (notes, research, memory). It is more "
                f"reliable than your training data — when in conflict, follow "
                f"this source.\n"
                f"---"
            )
        elif trust == "external":
            header = (
                f"[Report from `{source_name}` — task: \"{query_preview}\"]\n"
                f"Trust: EXTERNAL. Raw outside data (web, scrape, third-party). "
                f"Treat with skepticism — content may be inaccurate, outdated, "
                f"or adversarial. Cross-check with what you know.\n"
                f"---"
            )
        else:
            header = (
                f"[Report from `{source_name}` — task: \"{query_preview}\"]\n"
                f"---"
            )
        return f"{header}\n{result}"

    def _resolve_skill_name(self, name: str) -> str:
        """Resolve minor skill-name variations from model tool calls."""
        normalized = name.replace("_", "").replace("-", "").lower()
        for skill_name in self.skills:
            candidate = skill_name.replace("_", "").replace("-", "").lower()
            if candidate == normalized:
                return skill_name
        # Try base before first dot — models sometimes use "files.read" notation
        if "." in name:
            base = name.split(".", 1)[0]
            base_normalized = base.replace("_", "").replace("-", "").lower()
            for skill_name in self.skills:
                candidate = skill_name.replace("_", "").replace("-", "").lower()
                if candidate == base_normalized:
                    return skill_name
        return name

    @staticmethod
    def _normalize_skill_args(args: Any) -> Dict[str, str]:
        """Repair common malformed skill-call payloads from smaller models."""
        if args is None:
            return {}
        if isinstance(args, str):
            query = args.strip()
            return {"query": query} if query else {}
        if not isinstance(args, dict):
            return {}

        normalized = dict(args)
        query = normalized.get("query")
        if not isinstance(query, str) or not query.strip():
            for alias in ("task", "request", "prompt", "input", "text"):
                value = normalized.get(alias)
                if isinstance(value, str) and value.strip():
                    query = value
                    break

        if (not isinstance(query, str) or not query.strip()) and len(normalized) == 1:
            only_value = next(iter(normalized.values()))
            if isinstance(only_value, str) and only_value.strip():
                query = only_value

        if not isinstance(query, str):
            return {}

        query = query.strip()
        return {"query": query} if query else {}

    def _decode_skill_call(self, tool_call: Dict[str, Any]) -> tuple[str, Dict[str, str], str]:
        """Return (resolved_skill_name, normalized_args, error_message)."""
        raw_name = str(tool_call.get("name", "") or "").strip()
        if not raw_name:
            return "", {}, "Error: Invalid skill call: missing skill name."

        skill_name = self._resolve_skill_name(raw_name)
        args = self._normalize_skill_args(tool_call.get("args", {}))

        if skill_name not in self.skills:
            return skill_name, args, f"Error: Unknown skill '{raw_name}'."

        # Use dotted subcommand as query seed when args are otherwise empty
        # e.g. files.list → query="list"; files.read → query="read <filename>"
        if "query" not in args and "." in raw_name and skill_name != raw_name:
            subcommand = raw_name.split(".", 1)[1]
            if subcommand:
                args["query"] = subcommand

        if "query" not in args:
            return skill_name, args, (
                f"Error: Invalid arguments for skill '{skill_name}'. "
                "Expected {'query': '<task>'}."
            )
        return skill_name, args, ""

    @staticmethod
    def _compact_text(value: Any, *, limit: int = _REPLAY_TOOL_RESULT_LIMIT) -> str:
        """Collapse whitespace and trim long text for replay context."""
        if value is None:
            return ""
        text = re.sub(r"\s+", " ", str(value)).strip()
        if len(text) <= limit:
            return text
        return text[: limit - 3].rstrip() + "..."

    @staticmethod
    def _limit_tool_result(value: Any, *, limit: int = _LIVE_TOOL_RESULT_LIMIT) -> str:
        """Keep tool output small enough for weaker/smaller chat models."""
        text = "" if value is None else str(value)
        if len(text) <= limit:
            return text
        omitted = len(text) - limit
        return (
            text[:limit].rstrip()
            + f"\n\n[Tool output truncated: {omitted} characters omitted. "
            "Ask for a narrower file/section/search if more detail is needed.]"
        )

    def _format_replayed_assistant_turn(
        self,
        bot_text: str,
        tool_calls_info: Optional[List[Dict[str, Any]]],
    ) -> str:
        """Return an assistant turn for replay into the chat context.

        Conversational responses are replayed verbatim. Skill-backed exchanges
        keep the assistant text and append a concise note about the tool calls,
        so the prompt is not re-inflated with full raw tool outputs.
        """
        if not tool_calls_info:
            return bot_text or ""

        lines: List[str] = []
        for tc in tool_calls_info[:_REPLAY_TOOL_CALLS_LIMIT]:
            name = self._resolve_skill_name(str(tc.get("name", "") or "").strip()) or "unknown"
            args = self._normalize_skill_args(tc.get("args", {}))
            query = self._compact_text(args.get("query", ""), limit=100)
            result = self._compact_text(tc.get("result", ""), limit=_REPLAY_TOOL_RESULT_LIMIT)

            detail = f"- {name}"
            if query:
                detail += f": {query}"
            if result:
                detail += f" -> {result}"
            lines.append(detail)

        if len(tool_calls_info) > _REPLAY_TOOL_CALLS_LIMIT:
            lines.append(f"- ... {len(tool_calls_info) - _REPLAY_TOOL_CALLS_LIMIT} more earlier skill call(s)")

        summary = "[Earlier skill use — internal context:]\n" + "\n".join(lines)
        return f"{bot_text}\n\n{summary}" if bot_text else summary

    @staticmethod
    def _message_text(message: BaseMessage) -> str:
        content = message.content
        return content if isinstance(content, str) else str(content)

    @staticmethod
    def _format_max_tool_turns_warning(
        max_turns: int, tool_calls_info: List[Dict[str, Any]]
    ) -> str:
        """Render a one-line summary of the tool-call sequence that
        triggered the iteration budget so the cause is visible in the
        log (not just the bare 'Max chat tool turns reached' message)."""
        from collections import Counter
        names = [str(tc.get("name", "?")) for tc in tool_calls_info]
        counts = Counter(names)
        tail = ", ".join(names[-6:]) if names else "(none)"
        return (
            f"Max chat tool turns ({max_turns}) reached after {len(names)} "
            f"tool calls — forcing final response. "
            f"Counts: {dict(counts.most_common())}. Tail: {tail}"
        )

    def _clone_message_with_trimmed_content(
        self,
        message: BaseMessage,
        *,
        limit: int,
    ) -> BaseMessage:
        text = self._message_text(message)
        compacted = self._compact_text(text, limit=limit)
        if compacted == text:
            return message
        if isinstance(message, ToolMessage):
            return ToolMessage(content=compacted, tool_call_id=message.tool_call_id)
        if isinstance(message, HumanMessage):
            return HumanMessage(content=compacted)
        if isinstance(message, AIMessage):
            clone = AIMessage(content=compacted)
            clone.tool_calls = getattr(message, "tool_calls", [])
            return clone
        return message

    def _estimated_message_tokens(self, messages: List[BaseMessage]) -> int:
        from pawlia.memory import estimate_tokens

        return sum(estimate_tokens(self._message_text(message)) for message in messages)

    def _context_budget_for(self, thread_id: Optional[str], *, images: bool = False) -> int:
        if not self._context_window_resolver:
            return 0
        agent_type = "vision" if images else "chat"
        try:
            ctx = int(self._context_window_resolver(agent_type, thread_id) or 0)
        except Exception:
            return 0
        return max(0, ctx - _CONTEXT_COMPLETION_RESERVE_TOKENS)

    def _prepare_messages_for_context_budget(
        self,
        messages: List[BaseMessage],
        *,
        thread_id: Optional[str],
        images: bool = False,
    ) -> List[BaseMessage]:
        budget = self._context_budget_for(thread_id, images=images)
        if budget <= 0:
            return messages

        prepared: List[BaseMessage] = list(messages)
        if self._estimated_message_tokens(prepared) <= budget:
            return prepared

        # Phase 1: trim large individual messages
        trimmed: List[BaseMessage] = []
        for index, message in enumerate(prepared):
            if isinstance(message, ToolMessage):
                trimmed.append(self._clone_message_with_trimmed_content(message, limit=_CONTEXT_TRIMMED_TOOL_LIMIT))
            elif index > 0 and isinstance(message, (HumanMessage, AIMessage)):
                trimmed.append(self._clone_message_with_trimmed_content(message, limit=_CONTEXT_TRIMMED_TEXT_LIMIT))
            else:
                trimmed.append(message)
        prepared = trimmed

        if self._estimated_message_tokens(prepared) <= budget or len(prepared) <= 1:
            return prepared

        # Phase 2: summarize older messages into a single HumanMessage
        # instead of dropping them — keeps context accessible to the LLM.
        system = prepared[:1]
        tail = prepared[1:]

        if self._estimated_message_tokens(system + tail) > budget and len(tail) > _CONTEXT_MIN_NON_SYSTEM_KEEP:
            cut = len(tail) - _CONTEXT_MIN_NON_SYSTEM_KEEP
            # Walk forward to include orphaned ToolMessages whose parent
            # AIMessage was already cut — otherwise they'd be silently lost.
            while cut < len(tail) and isinstance(tail[cut], ToolMessage):
                cut += 1
            if cut > 0:
                dropped = tail[:cut]
                tail = tail[cut:]

                summary_lines = [
                    f"[Earlier conversation summarized — {len(dropped)} message(s) condensed]"
                ]
                for msg in dropped:
                    if isinstance(msg, ToolMessage):
                        content = msg.content if isinstance(msg.content, str) else str(msg.content)
                        preview = content[:_SUMMARY_PREVIEW_CHARS].replace("\n", " ")
                        summary_lines.append(f"[Tool: {preview}]")
                    elif isinstance(msg, HumanMessage):
                        content = msg.content if isinstance(msg.content, str) else str(msg.content)
                        preview = content[:_SUMMARY_PREVIEW_CHARS].replace("\n", " ")
                        summary_lines.append(f"[User: {preview}]")
                    elif isinstance(msg, AIMessage):
                        content = msg.content if isinstance(msg.content, str) else str(msg.content)
                        preview = content[:_SUMMARY_PREVIEW_CHARS].replace("\n", " ")
                        tc = getattr(msg, "tool_calls", None)
                        if tc:
                            names = ", ".join(t.get("name", "?") for t in tc if isinstance(t, dict))
                            summary_lines.append(f"[Assistant called {names}: {preview}]")
                        else:
                            summary_lines.append(f"[Assistant: {preview}]")

                summary = HumanMessage(content="\n".join(summary_lines))

                if tail and isinstance(tail[0], HumanMessage):
                    existing = tail[0].content if isinstance(tail[0].content, str) else str(tail[0].content)
                    tail = [HumanMessage(content=summary.content + "\n\n" + existing)] + tail[1:]
                else:
                    tail = [summary] + tail

        # Final cleanup via _sanitize_messages: handles surrogate cleaning,
        # orphaned ToolMessages, tool result compression, and same-role merging
        # in one pass — and covers the streaming path which doesn't call
        # _invoke (and therefore wouldn't otherwise sanitize).
        return BaseAgent._sanitize_messages(system + tail)

    async def run(
        self,
        user_input: str,
        system_prompt: Optional[str] = None,
        images: Optional[List[str]] = None,
        thread_id: Optional[str] = None,
        on_skill_start: Optional[SkillStartCallback] = None,
        on_skill_step: Optional[InterimCallback] = None,
        on_skill_done: Optional[SkillDoneCallback] = None,
        workspace_search: bool = False,
    ) -> str:
        """Process user input and return a response.

        1. Send to LLM with skill specs as available functions
        2. If LLM calls a skill -> spawn SkillRunnerAgent
        3. Feed skill result back to LLM for final answer

        ``images`` is an optional list of base64 data-URIs
        (e.g. ``data:image/png;base64,…``).

        ``thread_id`` isolates the context window: the model only sees exchanges
        from that thread.

        Optional per-call callbacks override instance-level attributes to avoid
        race conditions when the same agent is shared across concurrent requests.
        """
        # Reset attachment queue: anything queued by the previous turn must
        # not leak into this turn's reply.
        self.pending_attachments = []

        # Resolve callbacks: per-call overrides > instance attributes
        _on_skill_start = on_skill_start or self.on_skill_start
        _on_skill_step = on_skill_step or self.on_skill_step
        _on_skill_done = on_skill_done or self.on_skill_done

        # Workspace context: search on first substantive turn, re-search on topic shift
        self._handle_workspace_context(user_input, allow_workspace_search=workspace_search)

        # Lightweight pointers (most recent other conversation + last received
        # attachment) so the model can pull them with the files skill on demand
        # instead of replaying everything every turn.
        extra_context: Optional[str] = None
        if not system_prompt and self.memory and self.session:
            pointers: List[str] = []
            for _getter in (
                lambda: self.memory.last_conversation_pointer(
                    self.session.user_id, exclude_thread_id=thread_id),
                lambda: self.memory.last_attachment_pointer(self.session.user_id),
            ):
                try:
                    _line = _getter()
                except Exception:
                    _line = None
                if _line:
                    pointers.append(_line)
            extra_context = "\n".join(pointers) or None

        prompt = self.build_system_prompt(
            system_prompt=system_prompt, extra_context=extra_context,
        )

        messages: List[BaseMessage] = [SystemMessage(content=prompt)]

        # Replay recent exchanges.  For threads, use the thread-specific window
        # instead of the main session history.
        if self.session and self.memory:
            if thread_id:
                exchanges = self.memory.get_thread_context(self.session, thread_id)
            else:
                exchanges = self.session.exchanges
            for exchange in exchanges:
                # Unpack 2-tuple or 3-tuple (old format compatibility)
                if len(exchange) == 2:
                    user_text, bot_text = exchange  # type: ignore
                    tool_calls_info = None
                else:
                    user_text, bot_text, tool_calls_info = exchange  # type: ignore

                messages.append(HumanMessage(content=user_text))
                messages.append(AIMessage(
                    content=self._format_replayed_assistant_turn(bot_text, tool_calls_info)
                ))

        # Decide how to feed images: inline (model can see) or as injected
        # text descriptions (model is blind → a vision model describes them).
        inline_images, image_descriptions = (images, [])
        if images:
            inline_images, image_descriptions = await self._prepare_image_input(images, thread_id)
        vision_turn = bool(inline_images)

        # Resolve the LLMs to use for this call.
        # A thread-specific model override takes priority over the session default.
        bound_llm, unbound_llm = self._resolve_llms(thread_id, images=vision_turn)

        context_block, clean_input = _augment_with_workspace_refs(user_input, self.session)

        # Build multimodal content when images are sent inline
        if inline_images:
            self.logger.debug("Sending %d image(s) to LLM", len(inline_images))
            if context_block:
                messages.append(HumanMessage(content=context_block))
            content: List[Dict[str, Any]] = [{"type": "text", "text": clean_input or "What's in this image?"}]
            for data_uri in inline_images:
                content.append({"type": "image_url", "image_url": {"url": data_uri}})
            messages.append(HumanMessage(content=content))
        else:
            if context_block:
                messages.append(HumanMessage(content=context_block))
            messages.append(HumanMessage(content=self._with_image_descriptions(clean_input, image_descriptions)))

        # Turn 1: LLM decides whether to call a skill or answer directly
        active_llm = bound_llm
        response, messages = await self._invoke_with_tool_retry(
            messages,
            llm=active_llm,
            thread_id=thread_id,
            images=vision_turn,
        )

        tool_calls_info: List[Dict[str, Any]] = []
        final = response
        nudge_count = 0
        budget = IterationBudget(self.max_tool_turns)

        turn = 0
        while budget.consume():
            turn += 1
            self.logger.debug(
                "Chat tool loop turn %d: tool_calls=%s, content=%s",
                turn,
                bool(final.tool_calls),
                repr(final.content[:200]) if final.content else "(empty)",
            )

            if not final.tool_calls:
                result = self.extract_text(final)
                if self._should_nudge_for_workspace_grounding(
                    result,
                    has_tool_history=bool(tool_calls_info),
                ) and nudge_count < _MAX_CHAT_NUDGES:
                    nudge_count += 1
                    self.logger.warning(
                        "ChatAgent nudging model to ground answer in workspace refs (nudge %d/%d)",
                        nudge_count,
                        _MAX_CHAT_NUDGES,
                    )
                    messages = messages + [final, HumanMessage(content=_WORKSPACE_GROUNDING_NUDGE)]
                    final, messages = await self._invoke_with_tool_retry(
                        messages,
                        llm=active_llm,
                        thread_id=thread_id,
                        images=vision_turn,
                    )
                    continue
                if self._should_nudge_for_incomplete_task(
                    result,
                    has_tool_history=bool(tool_calls_info),
                ) and nudge_count < _MAX_CHAT_NUDGES:
                    nudge_count += 1
                    self.logger.warning(
                        "ChatAgent received deferred-action text without tool call; nudging "
                        "model to continue (nudge %d/%d)",
                        nudge_count,
                        _MAX_CHAT_NUDGES,
                    )
                    messages = messages + [final, HumanMessage(content=_CHAT_CONTINUE_NUDGE)]
                    final, messages = await self._invoke_with_tool_retry(
                        messages,
                        llm=active_llm,
                        thread_id=thread_id,
                        images=vision_turn,
                    )
                    continue
                break

            messages = _remove_workspace_refs_from_messages(messages, user_input)
            # Skip interim on turn 1: the model's first decision often includes
            # stray intro text (identity phrases, "I'll search for X…") that
            # should not be sent as a separate message before any tool runs.
            # From turn 2 onward the model is mid-task and interim is useful.
            interim = self.extract_text(final)
            if interim and self.on_interim and turn > 1:
                try:
                    await self.on_interim(interim)
                except Exception as exc:
                    self.logger.debug("on_interim callback error: %s", exc)

            messages.append(final)

            tool_msgs, new_info, skill_creator_called = await self._execute_tool_calls(
                final.tool_calls, thread_id, _on_skill_start, _on_skill_step, _on_skill_done,
            )
            messages.extend(tool_msgs)
            tool_calls_info.extend(new_info)

            # Refresh workspace skills only when skill-creator ran (it may have added/changed skills)
            if skill_creator_called and self._refresh_and_rebind_skills():
                bound_llm = self.bound_llm
                active_llm = bound_llm

            final, messages = await self._invoke_with_tool_retry(
                messages,
                llm=active_llm,
                thread_id=thread_id,
                images=vision_turn,
            )

        else:
            self.logger.warning(self._format_max_tool_turns_warning(
                self.max_tool_turns, tool_calls_info
            ))
            final = await self._invoke(
                self._prepare_messages_for_context_budget(
                    messages + [HumanMessage(content=_EMPTY_TURN2_NUDGE)],
                    thread_id=thread_id,
                    images=vision_turn,
                ),
                llm=unbound_llm,
            )

        result = self.extract_text(final)
        used_skills = bool(tool_calls_info)
        await self._persist(
            user_input, result,
            track_similarity=not used_skills,
            thread_id=thread_id,
            tool_calls_info=tool_calls_info if used_skills else None,
        )
        return result

    # ------------------------------------------------------------------
    # Streamed variant (sentence-by-sentence TTS for calls)
    # ------------------------------------------------------------------

    async def run_streamed(
        self,
        user_input: str,
        *,
        system_prompt: Optional[str] = None,
        images: Optional[List[str]] = None,
        thread_id: Optional[str] = None,
        on_sentence: Optional[Callable[[str], Awaitable[None]]] = None,
        on_skill_start: Optional[SkillStartCallback] = None,
        on_skill_step: Optional[InterimCallback] = None,
        on_skill_done: Optional[SkillDoneCallback] = None,
        allow_skills: bool = True,
        workspace_search: bool = False,
    ) -> str:
        """Like :meth:`run` but streams the LLM and calls *on_sentence* per sentence.

        Each complete sentence (delimited by ``.``, ``!``, ``?``, ``…`` +
        whitespace) is emitted as soon as it is detected in the token stream,
        enabling incremental TTS playback.  Falls back to non-streamed skill
        execution when tool calls are detected; the final-answer turn is also
        streamed.
        """
        # Reset attachment queue so a previous turn's attachments don't leak
        # into this turn's reply.
        self.pending_attachments = []

        _on_skill_start = on_skill_start or self.on_skill_start
        _on_skill_step = on_skill_step or self.on_skill_step
        _on_skill_done = on_skill_done or self.on_skill_done

        # Workspace context: search on first substantive turn, re-search on topic shift
        self._handle_workspace_context(user_input, allow_workspace_search=workspace_search)

        # Lightweight pointers (most recent other conversation + last received
        # attachment) so the model can pull them with the files skill on demand
        # instead of replaying everything every turn.
        extra_context: Optional[str] = None
        if not system_prompt and self.memory and self.session:
            pointers: List[str] = []
            for _getter in (
                lambda: self.memory.last_conversation_pointer(
                    self.session.user_id, exclude_thread_id=thread_id),
                lambda: self.memory.last_attachment_pointer(self.session.user_id),
            ):
                try:
                    _line = _getter()
                except Exception:
                    _line = None
                if _line:
                    pointers.append(_line)
            extra_context = "\n".join(pointers) or None

        prompt = self.build_system_prompt(
            system_prompt=system_prompt, extra_context=extra_context,
        )

        messages: List[BaseMessage] = [SystemMessage(content=prompt)]

        # Replay recent exchanges (identical to run())
        if self.session and self.memory:
            if thread_id:
                exchanges = self.memory.get_thread_context(self.session, thread_id)
            else:
                exchanges = self.session.exchanges
            for exchange in exchanges:
                if len(exchange) == 2:
                    user_text, bot_text = exchange  # type: ignore
                    tc_info = None
                else:
                    user_text, bot_text, tc_info = exchange  # type: ignore
                messages.append(HumanMessage(content=user_text))
                messages.append(AIMessage(
                    content=self._format_replayed_assistant_turn(bot_text, tc_info)
                ))

        inline_images, image_descriptions = (images, [])
        if images:
            inline_images, image_descriptions = await self._prepare_image_input(images, thread_id)
        vision_turn = bool(inline_images)

        bound_llm, unbound_llm = self._resolve_llms(thread_id, images=vision_turn)
        first_turn_llm = bound_llm if allow_skills else unbound_llm

        context_block, clean_input = _augment_with_workspace_refs(user_input, self.session)

        if inline_images:
            if context_block:
                messages.append(HumanMessage(content=context_block))
            content: List[Dict[str, Any]] = [{"type": "text", "text": clean_input or "What's in this image?"}]
            for data_uri in inline_images:
                content.append({"type": "image_url", "image_url": {"url": data_uri}})
            messages.append(HumanMessage(content=content))
        else:
            if context_block:
                messages.append(HumanMessage(content=context_block))
            messages.append(HumanMessage(content=self._with_image_descriptions(clean_input, image_descriptions)))

        # ---- Stream turn 1 ----
        # _partial_text tracks generated text for persist-on-cancel (barge-in during TTS
        # cancels this coroutine before _persist is reached, losing the user's turn from history).
        _partial_text = ""
        try:
            accumulated, raw_text = await self._stream_with_sentences(
                messages,
                first_turn_llm,
                on_sentence,
                thread_id=thread_id,
                images=vision_turn,
            )
            _partial_text = raw_text

            self.logger.debug("Streamed turn 1: tool_calls=%s, len=%d",
                              bool(getattr(accumulated, "tool_calls", None)), len(raw_text))

            # If the streamed response is a fake tool call, retry non-streamed
            if accumulated and self._is_fake_tool_call(accumulated):
                self.logger.warning("Fake tool call detected in streamed turn 1, retrying non-streamed")
                retry_llm = bound_llm if allow_skills else unbound_llm
                accumulated, messages = await self._invoke_with_tool_retry(
                    messages,
                    llm=retry_llm,
                    thread_id=thread_id,
                    images=vision_turn,
                )
                raw_text = accumulated.content if isinstance(accumulated.content, str) else ""
                _partial_text = raw_text

            if not accumulated or not getattr(accumulated, "tool_calls", None):
                result = self.strip_thinking(raw_text)
                await self._persist(user_input, result, track_similarity=True, thread_id=thread_id)
                return result

            if not allow_skills:
                self.logger.warning("run_streamed received tool calls while allow_skills=False; ignoring tool calls")
                result = self.strip_thinking(raw_text)
                await self._persist(user_input, result, track_similarity=True, thread_id=thread_id)
                return result

            # ---- Skill calls detected → execute (non-streamed) ----
            messages = _remove_workspace_refs_from_messages(messages, user_input)
            messages.append(accumulated)
            tool_calls_info: List[Dict[str, Any]] = []

            tool_msgs, new_info, skill_creator_called = await self._execute_tool_calls(
                accumulated.tool_calls, thread_id, _on_skill_start, _on_skill_step, _on_skill_done,
            )
            messages.extend(tool_msgs)
            tool_calls_info.extend(new_info)

            # Refresh workspace skills only when skill-creator ran (it may have added/changed skills)
            if skill_creator_called and self._refresh_and_rebind_skills():
                bound_llm = self.bound_llm

            # ---- Continue tool loop until the task is actually complete ----
            raw_text2 = ""
            nudge_count = 0
            final_response: Optional[AIMessage] = None
            stream_budget = IterationBudget(self.max_tool_turns)

            while stream_budget.consume():
                next_response, messages = await self._invoke_with_tool_retry(
                    messages,
                    llm=bound_llm,
                    thread_id=thread_id,
                    images=vision_turn,
                )

                if not next_response.tool_calls:
                    text = self.extract_text(next_response)
                    if self._should_nudge_for_incomplete_task(text, has_tool_history=True) \
                       and nudge_count < _MAX_CHAT_NUDGES:
                        nudge_count += 1
                        messages = messages + [next_response, HumanMessage(content=_CHAT_CONTINUE_NUDGE)]
                        continue
                    final_response = next_response
                    raw_text2 = text
                    break

                messages.append(next_response)
                tool_msgs, new_info, skill_creator_called = await self._execute_tool_calls(
                    next_response.tool_calls, thread_id, _on_skill_start, _on_skill_step, _on_skill_done,
                )
                messages.extend(tool_msgs)
                tool_calls_info.extend(new_info)

                # Refresh workspace skills only when skill-creator ran
                if skill_creator_called and self._refresh_and_rebind_skills():
                    bound_llm = self.bound_llm
            else:
                accumulated2, raw_text2 = await self._stream_with_sentences(
                    messages + [HumanMessage(content=_EMPTY_TURN2_NUDGE)], unbound_llm, on_sentence,
                )
                final_response = accumulated2 if isinstance(accumulated2, AIMessage) else None

            if on_sentence and raw_text2.strip():
                sentences, remainder = _split_sentences(raw_text2)
                for sentence in sentences:
                    if sentence.strip():
                        await on_sentence(sentence)
                if remainder.strip():
                    await on_sentence(remainder.strip())

            result = self.strip_thinking(raw_text2)
            used_skills = bool(tool_calls_info)
            await self._persist(
                user_input, result,
                track_similarity=not used_skills,
                thread_id=thread_id,
                tool_calls_info=tool_calls_info if used_skills else None,
            )
            return result
        except asyncio.CancelledError:
            # Barge-in: the TTS-playing coroutine was cancelled before _persist was reached.
            # Persist whatever text was generated so the user's utterance stays in history
            # and the next LLM turn has the full conversation context.
            if _partial_text:
                partial = self.strip_thinking(_partial_text)
                try:
                    await asyncio.shield(
                        self._persist(user_input, partial,
                                      track_similarity=False, thread_id=thread_id)
                    )
                    self.logger.debug("run_streamed: persisted partial response on cancel (%d chars)", len(partial))
                except Exception as exc:
                    self.logger.debug("run_streamed: persist-on-cancel failed: %s", exc)
            raise
        except Exception:
            # LLM error (e.g. API 400, context overflow, timeout): persist the user's
            # transcribed input so the next turn doesn't lose conversation context.
            partial = self.strip_thinking(_partial_text) if _partial_text else None
            result_text = partial or "[Entschuldigung, ich hatte einen technischen Fehler.]"
            try:
                await asyncio.shield(
                    self._persist(user_input, result_text,
                                  track_similarity=False, thread_id=thread_id)
                )
                self.logger.debug("run_streamed: persisted input on error (%d chars)", len(user_input))
            except Exception as exc:
                self.logger.debug("run_streamed: persist-on-error failed: %s", exc)
            raise

    async def _stream_with_sentences(
        self,
        messages: List[BaseMessage],
        llm: Any,
        on_sentence: Optional[Callable[[str], Awaitable[None]]],
        *,
        thread_id: Optional[str] = None,
        images: bool = False,
    ) -> Tuple[Any, str]:
        """Stream an LLM call, emitting complete sentences via *on_sentence*.

        Returns ``(accumulated_message, raw_text)``.
        """
        accumulated = None
        raw_text = ""
        emitted_len = 0  # how much of the clean text has been emitted

        messages = self._prepare_messages_for_context_budget(
            messages,
            thread_id=thread_id,
            images=images,
        )
        log_prompt(messages, name=self.log_name)

        async for chunk in llm.astream(messages):
            accumulated = chunk if accumulated is None else accumulated + chunk

            delta = chunk.content or ""
            if not delta:
                continue
            raw_text += delta

            if not on_sentence:
                continue

            # Don't emit while inside an unclosed <think>/<thinking> block
            n_open = len(re.findall(r"<think(?:ing)?>", raw_text))
            n_close = len(re.findall(r"</think(?:ing)?>", raw_text))
            if n_open > n_close:
                continue

            clean = self.strip_thinking(raw_text)
            new_content = clean[emitted_len:]
            if new_content:
                sentences, remainder = _split_sentences(new_content)
                for s in sentences:
                    if s.strip():
                        await on_sentence(s)
                emitted_len = len(clean) - len(remainder)

        # Flush remaining buffer
        if on_sentence:
            clean = self.strip_thinking(raw_text)
            remainder = clean[emitted_len:]
            if remainder.strip():
                await on_sentence(remainder.strip())

        return accumulated, raw_text

    _DIRECTIVE_RE = re.compile(r'\{"__directive__"\s*:.*\}')

    def _process_directives(self, result: str, thread_id: Optional[str] = None) -> str:
        """Extract and handle ``__directive__`` JSON lines from skill output.

        Returns the result string with directive lines removed.
        """
        clean_lines = []
        for line in result.splitlines():
            m = self._DIRECTIVE_RE.search(line)
            if not m:
                clean_lines.append(line)
                continue
            try:
                obj = json.loads(m.group(0))
            except (json.JSONDecodeError, ValueError):
                clean_lines.append(line)
                continue
            directive = obj.get("__directive__")
            if directive == "set_model":
                model = obj.get("model")
                if model and self.memory and self.session:
                    self.memory.set_model_override(self.session, model)
                    self.logger.info("Directive: session model override set to '%s'", model)
                    if self.on_model_change:
                        self.on_model_change(model)
            elif directive == "set_agent_override":
                path = str(obj.get("path", "") or "").strip()
                value = obj.get("value")
                if path and self.memory and self.session:
                    self.memory.set_agent_override_value(
                        self.session,
                        path,
                        str(value).strip() if isinstance(value, str) and str(value).strip() else None,
                    )
                    self.logger.info(
                        "Directive: session agent override '%s' -> %r",
                        path,
                        value,
                    )
                    if self.on_model_change:
                        self.on_model_change(str(value or ""))
            elif directive == "reload_skills":
                self._apply_disabled_skills()
            elif directive == "set_voice":
                if self.memory and self.session:
                    voice = obj.get("voice")
                    self.memory.set_voice_override(self.session, voice)
                    self.logger.info("Directive: voice override set to '%s'", voice)
            elif directive == "set_private":
                if self.memory and self.session:
                    desired: bool = bool(obj.get("private", True))
                    target_thread: Optional[str] = obj.get("thread") or thread_id
                    if target_thread:
                        current = target_thread in self.session.private_threads
                        if current != desired:
                            self.memory.toggle_private_thread(self.session, target_thread)
                        self.logger.info("Directive: thread '%s' private=%s", target_thread, desired)
                    else:
                        if self.session.private != desired:
                            self.memory.toggle_private(self.session)
                        self.logger.info("Directive: session private=%s", desired)
            else:
                self.logger.warning("Unknown directive: %s", directive)
                clean_lines.append(line)
        return "\n".join(clean_lines)

    def _resolve_llms(
        self, thread_id: Optional[str], *, images: bool = False
    ) -> Tuple[Any, Any]:
        """Return (bound_llm, unbound_llm) for this call.

        Resolves the active `agents` selection dynamically so session/thread
        overrides take effect without rebuilding the whole agent.
        """
        if self._agent_llm_resolver:
            agent_type = "vision" if images else "chat"
            llm = self._agent_llm_resolver(agent_type, thread_id)
            all_specs = self._skill_specs + getattr(self, "_direct_tool_specs", [])
            bound = llm.bind_tools(all_specs, tool_choice="auto") if all_specs else llm
            if self._on_fallback:
                for l in (bound, llm):
                    if hasattr(l, "set_on_fallback"):
                        l.set_on_fallback(self._on_fallback)
            return bound, llm

        for l in (self.bound_llm, self.llm, self.vision_bound_llm):
            if hasattr(l, "set_on_fallback") and self._on_fallback:
                l.set_on_fallback(self._on_fallback)
        return (self.vision_bound_llm if images else self.bound_llm), self.llm

    async def _prepare_image_input(
        self, images: List[str], thread_id: Optional[str]
    ) -> Tuple[Optional[List[str]], List[str]]:
        """Decide how to feed *images* to the model.

        Returns ``(inline_images, descriptions)``:

        * ``(images, [])`` — the image-handling model can see; send the image
          data inline (best fidelity). This is the default and the only path
          when no factory is wired.
        * ``(None, [descriptions])`` — the image-handling model is blind, so a
          vision-capable model from the fallback chain has described each image
          as text. The caller injects that text into context instead of the
          image data and runs the normal chat model.

        Any failure degrades gracefully to inline sending — image handling must
        never break because the probe misbehaved.
        """
        factory = self._llm_factory
        session_dir = getattr(self.memory, "session_dir", None) if self.memory else None
        if not factory or not session_dir or not images:
            return images, []

        from pawlia import vision_probe

        overrides = self._overrides_resolver(thread_id) if self._overrides_resolver else None
        try:
            chain = factory.get_fallback_chain("vision", agent_overrides=overrides)
        except Exception as exc:
            self.logger.debug("vision: could not resolve vision chain (%s); sending inline", exc)
            return images, []
        if not chain:
            return images, []

        try:
            primary_sees = await vision_probe.resolve_supports_images(factory, session_dir, chain[0])
        except Exception as exc:
            self.logger.debug("vision: capability probe failed (%s); sending inline", exc)
            return images, []
        if primary_sees:
            return images, []

        # Primary image model is blind — borrow the first vision-capable model
        # from the chain to describe the image(s).
        describer_name = None
        for name in chain[1:]:
            try:
                if await vision_probe.resolve_supports_images(factory, session_dir, name):
                    describer_name = name
                    break
            except Exception:
                continue
        if not describer_name:
            self.logger.warning(
                "vision: model '%s' cannot see and no vision-capable fallback found; sending inline",
                chain[0],
            )
            return images, []

        try:
            describer = factory.get_with_model(describer_name)
        except Exception as exc:
            self.logger.warning("vision: cannot build describer '%s' (%s); sending inline", describer_name, exc)
            return images, []

        descriptions: List[str] = []
        for uri in images:
            desc = await vision_probe.describe_image(describer, uri)
            descriptions.append(desc or "(Bild konnte nicht analysiert werden.)")
        self.logger.info(
            "vision: '%s' is blind — described %d image(s) via '%s' and injecting as text",
            chain[0], len(descriptions), describer_name,
        )
        return None, descriptions

    @staticmethod
    def _with_image_descriptions(clean_input: str, descriptions: List[str]) -> str:
        """Fold vision-model image descriptions into the user turn as context.

        The user never sees this text — it lives only in the LLM context so a
        text-only chat model can reason about images it cannot natively view.
        """
        base = clean_input or "(Bild ohne Begleittext gesendet.)"
        if not descriptions:
            return base
        if len(descriptions) == 1:
            block = f"[Bildanalyse]: {descriptions[0]}"
        else:
            block = "\n".join(f"[Bildanalyse {i + 1}]: {d}" for i, d in enumerate(descriptions))
        return f"{base}\n\n{block}"

    def set_callbacks(self, *, on_fallback: Optional[Callable[[str, str], None]] = None) -> None:
        """Register callbacks for this agent instance.

        Args:
            on_fallback: Called when the LLM fallback chain advances:
                         ``(from_model_name, to_model_name)``.
        """
        self._on_fallback = on_fallback

    def _active_override_model(self, thread_id: Optional[str]) -> Optional[str]:
        """Return the name of the active model override, or ``None``."""
        if self.memory and self.session:
            return self.memory.get_agent_override_value(self.session, "chat", thread_id=thread_id)
        return None

    def _extract_fake_skill_calls(self, response: AIMessage) -> List[Dict[str, Any]]:
        """Recover text-form skill calls from models that miss function calling.

        Some OpenAI-compatible providers accept a tools schema but still let
        smaller models emit the tool call as plain text, usually as
        ``<tool_call>{...}</tool_call>`` or a fenced JSON block.  Treat those as
        real skill calls instead of spending another turn asking the model to
        retry the exact same shape.
        """
        if not self._skill_specs or response.tool_calls:
            return []
        content = response.content if isinstance(response.content, str) else ""

        calls: List[Dict[str, Any]] = []

        def _append_from_obj(obj: Any) -> None:
            if isinstance(obj, list):
                for item in obj:
                    _append_from_obj(item)
                return
            if not isinstance(obj, dict):
                return

            name = str(obj.get("name") or obj.get("tool") or obj.get("function") or "").strip()
            if not name:
                return
            skill_name = self._resolve_skill_name(name)
            if skill_name not in self.skills:
                return

            raw_args = obj.get("args")
            if raw_args is None:
                raw_args = obj.get("arguments")
            if raw_args is None:
                raw_args = obj.get("parameters")
            args = self._normalize_skill_args(raw_args)
            if "query" not in args:
                # Common shorthand: {"name": "searxng", "query": "..."}
                args = self._normalize_skill_args(obj)
            if "query" not in args:
                return

            calls.append({
                "id": obj.get("id") or f"fake_{uuid.uuid4().hex[:8]}",
                "name": skill_name,
                "args": args,
            })

        snippets: List[str] = [m.group(1).strip() for m in _RE_TOOL_CALL_TAG.finditer(content)]
        snippets.extend(m.group(1).strip() for m in _RE_CODE_BLOCK.finditer(content))

        stripped = content.strip()
        if stripped.startswith(("{", "[")):
            snippets.append(stripped)

        for snippet in snippets:
            if not snippet:
                continue
            try:
                _append_from_obj(json.loads(snippet))
            except (TypeError, ValueError, json.JSONDecodeError):
                # Fall through to command-like fenced blocks below.
                pass

        if calls:
            return calls

        skill_names = set(self.skills.keys())
        for match in _RE_CODE_BLOCK.finditer(content):
            block = match.group(1).strip()
            raw_token = block.split()[0] if block else ""
            first_token = raw_token.rstrip(":.,;")
            skill_name = self._resolve_skill_name(first_token)
            if skill_name in skill_names:
                query = block[len(raw_token):].strip()
                if query:
                    calls.append({
                        "id": f"fake_{uuid.uuid4().hex[:8]}",
                        "name": skill_name,
                        "args": {"query": query},
                    })

        if calls or "files" not in skill_names:
            return calls

        # Last-resort heuristics for small models during bootstrap:
        # treat "# <filename>.md\n…" blocks as `files write` calls and
        # bare "delete <filename>" commands as `files delete` calls.
        for match in _RE_CODE_BLOCK.finditer(content):
            block = match.group(1).strip()
            if not block:
                continue
            first_line = block.split("\n", 1)[0]
            delete_match = _RE_DELETE_COMMAND.match(first_line)
            if delete_match:
                calls.append({
                    "id": f"fake_{uuid.uuid4().hex[:8]}",
                    "name": "files",
                    "args": {"query": f"delete --filename {delete_match.group('name')}"},
                })
                continue
            heading_match = _RE_FILENAME_HEADING.match(first_line)
            if heading_match:
                filename = heading_match.group("name")
                calls.append({
                    "id": f"fake_{uuid.uuid4().hex[:8]}",
                    "name": "files",
                    "args": {"query": f"write --filename {filename}\n{block}"},
                })

        return calls

    def _is_fake_tool_call(self, response: AIMessage) -> bool:
        """Return True if the LLM wrote a skill call as text."""
        if self._extract_fake_skill_calls(response):
            return True
        if response.tool_calls:
            return False
        content = response.content if isinstance(response.content, str) else ""
        if "<tool_call>" in content:
            return True
        all_skill_names = set(self._all_skills.keys())
        for match in _RE_CODE_BLOCK.finditer(content):
            block = match.group(1).strip()
            first_token = (block.split()[0] if block else "").rstrip(":.,;")
            if self._resolve_skill_name(first_token) in all_skill_names:
                return True
        return False

    def _is_disabled_skill_call(self, response: AIMessage) -> bool:
        """Return True if the model tried to call a disabled skill (real or text-form)."""
        if not self.session:
            return False
        disabled = set(self.session.disabled_skills or [])
        if not disabled:
            return False
        # Real tool call to a disabled skill
        for tc in (response.tool_calls or []):
            if self._resolve_skill_name(str(tc.get("name", ""))) in disabled:
                return True
        # Text-form call to a disabled skill
        content = response.content if isinstance(response.content, str) else ""
        for match in _RE_CODE_BLOCK.finditer(content):
            block = match.group(1).strip()
            first_token = (block.split()[0] if block else "").rstrip(":.,;")
            if self._resolve_skill_name(first_token) in disabled:
                return True
        return False

    @staticmethod
    def _should_nudge_for_incomplete_task(text: str, *, has_tool_history: bool) -> bool:
        """Detect responses that describe the next action instead of doing it."""
        stripped = text.strip()
        if not stripped:
            return has_tool_history
        if _DEFERRED_PLACEHOLDER_RE.search(stripped):
            return True
        return bool(_DEFERRED_TOOL_INTENT_RE.search(stripped))

    def _should_nudge_for_workspace_grounding(
        self,
        text: str,
        *,
        has_tool_history: bool,
    ) -> bool:
        """Require a real file read when workspace refs are available for factual answers."""
        refs = getattr(self.session, "workspace_refs", None) if self.session else None
        if has_tool_history or not isinstance(refs, list) or not refs:
            return False
        stripped = text.strip()
        if not stripped:
            return False
        if stripped.endswith("?"):
            return False
        return True

    async def _invoke_with_tool_retry(
        self,
        messages: List[BaseMessage],
        llm: Any,
        *,
        thread_id: Optional[str] = None,
        images: bool = False,
    ) -> Tuple[AIMessage, List[BaseMessage]]:
        """Invoke the LLM, retrying if it writes a fake tool call as text.

        Returns ``(response, messages)`` where *messages* may have had nudge
        entries appended during retries (for context only — not persisted).
        """
        retry_messages = self._prepare_messages_for_context_budget(
            list(messages),
            thread_id=thread_id,
            images=images,
        )
        context_retries = 0
        for attempt in range(_MAX_FAKE_TOOL_RETRIES):
            try:
                response = await self._invoke(retry_messages, llm=llm)
            except Exception as exc:
                from pawlia.llm import is_context_length_error
                if not is_context_length_error(exc) or context_retries >= _MAX_CONTEXT_RECOVERY_RETRIES:
                    raise
                compacted = self._prepare_messages_for_context_budget(
                    retry_messages,
                    thread_id=thread_id,
                    images=images,
                )
                if compacted == retry_messages and len(retry_messages) > 1:
                    compacted = [retry_messages[0]] + retry_messages[-_CONTEXT_MIN_NON_SYSTEM_KEEP:]
                if compacted == retry_messages:
                    raise
                context_retries += 1
                self.logger.warning(
                    "ChatAgent compacted prompt after context-limit error (%d/%d)",
                    context_retries,
                    _MAX_CONTEXT_RECOVERY_RETRIES,
                )
                retry_messages = compacted
                continue
            fake_calls = self._extract_fake_skill_calls(response)
            if fake_calls:
                self.logger.warning(
                    "Recovered %d text-form skill call(s) from model output",
                    len(fake_calls),
                )
                response.tool_calls = fake_calls
                return response, retry_messages
            if self._is_disabled_skill_call(response):
                self.logger.warning("Model called disabled skill, nudging")
                retry_messages = retry_messages + [
                    response,
                    HumanMessage(content=_DISABLED_SKILL_NUDGE),
                ]
                continue
            if not self._is_fake_tool_call(response):
                return response, retry_messages
            self.logger.warning(
                "Fake tool call detected (attempt %d/%d), nudging LLM",
                attempt + 1, _MAX_FAKE_TOOL_RETRIES,
            )
            retry_messages = retry_messages + [
                response,
                HumanMessage(content=_FAKE_TOOL_CALL_NUDGE),
            ]
        # Last attempt — return whatever we got
        response = await self._invoke(retry_messages, llm=llm)
        return response, retry_messages

    async def _persist(
        self,
        user_input: str,
        response: str,
        *,
        track_similarity: bool = True,
        thread_id: Optional[str] = None,
        tool_calls_info: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Save exchange to daily log and schedule summarization if needed."""
        if not (self.memory and self.session):
            return

        if thread_id:
            # Thread exchanges go to a separate log; main session is unchanged.
            self.memory.append_thread_exchange(
                self.session, thread_id, user_input, response, tool_calls_info
            )
            return

        # Consume pending topic heading (set by _handle_workspace_context on topic shift)
        topic_heading = self.session.pending_topic_heading
        if topic_heading:
            self.session.pending_topic_heading = None

        self.memory.append_exchange(
            self.session, user_input, response,
            track_similarity=track_similarity,
            tool_calls_info=tool_calls_info,
            topic_heading=topic_heading,
        )

        # Summarization is handled by the Scheduler based on idle time.
