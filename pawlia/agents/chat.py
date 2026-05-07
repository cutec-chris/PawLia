"""ChatAgent - conversational dispatcher that delegates work to SkillRunners.

The ChatAgent has NO tools of its own. It only knows about available skills
(via their OpenAI function specs). When the LLM decides a skill is needed,
the ChatAgent spawns a SkillRunnerAgent to do the actual work, then
incorporates the result into its final response.
"""

import asyncio
import json
import logging
import re
import uuid
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, List, Optional, Tuple

# Callback types
InterimCallback = Callable[[str], Awaitable[None]]
SkillStartCallback = Callable[[str, str], Awaitable[None]]  # (skill_name, query)

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI

from pawlia.agents.base import BaseAgent, log_prompt
from pawlia.prompt_utils import load_system_prompt
from pawlia.skills.loader import AgentSkill

if TYPE_CHECKING:
    from pawlia.memory import MemoryManager, Session

_SENTENCE_RE = re.compile(r'[.!?…]\s')
_RE_CODE_BLOCK = re.compile(r'```[^\n]*\n(.*?)(?:```|$)', re.DOTALL)
_RE_TOOL_CALL_TAG = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.DOTALL)

_FAKE_TOOL_CALL_NUDGE = (
    "You wrote a tool call as plain text or a code block instead of using the "
    "actual function-call mechanism. Do NOT write commands as text. "
    "Use a real tool call now."
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
_MAX_CHAT_TOOL_TURNS = 16
_MAX_CHAT_NUDGES = 3
_REPLAY_TOOL_RESULT_LIMIT = 240
_REPLAY_TOOL_CALLS_LIMIT = 3
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
    ):
        super().__init__(llm, logger)
        self.skills = skills
        self.skill_runner_factory = skill_runner_factory
        self.memory = memory
        self.session = session
        self.on_interim = on_interim
        self._workspace_search_cfg: Dict[str, Any] = workspace_search_cfg or {}
        self.on_skill_start: Optional[SkillStartCallback] = None  # (skill_name, query)
        self.on_skill_step: Optional[InterimCallback] = None      # (step_description)
        self.on_skill_done: Optional[InterimCallback] = None      # (skill_name)
        self.on_model_change: Optional[Callable[[str], None]] = None  # (new_model)

        # Bind skill specs as "tools" so the LLM can call them
        self._skill_specs = [s.as_openai_spec() for s in skills.values()]
        if self._skill_specs:
            self.bound_llm = llm.bind_tools(self._skill_specs, tool_choice="auto")
            self.vision_bound_llm = (vision_llm or llm).bind_tools(self._skill_specs, tool_choice="auto")
        else:
            self.bound_llm = llm
            self.vision_bound_llm = vision_llm or llm

        # Resolver for session/thread-specific agent selection at run() time.
        # Set by App.make_agent after construction.
        self._agent_llm_resolver: Optional[Callable[[str, Optional[str]], Any]] = None
        # Resolves config keys (e.g. "fast") to actual model names (e.g. "qwen3.5:4b").
        self._model_name_resolver: Optional[Callable[[str], str]] = None

        # Callback to re-discover workspace skills (set by App.make_agent).
        # Called after each skill returns so that skills created at runtime
        # (e.g. by skill-creator) become available immediately.
        self._skills_refresher: Optional[Callable[[], None]] = None

    def _run_workspace_search(self, query: str) -> None:
        """Run BM25 search over workspace on the first turn and cache results on session."""
        if not (self.memory and self.session):
            return
        try:
            from pawlia.workspace_search import WorkspaceSearch
            workspace = self.memory._workspace_dir(self.session.user_id)
            hits = WorkspaceSearch(workspace, config=self._workspace_search_cfg).search(query)
            self.session.workspace_refs = hits
            if hits:
                self.logger.debug(
                    "Workspace search: %d hit(s) for %r", len(hits), query[:60]
                )
        except Exception as exc:
            self.logger.warning("Workspace search failed: %s", exc)
            self.session.workspace_refs = []  # prevent retry on next turn

    def build_system_prompt(
        self,
        *,
        mode: str = "chat",
        system_prompt: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> str:
        """Resolve the system prompt for a chat or call context."""
        if system_prompt:
            return system_prompt
        if self.memory and self.session:
            return self.memory.build_system_prompt(
                self.session,
                skills=self.skills,
                mode=mode,
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
        self._skills_refresher()

        if len(self.skills) == prev_count:
            return False

        # New skills appeared — rebuild specs and rebind tools
        old_names = {s["function"]["name"] for s in self._skill_specs}
        self._skill_specs = [s.as_openai_spec() for s in self.skills.values()]
        if self._skill_specs:
            base_llm = self.llm  # underlying ChatOpenAI without tools
            self.bound_llm = base_llm.bind_tools(self._skill_specs, tool_choice="auto")
            self.vision_bound_llm = (
                (self.vision_llm or base_llm).bind_tools(self._skill_specs, tool_choice="auto")
                if hasattr(self, "vision_llm") else self.bound_llm
            )
        new_names = {s["function"]["name"] for s in self._skill_specs} - old_names
        self.logger.info(
            "Skills rebound (%d → %d), new: %s",
            prev_count, len(self.skills),
            ", ".join(new_names),
        )
        return True

    def _resolve_skill_name(self, name: str) -> str:
        """Resolve minor skill-name variations from model tool calls."""
        normalized = name.replace("_", "").replace("-", "").lower()
        for skill_name in self.skills:
            candidate = skill_name.replace("_", "").replace("-", "").lower()
            if candidate == normalized:
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

        summary = "Earlier skill use:\n" + "\n".join(lines)
        return f"{bot_text}\n\n{summary}" if bot_text else summary

    async def run(
        self,
        user_input: str,
        system_prompt: Optional[str] = None,
        images: Optional[List[str]] = None,
        thread_id: Optional[str] = None,
        on_skill_start: Optional[SkillStartCallback] = None,
        on_skill_step: Optional[InterimCallback] = None,
        on_skill_done: Optional[InterimCallback] = None,
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
        # Resolve callbacks: per-call overrides > instance attributes
        _on_skill_start = on_skill_start or self.on_skill_start
        _on_skill_step = on_skill_step or self.on_skill_step
        _on_skill_done = on_skill_done or self.on_skill_done

        # First-turn workspace context search (cached for the rest of the session)
        if self.session is not None and self.session.workspace_refs is None:
            self._run_workspace_search(user_input)

        prompt = self.build_system_prompt(system_prompt=system_prompt)

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

        # Resolve the LLMs to use for this call.
        # A thread-specific model override takes priority over the session default.
        bound_llm, unbound_llm = self._resolve_llms(thread_id, images=bool(images))

        # Build multimodal content when images are present
        if images:
            self.logger.debug("Sending %d image(s) to LLM", len(images))
            content: List[Dict[str, Any]] = [{"type": "text", "text": user_input or "What's in this image?"}]
            for data_uri in images:
                content.append({"type": "image_url", "image_url": {"url": data_uri}})
            messages.append(HumanMessage(content=content))
        else:
            messages.append(HumanMessage(content=user_input))

        # Turn 1: LLM decides whether to call a skill or answer directly
        active_llm = bound_llm
        response, messages = await self._invoke_with_tool_retry(messages, llm=active_llm)

        tool_calls_info: List[Dict[str, Any]] = []
        final = response
        nudge_count = 0

        for turn in range(_MAX_CHAT_TOOL_TURNS):
            self.logger.debug(
                "Chat tool loop turn %d: tool_calls=%s, content=%s",
                turn,
                bool(final.tool_calls),
                repr(final.content[:200]) if final.content else "(empty)",
            )

            if not final.tool_calls:
                result = self.extract_text(final)
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
                    final, messages = await self._invoke_with_tool_retry(messages, llm=active_llm)
                    continue
                break

            interim = self.extract_text(final)
            if interim and self.on_interim:
                try:
                    await self.on_interim(interim)
                except Exception as exc:
                    self.logger.debug("on_interim callback error: %s", exc)

            messages.append(final)

            for tool_call in final.tool_calls:
                skill_name, normalized_args, error = self._decode_skill_call(tool_call)
                query = normalized_args.get("query", "")
                skill = self.skills.get(skill_name)

                if error:
                    self.logger.warning("Skill call rejected: %s", error)
                    result = error
                elif skill:
                    self.logger.info("Delegating to skill '%s': %s", skill_name, query[:80])
                    if _on_skill_start:
                        try:
                            await _on_skill_start(skill_name, query)
                        except Exception as exc:
                            self.logger.debug("on_skill_start error: %s", exc)
                    runner = self.skill_runner_factory(skill, thread_id)
                    runner.on_step = _on_skill_step
                    result = await runner.run(query=query)
                    result = self._process_directives(result, thread_id)
                    if _on_skill_done:
                        try:
                            await _on_skill_done(skill_name)
                        except Exception as exc:
                            self.logger.debug("on_skill_done error: %s", exc)
                else:
                    self.logger.warning("Unknown skill called: %s", skill_name)
                    result = f"Error: Unknown skill '{skill_name}'."

                tool_calls_info.append({
                    "name": skill_name,
                    "args": normalized_args,
                    "result": result,
                })

                messages.append(ToolMessage(
                    content=result,
                    tool_call_id=tool_call.get("id", ""),
                ))

            # Refresh workspace skills (e.g. skill-creator may have added one)
            if self._refresh_and_rebind_skills():
                bound_llm = self.bound_llm
                active_llm = bound_llm

            final, messages = await self._invoke_with_tool_retry(messages, llm=active_llm)
        else:
            self.logger.warning("Max chat tool turns reached, forcing final response")
            final = await self._invoke(
                messages + [HumanMessage(content=_EMPTY_TURN2_NUDGE)],
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
        on_skill_done: Optional[InterimCallback] = None,
    ) -> str:
        """Like :meth:`run` but streams the LLM and calls *on_sentence* per sentence.

        Each complete sentence (delimited by ``.``, ``!``, ``?``, ``…`` +
        whitespace) is emitted as soon as it is detected in the token stream,
        enabling incremental TTS playback.  Falls back to non-streamed skill
        execution when tool calls are detected; the final-answer turn is also
        streamed.
        """
        _on_skill_start = on_skill_start or self.on_skill_start
        _on_skill_step = on_skill_step or self.on_skill_step
        _on_skill_done = on_skill_done or self.on_skill_done

        # First-turn workspace context search (cached for the rest of the session)
        if self.session is not None and self.session.workspace_refs is None:
            self._run_workspace_search(user_input)

        prompt = self.build_system_prompt(system_prompt=system_prompt)

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

        bound_llm, unbound_llm = self._resolve_llms(thread_id, images=bool(images))

        if images:
            content: List[Dict[str, Any]] = [{"type": "text", "text": user_input or "What's in this image?"}]
            for data_uri in images:
                content.append({"type": "image_url", "image_url": {"url": data_uri}})
            messages.append(HumanMessage(content=content))
        else:
            messages.append(HumanMessage(content=user_input))

        # ---- Stream turn 1 ----
        # _partial_text tracks generated text for persist-on-cancel (barge-in during TTS
        # cancels this coroutine before _persist is reached, losing the user's turn from history).
        _partial_text = ""
        try:
            accumulated, raw_text = await self._stream_with_sentences(
                messages, bound_llm, on_sentence,
            )
            _partial_text = raw_text

            self.logger.debug("Streamed turn 1: tool_calls=%s, len=%d",
                              bool(getattr(accumulated, "tool_calls", None)), len(raw_text))

            # If the streamed response is a fake tool call, retry non-streamed
            if accumulated and self._is_fake_tool_call(accumulated):
                self.logger.warning("Fake tool call detected in streamed turn 1, retrying non-streamed")
                accumulated, messages = await self._invoke_with_tool_retry(messages, llm=bound_llm)
                raw_text = accumulated.content if isinstance(accumulated.content, str) else ""
                _partial_text = raw_text

            if not accumulated or not getattr(accumulated, "tool_calls", None):
                result = self.strip_thinking(raw_text)
                await self._persist(user_input, result, track_similarity=True, thread_id=thread_id)
                return result

            # ---- Skill calls detected → execute (non-streamed) ----
            messages.append(accumulated)
            tool_calls_info: List[Dict[str, Any]] = []

            for tool_call in accumulated.tool_calls:
                skill_name, normalized_args, error = self._decode_skill_call(tool_call)
                query = normalized_args.get("query", "")
                skill = self.skills.get(skill_name)

                if error:
                    self.logger.warning("Skill call rejected: %s", error)
                    skill_result = error
                elif skill:
                    self.logger.info("Delegating to skill '%s': %s", skill_name, query[:80])
                    if _on_skill_start:
                        try:
                            await _on_skill_start(skill_name, query)
                        except Exception:
                            pass
                    runner = self.skill_runner_factory(skill, thread_id)
                    runner.on_step = _on_skill_step
                    skill_result = await runner.run(query=query)
                    skill_result = self._process_directives(skill_result, thread_id)
                    if _on_skill_done:
                        try:
                            await _on_skill_done(skill_name)
                        except Exception:
                            pass
                else:
                    self.logger.warning("Unknown skill called: %s", skill_name)
                    skill_result = f"Error: Unknown skill '{skill_name}'."

                tool_calls_info.append({
                    "name": skill_name,
                    "args": normalized_args,
                    "result": skill_result,
                })
                messages.append(ToolMessage(
                    content=skill_result,
                    tool_call_id=tool_call.get("id", ""),
                ))

            # Refresh workspace skills (e.g. skill-creator may have added one)
            if self._refresh_and_rebind_skills():
                bound_llm = self.bound_llm

            # ---- Continue tool loop until the task is actually complete ----
            raw_text2 = ""
            nudge_count = 0
            final_response: Optional[AIMessage] = None

            for _turn in range(_MAX_CHAT_TOOL_TURNS):
                next_response, messages = await self._invoke_with_tool_retry(messages, llm=bound_llm)

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
                for tool_call in next_response.tool_calls:
                    skill_name, normalized_args, error = self._decode_skill_call(tool_call)
                    query = normalized_args.get("query", "")
                    skill = self.skills.get(skill_name)

                    if error:
                        self.logger.warning("Skill call rejected: %s", error)
                        skill_result = error
                    elif skill:
                        self.logger.info("Delegating to skill '%s': %s", skill_name, query[:80])
                        if _on_skill_start:
                            try:
                                await _on_skill_start(skill_name, query)
                            except Exception:
                                pass
                        runner = self.skill_runner_factory(skill, thread_id)
                        runner.on_step = _on_skill_step
                        skill_result = await runner.run(query=query)
                        skill_result = self._process_directives(skill_result, thread_id)
                        if _on_skill_done:
                            try:
                                await _on_skill_done(skill_name)
                            except Exception:
                                pass
                    else:
                        self.logger.warning("Unknown skill called: %s", skill_name)
                        skill_result = f"Error: Unknown skill '{skill_name}'."

                    tool_calls_info.append({
                        "name": skill_name,
                        "args": normalized_args,
                        "result": skill_result,
                    })
                    messages.append(ToolMessage(
                        content=skill_result,
                        tool_call_id=tool_call.get("id", ""),
                    ))

                # Refresh workspace skills after each skill return
                if self._refresh_and_rebind_skills():
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

    async def _stream_with_sentences(
        self,
        messages: List[BaseMessage],
        llm: Any,
        on_sentence: Optional[Callable[[str], Awaitable[None]]],
    ) -> Tuple[Any, str]:
        """Stream an LLM call, emitting complete sentences via *on_sentence*.

        Returns ``(accumulated_message, raw_text)``.
        """
        accumulated = None
        raw_text = ""
        emitted_len = 0  # how much of the clean text has been emitted

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
                    if thread_id:
                        self.memory.set_thread_model_override(self.session, thread_id, model)
                        self.logger.info("Directive: thread '%s' model override set to '%s'", thread_id, model)
                    else:
                        self.memory.set_model_override(self.session, model)
                        self.logger.info("Directive: model override set to '%s'", model)
                    if self.on_model_change:
                        self.on_model_change(model)
            elif directive == "set_agent_override":
                path = str(obj.get("path", "") or "").strip()
                value = obj.get("value")
                if path and self.memory and self.session:
                    target_thread = obj.get("thread") or thread_id
                    self.memory.set_agent_override_value(
                        self.session,
                        path,
                        str(value).strip() if isinstance(value, str) and str(value).strip() else None,
                        thread_id=target_thread,
                    )
                    self.logger.info(
                        "Directive: %s agent override '%s' -> %r",
                        f"thread '{target_thread}'" if target_thread else "session",
                        path,
                        value,
                    )
                    if self.on_model_change:
                        self.on_model_change(str(value or ""))
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
            bound = llm.bind_tools(self._skill_specs, tool_choice="auto") if self._skill_specs else llm
            return bound, llm

        return (self.vision_bound_llm if images else self.bound_llm), self.llm

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
            first_token = block.split()[0] if block else ""
            skill_name = self._resolve_skill_name(first_token)
            if skill_name in skill_names:
                query = block[len(first_token):].strip()
                if query:
                    calls.append({
                        "id": f"fake_{uuid.uuid4().hex[:8]}",
                        "name": skill_name,
                        "args": {"query": query},
                    })

        return calls

    def _is_fake_tool_call(self, response: AIMessage) -> bool:
        """Return True if the LLM wrote a skill call as text."""
        if self._extract_fake_skill_calls(response):
            return True
        if not self._skill_specs or response.tool_calls:
            return False
        content = response.content if isinstance(response.content, str) else ""
        if "<tool_call>" in content:
            return True
        skill_names = set(self.skills.keys())
        for match in _RE_CODE_BLOCK.finditer(content):
            block = match.group(1).strip()
            first_token = block.split()[0] if block else ""
            if self._resolve_skill_name(first_token) in skill_names:
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

    async def _invoke_with_tool_retry(
        self,
        messages: List[BaseMessage],
        llm: Any,
    ) -> Tuple[AIMessage, List[BaseMessage]]:
        """Invoke the LLM, retrying if it writes a fake tool call as text.

        Returns ``(response, messages)`` where *messages* may have had nudge
        entries appended during retries (for context only — not persisted).
        """
        retry_messages = list(messages)
        for attempt in range(_MAX_FAKE_TOOL_RETRIES):
            response = await self._invoke(retry_messages, llm=llm)
            fake_calls = self._extract_fake_skill_calls(response)
            if fake_calls:
                self.logger.warning(
                    "Recovered %d text-form skill call(s) from model output",
                    len(fake_calls),
                )
                response.tool_calls = fake_calls
                return response, retry_messages
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

        self.memory.append_exchange(
            self.session, user_input, response,
            track_similarity=track_similarity,
            tool_calls_info=tool_calls_info,
        )

        # Summarization is handled by the Scheduler based on idle time.
