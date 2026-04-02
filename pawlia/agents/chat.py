"""ChatAgent - conversational dispatcher that delegates work to SkillRunners.

The ChatAgent has NO tools of its own. It only knows about available skills
(via their OpenAI function specs). When the LLM decides a skill is needed,
the ChatAgent spawns a SkillRunnerAgent to do the actual work, then
incorporates the result into its final response.
"""

import json
import logging
import re
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

_FAKE_TOOL_CALL_NUDGE = (
    "You wrote a tool call as plain text or a code block instead of using the "
    "actual function-call mechanism. Do NOT write commands as text. "
    "Use a real tool call now."
)
_MAX_FAKE_TOOL_RETRIES = 5
_EMPTY_TURN2_NUDGE = "The tool finished. Please respond to the user now."


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
        skill_runner_factory: Callable[[AgentSkill], Any],
        logger: Optional[logging.Logger] = None,
        memory: Optional["MemoryManager"] = None,
        session: Optional["Session"] = None,
        on_interim: Optional[InterimCallback] = None,
        vision_llm: Optional[ChatOpenAI] = None,
    ):
        super().__init__(llm, logger)
        self.skills = skills
        self.skill_runner_factory = skill_runner_factory
        self.memory = memory
        self.session = session
        self.on_interim = on_interim
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

        # Resolver for per-thread model overrides: model_name -> ChatOpenAI
        # Set by App.make_agent after construction.
        self._llm_resolver: Optional[Callable[[str], Any]] = None
        # Resolver for default agent-type LLMs (e.g. "chat", "vision").
        # Used to fall back when an override model is unreachable.
        self._fallback_resolver: Optional[Callable[[str], Any]] = None
        # Resolves config keys (e.g. "fast") to actual model names (e.g. "qwen3.5:4b").
        self._model_name_resolver: Optional[Callable[[str], str]] = None

    def build_system_prompt(
        self,
        *,
        mode: str = "chat",
        system_prompt: Optional[str] = None,
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

                # Restore tool calls if present
                if tool_calls_info:
                    # Create AIMessage with reconstructed tool_calls
                    reconstructed_tool_calls = []
                    for tc in tool_calls_info:
                        reconstructed_tool_calls.append({
                            "name": tc["name"],
                            "args": tc["args"],
                            "id": f"restored_{len(reconstructed_tool_calls)}",
                        })
                    messages.append(AIMessage(
                        content=bot_text,
                        tool_calls=reconstructed_tool_calls,
                    ))
                    # Add ToolMessage for each restored tool call with its result
                    for i, tc in enumerate(tool_calls_info):
                        messages.append(ToolMessage(
                            content=tc["result"],
                            tool_call_id=f"restored_{i}",
                        ))
                else:
                    messages.append(AIMessage(content=bot_text))

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
        _override_notice = ""
        override_model = self._active_override_model(thread_id)
        try:
            response, messages = await self._invoke_with_tool_retry(messages, llm=active_llm)
        except Exception as exc:
            if not override_model:
                raise
            self.logger.warning(
                "Override model '%s' unreachable (%s), falling back to default",
                override_model, exc,
            )
            _override_notice = (
                f"⚠️ Modell *{override_model}* war nicht erreichbar – "
                f"Override wurde deaktiviert.\n\n"
            )
            bound_llm, unbound_llm = self._clear_override_and_fallback(
                thread_id, images=bool(images),
            )
            active_llm = bound_llm
            response, messages = await self._invoke_with_tool_retry(messages, llm=active_llm)

        self.logger.debug("LLM response: tool_calls=%s, content=%s",
                          bool(response.tool_calls),
                          repr(response.content[:200]) if response.content else "(empty)")

        if not response.tool_calls:
            result = self.extract_text(response)
            if _override_notice:
                result = _override_notice + result
            await self._persist(user_input, result, track_similarity=True, thread_id=thread_id)
            return result

        # Send interim message if the LLM included text alongside tool calls
        interim = self.extract_text(response)
        if interim and self.on_interim:
            try:
                await self.on_interim(interim)
            except Exception as exc:
                self.logger.debug("on_interim callback error: %s", exc)

        # Skill calls detected -> execute via SkillRunners
        messages.append(response)  # AIMessage with tool_calls

        # Collect tool call information for persistent storage
        tool_calls_info: List[Dict[str, Any]] = []

        for tool_call in response.tool_calls:
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
                runner = self.skill_runner_factory(skill)
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

            # Store tool call info for persistent storage
            tool_calls_info.append({
                "name": skill_name,
                "args": normalized_args,
                "result": result,
            })

            messages.append(ToolMessage(
                content=result,
                tool_call_id=tool_call.get("id", ""),
            ))

        # Turn 2: LLM formulates final answer incorporating skill results
        # Use unbound LLM (no tools) for the final response
        self.logger.debug("Turn 2: sending %d messages to LLM for final answer", len(messages))
        try:
            final = await self._invoke(messages, llm=unbound_llm)
        except Exception as exc:
            error_str = str(exc)
            if "tool_use_failed" in error_str or \
               ("Tool choice is none" in error_str and "called a tool" in error_str):
                # Model output tool calls as JSON. Give it the tool results again
                # with a clear hint that no more tool calls are needed.
                self.logger.warning("Turn 2: model output tool calls as JSON, "
                                    "retrying with explicit guidance")
                result_summary = "\n".join(
                    f"[{tc['name']}] {tc['result'][:200]}"
                    for tc in tool_calls_info if tc['result']
                )
                guidance = (
                    f"The tools have been executed. Here are the results:\n\n"
                    f"{result_summary}\n\n"
                    f"No more tool calls are needed. "
                    f"Now respond to the user with a natural text answer "
                    f"based on these results."
                )
                retry_messages = messages + [
                    HumanMessage(content=guidance),
                ]
                final = await self._invoke(retry_messages, llm=unbound_llm)
            else:
                raise

        self.logger.debug("Turn 2 response: content=%s",
                          repr(final.content[:200]) if final.content else "(empty)")
        if not final.content:
            self.logger.warning("Turn 2 returned empty response, nudging LLM")
            final = await self._invoke(
                messages + [final, HumanMessage(content=_EMPTY_TURN2_NUDGE)],
                llm=unbound_llm,
            )
            self.logger.debug("Turn 2 nudge response: content=%s",
                              repr(final.content[:200]) if final.content else "(empty)")
        result = self.extract_text(final)
        if _override_notice:
            result = _override_notice + result
        await self._persist(
            user_input, result,
            track_similarity=False,
            thread_id=thread_id,
            tool_calls_info=tool_calls_info,
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
                if tc_info:
                    reconstructed = [
                        {"name": tc["name"], "args": tc["args"], "id": f"restored_{i}"}
                        for i, tc in enumerate(tc_info)
                    ]
                    messages.append(AIMessage(content=bot_text, tool_calls=reconstructed))
                    for i, tc in enumerate(tc_info):
                        messages.append(ToolMessage(content=tc["result"], tool_call_id=f"restored_{i}"))
                else:
                    messages.append(AIMessage(content=bot_text))

        bound_llm, unbound_llm = self._resolve_llms(thread_id, images=bool(images))

        if images:
            content: List[Dict[str, Any]] = [{"type": "text", "text": user_input or "What's in this image?"}]
            for data_uri in images:
                content.append({"type": "image_url", "image_url": {"url": data_uri}})
            messages.append(HumanMessage(content=content))
        else:
            messages.append(HumanMessage(content=user_input))

        # ---- Stream turn 1 ----
        _override_notice = ""
        override_model = self._active_override_model(thread_id)
        try:
            accumulated, raw_text = await self._stream_with_sentences(
                messages, bound_llm, on_sentence,
            )
        except Exception as exc:
            if not override_model:
                raise
            self.logger.warning(
                "Override model '%s' unreachable (%s), falling back to default",
                override_model, exc,
            )
            _override_notice = (
                f"⚠️ Modell *{override_model}* war nicht erreichbar – "
                f"Override wurde deaktiviert.\n\n"
            )
            bound_llm, unbound_llm = self._clear_override_and_fallback(
                thread_id, images=bool(images),
            )
            if on_sentence:
                await on_sentence(_override_notice.strip())
            accumulated, raw_text = await self._stream_with_sentences(
                messages, bound_llm, on_sentence,
            )

        self.logger.debug("Streamed turn 1: tool_calls=%s, len=%d",
                          bool(getattr(accumulated, "tool_calls", None)), len(raw_text))

        # If the streamed response is a fake tool call, retry non-streamed
        if accumulated and self._is_fake_tool_call(accumulated):
            self.logger.warning("Fake tool call detected in streamed turn 1, retrying non-streamed")
            accumulated, messages = await self._invoke_with_tool_retry(messages, llm=bound_llm)
            raw_text = accumulated.content if isinstance(accumulated.content, str) else ""

        if not accumulated or not getattr(accumulated, "tool_calls", None):
            result = self.strip_thinking(raw_text)
            if _override_notice:
                result = _override_notice + result
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
                runner = self.skill_runner_factory(skill)
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

        # ---- Stream turn 2 (final answer) ----
        try:
            accumulated2, raw_text2 = await self._stream_with_sentences(
                messages, unbound_llm, on_sentence,
            )
        except Exception as exc:
            error_str = str(exc)
            if "tool_use_failed" in error_str or \
               ("Tool choice is none" in error_str and "called a tool" in error_str):
                self.logger.warning("Streamed turn 2: model output tool calls as JSON")
                result_parts = []
                for tc_info in tool_calls_info:
                    res = tc_info["result"]
                    if res and not res.startswith("Error:"):
                        result_parts.append(f"[{tc_info['name']}] {res[:300]}")
                raw_text2 = "\n".join(result_parts) if result_parts else \
                    "Task completed. (Model was unable to generate a text response.)"
            else:
                raise

        result = self.strip_thinking(raw_text2)
        if _override_notice:
            result = _override_notice + result
        await self._persist(
            user_input, result,
            track_similarity=False,
            thread_id=thread_id,
            tool_calls_info=tool_calls_info,
        )
        return result

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

        Checks for a thread-specific model override first, then a session-level
        override, and finally falls back to the agent's default LLMs.
        Both overrides are resolved dynamically so directive-based changes
        (set_model) take effect without requiring an agent restart.
        """
        if self._llm_resolver and self.memory and self.session:
            model: Optional[str] = None
            if thread_id:
                model = self.memory.get_thread_model_override(self.session, thread_id)
            if not model:
                model = self.session.model_override
            if model:
                llm = self._llm_resolver(model)
                bound = llm.bind_tools(self._skill_specs, tool_choice="auto") if self._skill_specs else llm
                return bound, llm

        return (self.vision_bound_llm if images else self.bound_llm), self.llm

    def _active_override_model(self, thread_id: Optional[str]) -> Optional[str]:
        """Return the name of the active model override, or ``None``."""
        if thread_id and self.memory and self.session:
            model = self.memory.get_thread_model_override(self.session, thread_id)
            if model:
                return model
        if self.session and self.session.model_override:
            return self.session.model_override
        return None

    def _clear_override_and_fallback(
        self, thread_id: Optional[str], *, images: bool = False,
    ) -> Tuple[Any, Any]:
        """Clear all active overrides and return default ``(bound, unbound)`` LLMs."""
        if thread_id and self.memory and self.session:
            if self.memory.get_thread_model_override(self.session, thread_id):
                self.memory.set_thread_model_override(self.session, thread_id, None)

        if self.session and self.session.model_override and self.memory:
            self.memory.set_model_override(self.session, None)

        if self._fallback_resolver:
            llm = self._fallback_resolver("chat")
            vision = self._fallback_resolver("vision")
        else:
            llm = self.llm
            vision = self.llm

        if self._skill_specs:
            bound = llm.bind_tools(self._skill_specs, tool_choice="auto")
            vision_bound = vision.bind_tools(self._skill_specs, tool_choice="auto")
        else:
            bound = llm
            vision_bound = vision

        # Update agent LLMs so subsequent calls in this session use defaults
        self.llm = llm
        self.bound_llm = bound
        self.vision_bound_llm = vision_bound

        return (vision_bound if images else bound), llm

    def _is_fake_tool_call(self, response: AIMessage) -> bool:
        """Return True if the LLM wrote a tool call as text instead of calling it.

        Detects fenced code blocks whose first token matches a known skill name.
        Only relevant when tools are bound and no real tool_calls are present.
        """
        if not self._skill_specs or response.tool_calls:
            return False
        content = response.content if isinstance(response.content, str) else ""
        skill_names = set(self.skills.keys())
        for match in _RE_CODE_BLOCK.finditer(content):
            first_token = match.group(1).strip().split()[0] if match.group(1).strip() else ""
            if first_token in skill_names:
                return True
        return False

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
