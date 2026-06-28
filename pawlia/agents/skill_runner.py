"""SkillRunnerAgent - executes a single skill with real tools.

Supports two modes:
- Tool-call mode: LLM calls bash/tools directly (larger models)
- Command mode: LLM outputs a shell command as text, we execute it
  (fallback when the model ignores tool calling)
"""

import asyncio
import json
import logging
import os
import re
import subprocess
import sys
import uuid
from typing import Any, Dict, List, Optional

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI

from pawlia.agents.base import BaseAgent
from pawlia import credentials as credstore
from pawlia.prompt_utils import load_system_prompt
from pawlia.skills.executor import WorkflowExecutor
from pawlia.skills.loader import AgentSkill
from pawlia.tools.base import ToolExecutionResult, ToolRegistry

_RE_CODE_BLOCK = re.compile(r"```(?:bash|sh)?\s*\n(.+?)```", re.DOTALL)
_RE_ANY_CODE_BLOCK = re.compile(r"```[^\n]*\n(.+?)```", re.DOTALL)
from pawlia.agents.chat import _RE_TOOL_CALL_TAG


def _repair_tool_args(args: Any) -> Dict[str, Any]:
    """Normalize malformed tool-call arguments from small models.

    Common failure modes:
    - ``args`` is a bare string instead of a dict
    - ``args`` is a list (first element is the real payload)
    - ``args`` is a dict with wrong key casing or extra junk
    """
    if args is None:
        return {}
    if isinstance(args, str):
        stripped = args.strip()
        if stripped:
            return {"command": stripped} if stripped else {}
        return {}
    if isinstance(args, list):
        if args:
            return _repair_tool_args(args[0])
        return {}
    if not isinstance(args, dict):
        return {}

    return dict(args)


class SkillRunnerAgent(BaseAgent):
    """Worker agent that executes a skill using real tools.

    Tries tool-call mode first. If ``command_fallback`` is enabled and
    tool-call mode produces no output, falls back to command mode.

    Context policy: the skill's tool-call history is kept **complete** for
    the entire run. Individual tool outputs larger than 4 kB are truncated
    (the LLM only needs success status and key fields; raw output is in
    the Docker logs). Compaction is allowed ONLY as a guardrail when the
    real LLM context window is about to overflow — and even then via
    summary, never by dropping messages.
    """

    MAX_TOOL_TURNS = 30
    MAX_RETRIES = 2
    # Max chars of a single tool result fed back to the model. Large enough that
    # one paginated files read page (capped at _READ_BYTE_BUDGET=10 kB content,
    # plus JSON envelope/escaping) survives intact — otherwise the model sees a
    # truncation marker and re-reads the same range in a loop. The full raw
    # output is always in the Docker logs for debugging.
    MAX_RESULT = 16_000
    # No-progress circuit breakers. Without these a model can grind for the
    # full turn budget re-reading the same files / bloating the context until
    # every call overflows — burning tokens and never reporting back.
    _MAX_CONSECUTIVE_OVERFLOW = 3   # turns the prompt couldn't be made to fit
    _NUDGE_SAME_TOOL_CALL = 4       # identical tool call repeats → nudge once
    _ABORT_SAME_TOOL_CALL = 6       # identical tool call repeats → give up
    # Silent-tool-call circuit breaker: if the model issues N consecutive turns
    # whose visible content is empty (all reasoning hidden in <think> blocks or
    # the model just emits tool calls with no intermediate narration) it is
    # likely stuck in a verification / exploration loop.  Nudge it once at
    # _NUDGE_SILENT_TURNS; abort at _ABORT_SILENT_TURNS.
    _NUDGE_SILENT_TURNS = 6         # consecutive silent tool-call turns → nudge
    _ABORT_SILENT_TURNS = 10        # consecutive silent tool-call turns → abort
    # The skill's own script crashing with the same traceback N times means the
    # script is broken, not that the model is choosing badly — so the identical-
    # tool-call breaker above never catches it (the model keeps trying *different*
    # commands around the same broken script). Stop early and offer a repair
    # instead of burning the whole turn budget on manual workarounds.
    _ABORT_REPEATED_ERROR = 3

    def __init__(
        self,
        llm: ChatOpenAI,
        skill: AgentSkill,
        tool_registry: ToolRegistry,
        context: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None,
        command_fallback: bool = True,
        max_tool_turns: Optional[int] = None,
    ):
        super().__init__(llm, logger)
        self.log_name = f"skill_{skill.name}"
        self.skill = skill
        self.tool_registry = tool_registry
        self.context = context or {}
        # Admin skills (skill-creator) operate in the user's workspace, not the
        # bundled skills directory, so the sub-agent never sees /app/skills/ paths.
        ws_cwd = None
        if skill.name == "skill-creator":
            session_dir = (context or {}).get("session_dir", "")
            user_id = (context or {}).get("user_id", "")
            if session_dir and user_id:
                ws_cwd = os.path.join(
                    session_dir, user_id, "workspace", "skills", "skill-creator"
                )
                try:
                    os.makedirs(ws_cwd, exist_ok=True)
                except OSError:
                    ws_cwd = None

        self.context["cwd"] = ws_cwd or skill.base_dir
        self.context["skills_root"] = os.path.dirname(skill.base_dir)
        self.command_fallback = command_fallback
        self.max_tool_turns = max_tool_turns if (isinstance(max_tool_turns, int) and max_tool_turns > 0) else self.MAX_TOOL_TURNS

        # Load matching credentials into context
        self._load_credentials()
        self.on_step = None  # Optional[Callable[[str], Awaitable[None]]]
        self._directives: List[str] = []  # collected __directive__ lines from tool output
        # Track repeated crashes of the skill's own script (see
        # _ABORT_REPEATED_ERROR). Keyed by error signature → count.
        self._error_sig_counts: Dict[str, int] = {}
        self._broken_error_excerpt: str = ""
        self._broken_failing_command: str = ""
        self._broken_skill: bool = False

        # Bind real tools to the LLM
        tool_specs = tool_registry.get_specs()
        if tool_specs:
            self.bound_llm = llm.bind_tools(tool_specs, tool_choice="auto")
        else:
            self.bound_llm = llm

    def _load_credentials(self) -> None:
        """Load matching credentials into context as CRED_* env vars.

        Only the keys this skill declared via ``requires_credentials`` are
        exposed. The credential store lives outside the bash sandbox, so
        ``PAWLIA_CREDENTIALS_FILE`` is also set so that management scripts
        (``credentials.py``) can locate it without re-implementing the
        path logic.
        """
        session_dir = self.context.get("session_dir", "")
        user_id = self.context.get("user_id", "")
        if session_dir and user_id:
            # Migrate the legacy file out of the bash-writable area on first
            # contact — runs from PawLia's own process (outside the sandbox).
            credstore.migrate_if_needed(session_dir, user_id)
        if not self.skill.requires_credentials:
            return
        env_extra = credstore.build_env_extra(session_dir, user_id, self.skill.requires_credentials)
        if env_extra:
            self.context.setdefault("env_extra", {}).update(env_extra)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run(self, query: str) -> str:
        """Execute the skill task with retries.

        Only retries when the result is completely empty. Error messages
        from tools are returned as-is since the tool loop already had
        a chance to recover from them.
        """
        self._directives.clear()
        self._error_sig_counts.clear()
        self._broken_error_excerpt = ""
        self._broken_failing_command = ""
        self._broken_skill = False

        # Fast path: if this is the skill-creator and the query looks like a
        # "implement/fix <skill>" instruction, hand it straight to
        # ``creator.py implement|fix`` (which runs the configured coding
        # backend, e.g. opencode) instead of letting the sub-agent LLM
        # iterate manually for 10+ tool calls. Returns ``None`` for anything
        # the pattern does not confidently match (config queries, sync ops,
        # credential management, "validate"/"list" requests, …) so the
        # normal LLM loop handles those.
        direct = await self._try_direct_coding_backend(query)
        if direct is not None:
            return direct

        for attempt in range(1, self.MAX_RETRIES + 1):
            result = await self._attempt(query)
            if result.strip():
                if self._directives:
                    result = "\n".join(self._directives) + "\n" + result
                return result
            self.logger.info("Attempt %d produced no output — retrying", attempt)

        self.logger.warning("All %d attempts produced no output", self.MAX_RETRIES)
        if self._directives:
            return "\n".join(self._directives)
        return ""

    # ------------------------------------------------------------------
    # Direct coding-backend passthrough (skips the sub-agent LLM loop)
    # ------------------------------------------------------------------

    # Verbs that signal "I want code written/changed", not just inspection.
    # ``validate``, ``list``, ``audit``, ``package``, ``compile`` etc. are
    # intentionally absent — those are read-only / structural operations
    # where the LLM loop is fine.
    _DIRECT_CODE_VERBS = (
        # English
        "implement", "rewrite", "refactor",
        "improve", "enhance", "extend", "update", "modernize",
        "fix", "repair", "debug",
        "create", "scaffold", "build",
        "add",  # "add a feature to <skill>"
        # German imperatives (du-Form, singular). The German
        # dispatcher often appends the verb after a description, so
        # these must be recognised anywhere in the query — not just
        # at the start.
        "implementiere", "repariere", "behebe", "verbessere",
        "aktualisiere", "erstelle", "baue", "erweitere",
        "korrigiere", "debugge", "modernisiere",
        "fixen",  # German colloquial infinitive: "den skill fixen"
    )

    # Phrases that disqualify a query from direct passthrough even if a
    # code-verb is present. These are config/inspection operations that
    # happen to use words like "fix" or "change" in a non-coding sense.
    _DIRECT_CODE_BLOCKERS = (
        "coding backend", "coding_backend", "backend to",
        "sync --workspace", "sync workspace",
        "check current", "show current",
    )

    @staticmethod
    def _strip_skill_word(name: str) -> str:
        """Strip a trailing ``skill`` token from a captured skill name."""
        return name[:-6].strip() if name.lower().endswith(" skill") else name

    async def _try_direct_coding_backend(self, query: str) -> Optional[str]:
        """If the query is a clear "write/fix code for <skill>" request,
        invoke ``creator.py implement|fix`` directly and return its output.

        Returns ``None`` when the query does not confidently match — callers
        then fall back to the normal sub-agent LLM loop. Also returns
        ``None`` for any skill other than ``skill-creator``.

        The implementation is deliberately conservative: a false negative
        just means the LLM loop handles it (status quo), while a false
        positive would invoke the coding backend on something the user did
        not want changed. When in doubt → ``None``.
        """
        if self.skill.name != "skill-creator":
            return None

        q = query.strip()
        if not q:
            return None

        # Reject obvious config / sync queries up front — those use "fix"
        # or "change" in a non-coding sense ("change the coding backend",
        # "fix the sync") and must go through the LLM loop.
        q_lower = q.lower()
        if any(b in q_lower for b in self._DIRECT_CODE_BLOCKERS):
            return None

        # Find a code-verb anywhere in the query. German dispatchers
        # often append the verb after a description ("Der X Skill ...
        # Bitte behebe das."), so a head-only search would miss them.
        # Case-insensitive.
        verb_found: Optional[str] = None
        for verb in self._DIRECT_CODE_VERBS:
            m = re.search(rf"\b{re.escape(verb)}\b", q_lower)
            if m:
                verb_found = verb
                break
        if not verb_found:
            return None

        # Disambiguate implement vs fix. Anything containing an error
        # report ("error:", "fails with", "crashes", "traceback", "exit
        # code") routes to ``fix``; otherwise ``implement``. ``fix`` is
        # strictly more informative when error context is present (the
        # backend uses it to target the root cause), so we err on the
        # side of ``implement`` when unsure.
        is_fix = bool(re.search(
            r"\b(error|fails?|crash|traceback|exit code|broken|exception)\b",
            q_lower,
        ))
        mode = "fix" if is_fix else "implement"

        # Common words that are not skill names but can precede ``skill``
        # ("the skill", "a skill", "new skill", "this skill" …). Captures
        # matching one of these are skipped during name extraction below.
        non_name_words = {
            "the", "a", "an", "this", "that", "these", "those",
            "new", "other", "every", "any", "some", "all",
            "no",  # "no skill"
        }

        # Tokenise into hyphen-aware words so multi-part names like
        # "skill-creator" survive as a single token.
        tokens = re.findall(r"[a-z][a-z0-9_-]{2,59}", q_lower)

        # (1) Look for an explicit "<name> skill" pair. Iterating tokens
        #     (instead of regex finditer) avoids the pitfall where matching
        #     "the skill" consumes the "skill" token so that a following
        #     "creator skill" inside "skill-creator skill" gets captured
        #     as the bare name "creator". With tokenisation the predecessor
        #     of "skill" is the full previous token.
        name: Optional[str] = None
        rest = q
        for i, tok in enumerate(tokens):
            if tok == "skill" and i > 0:
                prev = tokens[i - 1]
                if prev in non_name_words:
                    continue
                # Skip if the predecessor is the code-verb itself
                # ("create skill" with no name between them — let the
                # LLM elicit the name from the user).
                if prev == verb_found:
                    continue
                name = prev
                # ``rest`` = everything after the "skill" token. Find
                # its position in the original (case-preserved) query.
                skill_pos = q_lower.find(f"{prev} skill") + len(prev) + len(" skill")
                rest = q[skill_pos:]
                break

        if name is None:
            # (2) Token right after the verb (allowing an optional
            #     "the"/"a" article and an optional trailing colon).
            m_after = re.search(
                rf"\b{re.escape(verb_found)}\b\s+(?:the\s+|a\s+)?"
                rf"([a-z][a-z0-9_-]{{2,59}})\s*?:?",
                q_lower,
            )
            if m_after:
                cand = m_after.group(1)
                if cand not in non_name_words:
                    name = cand
                    rest = q[m_after.end():]

        if not name:
            # No skill name identified → don't guess, let the LLM handle it.
            return None

        # Clean up the task description: drop leading colons, dashes,
        # "Use X backend."-style prefixes the chat agent sometimes adds,
        # and surrounding whitespace.
        task = re.sub(r"^\s*[:\-–—\s]+", "", rest).strip()
        task = re.sub(r"^use\s+\S+\s+backend\.?\s*", "", task, flags=re.IGNORECASE)
        if not task:
            # Verb + skill but no actual instruction — let the LLM ask
            # the user for clarification rather than firing off an empty
            # implement call.
            return None

        # Resolve the script directory the same way SkillRunner sets up
        # ``cwd`` for the sub-agent: bundled skills under /app/skills,
        # workspace skills under the per-user session dir.
        scripts_dir = os.path.join(self.skill.skill_path, "scripts")
        creator_py = os.path.join(scripts_dir, "creator.py")
        if not os.path.isfile(creator_py):
            # skill-creator not installed in this layout — bail out, the
            # LLM loop will produce a sensible error message.
            return None

        # Build the creator.py invocation. ``--task``/`--error`` get the
        # full sentence so the coding backend (opencode/aider/llm) sees
        # the same context the sub-agent would have assembled manually.
        cmd = [
            sys.executable, creator_py, mode, "--name", name,
        ]
        if mode == "fix":
            cmd += ["--error", task, "--failed-cmd", ""]
        else:
            cmd += ["--task", task]

        self.logger.info(
            "skill-creator direct passthrough: %s name=%r task=%r",
            mode, name, task[:120],
        )
        if self.on_step:
            try:
                asyncio.ensure_future(
                    self.on_step(f"Coding-Backend ({mode}): {name}")
                )
            except Exception as exc:
                self.logger.debug("on_step scheduling failed: %s", exc)

        # creator.py + opencode typically take 30–120s. Run in a worker
        # thread so the asyncio event loop stays responsive (Matrix
        # messages, heartbeats, other user requests).
        try:
            proc = await asyncio.to_thread(
                subprocess.run,
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
                cwd=self.context.get("cwd") or self.skill.base_dir,
                env={**os.environ},
            )
        except subprocess.TimeoutExpired:
            self.logger.warning(
                "skill-creator direct passthrough timed out after 600s"
            )
            return None  # fall back to LLM loop (which has its own budget)
        except Exception as exc:
            self.logger.warning(
                "skill-creator direct passthrough failed: %s", exc
            )
            return None

        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()
        if proc.returncode != 0:
            self.logger.warning(
                "skill-creator direct passthrough rc=%d: %s",
                proc.returncode, (err or out)[:300],
            )
            return None  # let the LLM loop surface a real error

        # creator.py prints a JSON result on stdout, but other lines
        # (npm deprecation warnings, Python warnings, blank lines) may
        # follow. Walk backwards and take the first line that parses as
        # JSON; fall back to the raw tail if none do.
        payload = None
        for line in reversed(out.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                break
            except ValueError:
                continue
        if payload is None:
            return out[-2000:]

        backend = payload.get("backend", "?")
        files_written = payload.get("files_written") or []
        files_modified = payload.get("files_modified") or []
        all_files = sorted(set(files_written) | set(files_modified))
        ok = bool(payload.get("success", payload.get("ok", False)))
        error_msg = payload.get("error") or payload.get("output") or ""

        if not ok:
            self.logger.warning(
                "skill-creator direct passthrough reported failure: %s",
                str(error_msg)[:300],
            )
            return None

        summary_lines = [
            f"Skill **{name}** aktualisiert via **{backend}**-Backend.",
        ]
        if all_files:
            listing = ", ".join(f"`{f}`" for f in all_files[:10])
            summary_lines.append(f"Geänderte Dateien: {listing}.")
        if error_msg and len(error_msg) < 500:
            summary_lines.append(str(error_msg).strip())

        return "\n\n".join(summary_lines)

    async def _attempt(self, query: str) -> str:
        """Single attempt: workflow mode, then tool-call, then command mode."""
        # Prefer compiled workflow if available
        if self.skill.workflow:
            result = await self._workflow_mode(query)
            if result.strip():
                return result
            self.logger.info("Workflow mode produced no result, falling back")

        result = await self._tool_call_mode(query)
        if result.strip():
            return result

        if not self.command_fallback:
            return result

        self.logger.info("Falling back to command mode")
        return await self._command_mode(query)

    # ------------------------------------------------------------------
    # Mode 0: Workflow mode (compiled building blocks + dynamic planning)
    # ------------------------------------------------------------------

    async def _workflow_mode(self, query: str) -> str:
        """Execute using the compiled workflow with building blocks."""
        compiled = self.skill.workflow
        if not compiled:
            return ""

        executor = WorkflowExecutor(
            tool_registry=self.tool_registry,
            context=self.context,
            llm=self.llm,
            logger=self.logger,
            log_name=self.log_name,
        )
        executor.on_step = self.on_step

        workflow = await executor.select_workflow(compiled.workflows, query)
        if not workflow:
            return ""

        self.logger.info("Executing workflow '%s' for skill '%s'", workflow.id, self.skill.name)
        return await executor.execute(workflow, query)

    # ------------------------------------------------------------------
    # Mode 1: Tool-call mode (for models that support it)
    # ------------------------------------------------------------------

    async def _tool_call_mode(self, query: str) -> str:
        """Let the LLM use tools directly via bind_tools."""
        system = self._build_tool_prompt()
        messages: List[BaseMessage] = [
            SystemMessage(content=system),
            HumanMessage(content=f"Task: {query}"),
        ]

        try:
            response, messages = await self._invoke(messages, llm=self.bound_llm)
        except Exception as exc:
            self.logger.error("LLM error in tool-call mode: %s", exc)
            return ""

        self.logger.debug(
            "Tool-call mode turn 0: tool_calls=%s, content=%s",
            bool(response.tool_calls),
            repr((response.content or "")[:200]),
        )

        if response.tool_calls:
            return await self._tool_call_loop(messages, response)

        recovered_calls = self._recover_text_tool_calls(response)
        if recovered_calls:
            self.logger.warning(
                "Recovered %d text-form tool call(s) in skill '%s'",
                len(recovered_calls),
                self.skill.name,
            )
            response.tool_calls = recovered_calls
            return await self._tool_call_loop(messages, response)

        # Model answered directly without tools
        return self.extract_text(response)

    async def _tool_call_loop(self, messages: List[BaseMessage],
                              first_response: AIMessage) -> str:
        """Execute tool calls and continue the loop.

        When a tool returns an error the loop continues so the LLM can
        analyse what went wrong and try a different approach (e.g. use
        ``show`` to re-read the page, pick a different element ID, etc.).
        """
        response = first_response
        messages.append(response)

        has_error = False
        retryable_error = False
        for tc in response.tool_calls:
            result = await self._execute_tool_call(tc, messages)
            if not result.ok:
                has_error = True
                retryable_error = retryable_error or result.retryable

        if retryable_error:
            messages.append(HumanMessage(content=self._retry_guidance()))

        nudge_count = 0
        total_tool_calls = len(first_response.tool_calls)
        max_turns = self.skill.max_tool_turns or self.max_tool_turns
        consecutive_overflow = 0
        consecutive_silent = 0   # turns where tool calls were made with no visible content
        sig_counts: Dict[str, int] = {}
        for tc in first_response.tool_calls:
            sig = self._tool_call_signature(tc)
            sig_counts[sig] = sig_counts.get(sig, 0) + 1
        repeat_nudged = False
        silent_nudged = False
        abort_note = ""
        for _turn in range(1, max_turns):
            response, messages = await self._invoke(messages, llm=self.bound_llm)

            # Context-overflow circuit breaker: when the conversation has
            # bloated so far that the intended model can no longer be made to
            # fit (even after summarization) for several turns running, stop —
            # otherwise the loop grinds on degraded fallbacks and never reports.
            if getattr(self.bound_llm, "last_invoke_context_skipped", False):
                consecutive_overflow += 1
                if consecutive_overflow >= self._MAX_CONSECUTIVE_OVERFLOW:
                    abort_note = (
                        f"Kontext übergelaufen — nach {_turn} Schritten passt das "
                        "Gespräch in kein Modell mehr."
                    )
                    self.logger.warning("skill '%s': %s — Loop abgebrochen", self.skill.name, abort_note)
                    break
            else:
                consecutive_overflow = 0

            self.logger.debug(
                "Tool-call mode turn %d: tool_calls=%s, content=%s",
                _turn, bool(response.tool_calls),
                repr((response.content or "")[:200]),
            )
            if not response.tool_calls:
                text = self.extract_text(response)
                # Nudge the model to keep using tools when it stops too
                # early: no output, after an error, hallucinated code,
                # or if it never called a tool at all.
                # Allow up to 2 nudges before accepting the answer.
                should_nudge = (
                    not text.strip()
                    or has_error
                    or text.lstrip().startswith(("```", "<!"))
                    or total_tool_calls == 0
                )
                if should_nudge and nudge_count < 2:
                    nudge_count += 1
                    if self.on_step:
                        asyncio.ensure_future(self.on_step(f"↩ nudge {nudge_count}"))
                    messages.append(response)
                    messages.append(HumanMessage(
                        content=load_system_prompt("skills/runner_continue_nudge.md")
                    ))
                    self.logger.info("Nudging LLM to continue (turn %d, nudge %d)", _turn, nudge_count)
                    continue
                break
            messages.append(response)
            has_error = False
            retryable_error = False
            total_tool_calls += len(response.tool_calls)

            # Silent-tool-call circuit breaker: if the model issues tool calls
            # with no visible content for several turns in a row it is likely
            # stuck in an exploration / verification loop (all reasoning hidden
            # inside <think> blocks or simply absent).  Nudge once; abort if
            # it continues without producing output.
            visible = self.strip_thinking(response.content or "").strip()
            if response.tool_calls and not visible:
                consecutive_silent += 1
            else:
                consecutive_silent = 0
            if consecutive_silent >= self._ABORT_SILENT_TURNS:
                abort_note = (
                    f"Kein Fortschritt — {consecutive_silent} aufeinanderfolgende "
                    f"Schritte ohne sichtbaren Output bis Schritt {_turn}."
                )
                self.logger.warning("skill '%s': %s — Loop abgebrochen", self.skill.name, abort_note)
                break
            do_silent_nudge = consecutive_silent >= self._NUDGE_SILENT_TURNS and not silent_nudged

            # No-progress circuit breaker: detect the model repeating the exact
            # same tool call (e.g. re-reading one file). Nudge once, then abort.
            for tc in response.tool_calls:
                sig = self._tool_call_signature(tc)
                sig_counts[sig] = sig_counts.get(sig, 0) + 1
            worst = max(sig_counts.values()) if sig_counts else 0
            if worst >= self._ABORT_SAME_TOOL_CALL:
                abort_note = (
                    f"Kein Fortschritt — derselbe Tool-Aufruf wiederholte sich "
                    f"{worst}× bis Schritt {_turn}."
                )
                self.logger.warning("skill '%s': %s — Loop abgebrochen", self.skill.name, abort_note)
                break
            do_repeat_nudge = worst >= self._NUDGE_SAME_TOOL_CALL and not repeat_nudged

            # Execute tool calls BEFORE appending any nudge HumanMessage.
            # _sanitize_messages resets known_tool_ids on any non-AI/ToolMessage,
            # so a HumanMessage inserted between the AIMessage(tool_calls) and the
            # resulting ToolMessages would cause those ToolMessages to be dropped.
            for tc in response.tool_calls:
                result = await self._execute_tool_call(tc, messages)
                if not result.ok:
                    has_error = True
                    retryable_error = retryable_error or result.retryable

            # Append nudges after ToolMessages are already in the message list.
            if do_silent_nudge:
                silent_nudged = True
                self.logger.info(
                    "skill '%s': %d stille Schritte — einmaliger Nudge", self.skill.name, consecutive_silent
                )
                messages.append(HumanMessage(content=self._silent_guidance()))
            elif do_repeat_nudge:
                repeat_nudged = True
                self.logger.info(
                    "skill '%s': identischer Tool-Aufruf %d× — einmaliger Nudge", self.skill.name, worst
                )
                messages.append(HumanMessage(content=self._repeat_guidance()))
            # Broken-skill circuit breaker: the skill's own script keeps crashing
            # with the same traceback. Stop and offer a repair instead of letting
            # the model invent more workarounds.
            broken_note = self._broken_skill_note()
            if broken_note:
                abort_note = broken_note
                self.logger.warning(
                    "skill '%s': eigenes Skript %d× mit gleichem Fehler abgestürzt — "
                    "Loop abgebrochen, Reparatur angeboten",
                    self.skill.name, max(self._error_sig_counts.values(), default=0),
                )
                break
            if retryable_error:
                messages.append(HumanMessage(content=self._retry_guidance()))
        else:
            if response.tool_calls:
                response, messages = await self._invoke(messages, llm=self.llm)

        result_text = self.extract_text(response)
        if abort_note:
            # Always leave a traceable outcome so on_skill_done posts something
            # to the thread instead of the run vanishing silently.
            if self._broken_skill:
                # Already a complete, model-facing repair instruction — return it
                # verbatim so chat asks the user about a skill-creator repair.
                return abort_note
            prefix = f"⚠ {self.skill.name} gestoppt: {abort_note}"
            return f"{prefix}\n\n{result_text}".strip() if result_text.strip() else prefix
        return result_text

    async def _execute_tool_call(self, tc: dict, messages: List[BaseMessage]) -> ToolExecutionResult:
        """Execute a single tool call, append result to messages, and return it.

        The actual tool runs in a worker thread (``asyncio.to_thread``) because
        BashTool uses a blocking ``subprocess.run``; running it inline would
        freeze the whole event loop for the command's duration — stalling every
        other thread, live calls, and the //stop command itself. Offloading
        keeps the loop responsive and lets a cancel land promptly between turns.
        """
        tc_name = str(tc.get("name", "") or "").strip()
        tc_args = _repair_tool_args(tc.get("args", {}))
        tc_id = tc.get("id", "")

        if not tc_name:
            result = ToolExecutionResult(
                ok=False,
                tool_name="",
                normalized_args={},
                error="Invalid tool call: missing tool name.",
                error_code="tool_call_missing_name",
                retryable=True,
                hint="Call one of the available tools and include its exact name.",
            )
            messages.append(ToolMessage(content=result.to_tool_message(), tool_call_id=tc_id))
            return result

        self.logger.debug("Tool call: %s(%s)", tc_name, json.dumps(tc_args)[:200])
        if self.on_step:
            step = self._friendly_step(tc_name, tc_args)
            asyncio.ensure_future(self.on_step(step[:120]))
        result = await asyncio.to_thread(
            self.tool_registry.execute_detailed, tc_name, tc_args, self.context
        )
        result_str = result.to_tool_message()
        self.logger.debug("Tool result: %s", result_str[:200])

        # Extract __directive__ lines before passing to LLM
        clean_lines = []
        for line in result_str.splitlines():
            if '"__directive__"' in line:
                self._directives.append(line)
            else:
                clean_lines.append(line)
        result_str = "\n".join(clean_lines)

        # Record crashes of the skill's own script so the loop can give up early
        # and offer a repair instead of flailing through manual workarounds.
        self._record_error_signature(result_str, tc_args)

        # Keep tool results bounded for the model context window. The cap lives
        # on the class (MAX_RESULT) so it stays in sync with the files skill's
        # read-page byte budget — see the constant's comment.
        MAX_RESULT = self.MAX_RESULT
        if len(result_str) > MAX_RESULT:
            omitted = len(result_str) - MAX_RESULT
            result_str = result_str[:MAX_RESULT].rstrip()
            result_str += f"\n\n[Tool output truncated: {omitted} characters omitted]"

        messages.append(ToolMessage(content=result_str, tool_call_id=tc_id))
        return result

    _TRACEBACK_FILE_RE = re.compile(r'File "([^"]+)", line (\d+)')

    def _record_error_signature(self, result_str: str, tc_args: Dict[str, Any]) -> None:
        """Count repeated crashes of the skill's *own* script.

        Targets Python tracebacks whose frames point inside this skill's
        directory — i.e. the skill's script itself is broken, not a transient
        external error. A signature is ``script.py:line:ExceptionType`` so the
        same crash counts even when the model varies the surrounding command.
        """
        if "Traceback (most recent call last)" not in result_str:
            return
        frames = self._TRACEBACK_FILE_RE.findall(result_str)
        if not frames:
            return
        skill_dir = os.path.abspath(self.skill.skill_path or "")
        scripts_dir = os.path.abspath(self.skill.scripts_dir or "") if self.skill.scripts_dir else ""
        in_skill = [
            (path, line) for path, line in frames
            if (skill_dir and skill_dir in os.path.abspath(path))
            or (scripts_dir and scripts_dir in os.path.abspath(path))
        ]
        if not in_skill:
            return
        path, line = in_skill[-1]  # deepest frame inside the skill
        last_line = ""
        for ln in reversed(result_str.strip().splitlines()):
            if ln.strip():
                last_line = ln.strip()
                break
        exc_type = last_line.split(":", 1)[0].strip()[:60]
        sig = f"{os.path.basename(path)}:{line}:{exc_type}"
        self._error_sig_counts[sig] = self._error_sig_counts.get(sig, 0) + 1
        # Remember a short, human-readable excerpt + the failing command for the
        # repair offer (latest occurrence wins).
        self._broken_error_excerpt = f"{os.path.basename(path)}, Zeile {line} — {last_line[:160]}"
        cmd = tc_args.get("command") if isinstance(tc_args, dict) else None
        if isinstance(cmd, str) and cmd.strip():
            self._broken_failing_command = cmd.strip()[:200]

    def _broken_skill_note(self) -> Optional[str]:
        """Return a repair-offer note if the skill's own script keeps crashing.

        Returns ``None`` when the threshold isn't reached, or for skill-creator
        itself (it is the repair tool — offering to repair it with itself would
        loop).
        """
        if self.skill.name == "skill-creator":
            return None
        worst = max(self._error_sig_counts.values(), default=0)
        if worst < self._ABORT_REPEATED_ERROR:
            return None
        self._broken_skill = True
        parts = [
            f"Die Skill »{self.skill.name}« scheint defekt — derselbe Fehler trat "
            f"{worst}× auf:",
            self._broken_error_excerpt or "wiederholter Skript-Fehler",
        ]
        if self._broken_failing_command:
            parts.append(f"Fehlgeschlagener Befehl: {self._broken_failing_command}")
        parts.append(
            "Versuche NICHT, den Fehler manuell zu umgehen. Frag den Nutzer, ob du "
            f"die Skill »{self.skill.name}« vom skill-creator reparieren lassen sollst. "
            "Wenn der Nutzer zustimmt, ruf die Skill »skill-creator« mit Skillname und "
            "der Fehlermeldung als Reparatur-Auftrag auf."
        )
        return "\n".join(parts)

    @staticmethod
    def _retry_guidance() -> str:
        return load_system_prompt("skills/runner_retry_guidance.md")

    @staticmethod
    def _repeat_guidance() -> str:
        return (
            "Du hast denselben Tool-Aufruf mehrfach mit identischen Argumenten "
            "ausgeführt, ohne Fortschritt. Wiederhole ihn nicht erneut. Arbeite "
            "mit dem Ergebnis, das du bereits hast, oder gib jetzt dein "
            "abschließendes Ergebnis aus."
        )

    @staticmethod
    def _silent_guidance() -> str:
        return (
            "Du führst seit mehreren Schritten Tool-Aufrufe aus, ohne ein "
            "sichtbares Zwischenergebnis oder eine Einschätzung zu formulieren. "
            "Wenn du bereits genug Informationen gesammelt hast, gib jetzt dein "
            "abschließendes Ergebnis aus. Wenn noch etwas fehlt, erkläre kurz "
            "was und führe dann den nächsten gezielten Schritt aus."
        )

    @staticmethod
    def _tool_call_signature(tc: Dict[str, Any]) -> str:
        """Stable signature of a tool call (name + normalized args) for
        detecting a model stuck repeating the exact same action."""
        name = str(tc.get("name", "") or "").strip()
        try:
            args = json.dumps(_repair_tool_args(tc.get("args", {})), sort_keys=True)
        except Exception:
            args = str(tc.get("args", {}))
        return f"{name}|{args}"

    # ------------------------------------------------------------------
    # Mode 2: Command mode (for small models that can't do tool calls)
    # ------------------------------------------------------------------

    async def _command_mode(self, query: str) -> str:
        """Ask LLM to output a shell command, execute it, return result."""
        system = self._build_command_prompt()
        messages: List[BaseMessage] = [
            SystemMessage(content=system),
            HumanMessage(content=f"Task: {query}"),
        ]

        response, _ = await self._invoke(messages, llm=self.llm)
        content = response.content or ""
        self.logger.debug("Command mode response: %s", repr(content[:200]))

        command = self._extract_command(content)
        if not command:
            self.logger.warning("Could not extract command from LLM response")
            return self.extract_text(response) or "Error: could not determine command."

        self.logger.debug("Executing: %s", command[:200])
        result = await asyncio.to_thread(
            self.tool_registry.execute, "bash", {"command": command}, self.context
        )
        result_str = str(result)
        self.logger.debug("Result: %s", result_str[:300])

        return result_str

    @staticmethod
    def _extract_command(text: str) -> str:
        """Extract a shell command from LLM text output."""
        m = _RE_CODE_BLOCK.search(text)
        if m:
            for line in m.group(1).strip().splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    return line

        for line in text.splitlines():
            line = line.strip()
            if line.startswith(("node ", "python", "bash ", "sh ", "curl ", "./")):
                return line

        return ""

    def _recover_text_tool_calls(self, response: AIMessage) -> List[Dict[str, Any]]:
        """Recover tool calls that were emitted as plain text.

        GLM/Qwen-style local models sometimes write a bash command or
        ``<tool_call>{...}</tool_call>`` even though tools are bound.  Turning
        that into a real ToolMessage keeps the same recovery loop alive without
        adding another agent layer.
        """
        if response.tool_calls:
            return []
        content = response.content if isinstance(response.content, str) else ""
        valid_names = set(self.tool_registry.names())
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
            resolved = self.tool_registry._resolve(name)
            if resolved not in valid_names:
                return
            args = obj.get("args")
            if args is None:
                args = obj.get("arguments")
            if args is None:
                args = obj.get("parameters")
            if args is None:
                args = {
                    key: value
                    for key, value in obj.items()
                    if key not in {"id", "name", "tool", "function"}
                }
            calls.append({
                "id": obj.get("id") or f"fake_{uuid.uuid4().hex[:8]}",
                "name": resolved,
                "args": args,
            })

        snippets: List[str] = [m.group(1).strip() for m in _RE_TOOL_CALL_TAG.finditer(content)]
        snippets.extend(m.group(1).strip() for m in _RE_ANY_CODE_BLOCK.finditer(content))

        stripped = content.strip()
        if stripped.startswith(("{", "[")):
            snippets.append(stripped)

        for snippet in snippets:
            if not snippet:
                continue
            try:
                _append_from_obj(json.loads(snippet))
            except (TypeError, ValueError, json.JSONDecodeError):
                pass

        if calls:
            return calls

        command = self._extract_command(content)
        if command and "bash" in valid_names:
            return [{
                "id": f"fake_{uuid.uuid4().hex[:8]}",
                "name": "bash",
                "args": {"command": command},
            }]

        return []

    # ------------------------------------------------------------------
    # Step display
    # ------------------------------------------------------------------

    def _friendly_step(self, tc_name: str, tc_args: dict) -> str:
        """Return a short, user-friendly description of a tool call."""
        if tc_name != "bash":
            return tc_name

        cmd = tc_args.get("command", "")
        # Extract the script basename (e.g. "memory.py", "researcher.py")
        parts = cmd.split()
        script = ""
        for p in parts:
            base = os.path.basename(p)
            if base.endswith((".py", ".mjs", ".js", ".sh")):
                script = base.removesuffix(".py").removesuffix(".mjs").removesuffix(".js").removesuffix(".sh")
                break

        # Extract the sub-command (e.g. "search", "index", "status")
        action = ""
        if script:
            # Sub-command is typically the argument after the script path
            found_script = False
            for p in parts:
                if found_script:
                    if not p.startswith("-") and not p.startswith("/") and ":" not in p:
                        action = p
                        break
                if os.path.basename(p).startswith(script):
                    found_script = True

        if script and action:
            return f"{script} → {action}"
        if script:
            return script
        # Fallback: show just the command name
        return os.path.basename(parts[0]) if parts else tc_name

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def _build_tool_prompt(self) -> str:
        """System prompt for tool-call mode."""
        parts = [load_system_prompt("skills/runner_tool.md", skill_name=self.skill.name)]
        parts.append(load_system_prompt("skills/credentials.md"))
        self._append_skill_context(parts)
        return "\n".join(parts)

    def _build_command_prompt(self) -> str:
        """System prompt for command mode (text-only, no tools)."""
        parts = [load_system_prompt("skills/runner_command.md", skill_name=self.skill.name)]
        parts.append(load_system_prompt("skills/credentials.md"))
        self._append_skill_context(parts)
        return "\n".join(parts)

    def _append_skill_context(self, parts: List[str]) -> None:
        """Append working directory, scripts, config, and instructions."""
        cwd = self.context.get("cwd") or os.path.abspath(self.skill.skill_path)
        parts.append(
            f"\nWorking directory: {cwd}"
            "\nUse relative paths (e.g. scripts/route.py, scripts/bahn.mjs)."
        )

        if self.skill.scripts_dir and os.path.isdir(self.skill.scripts_dir):
            try:
                scripts = ", ".join(os.listdir(self.skill.scripts_dir))
            except OSError:
                scripts = "(could not list)"
            parts.append(f"Available scripts: {scripts}")

        skill_cfg = self.context.get("skill_config", {})
        if skill_cfg:
            parts.append(f"\nConfiguration: {json.dumps(skill_cfg)}")

        instructions = self.skill.instructions
        instructions = instructions.replace("<user_id>", self.context.get("user_id", ""))
        instructions = instructions.replace("<session_dir>", self.context.get("session_dir", ""))
        if self.skill.scripts_dir:
            instructions = instructions.replace("<scripts_dir>", os.path.abspath(self.skill.scripts_dir))
        parts.append(f"\n{instructions}")
