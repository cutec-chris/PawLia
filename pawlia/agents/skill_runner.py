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
    the entire run. No trimming, no summarization, no truncation of tool
    outputs between turns. Compaction is allowed ONLY as a guardrail when
    the real LLM context window is about to overflow — and even then via
    summary, never by dropping messages. Do not introduce prune/shrink
    heuristics in this loop.
    """

    MAX_TOOL_TURNS = 30
    MAX_RETRIES = 2

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
        self.context["cwd"] = skill.base_dir
        self.context["skills_root"] = os.path.dirname(skill.base_dir)
        self.command_fallback = command_fallback
        self.max_tool_turns = max_tool_turns if (isinstance(max_tool_turns, int) and max_tool_turns > 0) else self.MAX_TOOL_TURNS

        # Load matching credentials into context
        self._load_credentials()
        self.on_step = None  # Optional[Callable[[str], Awaitable[None]]]
        self._directives: List[str] = []  # collected __directive__ lines from tool output

        # Bind real tools to the LLM
        tool_specs = tool_registry.get_specs()
        if tool_specs:
            self.bound_llm = llm.bind_tools(tool_specs, tool_choice="auto")
        else:
            self.bound_llm = llm

    def _load_credentials(self) -> None:
        """Load matching credentials into context as CRED_* env vars."""
        if not self.skill.requires_credentials:
            return
        session_dir = self.context.get("session_dir", "")
        user_id = self.context.get("user_id", "")
        if not session_dir or not user_id:
            return
        cred_path = os.path.join(session_dir, user_id, ".credentials.json")
        if not os.path.isfile(cred_path):
            return
        try:
            with open(cred_path, encoding="utf-8") as f:
                all_creds = json.load(f)
        except (json.JSONDecodeError, OSError):
            return
        cred_env = {}
        for key in self.skill.requires_credentials:
            if key in all_creds:
                env_key = "CRED_" + re.sub(r"[^A-Za-z0-9]", "_", key).upper()
                cred_env[env_key] = str(all_creds[key])
        if cred_env:
            self.context.setdefault("env_extra", {}).update(cred_env)

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
            response = await self._invoke(messages, llm=self.bound_llm)
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
            result = self._execute_tool_call(tc, messages)
            if not result.ok:
                has_error = True
                retryable_error = retryable_error or result.retryable

        if retryable_error:
            messages.append(HumanMessage(content=self._retry_guidance()))

        nudge_count = 0
        total_tool_calls = len(first_response.tool_calls)
        max_turns = self.skill.max_tool_turns or self.max_tool_turns
        for _turn in range(1, max_turns):
            response = await self._invoke(messages, llm=self.bound_llm)
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
            for tc in response.tool_calls:
                result = self._execute_tool_call(tc, messages)
                if not result.ok:
                    has_error = True
                    retryable_error = retryable_error or result.retryable
            if retryable_error:
                messages.append(HumanMessage(content=self._retry_guidance()))
        else:
            if response.tool_calls:
                response = await self._invoke(messages, llm=self.llm)

        return self.extract_text(response)

    def _execute_tool_call(self, tc: dict, messages: List[BaseMessage]) -> ToolExecutionResult:
        """Execute a single tool call, append result to messages, and return it."""
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
        result = self.tool_registry.execute_detailed(tc_name, tc_args, self.context)
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

        messages.append(ToolMessage(content=result_str, tool_call_id=tc_id))
        return result

    @staticmethod
    def _retry_guidance() -> str:
        return load_system_prompt("skills/runner_retry_guidance.md")

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

        response = await self._invoke(messages, llm=self.llm)
        content = response.content or ""
        self.logger.debug("Command mode response: %s", repr(content[:200]))

        command = self._extract_command(content)
        if not command:
            self.logger.warning("Could not extract command from LLM response")
            return self.extract_text(response) or "Error: could not determine command."

        self.logger.debug("Executing: %s", command[:200])
        result = self.tool_registry.execute("bash", {"command": command}, self.context)
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
        self._append_skill_context(parts)
        return "\n".join(parts)

    def _build_command_prompt(self) -> str:
        """System prompt for command mode (text-only, no tools)."""
        parts = [load_system_prompt("skills/runner_command.md", skill_name=self.skill.name)]
        self._append_skill_context(parts)
        return "\n".join(parts)

    def _append_skill_context(self, parts: List[str]) -> None:
        """Append working directory, scripts, config, and instructions."""
        parts.append(
            f"\nWorking directory: {os.path.abspath(self.skill.skill_path)}"
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
