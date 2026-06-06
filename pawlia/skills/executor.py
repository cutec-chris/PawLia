"""Workflow executor — runs compiled workflows via native tool calls.

Building blocks from workflow.yaml become tool definitions. The LLM calls
them directly via tool_calls — no JSON planning, no free-form parsing.
The loop continues until the LLM responds with text (= done) or max_steps
is reached.
"""

import asyncio
import json
import logging
import os
import re
import shlex
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

logger = logging.getLogger(__name__)


def _is_tool_choice_error(exc: Exception) -> bool:
    """Check if an exception is a tool-choice error from the API."""
    error_str = str(exc)
    return ("tool_use_failed" in error_str or
            ("Tool choice is none" in error_str and "called a tool" in error_str))


def _extract_tool_name(error_str: str) -> str:
    """Extract the tool name from the failed_generation error."""
    name_match = re.search(r'"name"\s*:\s*"([^"]+)"', error_str)
    return name_match.group(1) if name_match else ""

from pawlia.agents.base import log_prompt
from pawlia.prompt_utils import load_system_prompt
from pawlia.skills.workflow_schema import (
    BuildingBlock,
    CompiledWorkflow,
    VerifySpec,
    Workflow,
)
from pawlia.tools.base import ToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class StepResult:
    output: str
    exit_code: int


class WorkflowExecutor:
    """Executes a compiled workflow using tool calls."""

    def __init__(
        self,
        tool_registry: ToolRegistry,
        context: Dict[str, Any],
        llm: Any,
        logger: Optional[logging.Logger] = None,
        log_name: str = "prompt",
    ):
        self.tool_registry = tool_registry
        self.context = context
        self.llm = llm
        self.logger = logger or logging.getLogger(__name__)
        self.log_name = log_name
        self.on_step: Any = None  # Optional async callback

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def select_workflow(
        self, workflows: List[Workflow], query: str
    ) -> Optional[Workflow]:
        """Let the LLM pick the best workflow by calling it as a tool."""
        if len(workflows) == 1:
            return workflows[0]

        # Each workflow becomes a callable tool
        tools = [
            {
                "type": "function",
                "function": {
                    "name": w.id,
                    "description": w.trigger,
                    "parameters": {
                        "type": "object",
                        "properties": {},
                    },
                },
            }
            for w in workflows
        ]

        bound = self.llm.bind_tools(tools, tool_choice="required")
        messages = [
            SystemMessage(content=load_system_prompt("workflow/select.md")),
            HumanMessage(content=query),
        ]
        log_prompt(messages, name=self.log_name)

        for attempt in range(3):
            try:
                response = await bound.ainvoke(messages)
            except Exception as exc:
                if _is_tool_choice_error(exc) and attempt < 2:
                    self.logger.info(
                        "Model output a tool call as text in workflow select "
                        "(attempt %d/3), retrying", attempt + 1)
                    messages = messages + [
                        HumanMessage(
                            content="Do NOT output tool calls as text or JSON. "
                            "Select a workflow by using the proper tool call mechanism. "
                            "Call exactly one of the available workflow tools."
                        ),
                    ]
                    continue
                self.logger.error("LLM error selecting workflow: %s", exc)
                return None

            if response.tool_calls:
                chosen_id = response.tool_calls[0]["name"]
                for w in workflows:
                    if w.id == chosen_id:
                        return w

            self.logger.warning("Workflow selection failed on attempt %d", attempt + 1)
            messages.append(response)
            messages.append(HumanMessage(
                content=load_system_prompt("workflow/select_retry.md")
            ))

        return None

    async def execute(self, workflow: Workflow, query: str) -> str:
        """Execute a workflow via tool-call loop."""
        tools = self._blocks_to_tools(workflow)
        bound_llm = self.llm.bind_tools(tools)

        # Build system prompt with config context
        now = datetime.now()
        skill_config = self.context.get("skill_config")
        skill_config_block = ""
        if skill_config:
            skill_config_block = (
                f"Config values: {json.dumps(skill_config, ensure_ascii=False)}"
            )
        system = load_system_prompt(
            "workflow/execute.md",
            current_date=now.strftime("%Y-%m-%d"),
            current_day=now.strftime("%A"),
            current_time=now.strftime("%H:%M"),
            skill_config_block=skill_config_block,
        )

        messages: List[Any] = [
            SystemMessage(content=system),
            HumanMessage(content=query),
        ]
        log_prompt(messages, name=self.log_name)

        outputs: List[str] = []
        directives: List[str] = []

        for step in range(workflow.max_steps):
            response = None
            step_retries = 0
            max_step_retries = 3
            while True:
                try:
                    response = await bound_llm.ainvoke(messages)
                    break
                except Exception as exc:
                    if _is_tool_choice_error(exc) and step_retries < max_step_retries:
                        step_retries += 1
                        self.logger.info(
                            "Model output a tool call as text in workflow step %d "
                            "(retry %d/%d)", step, step_retries, max_step_retries)
                        messages = messages + [
                            HumanMessage(
                                content="Do NOT output tool calls as text or JSON. "
                                "Use the proper tool_calls mechanism, or respond with "
                                "plain text when the workflow is complete."
                            ),
                        ]
                        continue
                    step_retries += 1
                    self.logger.error("LLM error in workflow step %d (attempt %d): %s",
                                      step, step_retries, exc)
                    break

            if response is None:
                break

            # No tool calls → LLM is done, return its text
            if not response.tool_calls:
                text = (response.content or "").strip()
                if text:
                    if directives:
                        return "\n".join(directives) + "\n" + text
                    return text
                break

            messages.append(response)

            for tc in response.tool_calls:
                block_id = tc["name"]
                params = tc.get("args", {})
                block = self._find_block(workflow, block_id)

                if not block:
                    self.logger.warning("Unknown block '%s' — skipping", block_id)
                    messages.append(ToolMessage(
                        content=f"Error: unknown tool '{block_id}'",
                        tool_call_id=tc["id"],
                    ))
                    continue

                # Status callback
                if self.on_step:
                    status = (
                        self._substitute(block.status_desc, params)
                        if block.status_desc
                        else block.description
                    )
                    asyncio.ensure_future(self.on_step(status))

                # Execute command — env_params are passed as env vars,
                # not substituted into the command string (avoids
                # shell escaping issues with multiline content).
                env_extra = {}
                cmd_params = params
                if block.env_params:
                    env_extra = {
                        p.upper(): str(params[p])
                        for p in block.env_params
                        if p in params
                    }
                    cmd_params = {
                        k: v for k, v in params.items()
                        if k not in block.env_params
                    }
                command = self._substitute(block.command, cmd_params)
                result = await self._run_bash(command, env_extra=env_extra)
                outputs.append(result.output)
                for line in result.output.splitlines():
                    if '"__directive__"' in line:
                        directives.append(line)

                # Programmatic verification
                if block.verify and not self._verify(
                    result.output, result.exit_code, block.verify
                ):
                    self.logger.info(
                        "Block '%s' failed verification (exit=%d)",
                        block_id, result.exit_code,
                    )
                    error_content = f"ERROR: {result.output}"
                    if block.on_error:
                        error_block = self._find_block(workflow, block.on_error)
                        if error_block:
                            recovery = await self._run_bash(
                                self._substitute(error_block.command, {})
                            )
                            error_content += f"\n\nRecovery:\n{recovery.output}"
                    messages.append(ToolMessage(
                        content=error_content,
                        tool_call_id=tc["id"],
                    ))
                else:
                    messages.append(ToolMessage(
                        content=result.output,
                        tool_call_id=tc["id"],
                    ))

        last = outputs[-1] if outputs else ""
        if directives:
            return "\n".join(directives) + "\n" + last
        return last

    # ------------------------------------------------------------------
    # Tool generation
    # ------------------------------------------------------------------

    def _blocks_to_tools(self, workflow: Workflow) -> List[Dict[str, Any]]:
        """Convert building blocks to OpenAI tool specs.

        Schema is strict on the *required* side (provider rejects missing
        required params) but lenient on the *extras* side. The LLM often
        adds hallucinated arguments (e.g. ``path``/``depth`` for a tool
        that takes none) and a strict ``additionalProperties: false``
        turns that into a 400 that knocks the whole request over to a
        fallback model. The executor silently ignores keys not present
        in the command template, so accepting the extra args is safe.
        """
        tools = []
        config_params = self._skill_config_params()
        for block in workflow.building_blocks:
            # Extract {param} placeholders, excluding context vars,
            # config vars, plus any env_params (passed as env vars,
            # not in command).
            param_names = list(dict.fromkeys(
                [p for p in re.findall(r"\{(\w+)\}", block.command)
                 if p not in ("scripts_dir",) and p not in config_params]
                + block.env_params
            ))

            properties = {p: {"type": "string"} for p in param_names}

            tools.append({
                "type": "function",
                "function": {
                    "name": block.id,
                    "description": block.description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": param_names,
                    },
                },
            })
        return tools

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_block(
        self, workflow: Workflow, block_id: str
    ) -> Optional[BuildingBlock]:
        for b in workflow.building_blocks:
            if b.id == block_id:
                return b
        return None

    def _substitute(self, template: str, params: Dict[str, str]) -> str:
        """Replace {param} and <param> placeholders in a command template."""
        result = template

        # Resolve scripts_dir from context
        scripts_dir = self.context.get("cwd", "")
        if scripts_dir:
            scripts_dir = os.path.join(os.path.abspath(scripts_dir), "scripts")
        result = result.replace("{scripts_dir}", scripts_dir)
        result = result.replace("<scripts_dir>", scripts_dir)

        # Resolve additional context paths (e.g. skills_root)
        for key in ("skills_root",):
            value = self.context.get(key, "")
            if value:
                result = result.replace(f"{{{key}}}", value)
                result = result.replace(f"<{key}>", value)

        # Skill config values are system-provided, not model-provided. This
        # lets workflow commands use {url}, {timeout}, etc. without exposing
        # those as LLM parameters when they exist in skill-config.<skill>.
        for key, value in self._skill_config_params().items():
            result = result.replace(f"{{{key}}}", value)
            result = result.replace(f"<{key}>", value)

        # Replace both {key} and <key> for all params.
        # If the template wraps the placeholder in quotes (e.g. "{task}"),
        # strip those quotes and apply shlex.quote so that spaces and
        # shell metacharacters in LLM-generated arguments don't break the
        # command. This is backwards-compatible with existing workflow.yaml
        # files that already quote placeholders.
        for key, value in params.items():
            quoted = shlex.quote(str(value))
            for quoted_pat in [
                f'"{{{key}}}"', f"'{{{key}}}'",
                f'"<{key}>"', f"'<{key}>'",
            ]:
                result = result.replace(quoted_pat, quoted)
            result = result.replace(f"{{{key}}}", quoted)
            result = result.replace(f"<{key}>", quoted)

        return result

    def _skill_config_params(self) -> Dict[str, str]:
        """Return scalar skill-config values safe for command templating."""
        skill_config = self.context.get("skill_config") or {}
        if not isinstance(skill_config, dict):
            return {}
        result: Dict[str, str] = {}
        for key, value in skill_config.items():
            if isinstance(value, (str, int, float, bool)):
                result[str(key)] = str(value)
        return result

    async def _run_bash(
        self, command: str, env_extra: Optional[Dict[str, str]] = None
    ) -> StepResult:
        """Execute a bash command via the tool registry.

        Runs in a worker thread (``asyncio.to_thread``): BashTool blocks on
        ``subprocess.run``, so running it inline would freeze the event loop —
        and every other thread/call sharing it — for the command's duration.
        """
        self.logger.debug("Executing: %s", command[:200])
        ctx = self.context
        if env_extra:
            ctx = {**self.context, "env_extra": env_extra}
        raw = await asyncio.to_thread(
            self.tool_registry.execute_detailed, "bash", {"command": command}, ctx
        )
        output = raw.to_tool_message() if not raw.ok else str(raw.output)
        exit_code = 0 if raw.ok else 1
        self.logger.debug("Result (exit=%d): %s", exit_code, output)
        return StepResult(output=output, exit_code=exit_code)

    def _verify(self, output: str, exit_code: int, spec: VerifySpec) -> bool:
        """Programmatic verification — no LLM needed."""
        if exit_code != spec.exit_code:
            return False
        for s in spec.output_contains:
            if s not in output:
                return False
        for s in spec.output_not_contains:
            if s in output:
                return False
        if spec.output_regex and not re.search(spec.output_regex, output):
            return False
        return True
