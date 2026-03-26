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
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from pawlia.agents.base import log_prompt
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
    ):
        self.tool_registry = tool_registry
        self.context = context
        self.llm = llm
        self.logger = logger or logging.getLogger(__name__)
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
                    "parameters": {"type": "object", "properties": {}},
                },
            }
            for w in workflows
        ]

        bound = self.llm.bind_tools(tools)
        messages = [
            SystemMessage(content="Pick the workflow that matches the user's request by calling it."),
            HumanMessage(content=query),
        ]
        log_prompt(messages)

        try:
            response = await bound.ainvoke(messages)
            if response.tool_calls:
                chosen_id = response.tool_calls[0]["name"]
                for w in workflows:
                    if w.id == chosen_id:
                        return w
        except Exception as exc:
            self.logger.error("LLM error selecting workflow: %s", exc)

        return workflows[0]

    async def execute(self, workflow: Workflow, query: str) -> str:
        """Execute a workflow via tool-call loop."""
        tools = self._blocks_to_tools(workflow)
        bound_llm = self.llm.bind_tools(tools)

        # Build system prompt with config context
        system = "Use the available tools to fulfill the user's request.\n"
        skill_config = self.context.get("skill_config")
        if skill_config:
            system += f"Config values: {json.dumps(skill_config, ensure_ascii=False)}\n"
        system += "After getting results, respond with the answer."

        messages: List[Any] = [
            SystemMessage(content=system),
            HumanMessage(content=query),
        ]
        log_prompt(messages)

        outputs: List[str] = []

        for step in range(workflow.max_steps):
            try:
                response = await bound_llm.ainvoke(messages)
            except Exception as exc:
                self.logger.error("LLM error in workflow step %d: %s", step, exc)
                break

            # No tool calls → LLM is done, return its text
            if not response.tool_calls:
                text = (response.content or "").strip()
                if text:
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

                # Execute command
                command = self._substitute(block.command, params)
                result = self._run_bash(command)
                outputs.append(result.output)

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
                            recovery = self._run_bash(
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

        return outputs[-1] if outputs else ""

    # ------------------------------------------------------------------
    # Tool generation
    # ------------------------------------------------------------------

    def _blocks_to_tools(self, workflow: Workflow) -> List[Dict[str, Any]]:
        """Convert building blocks to OpenAI tool specs."""
        tools = []
        for block in workflow.building_blocks:
            # Extract {param} placeholders, excluding context vars
            param_names = list(dict.fromkeys(
                p for p in re.findall(r"\{(\w+)\}", block.command)
                if p not in ("scripts_dir",)
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
        """Replace {param} placeholders in a command template."""
        result = template
        if "{scripts_dir}" in result:
            scripts_dir = self.context.get("cwd", "")
            if scripts_dir:
                scripts_dir = os.path.join(os.path.abspath(scripts_dir), "scripts")
            result = result.replace("{scripts_dir}", scripts_dir)

        for key, value in params.items():
            result = result.replace(f"{{{key}}}", str(value))

        return result

    def _run_bash(self, command: str) -> StepResult:
        """Execute a bash command via the tool registry.

        Injects PAWLIA_USER_ID and PAWLIA_SESSION_DIR as environment
        variables so scripts can read them without needing CLI args.
        """
        command = self._inject_env(command)
        self.logger.debug("Executing: %s", command[:200])
        raw = self.tool_registry.execute("bash", {"command": command}, self.context)
        output = str(raw)
        exit_code = 1 if output.startswith("Error") else 0
        self.logger.debug("Result (exit=%d): %s", exit_code, output[:300])
        return StepResult(output=output, exit_code=exit_code)

    def _inject_env(self, command: str) -> str:
        """Prepend PAWLIA_USER_ID / PAWLIA_SESSION_DIR env vars."""
        env_parts = []
        user_id = self.context.get("user_id")
        session_dir = self.context.get("session_dir")
        if user_id:
            env_parts.append(f'PAWLIA_USER_ID="{user_id}"')
        if session_dir:
            env_parts.append(f'PAWLIA_SESSION_DIR="{session_dir}"')
        if not env_parts:
            return command
        return " ".join(env_parts) + " " + command

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
