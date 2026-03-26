"""Workflow executor — runs compiled workflows with dynamic planning.

The executor uses the LLM minimally:
1. One call to select a workflow and create an initial task plan
2. After each step: LLM decides next action (or adapts the plan on failure)
3. Goal check at the end (if configured)

All verification between steps is programmatic (exit codes, output patterns).
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from json_repair import repair_json

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
    passed: bool


@dataclass
class TaskPlan:
    goal: str
    tasks: List[Dict[str, Any]] = field(default_factory=list)

    def pending_tasks(self) -> List[Dict[str, Any]]:
        return [t for t in self.tasks if t.get("status") != "done"]

    def to_json(self) -> str:
        return json.dumps({"goal": self.goal, "tasks": self.tasks}, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict) -> "TaskPlan":
        return cls(goal=data.get("goal", ""), tasks=data.get("tasks", []))


class WorkflowExecutor:
    """Executes a compiled workflow using building blocks and LLM planning."""

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
        """Let the LLM pick the best-matching workflow for a query."""
        if len(workflows) == 1:
            return workflows[0]

        trigger_list = "\n".join(
            f"- {w.id}: {w.trigger}" for w in workflows
        )
        prompt = (
            f"Available workflows:\n{trigger_list}\n\n"
            f"User query: {query}\n\n"
            "Which workflow id best matches? Reply with ONLY the id, nothing else."
        )

        from langchain_core.messages import HumanMessage, SystemMessage
        messages = [
            SystemMessage(content="You select the best workflow for a user query. Reply with only the workflow id."),
            HumanMessage(content=prompt),
        ]

        try:
            response = await self.llm.ainvoke(messages)
            chosen_id = (response.content or "").strip().strip('"').strip("'")
        except Exception as exc:
            self.logger.error("LLM error selecting workflow: %s", exc)
            return workflows[0]

        for w in workflows:
            if w.id == chosen_id:
                return w

        self.logger.warning("LLM chose unknown workflow '%s', using first", chosen_id)
        return workflows[0]

    async def execute(self, workflow: Workflow, query: str) -> str:
        """Execute a workflow: create plan, run steps, check goal."""
        # 1. Create initial plan
        plan = await self._create_plan(workflow, query)
        if not plan:
            return ""

        outputs: List[str] = []
        steps_taken = 0

        # 2. Plan loop
        result = await self._run_plan_loop(workflow, plan, outputs, steps_taken)
        steps_taken = result["steps_taken"]
        outputs = result["outputs"]
        plan = result["plan"]

        # 3. Goal check
        if workflow.goal_check:
            for attempt in range(workflow.goal_check.max_retries + 1):
                goal_result = await self._check_goal(workflow, plan, outputs, query)
                if goal_result.get("reached"):
                    break
                self.logger.info(
                    "Goal not reached (attempt %d): %s",
                    attempt + 1, goal_result.get("reason", ""),
                )
                # Extend plan and run more steps
                plan = await self._extend_plan(
                    workflow, plan, goal_result, outputs, query
                )
                result = await self._run_plan_loop(
                    workflow, plan, outputs, steps_taken
                )
                steps_taken = result["steps_taken"]
                outputs = result["outputs"]
                plan = result["plan"]

        return outputs[-1] if outputs else ""

    # ------------------------------------------------------------------
    # Plan creation
    # ------------------------------------------------------------------

    async def _create_plan(
        self, workflow: Workflow, query: str
    ) -> Optional[TaskPlan]:
        """Create an initial task plan — LLM-based or direct for simple cases."""
        blocks_desc = "\n".join(
            f"- {b.id}: {b.description}  (command template: {b.command})"
            for b in workflow.building_blocks
        )

        prompt = (
            f"User query: {query}\n\n"
            f"Available blocks:\n{blocks_desc}\n\n"
            "Reply with ONLY a JSON object like this example:\n"
            '{"goal": "find pizza recipes", "tasks": [\n'
            '  {"block": "run_search", "params": {"query": "pizza recipes"}}\n'
            "]}\n\n"
            "Rules:\n"
            "- Use ONLY blocks listed above.\n"
            "- Params must match the {placeholders} in the command template.\n"
            "- ONLY valid JSON, no explanation."
        )

        from langchain_core.messages import HumanMessage, SystemMessage
        messages = [
            SystemMessage(content=(
                "You plan task execution using predefined building blocks. "
                "Reply with only JSON, no markdown fences, no explanation."
            )),
            HumanMessage(content=prompt),
        ]

        try:
            response = await self.llm.ainvoke(messages)
            raw = (response.content or "").strip()
            self.logger.debug("LLM plan response (first 500): %s", repr(raw[:500]))
            if not raw or (raw.startswith("<think") and "</think" not in raw):
                self.logger.error(
                    "Empty or truncated LLM response — max_tokens is likely too low"
                )
                return None
            data = self._parse_json(raw)
            plan = TaskPlan.from_dict(data)
            self.logger.debug("Initial plan: %s", plan.to_json()[:500])
            return plan
        except Exception as exc:
            self.logger.error("Failed to create plan: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Plan execution loop
    # ------------------------------------------------------------------

    async def _run_plan_loop(
        self,
        workflow: Workflow,
        plan: TaskPlan,
        outputs: List[str],
        steps_taken: int,
    ) -> Dict[str, Any]:
        """Execute pending tasks from the plan. Returns updated state."""
        while steps_taken < workflow.max_steps:
            # Get next task
            next_task = await self._get_next_task(workflow, plan, outputs)
            if next_task is None:
                break

            block_id = next_task.get("block", "")
            block = self._find_block(workflow, block_id)
            if not block:
                self.logger.warning("Unknown block '%s' in plan — skipping", block_id)
                self._mark_task_done(plan, next_task)
                continue

            # Substitute params and execute
            params = next_task.get("params", {})
            command = self._substitute(block.command, params)

            if self.on_step:
                import asyncio
                status = self._substitute(block.status_desc, params) if block.status_desc else block.description
                asyncio.ensure_future(self.on_step(status))

            result = self._run_bash(command)
            steps_taken += 1

            # Programmatic verification
            if block.verify and not self._verify(result.output, result.exit_code, block.verify):
                self.logger.info(
                    "Block '%s' failed verification (exit=%d)",
                    block_id, result.exit_code,
                )
                self._mark_task_done(plan, next_task)

                # Run on_error block if defined
                if block.on_error:
                    error_block = self._find_block(workflow, block.on_error)
                    if error_block:
                        recovery = self._run_bash(
                            self._substitute(error_block.command, {})
                        )
                        outputs.append(recovery.output)

                # Let LLM adapt the plan (may add new tasks)
                plan = await self._adapt_plan(workflow, plan, result, outputs)
                continue

            outputs.append(result.output)
            self._mark_task_done(plan, next_task)

        return {"plan": plan, "outputs": outputs, "steps_taken": steps_taken}

    async def _get_next_task(
        self, workflow: Workflow, plan: TaskPlan, outputs: List[str]
    ) -> Optional[Dict[str, Any]]:
        """Get the next pending task, letting the LLM decide/add steps."""
        pending = plan.pending_tasks()
        if not pending:
            return None

        # If there's a clear next step, just return it
        first_pending = pending[0]

        # Let the LLM see only the last output + pending tasks (lean context)
        if outputs:
            last_output = outputs[-1][:2000]
            pending_json = json.dumps(
                [t for t in plan.tasks if t.get("status") != "done"],
                ensure_ascii=False,
            )
            blocks_desc = "\n".join(
                f"- {b.id}: {b.description}" for b in workflow.building_blocks
            )
            prompt = (
                f"Goal: {plan.goal}\n\n"
                f"Pending tasks:\n{pending_json}\n\n"
                f"Last command output:\n{last_output}\n\n"
                f"Available blocks:\n{blocks_desc}\n\n"
                "Based on the output, what should happen next? Options:\n"
                '1. Execute the next pending task as-is → reply: {{"action": "continue"}}\n'
                '2. Modify pending tasks (add/change/remove) → reply: {{"action": "update", "tasks": [...updated pending tasks...]}}\n'
                '3. We are done → reply: {{"action": "done"}}\n\n'
                "Reply with ONLY JSON."
            )

            from langchain_core.messages import HumanMessage, SystemMessage
            messages = [
                SystemMessage(content=(
                    "You manage a task execution plan. Decide the next step. "
                    "Reply with only JSON, no explanation."
                )),
                HumanMessage(content=prompt),
            ]

            try:
                response = await self.llm.ainvoke(messages)
                decision = self._parse_json((response.content or "").strip())
                action = decision.get("action", "continue")

                if action == "done":
                    return None
                if action == "update" and "tasks" in decision:
                    plan.tasks = decision["tasks"]
                    pending = plan.pending_tasks()
                    if not pending:
                        return None
                    return pending[0]
            except Exception as exc:
                self.logger.debug("LLM next-task decision failed: %s — continuing", exc)

        return first_pending

    # ------------------------------------------------------------------
    # Plan adaptation (on failure)
    # ------------------------------------------------------------------

    async def _adapt_plan(
        self,
        workflow: Workflow,
        plan: TaskPlan,
        failed_result: StepResult,
        outputs: List[str],
    ) -> TaskPlan:
        """Let LLM adapt the plan after a step failure."""
        blocks_desc = "\n".join(
            f"- {b.id}: {b.description}" for b in workflow.building_blocks
        )
        last_output = outputs[-1][:2000] if outputs else "(no recovery output)"

        prompt = (
            f"Goal: {plan.goal}\n\n"
            f"FAILURE: The last command failed.\n"
            f"Exit code: {failed_result.exit_code}\n"
            f"Output: {failed_result.output[:1000]}\n\n"
            f"Recovery output (if any):\n{last_output}\n\n"
            f"Available blocks:\n{blocks_desc}\n\n"
            "Create a recovery plan. Reply with ONLY a JSON object:\n"
            '{"goal": "...", "tasks": [...]}\n'
        )

        from langchain_core.messages import HumanMessage, SystemMessage
        messages = [
            SystemMessage(content=(
                "A task step failed. Adjust the plan to recover. "
                "Reply with only the updated plan JSON."
            )),
            HumanMessage(content=prompt),
        ]

        try:
            response = await self.llm.ainvoke(messages)
            data = self._parse_json((response.content or "").strip())
            new_plan = TaskPlan.from_dict(data)
            self.logger.info("Plan adapted after failure")
            return new_plan
        except Exception as exc:
            self.logger.error("Failed to adapt plan: %s", exc)
            return plan

    # ------------------------------------------------------------------
    # Goal checking
    # ------------------------------------------------------------------

    async def _check_goal(
        self,
        workflow: Workflow,
        plan: TaskPlan,
        outputs: List[str],
        query: str,
    ) -> Dict[str, Any]:
        """Check if the user's goal was reached."""
        if not workflow.goal_check:
            return {"reached": True}

        last_output = outputs[-1][:3000] if outputs else "(no output)"

        prompt = (
            f"Original user request: {query}\n"
            f"Goal: {plan.goal}\n\n"
            f"Last output:\n{last_output}\n\n"
            f"{workflow.goal_check.prompt}\n\n"
            'Reply with JSON: {"reached": true/false, "reason": "..."}'
        )

        from langchain_core.messages import HumanMessage, SystemMessage
        messages = [
            SystemMessage(content=(
                "You are a skeptical QA reviewer. Do NOT assume success. "
                "Check if the output actually contains the information or result "
                "the user asked for. Reply with only JSON."
            )),
            HumanMessage(content=prompt),
        ]

        try:
            response = await self.llm.ainvoke(messages)
            return self._parse_json((response.content or "").strip())
        except Exception as exc:
            self.logger.error("Goal check failed: %s", exc)
            return {"reached": True}  # assume reached on error

    async def _extend_plan(
        self,
        workflow: Workflow,
        plan: TaskPlan,
        goal_result: Dict[str, Any],
        outputs: List[str],
        query: str,
    ) -> TaskPlan:
        """Extend the plan when the goal was not reached."""
        blocks_desc = "\n".join(
            f"- {b.id}: {b.description}" for b in workflow.building_blocks
        )
        last_output = outputs[-1][:2000] if outputs else ""

        prompt = (
            f"Original request: {query}\n"
            f"Goal: {plan.goal}\n\n"
            f"Last output:\n{last_output}\n\n"
            f"Goal NOT reached: {goal_result.get('reason', '')}\n\n"
            f"Available blocks:\n{blocks_desc}\n\n"
            "Create additional steps to reach the goal. "
            'Reply with ONLY JSON: {"goal": "...", "tasks": [...]}'
        )

        from langchain_core.messages import HumanMessage, SystemMessage
        messages = [
            SystemMessage(content=(
                "The goal was not reached. Extend the plan. "
                "Reply with only the updated plan JSON."
            )),
            HumanMessage(content=prompt),
        ]

        try:
            response = await self.llm.ainvoke(messages)
            data = self._parse_json((response.content or "").strip())
            return TaskPlan.from_dict(data)
        except Exception as exc:
            self.logger.error("Failed to extend plan: %s", exc)
            return plan

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
        # Context substitutions
        if "{scripts_dir}" in result:
            scripts_dir = self.context.get("cwd", "")
            if scripts_dir:
                scripts_dir = os.path.join(os.path.abspath(scripts_dir), "scripts")
            result = result.replace("{scripts_dir}", scripts_dir)

        # Parameter substitutions
        for key, value in params.items():
            result = result.replace(f"{{{key}}}", str(value))

        return result

    def _run_bash(self, command: str) -> StepResult:
        """Execute a bash command via the tool registry."""
        self.logger.debug("Executing: %s", command[:200])
        raw = self.tool_registry.execute("bash", {"command": command}, self.context)
        output = str(raw)

        # BashTool returns "Error: ..." on failure
        exit_code = 1 if output.startswith("Error") else 0

        self.logger.debug("Result (exit=%d): %s", exit_code, output[:300])
        return StepResult(output=output, exit_code=exit_code, passed=exit_code == 0)

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

    @staticmethod
    def _mark_task_done(plan: TaskPlan, task: Dict[str, Any]) -> None:
        """Mark a task as done in the plan."""
        for t in plan.tasks:
            if t is task:
                t["status"] = "done"
                break

    @staticmethod
    def _parse_json(text: str) -> Any:
        """Strip fences, repair broken JSON, parse."""
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return json.loads(repair_json(text))
