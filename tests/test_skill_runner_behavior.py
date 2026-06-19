"""Behavior of SkillRunnerAgent — the worker that actually uses tools.

Driven by a scripted LLM but a *real* ToolRegistry + BashTool, so the tool
loop, argument validation, error feedback, text-form recovery and the
command-mode fallback are all exercised for real. Also covers the static
command extractor and the WorkflowExecutor selection contract.
"""

import pytest

from langchain_core.messages import AIMessage

from pawlia.agents.skill_runner import SkillRunnerAgent
from pawlia.skills.executor import WorkflowExecutor
from pawlia.skills.workflow_schema import BuildingBlock, Workflow
from pawlia.tools.base import ToolRegistry
from support.llm import Reply, ScriptedLLM


class _OverflowLLM:
    """Always returns the same tool call and reports a context overflow, so the
    skill runner's overflow circuit breaker has to stop the loop."""

    model_name = "ovf"
    model = "ovf"
    temperature = 0.0
    last_invoke_context_skipped = True

    def __init__(self):
        self.calls = 0

    def invoke(self, messages, **kwargs):
        self.calls += 1
        msg = AIMessage(content="")
        msg.tool_calls = [{
            "id": f"c{self.calls}", "name": "bash",
            "args": {"command": "echo working"}, "type": "tool_call",
        }]
        return msg

    async def ainvoke(self, messages, **kwargs):
        return self.invoke(messages, **kwargs)

    def bind_tools(self, *args, **kwargs):
        return self

    def set_on_fallback(self, callback):
        pass


def _tool_message_texts(llm):
    """All ToolMessage contents seen across every invocation."""
    out = []
    for prompt in llm.calls:
        for m in prompt:
            if m.__class__.__name__ == "ToolMessage":
                out.append(str(m.content))
    return out


# ---------------------------------------------------------------------------
# Answering / tool loop
# ---------------------------------------------------------------------------
async def test_model_answer_without_tools_is_returned_directly(make_skill_runner):
    llm = ScriptedLLM().on("find", Reply(text="Search results: 1, 2, 3"))
    runner = make_skill_runner(llm=llm)

    result = await runner.run("find something")

    assert "Search results" in result


async def test_tool_mode_runs_bash_and_feeds_the_output_back(make_skill_runner):
    llm = ScriptedLLM().on(
        "run the test",
        ScriptedLLM.tool("bash", command="echo test_output"),
        Reply(text="The script printed test_output."),
    )
    runner = make_skill_runner(llm=llm)

    result = await runner.run("run the test")

    # Real bash ran and its output was fed back to the model.
    assert any("test_output" in t for t in _tool_message_texts(llm))
    assert "test_output" in result


async def test_bash_code_block_in_tool_mode_is_recovered_and_executed(make_skill_runner):
    llm = ScriptedLLM().on(
        "run command",
        Reply(text="```bash\necho recovered_output\n```"),
        Reply(text="The result is: recovered_output"),
    )
    runner = make_skill_runner(llm=llm)

    result = await runner.run("run command")

    assert any("recovered_output" in t for t in _tool_message_texts(llm))
    assert "recovered_output" in result


async def test_command_mode_fallback_when_tool_mode_is_empty(make_skill_runner):
    # Tool-call turn yields nothing -> fall back to command mode, which emits a
    # shell command that is executed.
    llm = ScriptedLLM().on(
        "do something",
        Reply(text=""),  # tool-call mode: empty
        Reply(text="```bash\necho fallback_output\n```"),  # command mode
    )
    runner = make_skill_runner(llm=llm, command_fallback=True)

    result = await runner.run("do something")

    assert "fallback_output" in result


async def test_no_fallback_returns_empty_when_tool_mode_is_empty(make_skill_runner):
    llm = ScriptedLLM().default(Reply(text=""))
    runner = make_skill_runner(llm=llm, command_fallback=False)

    result = await runner.run("do something")

    assert result == ""


async def test_empty_first_attempt_is_retried(make_skill_runner):
    # Attempt 1 (tool mode) empty -> run() retries the whole attempt; attempt 2
    # produces a real answer.
    llm = ScriptedLLM().on(
        "find data",
        Reply(text=""),       # attempt 1
        Reply(text="Got it: data here"),  # attempt 2
    )
    runner = make_skill_runner(llm=llm, command_fallback=False)

    result = await runner.run("find data")

    assert "data here" in result


async def test_invalid_tool_args_produce_a_structured_error(make_skill_runner):
    llm = (
        ScriptedLLM()
        # First turn: a bash call with a wrong/empty argument.
        .on("run the test", Reply(tool_calls=[{"name": "bash", "args": {"cmd": ""}}]))
        # Everything after (error feedback, nudges) -> just answer.
        .default(Reply(text="The corrected result is ready."))
    )
    runner = make_skill_runner(llm=llm)

    result = await runner.run("run the test")

    assert "corrected result" in result
    assert any('"error_code": "invalid_arguments"' in t for t in _tool_message_texts(llm))


# ---------------------------------------------------------------------------
# Command extraction (pure static helper)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text,expected", [
    ("Here's the command:\n```bash\npython scripts/run.py\n```", "python scripts/run.py"),
    ("```sh\ncurl http://example.com\n```", "curl http://example.com"),
    ("```\nnode scripts/app.js\n```", "node scripts/app.js"),
    ("```bash\n# a comment\npython run.py\n```", "python run.py"),
    ("You should run:\npython scripts/main.py --flag", "python scripts/main.py --flag"),
    ("I don't know what to do.", ""),
    ("Run this:\n./scripts/run.sh", "./scripts/run.sh"),
])
def test_extract_command(text, expected):
    assert SkillRunnerAgent._extract_command(text) == expected


# ---------------------------------------------------------------------------
# WorkflowExecutor
# ---------------------------------------------------------------------------
def _workflows():
    return [
        Workflow(id="workflow_a", trigger="A", building_blocks=[
            BuildingBlock(id="step_a", command="echo a", description="A")]),
        Workflow(id="workflow_b", trigger="B", building_blocks=[
            BuildingBlock(id="step_b", command="echo b", description="B")]),
    ]


async def test_select_workflow_returns_none_when_model_does_not_call_a_tool():
    llm = ScriptedLLM().default(Reply(text="just chatting, no selection"))
    executor = WorkflowExecutor(tool_registry=ToolRegistry(), context={}, llm=llm)

    chosen = await executor.select_workflow(_workflows(), "do B")

    assert chosen is None


async def test_workflow_execute_runs_a_block_and_returns_its_output(tool_registry):
    llm = ScriptedLLM().on("go", Reply(tool_calls=[{"name": "step_a", "args": {}}]))
    executor = WorkflowExecutor(tool_registry=tool_registry, context={}, llm=llm)
    workflow = Workflow(id="wf", trigger="t", max_steps=1, building_blocks=[
        BuildingBlock(id="step_a", command="echo workflow_out", description="A")])

    result = await executor.execute(workflow, "go")

    assert "workflow_out" in result


def test_workflow_block_tools_accept_extra_arguments():
    """The schema stays strict on *required* params but lets the LLM pass
    extra keys without 400ing the whole request. Hallucinated args
    (path/depth on a tool that takes none) get silently ignored by the
    command-template substitution; missing required args still 400."""
    executor = WorkflowExecutor(tool_registry=ToolRegistry(), context={}, llm=ScriptedLLM())
    workflow = Workflow(id="workflow_a", trigger="A", building_blocks=[
        BuildingBlock(id="step_a", command="echo {term}", description="A")])

    tools = executor._blocks_to_tools(workflow)

    params = tools[0]["function"]["parameters"]
    assert "additionalProperties" not in params
    assert params["required"] == ["term"]
    assert "term" in params["properties"]


def test_workflow_block_tools_keep_required_for_missing_args():
    """Even with the lenient extras policy, required params are still
    enforced so the provider rejects a tool call that's missing them."""
    executor = WorkflowExecutor(tool_registry=ToolRegistry(), context={}, llm=ScriptedLLM())
    workflow = Workflow(id="workflow_a", trigger="A", building_blocks=[
        BuildingBlock(id="step_a", command="echo {term}", description="A")])

    tools = executor._blocks_to_tools(workflow)

    assert tools[0]["function"]["parameters"]["required"] == ["term"]


def test_workflow_execute_ignores_hallucinated_extra_arguments(tool_registry):
    """_substitute must be a no-op for keys that aren't placeholders in
    the command template — that's what makes the lenient
    additionalProperties schema safe: silently ignore, don't fail."""
    llm = ScriptedLLM().on("go", Reply(tool_calls=[{"name": "step_a", "args": {}}]))
    executor = WorkflowExecutor(tool_registry=tool_registry, context={}, llm=llm)

    # Hallucinated args (path/depth) are not placeholders, so they
    # must not appear in the substituted command.
    substituted = executor._substitute("echo {term}",
                                       {"term": "ok", "path": "x", "depth": 1})
    assert substituted == "echo ok"

    # And on a template without that placeholder, all extras are dropped.
    substituted = executor._substitute("echo workflow_out",
                                       {"path": "x", "depth": 1})
    assert substituted == "echo workflow_out"


# ---------------------------------------------------------------------------
# No-progress circuit breakers
# ---------------------------------------------------------------------------
async def test_repeated_identical_tool_call_aborts_with_outcome(make_skill_runner):
    # The model keeps issuing the exact same tool call, making no progress.
    llm = ScriptedLLM().default(ScriptedLLM.tool("bash", command="echo same"))
    runner = make_skill_runner(llm=llm)

    result = await runner.run("do the task")

    # Bailed out with a traceable outcome instead of grinding the turn budget.
    assert "gestoppt" in result.lower()
    assert "fortschritt" in result.lower()
    assert llm.call_count <= 10


async def test_context_overflow_aborts_loop_with_outcome(make_skill_runner):
    fake = _OverflowLLM()
    runner = make_skill_runner(llm=fake)

    result = await runner.run("do the task")

    assert "gestoppt" in result.lower()
    assert "übergelaufen" in result.lower()
    # Aborted after a few overflow turns, not after the full turn budget.
    assert fake.calls <= 6


class _BrokenScriptLLM:
    """Runs the skill's own (crashing) script every turn, varying the surrounding
    command so the identical-tool-call breaker can't catch it — only the
    repeated-error-signature breaker should."""

    model_name = "broken"
    model = "broken"
    temperature = 0.0
    last_invoke_context_skipped = False

    def __init__(self, script_path):
        self.calls = 0
        self.script_path = script_path

    def invoke(self, messages, **kwargs):
        self.calls += 1
        msg = AIMessage(content="")
        msg.tool_calls = [{
            "id": f"c{self.calls}", "name": "bash",
            # Different command text each turn (--try N), same broken script.
            "args": {"command": f"python3 {self.script_path} --try {self.calls}"},
            "type": "tool_call",
        }]
        return msg

    async def ainvoke(self, messages, **kwargs):
        return self.invoke(messages, **kwargs)

    def bind_tools(self, *args, **kwargs):
        return self

    def set_on_fallback(self, callback):
        pass


async def test_broken_own_script_aborts_and_offers_repair(make_skill_runner, session_dir):
    # The skill's own script crashes with the same traceback every turn.
    script = session_dir / "broken.py"
    script.write_text("import totally_missing_module_xyz  # noqa\n", encoding="utf-8")
    llm = _BrokenScriptLLM(str(script))
    runner = make_skill_runner(llm=llm, name="weatherish")

    result = await runner.run("get the weather")

    # Offers a skill-creator repair for the named skill instead of flailing.
    assert "skill-creator" in result.lower()
    assert "weatherish" in result
    assert "defekt" in result.lower()
    # Stopped at the 3rd identical crash, well before the turn budget.
    assert llm.calls <= 4


def test_broken_skill_note_skips_skill_creator(make_skill_runner):
    # skill-creator is the repair tool; offering to repair it with itself loops.
    runner = make_skill_runner(llm=ScriptedLLM(), name="skill-creator")
    runner._error_sig_counts = {"creator.py:5:ModuleNotFoundError": 9}
    assert runner._broken_skill_note() is None


def test_broken_skill_note_fires_at_threshold_for_normal_skill(make_skill_runner):
    runner = make_skill_runner(llm=ScriptedLLM(), name="weatherish")
    runner._broken_error_excerpt = "weather.py, Zeile 22 — ModuleNotFoundError: No module named 'ha'"
    runner._broken_failing_command = "python3 weather.py --location Biederitz"
    # Below threshold → no offer yet.
    runner._error_sig_counts = {"weather.py:22:ModuleNotFoundError": 2}
    assert runner._broken_skill_note() is None
    # At threshold → repair offer naming the skill and skill-creator.
    runner._error_sig_counts = {"weather.py:22:ModuleNotFoundError": 3}
    note = runner._broken_skill_note()
    assert note is not None
    assert "skill-creator" in note.lower()
    assert "weatherish" in note
    assert "ha" in note
