"""Behavior of SkillRunnerAgent — the worker that actually uses tools.

Driven by a scripted LLM but a *real* ToolRegistry + BashTool, so the tool
loop, argument validation, error feedback, text-form recovery and the
command-mode fallback are all exercised for real. Also covers the static
command extractor and the WorkflowExecutor selection contract.
"""

import pytest

from pawlia.agents.skill_runner import SkillRunnerAgent
from pawlia.skills.executor import WorkflowExecutor
from pawlia.skills.workflow_schema import BuildingBlock, Workflow
from pawlia.tools.base import ToolRegistry
from support.llm import Reply, ScriptedLLM


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


def test_workflow_block_tools_disallow_extra_arguments():
    executor = WorkflowExecutor(tool_registry=ToolRegistry(), context={}, llm=ScriptedLLM())
    workflow = Workflow(id="workflow_a", trigger="A", building_blocks=[
        BuildingBlock(id="step_a", command="echo {term}", description="A")])

    tools = executor._blocks_to_tools(workflow)

    assert tools[0]["function"]["parameters"]["additionalProperties"] is False
