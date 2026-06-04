"""Pure helpers of the SKILL.md -> workflow.yaml compiler — no LLM.

The compile step itself needs a large LLM (covered by the simulation in
test_compiler.py), but the surrounding pure logic is deterministic and worth
pinning: extracting YAML from messy model output, normalising ``<param>``
placeholders, building the user prompt, and the semantic lint that decides
whether to retry. These rules keep a flaky model response from producing a
broken workflow.yaml.
"""

from pawlia.skills.compiler import (
    _build_user_prompt,
    _extract_yaml,
    _lint_compiled_workflow,
    _normalize_placeholders,
)
from pawlia.skills.workflow_schema import BuildingBlock, CompiledWorkflow, Workflow


# ---- _extract_yaml ---------------------------------------------------------
def test_extract_yaml_strips_markdown_fences():
    text = "```yaml\nskill: browser\nversion: '1'\n```"
    assert _extract_yaml(text) == "skill: browser\nversion: '1'"


def test_extract_yaml_strips_bare_fences():
    text = "```\nskill: browser\n```"
    assert _extract_yaml(text) == "skill: browser"


def test_extract_yaml_drops_thinking_block():
    text = "<think>let me reason</think>\nskill: browser"
    assert _extract_yaml(text) == "skill: browser"


def test_extract_yaml_recovers_from_unclosed_think_by_seeking_skill_line():
    text = "<think>reasoning that never closes\nblah blah\nskill: browser\nversion: '2'"
    out = _extract_yaml(text)
    assert out.startswith("skill: browser")
    assert "reasoning" not in out


def test_extract_yaml_passes_clean_input_through():
    assert _extract_yaml("skill: x") == "skill: x"


# ---- _normalize_placeholders ----------------------------------------------
def test_normalize_placeholders_rewrites_angle_to_brace_in_strings():
    assert _normalize_placeholders("open <url> now") == "open {url} now"


def test_normalize_placeholders_recurses_lists_and_dicts():
    value = {"cmd": "go <a>", "args": ["<b>", "plain"]}
    assert _normalize_placeholders(value) == {"cmd": "go {a}", "args": ["{b}", "plain"]}


def test_normalize_placeholders_leaves_non_identifier_angles_untouched():
    # "<3" is not a valid placeholder name, so it must be left alone.
    assert _normalize_placeholders("a <3 b") == "a <3 b"


def test_normalize_placeholders_passes_through_scalars():
    assert _normalize_placeholders(42) == 42
    assert _normalize_placeholders(None) is None


# ---- _build_user_prompt ----------------------------------------------------
def test_build_user_prompt_includes_metadata_and_scripts():
    prompt = _build_user_prompt(
        "browser", "1.2", "Do things.", ["nav.py", "click.py"], "2026-06-04")
    assert "# Skill: browser  (version 1.2)" in prompt
    assert "# Date: 2026-06-04" in prompt
    assert "Do things." in prompt
    assert "nav.py, click.py" in prompt


def test_build_user_prompt_marks_absent_scripts():
    prompt = _build_user_prompt("x", "1", "instr", [], "2026-06-04")
    assert "(none)" in prompt


# ---- _lint_compiled_workflow ----------------------------------------------
def _compiled(command="run nav.py --go", compiled_at="2026-06-04"):
    return CompiledWorkflow(
        skill="browser",
        version="1",
        compiled_at=compiled_at,
        compiled_by="test",
        workflows=[Workflow(
            id="wf",
            trigger="go",
            building_blocks=[BuildingBlock(id="b1", command=command, description="d")],
        )],
    )


def test_lint_clean_workflow_has_no_issues():
    issues = _lint_compiled_workflow(
        _compiled(), today="2026-06-04", scripts=["nav.py"])
    assert issues == []


def test_lint_flags_stale_compiled_at():
    issues = _lint_compiled_workflow(
        _compiled(compiled_at="2020-01-01"), today="2026-06-04", scripts=["nav.py"])
    assert any("compiled_at must be 2026-06-04" in i for i in issues)


def test_lint_flags_markdown_optional_brackets_in_command():
    issues = _lint_compiled_workflow(
        _compiled(command="run nav.py [--flag]"), today="2026-06-04", scripts=["nav.py"])
    assert any("markdown-style optional syntax" in i for i in issues)


def test_lint_flags_command_not_referencing_a_known_script():
    issues = _lint_compiled_workflow(
        _compiled(command="echo hello"), today="2026-06-04", scripts=["nav.py"])
    assert any("does not reference a known script" in i for i in issues)


def test_lint_skips_script_check_when_no_scripts_declared():
    issues = _lint_compiled_workflow(
        _compiled(command="echo hello"), today="2026-06-04", scripts=[])
    assert issues == []
