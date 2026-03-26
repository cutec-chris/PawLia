"""Tests for the workflow compiler (pawlia.skills.compiler).

Runs an actual LLM compilation of the browser skill and validates
the resulting workflow.yaml against the Pydantic schema.
"""

import asyncio
import os
import sys

import yaml

# Use the project venv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pawlia.config import load_config
from pawlia.skills.compiler import compile_skill
from pawlia.skills.workflow_schema import CompiledWorkflow
from pawlia.llm import LLMFactory


def simulate_compile_browser_skill():
    """Compile the browser skill with the real LLM and validate output."""
    pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    skill_path = os.path.join(pkg_dir, "skills", "browser")
    workflow_path = os.path.join(skill_path, "workflow.yaml")

    assert os.path.isdir(skill_path), f"Browser skill not found at {skill_path}"

    config = load_config()
    llm = LLMFactory(config).get("compiler")

    # Remove existing workflow to force recompilation
    if os.path.isfile(workflow_path):
        os.remove(workflow_path)

    # Run compilation
    compiled = asyncio.run(compile_skill(skill_path, llm, force=True))

    # --- Assertions ---
    assert compiled is not None, "Compilation returned None"
    assert isinstance(compiled, CompiledWorkflow)
    assert compiled.skill == "browser"
    assert compiled.version == "1.0"
    assert len(compiled.workflows) >= 1, "Expected at least 1 workflow"

    # Check that workflow.yaml was written
    assert os.path.isfile(workflow_path), "workflow.yaml not written"

    # Re-parse from disk to verify roundtrip
    with open(workflow_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    roundtrip = CompiledWorkflow(**data)
    assert roundtrip.skill == "browser"

    # Check workflow structure
    for wf in compiled.workflows:
        assert wf.id, "Workflow must have an id"
        assert wf.trigger, "Workflow must have a trigger"
        assert len(wf.building_blocks) >= 1, f"Workflow '{wf.id}' has no building blocks"

        # Check building blocks have required fields
        block_ids = set()
        for block in wf.building_blocks:
            assert block.id, "Block must have an id"
            assert block.command, "Block must have a command"
            assert block.description, "Block must have a description"
            assert "browser.py" in block.command, (
                f"Block '{block.id}' command doesn't reference browser.py: {block.command}"
            )
            block_ids.add(block.id)

        # on_error references must point to valid block ids
        for block in wf.building_blocks:
            if block.on_error:
                assert block.on_error in block_ids, (
                    f"Block '{block.id}' on_error references unknown block '{block.on_error}'"
                )

    print(f"\n--- Compilation successful ---")
    print(f"Skill: {compiled.skill} v{compiled.version}")
    print(f"Workflows: {len(compiled.workflows)}")
    for wf in compiled.workflows:
        print(f"  - {wf.id}: {len(wf.building_blocks)} blocks, trigger='{wf.trigger[:60]}'")
        for b in wf.building_blocks:
            verify_str = "verify" if b.verify else ""
            error_str = f"on_error={b.on_error}" if b.on_error else ""
            extras = " ".join(filter(None, [verify_str, error_str]))
            print(f"    [{b.id}] {b.description[:50]}  {extras}")


if __name__ == "__main__":
    test_compile_browser_skill()
    print("\nAll tests passed!")
