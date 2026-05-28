#!/usr/bin/env python3
"""E2E test: ARK workspace with direct file tools vs. skill fallback.

Creates a mock ARK workspace and tests whether the agent uses direct tools
(read_file, list_files, grep_files) for simple operations vs. the files skill
for complex ones.

Run:
    PYTHONPATH=. .venv/bin/pytest tests/e2e_ark_direct_tools.py -v
"""

import asyncio
import json
import os
import tempfile
from typing import Any, Dict

import pytest
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage

from pawlia.agents.chat import ChatAgent
from pawlia.memory import MemoryManager
from pawlia.skills.loader import AgentSkill
from pawlia.tools.files_tools import ReadFileTool, ListFilesTool, GrepFilesTool


def _mock_llm_with_tool_calls(responses):
    """Mock LLM that returns predefined responses with tool_calls."""
    llm = MagicMock()
    llm.invoke = MagicMock(side_effect=responses)
    llm.bind_tools = MagicMock(return_value=llm)
    return llm


def _make_skill(name, description):
    skill = MagicMock(spec=AgentSkill)
    skill.name = name
    skill.description = description
    skill.instructions = "Run the script."
    skill.skill_path = "/nonexistent"
    skill.scripts_dir = "/nonexistent"
    skill.base_dir = "/nonexistent"
    skill.workflow = None
    skill.max_tool_turns = None
    skill.requires_credentials = []
    skill.trust = "internal"
    skill.as_openai_spec.return_value = {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }
    return skill


def _make_ai_with_tool_call(tool_name, args, content=""):
    msg = AIMessage(content=content)
    msg.tool_calls = [{
        "id": "call_123",
        "name": tool_name,
        "args": args,
    }]
    return msg


@pytest.fixture
def ark_workspace():
    """Create a temporary workspace with ARK-themed files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        user_id = "ark_test_user"
        workspace = os.path.join(tmpdir, user_id, "workspace")
        os.makedirs(workspace)

        # Create ARK notes
        with open(os.path.join(workspace, "taming-guide.md"), "w") as f:
            f.write("""# ARK Taming Guide

## Knockout Taming
- Use tranq arrows or darts
- Keep dinos unconscious with narcotics
- Feed preferred food (mejoberries for herbivores, raw meat for carnivores)

## Passive Taming
- Feed by hand without knocking out
- Requires special foods
- Must stay close to dino
""")

        with open(os.path.join(workspace, "boss-strategy.md"), "w") as f:
            f.write("""# Boss Strategy Guide

## The Island Bosses
- Broodmother: Spider queen, bring Rex army
- Megapithecus: Giant ape, bring high-level dinos
- Dragon: Fire-breathing boss, use Therizinosaurus

## Recommended Dinos
- Rex (high health and damage)
- Therizinosaurus (versatile, good vs Dragon)
- Yutyrannus (roar buff for allies)
""")

        with open(os.path.join(workspace, "breeding-notes.md"), "w") as f:
            f.write("""# Breeding Notes

## Mating
- Enable mating on male and female
- Place close together
- Wait for mating bar to fill

## Egg Incubation
- Maintain correct temperature
- Use campfires or AC units
- Watch for egg health

## Mutations
- Small chance for stat boost
- Color mutations are cosmetic
- Stack mutations for super dinos
""")

        yield tmpdir, user_id


@pytest.mark.asyncio
async def test_ark_list_then_read_direct_tools(ark_workspace):
    """Agent uses list_files + read_file (direct tools) to discover and read ARK notes."""
    tmpdir, user_id = ark_workspace

    # Mock LLM: first list_files, then read_file, then answer
    llm = _mock_llm_with_tool_calls([
        _make_ai_with_tool_call("list_files", {}),
        _make_ai_with_tool_call("read_file", {"filename": "taming-guide.md"}),
        AIMessage(content="Taming in ARK involves knockout or passive methods."),
    ])

    memory = MemoryManager(tmpdir)
    session = memory.load_session(user_id)

    direct_tools = {
        "read_file": ReadFileTool(),
        "list_files": ListFilesTool(),
        "grep_files": GrepFilesTool(),
    }

    skill_runner_spawned = False
    def tracking_factory(skill, thread_id=None):
        nonlocal skill_runner_spawned
        skill_runner_spawned = True
        return MagicMock()

    agent = ChatAgent(
        llm=llm,
        skills={"files": _make_skill("files", "File operations")},
        skill_runner_factory=tracking_factory,
        memory=memory,
        session=session,
        direct_tools=direct_tools,
    )

    response = await agent.run("How do I tame dinos in ARK?")

    # Verify the response mentions taming
    assert "taming" in response.lower() or "knockout" in response.lower(), \
        f"Expected taming info, got: {response}"

    # Verify NO SkillRunner was spawned (direct tools used)
    assert not skill_runner_spawned, \
        "SkillRunner should NOT be spawned for direct tools"

    # Verify history: two tool calls (list_files, read_file)
    exchanges = session.exchanges
    assert len(exchanges) >= 1
    _, _, tool_calls_info = exchanges[-1]
    assert tool_calls_info is not None
    # The last exchange should be the final answer, but tool_calls_info
    # might be None if no tools were called in the final turn.
    # Let's check all exchanges for tool calls.
    all_tool_calls = []
    for ex in exchanges:
        if len(ex) == 3:
            _, _, tci = ex
            if tci:
                all_tool_calls.extend(tci)

    tool_names = [tc["name"] for tc in all_tool_calls]
    assert "list_files" in tool_names, f"Expected list_files in {tool_names}"
    assert "read_file" in tool_names, f"Expected read_file in {tool_names}"

    print("  PASS: ARK list+read via direct tools")


@pytest.mark.asyncio
async def test_ark_grep_for_boss_info(ark_workspace):
    """Agent uses grep_files to find boss info, then read_file to read it."""
    tmpdir, user_id = ark_workspace

    llm = _mock_llm_with_tool_calls([
        _make_ai_with_tool_call("grep_files", {"pattern": "boss|rex"}),
        _make_ai_with_tool_call("read_file", {"filename": "boss-strategy.md"}),
        AIMessage(content="For bosses, use Rex army. Broodmother, Megapithecus, Dragon."),
    ])

    memory = MemoryManager(tmpdir)
    session = memory.load_session(user_id)

    direct_tools = {
        "read_file": ReadFileTool(),
        "list_files": ListFilesTool(),
        "grep_files": GrepFilesTool(),
    }

    agent = ChatAgent(
        llm=llm,
        skills={},
        skill_runner_factory=lambda s, t=None: None,
        memory=memory,
        session=session,
        direct_tools=direct_tools,
    )

    response = await agent.run("What dinos should I use for bosses?")

    assert "rex" in response.lower() or "boss" in response.lower(), \
        f"Expected boss info, got: {response}"

    exchanges = session.exchanges
    all_tool_calls = []
    for ex in exchanges:
        if len(ex) == 3:
            _, _, tci = ex
            if tci:
                all_tool_calls.extend(tci)

    tool_names = [tc["name"] for tc in all_tool_calls]
    assert "grep_files" in tool_names, f"Expected grep_files in {tool_names}"
    assert "read_file" in tool_names, f"Expected read_file in {tool_names}"

    print("  PASS: ARK grep+read via direct tools")


@pytest.mark.asyncio
async def test_ark_complex_write_uses_skill(ark_workspace):
    """For complex write operations, agent still uses the files skill."""
    tmpdir, user_id = ark_workspace

    llm = _mock_llm_with_tool_calls([
        _make_ai_with_tool_call("files", {"query": "write new file ark-todo.md with checklist"}),
        AIMessage(content="Created ark-todo.md with your checklist."),
    ])

    memory = MemoryManager(tmpdir)
    session = memory.load_session(user_id)

    direct_tools = {
        "read_file": ReadFileTool(),
        "list_files": ListFilesTool(),
        "grep_files": GrepFilesTool(),
    }

    skill_runner_spawned = False
    def tracking_factory(skill, thread_id=None):
        nonlocal skill_runner_spawned
        skill_runner_spawned = True
        runner = MagicMock()
        async def mock_run(query):
            return f"SIMULATED: wrote file for query: {query}"
        runner.run = mock_run
        return runner

    agent = ChatAgent(
        llm=llm,
        skills={"files": _make_skill("files", "File operations")},
        skill_runner_factory=tracking_factory,
        memory=memory,
        session=session,
        direct_tools=direct_tools,
    )

    response = await agent.run("Create a new file ark-todo.md with a checklist")

    # Verify SkillRunner WAS spawned for complex write operation
    assert skill_runner_spawned, \
        "SkillRunner SHOULD be spawned for complex write via files skill"

    print("  PASS: Complex write still uses files skill")


@pytest.mark.asyncio
async def test_ark_topic_shift_detection(ark_workspace):
    """When topic shifts from taming to breeding, workspace search helps."""
    tmpdir, user_id = ark_workspace

    # First turn: taming
    llm1 = _mock_llm_with_tool_calls([
        _make_ai_with_tool_call("read_file", {"filename": "taming-guide.md"}),
        AIMessage(content="Taming involves knockout or passive methods."),
    ])

    memory = MemoryManager(tmpdir)
    session = memory.load_session(user_id)

    direct_tools = {
        "read_file": ReadFileTool(),
        "list_files": ListFilesTool(),
        "grep_files": GrepFilesTool(),
    }

    agent = ChatAgent(
        llm=llm1,
        skills={},
        skill_runner_factory=lambda s, t=None: None,
        memory=memory,
        session=session,
        direct_tools=direct_tools,
    )

    await agent.run("How does taming work?")

    # Second turn: breeding (topic shift)
    llm2 = _mock_llm_with_tool_calls([
        _make_ai_with_tool_call("read_file", {"filename": "breeding-notes.md"}),
        AIMessage(content="Breeding requires mating, incubation, and mutations."),
    ])

    agent.llm = llm2
    agent._bind_all_tools(llm2)

    response = await agent.run("Now tell me about breeding")

    assert "breeding" in response.lower() or "mating" in response.lower(), \
        f"Expected breeding info, got: {response}"

    print("  PASS: Topic shift handled with direct tools")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
