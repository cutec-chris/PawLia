"""E2E test for direct file tools (read_file, list_files, grep_files).

NOTE: As of the "files back to skill-only" change, production no longer wires
these direct tools into the ChatAgent (app.py passes `direct_tools={}`); reads
route through the `files` skill instead. This test still exercises the retained
ChatAgent direct-tool *plumbing* by passing `direct_tools` explicitly — it is a
regression test for that mechanism, not the production path.

Uses the real ChatAgent with real direct tools against a temporary workspace,
but mocks the LLM so the test is deterministic and fast.

Run:
    PYTHONPATH=. .venv/bin/pytest tests/e2e_direct_file_tools.py -v
    PYTHONPATH=. .venv/bin/python -m tests.e2e_direct_file_tools
"""

import asyncio
import json
import os
import sys
import tempfile
from typing import Any, Dict, List, Optional

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, ToolMessage

from pawlia.agents.chat import ChatAgent
from pawlia.memory import MemoryManager
from pawlia.skills.loader import AgentSkill
from pawlia.tools.files_tools import ReadFileTool, ListFilesTool, GrepFilesTool


def _mock_llm_with_tool_calls(responses: List[Any]) -> MagicMock:
    """Mock LLM that returns predefined responses, supporting tool_calls."""
    llm = MagicMock()
    llm.invoke = MagicMock(side_effect=responses)
    llm.bind_tools = MagicMock(return_value=llm)
    return llm


def _make_skill(name="test_skill", description="A test skill") -> MagicMock:
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


def _make_ai_with_tool_call(tool_name: str, args: Dict[str, Any], content: str = "") -> AIMessage:
    """Create an AIMessage with a tool_call."""
    msg = AIMessage(content=content)
    msg.tool_calls = [{
        "id": "call_123",
        "name": tool_name,
        "args": args,
    }]
    return msg


@pytest.mark.asyncio
async def test_list_files_direct_tool():
    """The LLM calls list_files directly — no SkillRunner involved."""
    with tempfile.TemporaryDirectory() as tmpdir:
        user_id = "e2e_test_user"
        workspace = os.path.join(tmpdir, user_id, "workspace")
        os.makedirs(workspace)
        with open(os.path.join(workspace, "brief.md"), "w") as f:
            f.write("# Test\nContent.\n")

        # Mock LLM: first returns a tool call for list_files, then final answer
        llm = _mock_llm_with_tool_calls([
            _make_ai_with_tool_call("list_files", {}),
            AIMessage(content="I found one file: brief.md"),
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
            skills={"dummy": _make_skill("dummy", "dummy")},
            skill_runner_factory=lambda s, t=None: None,
            memory=memory,
            session=session,
            direct_tools=direct_tools,
        )

        response = await agent.run("What files do I have?")

        # Verify response
        assert "brief.md" in response or "file" in response.lower(), \
            f"Expected mention of files, got: {response}"

        # Verify history: the exchange should have tool_calls_info
        exchanges = session.exchanges
        assert len(exchanges) >= 1
        last_exchange = exchanges[-1]
        assert len(last_exchange) == 3, "Expected tool_calls_info in exchange"
        user_input, bot_text, tool_calls_info = last_exchange
        assert tool_calls_info is not None
        assert len(tool_calls_info) == 1
        assert tool_calls_info[0]["name"] == "list_files"

        print("  PASS: list_files direct tool executed inline")


@pytest.mark.asyncio
async def test_read_file_direct_tool():
    """The LLM calls read_file directly to read a document."""
    with tempfile.TemporaryDirectory() as tmpdir:
        user_id = "e2e_test_user"
        workspace = os.path.join(tmpdir, user_id, "workspace")
        os.makedirs(workspace)
        with open(os.path.join(workspace, "notes.md"), "w") as f:
            f.write("# My Notes\nApples are red.\n")

        llm = _mock_llm_with_tool_calls([
            _make_ai_with_tool_call("read_file", {"filename": "notes.md"}),
            AIMessage(content="Your notes say that apples are red."),
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

        response = await agent.run("What do my notes say?")

        assert "apple" in response.lower(), f"Expected mention of apples, got: {response}"

        exchanges = session.exchanges
        last_exchange = exchanges[-1]
        _, _, tool_calls_info = last_exchange
        assert tool_calls_info[0]["name"] == "read_file"

        print("  PASS: read_file direct tool executed inline")


@pytest.mark.asyncio
async def test_grep_files_direct_tool():
    """The LLM calls grep_files directly to search for a pattern."""
    with tempfile.TemporaryDirectory() as tmpdir:
        user_id = "e2e_test_user"
        workspace = os.path.join(tmpdir, user_id, "workspace")
        os.makedirs(workspace)
        with open(os.path.join(workspace, "a.md"), "w") as f:
            f.write("Hello world from A.\n")
        with open(os.path.join(workspace, "b.md"), "w") as f:
            f.write("Hello world from B.\n")

        llm = _mock_llm_with_tool_calls([
            _make_ai_with_tool_call("grep_files", {"pattern": "world"}),
            AIMessage(content="Found 'world' in a.md and b.md."),
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

        response = await agent.run("Where do I mention 'world'?")

        assert "a.md" in response or "b.md" in response, f"Expected file names, got: {response}"

        exchanges = session.exchanges
        last_exchange = exchanges[-1]
        _, _, tool_calls_info = last_exchange
        assert tool_calls_info[0]["name"] == "grep_files"

        print("  PASS: grep_files direct tool executed inline")


@pytest.mark.asyncio
async def test_direct_tool_vs_skill_fallback():
    """When a direct tool is called, no SkillRunner is spawned."""
    with tempfile.TemporaryDirectory() as tmpdir:
        user_id = "e2e_test_user"
        workspace = os.path.join(tmpdir, user_id, "workspace")
        os.makedirs(workspace)
        with open(os.path.join(workspace, "test.md"), "w") as f:
            f.write("Test content.\n")

        skill_runner_spawned = False

        def tracking_factory(skill, thread_id=None):
            nonlocal skill_runner_spawned
            skill_runner_spawned = True
            return MagicMock()

        llm = _mock_llm_with_tool_calls([
            _make_ai_with_tool_call("read_file", {"filename": "test.md"}),
            AIMessage(content="The file says: Test content."),
        ])

        memory = MemoryManager(tmpdir)
        session = memory.load_session(user_id)

        direct_tools = {
            "read_file": ReadFileTool(),
        }

        agent = ChatAgent(
            llm=llm,
            skills={"files": _make_skill("files", "File operations")},
            skill_runner_factory=tracking_factory,
            memory=memory,
            session=session,
            direct_tools=direct_tools,
        )

        await agent.run("Read test.md")

        assert not skill_runner_spawned, \
            "SkillRunner should NOT be spawned for direct tools"

        print("  PASS: direct tool did not spawn SkillRunner")


async def main():
    print("=" * 60)
    print("E2E: Direct File Tools")
    print("=" * 60)
    await test_list_files_direct_tool()
    await test_read_file_direct_tool()
    await test_grep_files_direct_tool()
    await test_direct_tool_vs_skill_fallback()
    print("=" * 60)
    print("All E2E tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
