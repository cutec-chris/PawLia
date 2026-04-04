"""Tests for ChatAgent and SkillRunnerAgent with mock LLMs."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from pawlia.agents.chat import ChatAgent
from pawlia.agents.skill_runner import SkillRunnerAgent
from pawlia.agents.base import BaseAgent
from pawlia.skills.loader import AgentSkill
from pawlia.tools.base import Tool, ToolRegistry
from pawlia.tools.bash import BashTool
from pawlia.skills.executor import WorkflowExecutor
from pawlia.skills.workflow_schema import BuildingBlock, Workflow


def _make_ai_message(content: str, tool_calls=None) -> AIMessage:
    """Create an AIMessage, optionally with tool_calls."""
    msg = AIMessage(content=content)
    if tool_calls:
        msg.tool_calls = tool_calls
    return msg


def _mock_llm(responses: list) -> MagicMock:
    """Create a mock LLM that returns predefined responses in order."""
    llm = MagicMock()
    llm.invoke = MagicMock(side_effect=responses)
    llm.bind_tools = MagicMock(return_value=llm)
    return llm


def _make_skill(name="test_skill", description="A test skill",
                instructions="Run the script.") -> AgentSkill:
    """Create a mock AgentSkill."""
    skill = MagicMock(spec=AgentSkill)
    skill.name = name
    skill.description = description
    skill.instructions = instructions
    skill.skill_path = "/nonexistent"
    skill.scripts_dir = "/nonexistent"
    skill.base_dir = "/nonexistent"
    skill.workflow = None
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


class TestBaseAgent:
    def test_strip_thinking(self):
        assert BaseAgent.strip_thinking("<think>internal</think>Hello") == "Hello"
        assert BaseAgent.strip_thinking("<thinking>deep thought</thinking>Result") == "Result"
        assert BaseAgent.strip_thinking("No thinking here") == "No thinking here"

    def test_extract_text(self):
        msg = AIMessage(content="<think>hmm</think>The answer is 42")
        assert BaseAgent.extract_text(msg) == "The answer is 42"


class TestChatAgent:
    @pytest.mark.asyncio
    async def test_direct_response(self):
        """When LLM responds without tool calls, return text directly."""
        llm = _mock_llm([_make_ai_message("Hello! How can I help?")])

        agent = ChatAgent(
            llm=llm,
            skills={},
            skill_runner_factory=lambda s: None,
        )

        result = await agent.run("Hi there")
        assert result == "Hello! How can I help?"

    @pytest.mark.asyncio
    async def test_skill_delegation(self):
        """When LLM calls a skill, it should delegate to SkillRunner."""
        skill = _make_skill("searxng", "Web search")

        # Turn 1: LLM calls the skill
        turn1 = _make_ai_message("Let me search for that.", tool_calls=[
            {"id": "call_1", "name": "searxng", "args": {"query": "Python tutorials"}}
        ])
        # Turn 2: LLM formulates final response
        turn2 = _make_ai_message("Here are some Python tutorials: ...")

        llm = _mock_llm([turn1, turn2])

        # Mock SkillRunner
        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value="1. Tutorial A\n2. Tutorial B")

        agent = ChatAgent(
            llm=llm,
            skills={"searxng": skill},
            skill_runner_factory=lambda s: mock_runner,
        )

        result = await agent.run("Search for Python tutorials")

        # SkillRunner was called
        mock_runner.run.assert_called_once_with(query="Python tutorials")
        # Final response from Turn 2
        assert "Python tutorials" in result

    @pytest.mark.asyncio
    async def test_skill_delegation_repairs_string_args(self):
        """String tool args should be normalized into query."""
        skill = _make_skill("searxng", "Web search")

        turn1 = _make_ai_message("Let me search for that.", tool_calls=[
            {"id": "call_1", "name": "searxng", "args": "Python tutorials"}
        ])
        turn2 = _make_ai_message("Here are some Python tutorials: ...")

        llm = _mock_llm([turn1, turn2])
        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value="1. Tutorial A\n2. Tutorial B")

        agent = ChatAgent(
            llm=llm,
            skills={"searxng": skill},
            skill_runner_factory=lambda s: mock_runner,
        )

        await agent.run("Search for Python tutorials")
        mock_runner.run.assert_called_once_with(query="Python tutorials")

    @pytest.mark.asyncio
    async def test_skill_delegation_repairs_alias_args(self):
        """Common alias keys should be normalized into query."""
        skill = _make_skill("searxng", "Web search")

        turn1 = _make_ai_message("Let me search for that.", tool_calls=[
            {"id": "call_1", "name": "searxng", "args": {"request": "Python tutorials"}}
        ])
        turn2 = _make_ai_message("Here are some Python tutorials: ...")

        llm = _mock_llm([turn1, turn2])
        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value="1. Tutorial A\n2. Tutorial B")

        agent = ChatAgent(
            llm=llm,
            skills={"searxng": skill},
            skill_runner_factory=lambda s: mock_runner,
        )

        await agent.run("Search for Python tutorials")
        mock_runner.run.assert_called_once_with(query="Python tutorials")

    @pytest.mark.asyncio
    async def test_skill_name_is_fuzzy_resolved(self):
        """Minor name variations should still resolve to the correct skill."""
        skill = _make_skill("web_search", "Web search")

        turn1 = _make_ai_message("Let me search for that.", tool_calls=[
            {"id": "call_1", "name": "web-search", "args": {"query": "Python tutorials"}}
        ])
        turn2 = _make_ai_message("Here are some Python tutorials: ...")

        llm = _mock_llm([turn1, turn2])
        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value="1. Tutorial A\n2. Tutorial B")

        agent = ChatAgent(
            llm=llm,
            skills={"web_search": skill},
            skill_runner_factory=lambda s: mock_runner,
        )

        await agent.run("Search for Python tutorials")
        mock_runner.run.assert_called_once_with(query="Python tutorials")

    @pytest.mark.asyncio
    async def test_unknown_skill(self):
        """Unknown skill call should return error."""
        turn1 = _make_ai_message("", tool_calls=[
            {"id": "call_1", "name": "unknown_skill", "args": {"query": "test"}}
        ])
        turn2 = _make_ai_message("Sorry, I couldn't do that.")

        llm = _mock_llm([turn1, turn2])

        agent = ChatAgent(
            llm=llm,
            skills={},
            skill_runner_factory=lambda s: None,
        )

        result = await agent.run("Do something")
        assert "Sorry" in result or "couldn't" in result


class TestChatAgentInterim:
    @pytest.mark.asyncio
    async def test_interim_sent_on_skill_call(self):
        """When LLM returns text + tool_calls, interim callback fires."""
        skill = _make_skill("searxng", "Web search")

        turn1 = _make_ai_message("Let me search for that!", tool_calls=[
            {"id": "c1", "name": "searxng", "args": {"query": "test"}}
        ])
        turn2 = _make_ai_message("Here are the results.")

        llm = _mock_llm([turn1, turn2])

        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value="data")

        interim_messages = []

        async def on_interim(text):
            interim_messages.append(text)

        agent = ChatAgent(
            llm=llm,
            skills={"searxng": skill},
            skill_runner_factory=lambda s: mock_runner,
            on_interim=on_interim,
        )

        result = await agent.run("Search something")
        assert len(interim_messages) == 1
        assert "Let me search" in interim_messages[0]
        assert "results" in result

    @pytest.mark.asyncio
    async def test_no_interim_on_direct_response(self):
        """Direct responses should not trigger interim callback."""
        llm = _mock_llm([_make_ai_message("Simple answer")])

        interim_messages = []

        async def on_interim(text):
            interim_messages.append(text)

        agent = ChatAgent(
            llm=llm,
            skills={},
            skill_runner_factory=lambda s: None,
            on_interim=on_interim,
        )

        await agent.run("Hi")
        assert len(interim_messages) == 0

    @pytest.mark.asyncio
    async def test_no_interim_when_content_empty(self):
        """No interim if tool_calls present but content is empty."""
        skill = _make_skill("searxng", "Web search")

        turn1 = _make_ai_message("", tool_calls=[
            {"id": "c1", "name": "searxng", "args": {"query": "test"}}
        ])
        turn2 = _make_ai_message("Results here.")

        llm = _mock_llm([turn1, turn2])

        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value="data")

        interim_messages = []

        async def on_interim(text):
            interim_messages.append(text)

        agent = ChatAgent(
            llm=llm,
            skills={"searxng": skill},
            skill_runner_factory=lambda s: mock_runner,
            on_interim=on_interim,
        )

        await agent.run("Search")
        assert len(interim_messages) == 0


class TestChatAgentPersist:
    @pytest.mark.asyncio
    async def test_direct_response_persists(self):
        """Direct response should persist exchange to memory."""
        from pawlia.memory import MemoryManager
        import tempfile

        llm = _mock_llm([_make_ai_message("Hello!")])

        with tempfile.TemporaryDirectory() as tmpdir:
            mm = MemoryManager(tmpdir)
            session = mm.load_session("u1")

            agent = ChatAgent(
                llm=llm,
                skills={},
                skill_runner_factory=lambda s: None,
                memory=mm,
                session=session,
            )

            result = await agent.run("Hi")
            assert result == "Hello!"
            assert session.exchange_count == 1
            assert session.exchanges[0] == ("Hi", "Hello!", None)

    @pytest.mark.asyncio
    async def test_skill_response_persists_no_similarity(self):
        """Skill response should persist without similarity tracking."""
        from pawlia.memory import MemoryManager
        import tempfile

        skill = _make_skill("searxng", "Web search")
        turn1 = _make_ai_message("Searching.", tool_calls=[
            {"id": "c1", "name": "searxng", "args": {"query": "test"}}
        ])
        turn2 = _make_ai_message("Found results.")
        llm = _mock_llm([turn1, turn2])

        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value="result data")

        with tempfile.TemporaryDirectory() as tmpdir:
            mm = MemoryManager(tmpdir)
            session = mm.load_session("u1")

            agent = ChatAgent(
                llm=llm,
                skills={"searxng": skill},
                skill_runner_factory=lambda s: mock_runner,
                memory=mm,
                session=session,
            )

            await agent.run("Search test")
            assert session.exchange_count == 1
            # Skill responses don't track similarity
            assert len(session.recent_bot_responses) == 0

    @pytest.mark.asyncio
    async def test_replays_exchanges(self):
        """Session exchanges should be replayed as message pairs."""
        from pawlia.memory import MemoryManager
        import tempfile

        llm = _mock_llm([_make_ai_message("Response")])

        with tempfile.TemporaryDirectory() as tmpdir:
            mm = MemoryManager(tmpdir)
            session = mm.load_session("u1")
            session.exchanges = [("prev_q", "prev_a")]

            agent = ChatAgent(
                llm=llm,
                skills={},
                skill_runner_factory=lambda s: None,
                memory=mm,
                session=session,
            )

            await agent.run("New question")
            # Check that invoke was called with messages including history
            call_args = llm.invoke.call_args
            messages = call_args[0][0]
            # SystemMessage + HumanMessage(prev_q) + AIMessage(prev_a) + HumanMessage(new)
            assert len(messages) == 4
            assert messages[1].content == "prev_q"
            assert messages[2].content == "prev_a"
            assert messages[3].content == "New question"


class TestChatAgentMultiSkill:
    @pytest.mark.asyncio
    async def test_multiple_skill_calls(self):
        """LLM calling multiple skills in one turn."""
        skill_a = _make_skill("skill_a", "Skill A")
        skill_b = _make_skill("skill_b", "Skill B")

        turn1 = _make_ai_message("Let me use both.", tool_calls=[
            {"id": "c1", "name": "skill_a", "args": {"query": "query_a"}},
            {"id": "c2", "name": "skill_b", "args": {"query": "query_b"}},
        ])
        turn2 = _make_ai_message("Combined result from A and B.")

        llm = _mock_llm([turn1, turn2])

        runner_a = MagicMock()
        runner_a.run = AsyncMock(return_value="result_a")
        runner_b = MagicMock()
        runner_b.run = AsyncMock(return_value="result_b")

        runners = {"skill_a": runner_a, "skill_b": runner_b}

        agent = ChatAgent(
            llm=llm,
            skills={"skill_a": skill_a, "skill_b": skill_b},
            skill_runner_factory=lambda s: runners[s.name],
        )

        result = await agent.run("Use both skills")
        runner_a.run.assert_called_once_with(query="query_a")
        runner_b.run.assert_called_once_with(query="query_b")
        assert "Combined result" in result

    @pytest.mark.asyncio
    async def test_sequential_skill_rounds_continue_until_done(self):
        """ChatAgent should allow follow-up skill calls after earlier tool results."""
        skill_a = _make_skill("skill_a", "Skill A")
        skill_b = _make_skill("skill_b", "Skill B")

        turn1 = _make_ai_message("First I need skill A.", tool_calls=[
            {"id": "c1", "name": "skill_a", "args": {"query": "query_a"}},
        ])
        turn2 = _make_ai_message("Now I need skill B as well.", tool_calls=[
            {"id": "c2", "name": "skill_b", "args": {"query": "query_b"}},
        ])
        turn3 = _make_ai_message("Combined result from A and B.")

        llm = _mock_llm([turn1, turn2, turn3])

        runner_a = MagicMock()
        runner_a.run = AsyncMock(return_value="result_a")
        runner_b = MagicMock()
        runner_b.run = AsyncMock(return_value="result_b")
        runners = {"skill_a": runner_a, "skill_b": runner_b}

        agent = ChatAgent(
            llm=llm,
            skills={"skill_a": skill_a, "skill_b": skill_b},
            skill_runner_factory=lambda s: runners[s.name],
        )

        result = await agent.run("Use both skills")

        runner_a.run.assert_called_once_with(query="query_a")
        runner_b.run.assert_called_once_with(query="query_b")
        assert "Combined result" in result

    @pytest.mark.asyncio
    async def test_plain_text_tool_intent_is_nudged_into_real_skill_call(self):
        """A textual 'I'll search' response should be retried until a real tool call happens."""
        skill = _make_skill("searxng", "Web search")

        turn1 = _make_ai_message("I will search the web for that now.")
        turn2 = _make_ai_message("", tool_calls=[
            {"id": "c1", "name": "searxng", "args": {"query": "Python tutorials"}},
        ])
        turn3 = _make_ai_message("Here are some Python tutorials.")

        llm = _mock_llm([turn1, turn2, turn3])

        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value="1. Tutorial A\n2. Tutorial B")

        agent = ChatAgent(
            llm=llm,
            skills={"searxng": skill},
            skill_runner_factory=lambda s: mock_runner,
        )

        result = await agent.run("Search for Python tutorials")

        mock_runner.run.assert_called_once_with(query="Python tutorials")
        assert "Python tutorials" in result
        second_call_messages = llm.invoke.call_args_list[1][0][0]
        assert any(
            isinstance(msg, HumanMessage)
            and "If you need a skill, call it now" in msg.content
            for msg in second_call_messages
        )


class TestSkillRunnerAgent:
    @pytest.mark.asyncio
    async def test_direct_result(self):
        """SkillRunner returns result without tool calls."""
        llm = _mock_llm([
            _make_ai_message("Search results: 1, 2, 3"),  # task response
            _make_ai_message("PASS"),  # validation
        ])

        skill = _make_skill()
        tools = ToolRegistry()
        tools.register(BashTool())

        runner = SkillRunnerAgent(llm=llm, skill=skill, tool_registry=tools)
        result = await runner.run("find something")

        assert "Search results" in result

    @pytest.mark.asyncio
    async def test_tool_loop(self):
        """SkillRunner executes tool calls in a loop."""
        # Turn 1: tool call
        turn1 = _make_ai_message("Let me run the script.", tool_calls=[
            {"id": "tc1", "name": "bash", "args": {"command": "echo test_output"}}
        ])
        # Turn 2: final answer (nudge logic may re-prompt when total_tool_calls <= 1)
        turn2 = _make_ai_message("The result is: test_output")
        # Extra responses for nudge retries
        nudge_reply = _make_ai_message("The result is: test_output")

        llm = _mock_llm([turn1, turn2, nudge_reply, nudge_reply])

        skill = _make_skill()
        tools = ToolRegistry()
        tools.register(BashTool())

        runner = SkillRunnerAgent(llm=llm, skill=skill, tool_registry=tools)
        result = await runner.run("run the test")

        assert "test_output" in result

    @pytest.mark.asyncio
    async def test_command_fallback(self):
        """When tool-call mode returns empty, fall back to command mode."""
        # Tool-call mode: empty response
        empty = _make_ai_message("")
        # Command mode: outputs a bash block
        cmd_response = _make_ai_message("```bash\necho fallback_output\n```")

        llm = _mock_llm([empty, cmd_response])

        skill = _make_skill()
        tools = ToolRegistry()
        mock_bash = MagicMock(spec=BashTool)
        mock_bash.name = "bash"
        mock_bash.normalize_args.side_effect = lambda args: {"command": args} if isinstance(args, str) else args
        mock_bash.validate_args.return_value = None
        mock_bash.execute.return_value = "fallback_output"
        mock_bash.as_openai_spec.return_value = BashTool().as_openai_spec()
        tools.register(mock_bash)

        runner = SkillRunnerAgent(
            llm=llm, skill=skill, tool_registry=tools,
            command_fallback=True,
        )
        result = await runner.run("do something")
        assert "fallback_output" in result

    @pytest.mark.asyncio
    async def test_no_command_fallback(self):
        """With command_fallback=False, empty tool-call result retries."""
        empty = _make_ai_message("")

        llm = _mock_llm([empty, empty, empty, empty])

        skill = _make_skill()
        tools = ToolRegistry()
        tools.register(BashTool())

        runner = SkillRunnerAgent(
            llm=llm, skill=skill, tool_registry=tools,
            command_fallback=False,
        )
        result = await runner.run("do something")
        assert result == ""

    @pytest.mark.asyncio
    async def test_retry_on_empty_no_fallback(self):
        """Empty first attempt without fallback should retry."""
        empty = _make_ai_message("")
        good = _make_ai_message("Got it: data here")

        # Attempt 1: tool-call empty → retry
        # Attempt 2: tool-call returns good result
        llm = _mock_llm([empty, good])

        skill = _make_skill()
        tools = ToolRegistry()
        tools.register(BashTool())

        runner = SkillRunnerAgent(
            llm=llm, skill=skill, tool_registry=tools,
            command_fallback=False,
        )
        result = await runner.run("find data")
        assert "data here" in result

    @pytest.mark.asyncio
    async def test_invalid_tool_args_add_retry_guidance(self):
        """Malformed tool args should produce structured error feedback and retry guidance."""
        turn1 = _make_ai_message("", tool_calls=[
            {"id": "tc1", "name": "bash", "args": {"cmd": ""}}
        ])
        turn2 = _make_ai_message("The corrected result is ready.")
        turn3 = _make_ai_message("The corrected result is ready.")
        turn4 = _make_ai_message("The corrected result is ready.")

        llm = _mock_llm([turn1, turn2, turn3, turn4])
        skill = _make_skill()
        tools = ToolRegistry()
        tools.register(BashTool())

        runner = SkillRunnerAgent(llm=llm, skill=skill, tool_registry=tools)
        result = await runner.run("run the test")

        assert "corrected result" in result
        second_call_messages = llm.invoke.call_args_list[1][0][0]
        tool_message = next(msg for msg in second_call_messages if msg.__class__.__name__ == "ToolMessage")
        assert '"error_code": "invalid_arguments"' in tool_message.content
        assert any(
            msg.__class__.__name__ == "HumanMessage"
            and "error_code or hint fields" in msg.content
            for msg in second_call_messages
        )


class TestExtractCommand:
    def test_bash_block(self):
        text = "Here's the command:\n```bash\npython scripts/run.py\n```"
        assert SkillRunnerAgent._extract_command(text) == "python scripts/run.py"

    def test_sh_block(self):
        text = "```sh\ncurl http://example.com\n```"
        assert SkillRunnerAgent._extract_command(text) == "curl http://example.com"

    def test_plain_block(self):
        text = "```\nnode scripts/app.js\n```"
        assert SkillRunnerAgent._extract_command(text) == "node scripts/app.js"

    def test_skips_comments(self):
        text = "```bash\n# This is a comment\npython run.py\n```"
        assert SkillRunnerAgent._extract_command(text) == "python run.py"

    def test_inline_command(self):
        text = "You should run:\npython scripts/main.py --flag"
        assert SkillRunnerAgent._extract_command(text) == "python scripts/main.py --flag"

    def test_no_command(self):
        text = "I don't know what to do."
        assert SkillRunnerAgent._extract_command(text) == ""

    def test_relative_path(self):
        text = "Run this:\n./scripts/run.sh"
        assert SkillRunnerAgent._extract_command(text) == "./scripts/run.sh"


class DummyTool(Tool):
    name = "dummy"
    description = "Dummy tool"

    def parameters(self):
        return {}

    def execute(self, args, context=None):
        return "ok"


class TestWorkflowExecutor:
    @pytest.mark.asyncio
    async def test_select_workflow_returns_none_when_model_does_not_call_tool(self):
        llm = MagicMock()
        bound = MagicMock()
        bound.ainvoke = AsyncMock(side_effect=[
            _make_ai_message("workflow_a"),
            _make_ai_message("still text"),
        ])
        llm.bind_tools = MagicMock(return_value=bound)

        executor = WorkflowExecutor(
            tool_registry=ToolRegistry(),
            context={},
            llm=llm,
        )
        workflows = [
            Workflow(id="workflow_a", trigger="A", building_blocks=[
                BuildingBlock(id="step_a", command="echo a", description="A")
            ]),
            Workflow(id="workflow_b", trigger="B", building_blocks=[
                BuildingBlock(id="step_b", command="echo b", description="B")
            ]),
        ]

        chosen = await executor.select_workflow(workflows, "do B")
        assert chosen is None

    def test_block_tools_disallow_extra_arguments(self):
        executor = WorkflowExecutor(
            tool_registry=ToolRegistry(),
            context={},
            llm=MagicMock(),
        )
        workflow = Workflow(id="workflow_a", trigger="A", building_blocks=[
            BuildingBlock(id="step_a", command="echo {term}", description="A")
        ])

        tools = executor._blocks_to_tools(workflow)
        assert tools[0]["function"]["parameters"]["additionalProperties"] is False
