"""Shared fixtures for the system-test layer.

The goal is to construct *real* agents wired to a deterministic ``ScriptedLLM``
and exercise observable behavior through their public entry points
(``ChatAgent.run`` / ``run_streamed`` / ``SkillRunnerAgent.run`` /
``RouterAgent.run``) — not to mock internals.

We build ``ChatAgent`` / ``SkillRunnerAgent`` *directly* rather than through
``App.make_agent``: the App path attaches an ``_agent_llm_resolver`` that, at
run time, re-fetches the LLM from the real ``LLMFactory`` and would bypass our
injected double. Leaving the resolvers unset makes ``_resolve_llms`` fall back
to the injected ``bound_llm`` / ``llm`` (see chat.py ``_resolve_llms``).
"""

import os
import sys

# Make ``from support.<mod> import ...`` work from the test modules.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest

from pawlia.agents.chat import ChatAgent
from pawlia.agents.skill_runner import SkillRunnerAgent
from pawlia.memory import MemoryManager
from pawlia.skills.loader import AgentSkill, SkillLoader
from pawlia.tools.base import ToolRegistry
from pawlia.tools.bash import BashTool

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKILLS_DIR = os.path.join(REPO_ROOT, "skills")


# ---- session / memory -----------------------------------------------------
@pytest.fixture
def session_dir(tmp_path):
    d = tmp_path / "session"
    d.mkdir()
    return d


@pytest.fixture
def memory(session_dir):
    return MemoryManager(str(session_dir))


@pytest.fixture
def session(memory):
    return memory.load_session("tester")


# ---- skills / tools --------------------------------------------------------
@pytest.fixture(scope="session")
def real_skills():
    """All bundled skills, parsed from their SKILL.md (read-only, cheap)."""
    return SkillLoader.discover(SKILLS_DIR, config={})


@pytest.fixture
def tool_registry():
    reg = ToolRegistry()
    reg.register(BashTool())
    return reg


# ---- fake SkillRunner ------------------------------------------------------
class FakeRunnerFactory:
    """A ``skill_runner_factory`` that records delegations and returns canned
    results, so ChatAgent tests stay deterministic and never shell out."""

    def __init__(self, returns="ok", per_skill=None):
        self._returns = returns
        self._per_skill = per_skill or {}
        self.calls = []  # list of (skill_name, query)

    @property
    def last_query(self):
        return self.calls[-1][1] if self.calls else None

    def __call__(self, skill, thread_id=None):
        factory = self
        result = self._per_skill.get(getattr(skill, "name", None), self._returns)

        class _Runner:
            on_step = None

            async def run(self, query):
                factory.calls.append((getattr(skill, "name", None), query))
                return result

        return _Runner()


@pytest.fixture
def fake_runner():
    """Returns a builder: ``fake_runner(returns=..., per_skill={...})``."""
    return lambda returns="ok", per_skill=None: FakeRunnerFactory(returns, per_skill)


# ---- agent builders --------------------------------------------------------
@pytest.fixture
def make_chat_agent(memory, session, real_skills):
    def _make(*, llm, skills=None, runner=None, **kwargs):
        chosen = {}
        for name in (skills or []):
            chosen[name] = real_skills.get(name) or AgentSkill.from_instruction(
                name, f"The {name} skill."
            )
        return ChatAgent(
            llm=llm,
            skills=chosen,
            skill_runner_factory=runner or (lambda s, thread_id=None: None),
            memory=memory,
            session=session,
            **kwargs,
        )

    return _make


@pytest.fixture
def make_skill_runner(tool_registry, session_dir):
    def _make(*, llm, skill=None, name="probe", instruction="Do the task.", **kwargs):
        skill = skill or AgentSkill.from_instruction(name, instruction)
        return SkillRunnerAgent(
            llm=llm,
            skill=skill,
            tool_registry=tool_registry,
            context={"session_dir": str(session_dir), "user_id": "tester"},
            **kwargs,
        )

    return _make
