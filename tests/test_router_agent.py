"""Tests for the backend-dispatching RouterAgent."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from pawlia.agents.router import RouterAgent
from pawlia.memory import MemoryManager


class _FakeLLMFactory:
    def __init__(self):
        self.models = {
            "pawlia_model": {"model": "m-local", "provider": "ollama", "temperature": 0.3},
            "hermes": {"model": "hermes-agent", "provider": "hermes_local"},
        }
        self.providers = {
            "ollama": {"backend": "pawlia", "apiBase": "http://ollama.test/v1"},
            "hermes_local": {
                "backend": "hermes",
                "apiBase": "http://hermes.test/v1",
                "apiKey": "secret",
                "conversation_namespace": "pawlia",
            },
        }

    def default_model_name(self, agent_type: str = "chat", agent_overrides=None) -> str:
        if agent_overrides and agent_overrides.get(agent_type):
            return agent_overrides[agent_type]
        return "pawlia_model"

    def get_model_config(self, model_name: str):
        return dict(self.models[model_name])

    def get_provider_name_for_model(self, model_name: str) -> str:
        return str(self.models[model_name]["provider"])

    def get_provider_config(self, provider_name: str):
        return dict(self.providers[provider_name])

    def get_backend_for_model(self, model_name: str) -> str:
        provider = self.get_provider_name_for_model(model_name)
        return str(self.providers[provider]["backend"])


@pytest.mark.asyncio
async def test_router_agent_persists_hermes_turns(tmp_path):
    mm = MemoryManager(str(tmp_path))
    session = mm.load_session("alice")
    mm.set_agent_override_value(session, "chat", "hermes")

    local_agent = SimpleNamespace(
        llm=SimpleNamespace(model_name="m-local", model="m-local", temperature=0.3),
        run=AsyncMock(return_value="local"),
        run_streamed=AsyncMock(return_value="local"),
        on_interim=None,
        on_skill_start=None,
        on_skill_step=None,
        on_skill_done=None,
        on_model_change=None,
    )

    router = RouterAgent(
        user_id="alice",
        llm_factory=_FakeLLMFactory(),
        memory=mm,
        session=session,
        skills={"browser": object()},
        local_agent_factory=lambda: local_agent,
        logger=SimpleNamespace(getChild=lambda _: SimpleNamespace()),
    )

    hermes = AsyncMock(return_value="Antwort von Hermes")
    router._hermes_client = lambda thread_id=None, agent_type="chat": SimpleNamespace(run=hermes)

    result = await router.run("Hallo")

    assert result == "Antwort von Hermes"
    assert session.exchanges[-1][0] == "Hallo"
    assert session.exchanges[-1][1] == "Antwort von Hermes"
    assert router.list_skills() == []
    local_agent.run.assert_not_called()


@pytest.mark.asyncio
async def test_router_agent_uses_local_agent_for_local_backend(tmp_path):
    mm = MemoryManager(str(tmp_path))
    session = mm.load_session("bob")
    mm.set_agent_override_value(session, "chat", "pawlia_model")

    local_agent = SimpleNamespace(
        llm=SimpleNamespace(model_name="m-local", model="m-local", temperature=0.3),
        run=AsyncMock(return_value="lokal"),
        run_streamed=AsyncMock(return_value="lokal"),
        on_interim=None,
        on_skill_start=None,
        on_skill_step=None,
        on_skill_done=None,
        on_model_change=None,
    )

    router = RouterAgent(
        user_id="bob",
        llm_factory=_FakeLLMFactory(),
        memory=mm,
        session=session,
        skills={"browser": object()},
        local_agent_factory=lambda: local_agent,
        logger=SimpleNamespace(getChild=lambda _: SimpleNamespace()),
    )

    result = await router.run("Hi")

    assert result == "lokal"
    local_agent.run.assert_awaited_once()
    assert router.list_skills() == ["browser"]


@pytest.mark.asyncio
async def test_router_agent_forwards_allow_skills_for_streamed_local_backend(tmp_path):
    mm = MemoryManager(str(tmp_path))
    session = mm.load_session("bob")
    mm.set_agent_override_value(session, "chat", "pawlia_model")

    local_agent = SimpleNamespace(
        llm=SimpleNamespace(model_name="m-local", model="m-local", temperature=0.3),
        run=AsyncMock(return_value="lokal"),
        run_streamed=AsyncMock(return_value="lokal"),
        on_interim=None,
        on_skill_start=None,
        on_skill_step=None,
        on_skill_done=None,
        on_model_change=None,
    )

    router = RouterAgent(
        user_id="bob",
        llm_factory=_FakeLLMFactory(),
        memory=mm,
        session=session,
        skills={"browser": object()},
        local_agent_factory=lambda: local_agent,
        logger=SimpleNamespace(getChild=lambda _: SimpleNamespace()),
    )

    result = await router.run_streamed("Hi", allow_skills=False)

    assert result == "lokal"
    local_agent.run_streamed.assert_awaited_once_with(
        "Hi",
        system_prompt=None,
        images=None,
        thread_id=None,
        on_sentence=None,
        on_skill_start=None,
        on_skill_step=None,
        on_skill_done=None,
        allow_skills=False,
    )
