"""Tests for model fallback behavior in pawlia.llm.LLMFactory."""

from typing import Any, Dict, List

import pytest

from pawlia.llm import LLMFactory


class _DummyLLM:
    def __init__(self, model_name: str, fail: bool = False):
        self.model_name = model_name
        self.fail = fail
        self.calls = 0

    def invoke(self, messages: List[Any], **kwargs: Any) -> str:
        self.calls += 1
        if self.fail:
            raise RuntimeError(f"boom-{self.model_name}")
        return f"ok-{self.model_name}"

    async def ainvoke(self, messages: List[Any], **kwargs: Any) -> str:
        return self.invoke(messages, **kwargs)

    def bind_tools(self, *args: Any, **kwargs: Any) -> "_DummyLLM":
        return self


def _base_config() -> Dict[str, Any]:
    return {
        "providers": {
            "test": {
                "apiBase": "http://example.test/v1",
                "apiKey": "x",
            }
        },
        "models": {
            "m1": {"model": "m1", "provider": "test"},
            "m2": {"model": "m2", "provider": "test"},
        },
        "agents": {
            "default": "m1",
        },
    }


def test_agent_model_list_falls_back_on_error(monkeypatch: pytest.MonkeyPatch):
    config = _base_config()
    config["agents"]["chat"] = "m1,m2"

    built: Dict[str, _DummyLLM] = {}

    def fake_build(self: LLMFactory, model_cfg: Dict[str, Any]) -> _DummyLLM:
        model_name = str(model_cfg["model"])
        llm = _DummyLLM(model_name=model_name, fail=(model_name == "m1"))
        built[model_name] = llm
        return llm

    monkeypatch.setattr(LLMFactory, "_build", fake_build)

    factory = LLMFactory(config)
    llm = factory.get("chat")

    result = llm.invoke([])

    assert result == "ok-m2"
    assert built["m1"].calls == 1
    assert built["m2"].calls == 1


def test_agent_model_list_uses_default_chain_when_chat_missing(monkeypatch: pytest.MonkeyPatch):
    config = _base_config()
    config["agents"]["default"] = "m1, m2"

    built: Dict[str, _DummyLLM] = {}

    def fake_build(self: LLMFactory, model_cfg: Dict[str, Any]) -> _DummyLLM:
        model_name = str(model_cfg["model"])
        llm = _DummyLLM(model_name=model_name, fail=(model_name == "m1"))
        built[model_name] = llm
        return llm

    monkeypatch.setattr(LLMFactory, "_build", fake_build)

    factory = LLMFactory(config)
    llm = factory.get("chat")

    result = llm.invoke([])

    assert result == "ok-m2"
    assert built["m1"].calls == 1
    assert built["m2"].calls == 1


def test_agent_model_list_raises_last_error_if_all_fail(monkeypatch: pytest.MonkeyPatch):
    config = _base_config()
    config["agents"]["chat"] = "m1,m2"

    def fake_build(self: LLMFactory, model_cfg: Dict[str, Any]) -> _DummyLLM:
        return _DummyLLM(model_name=str(model_cfg["model"]), fail=True)

    monkeypatch.setattr(LLMFactory, "_build", fake_build)

    factory = LLMFactory(config)
    llm = factory.get("chat")

    with pytest.raises(RuntimeError, match="boom-m2"):
        llm.invoke([])
