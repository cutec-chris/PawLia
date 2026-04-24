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
                "backend": "pawlia",
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


def test_cache_key_distinguishes_think_setting(monkeypatch: pytest.MonkeyPatch):
    config = _base_config()
    config["models"] = {
        "plain": {"model": "shared", "provider": "test"},
        "no_think": {"model": "shared", "provider": "test", "think": False},
    }

    built: List[Dict[str, Any]] = []

    def fake_build(self: LLMFactory, model_cfg: Dict[str, Any]) -> _DummyLLM:
        built.append(dict(model_cfg))
        suffix = "no_think" if model_cfg.get("think") is False else "plain"
        return _DummyLLM(model_name=f"{model_cfg['model']}-{suffix}")

    monkeypatch.setattr(LLMFactory, "_build", fake_build)

    factory = LLMFactory(config)
    plain = factory.get_with_model("plain")
    no_think = factory.get_with_model("no_think")

    assert plain is not no_think
    assert len(built) == 2


def test_cache_key_distinguishes_max_tokens(monkeypatch: pytest.MonkeyPatch):
    config = _base_config()
    config["models"] = {
        "short": {"model": "shared", "provider": "test", "max_tokens": 64},
        "long": {"model": "shared", "provider": "test", "max_tokens": 512},
    }

    built: List[Dict[str, Any]] = []

    def fake_build(self: LLMFactory, model_cfg: Dict[str, Any]) -> _DummyLLM:
        built.append(dict(model_cfg))
        return _DummyLLM(model_name=f"{model_cfg['model']}-{model_cfg.get('max_tokens')}")

    monkeypatch.setattr(LLMFactory, "_build", fake_build)

    factory = LLMFactory(config)
    short = factory.get_with_model("short")
    long = factory.get_with_model("long")

    assert short is not long
    assert len(built) == 2


def test_agent_model_blacklists_after_three_failed_requests(monkeypatch: pytest.MonkeyPatch):
    config = _base_config()
    config["agents"]["chat"] = "m1,m2"

    built: Dict[str, _DummyLLM] = {}
    now = 1000.0

    def fake_build(self: LLMFactory, model_cfg: Dict[str, Any]) -> _DummyLLM:
        model_name = str(model_cfg["model"])
        llm = _DummyLLM(model_name=model_name, fail=(model_name == "m1"))
        built[model_name] = llm
        return llm

    monkeypatch.setattr("pawlia.llm.time.monotonic", lambda: now)
    monkeypatch.setattr(LLMFactory, "_build", fake_build)

    factory = LLMFactory(config)
    llm = factory.get("chat")

    for _ in range(4):
        assert llm.invoke([]) == "ok-m2"

    assert built["m1"].calls == 3
    assert built["m2"].calls == 4


def test_agent_model_blacklist_expires_after_cooldown(monkeypatch: pytest.MonkeyPatch):
    config = _base_config()
    config["agents"]["chat"] = "m1,m2"

    built: Dict[str, _DummyLLM] = {}
    clock = {"now": 1000.0}

    class _FlakyLLM(_DummyLLM):
        def __init__(self, model_name: str):
            super().__init__(model_name=model_name, fail=False)
            self.remaining_failures = 3

        def invoke(self, messages: List[Any], **kwargs: Any) -> str:
            self.calls += 1
            if self.remaining_failures > 0:
                self.remaining_failures -= 1
                raise RuntimeError(f"boom-{self.model_name}")
            return f"ok-{self.model_name}"

    def fake_build(self: LLMFactory, model_cfg: Dict[str, Any]) -> _DummyLLM:
        model_name = str(model_cfg["model"])
        if model_name == "m1":
            llm = _FlakyLLM(model_name=model_name)
        else:
            llm = _DummyLLM(model_name=model_name)
        built[model_name] = llm
        return llm

    monkeypatch.setattr("pawlia.llm.time.monotonic", lambda: clock["now"])
    monkeypatch.setattr(LLMFactory, "_build", fake_build)

    factory = LLMFactory(config)
    llm = factory.get("chat")

    for _ in range(4):
        assert llm.invoke([]) == "ok-m2"


def test_local_factory_skips_nonlocal_chat_models(monkeypatch: pytest.MonkeyPatch):
    config = _base_config()
    config["providers"]["hermes"] = {
        "backend": "hermes",
        "apiBase": "http://hermes.test/v1",
        "apiKey": "secret",
    }
    config["models"] = {
        "hermes": {"model": "hermes-agent", "provider": "hermes"},
        "pawlia_model": {"model": "m-local", "provider": "test"},
    }
    config["agents"]["chat"] = "hermes,pawlia_model"

    built: List[Dict[str, Any]] = []

    def fake_build(self: LLMFactory, model_cfg: Dict[str, Any]) -> _DummyLLM:
        built.append(dict(model_cfg))
        return _DummyLLM(model_name=str(model_cfg["model"]))

    monkeypatch.setattr(LLMFactory, "_build", fake_build)

    factory = LLMFactory(config)
    llm = factory.get("chat")

    assert llm.invoke([]) == "ok-m-local"
    assert [cfg["model"] for cfg in built] == ["m-local"]


def test_get_with_model_rejects_nonlocal_backend():
    config = _base_config()
    config["providers"]["hermes"] = {
        "backend": "hermes",
        "apiBase": "http://hermes.test/v1",
        "apiKey": "secret",
    }
    config["models"]["hermes"] = {"model": "hermes-agent", "provider": "hermes"}

    factory = LLMFactory(config)

    with pytest.raises(RuntimeError, match="cannot be built by LLMFactory"):
        factory.get_with_model("hermes")


def test_default_model_name_keeps_user_facing_selector():
    config = _base_config()
    config["providers"]["hermes"] = {
        "backend": "hermes",
        "apiBase": "http://hermes.test/v1",
        "apiKey": "secret",
    }
    config["models"]["hermes"] = {"model": "hermes-agent", "provider": "hermes"}
    config["agents"]["chat"] = "hermes,m1"

    factory = LLMFactory(config)

    assert factory.default_model_name("chat") == "hermes"
    assert factory.get_backend_for_model("hermes") == "hermes"
