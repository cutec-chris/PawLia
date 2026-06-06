"""Tests for model fallback behavior in pawlia.llm.LLMFactory."""

from typing import Any, Dict, List

import pytest

from langchain_core.messages import HumanMessage

from pawlia.llm import (
    LLMFactory,
    estimate_context_size,
    estimate_max_tool_turns,
    is_context_length_error,
)


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
        # Probing the (unreachable) test provider would add network latency;
        # the dedicated probe tests below exercise that path with mocks.
        "context-probe": {"enabled": False},
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


def test_context_length_error_is_detected() -> None:
    exc = RuntimeError(
        "Error code: 400 - {'error': {'message': 'Please reduce the length of the messages or completion.', "
        "'code': 'context_length_exceeded'}}"
    )

    assert is_context_length_error(exc) is True


def test_context_length_errors_do_not_blacklist_model(monkeypatch: pytest.MonkeyPatch):
    config = _base_config()
    config["agents"]["chat"] = "m1,m2"

    built: Dict[str, _DummyLLM] = {}

    class _ContextFailLLM(_DummyLLM):
        def invoke(self, messages: List[Any], **kwargs: Any) -> str:
            self.calls += 1
            raise RuntimeError(
                "Error code: 400 - {'error': {'message': 'Please reduce the length of the messages or completion.', "
                "'code': 'context_length_exceeded'}}"
            )

    def fake_build(self: LLMFactory, model_cfg: Dict[str, Any]) -> _DummyLLM:
        model_name = str(model_cfg["model"])
        llm = _ContextFailLLM(model_name=model_name) if model_name == "m1" else _DummyLLM(model_name=model_name)
        built[model_name] = llm
        return llm

    monkeypatch.setattr(LLMFactory, "_build", fake_build)

    factory = LLMFactory(config)
    llm = factory.get("chat")

    for _ in range(4):
        assert llm.invoke([]) == "ok-m2"

    assert built["m1"].calls == 4
    assert built["m2"].calls == 4


def test_task_specific_errors_do_not_blacklist_model(monkeypatch: pytest.MonkeyPatch):
    """A malformed/400 request is a property of *this* prompt, not the model's
    health — the model must stay available for other callers (e.g. a live call
    sharing the cached fallback chain), so it keeps getting retried."""
    config = _base_config()
    config["agents"]["chat"] = "m1,m2"

    built: Dict[str, _DummyLLM] = {}

    class _BadRequestError(RuntimeError):
        status_code = 400

    class _FormatFailLLM(_DummyLLM):
        def invoke(self, messages: List[Any], **kwargs: Any) -> str:
            self.calls += 1
            raise _BadRequestError("invalid request payload")

    def fake_build(self: LLMFactory, model_cfg: Dict[str, Any]) -> _DummyLLM:
        model_name = str(model_cfg["model"])
        llm = _FormatFailLLM(model_name=model_name) if model_name == "m1" else _DummyLLM(model_name=model_name)
        built[model_name] = llm
        return llm

    monkeypatch.setattr(LLMFactory, "_build", fake_build)

    factory = LLMFactory(config)
    llm = factory.get("chat")

    for _ in range(4):
        assert llm.invoke([]) == "ok-m2"

    # m1 never gets benched for a task-specific (format) error.
    assert built["m1"].calls == 4
    assert built["m2"].calls == 4


def test_rate_limit_errors_do_blacklist_model(monkeypatch: pytest.MonkeyPatch):
    """Rate limiting is provider-wide and transient — benching the model for a
    cooldown is correct, so m1 stops being called after the threshold."""
    config = _base_config()
    config["agents"]["chat"] = "m1,m2"

    built: Dict[str, _DummyLLM] = {}
    now = 1000.0

    class _RateLimitLLM(_DummyLLM):
        def invoke(self, messages: List[Any], **kwargs: Any) -> str:
            self.calls += 1
            raise RuntimeError("Error code: 429 - rate limit exceeded, try again in 20s")

    def fake_build(self: LLMFactory, model_cfg: Dict[str, Any]) -> _DummyLLM:
        model_name = str(model_cfg["model"])
        llm = _RateLimitLLM(model_name=model_name) if model_name == "m1" else _DummyLLM(model_name=model_name)
        built[model_name] = llm
        return llm

    monkeypatch.setattr("pawlia.llm.time.monotonic", lambda: now)
    monkeypatch.setattr(LLMFactory, "_build", fake_build)

    factory = LLMFactory(config)
    llm = factory.get("chat")

    for _ in range(4):
        assert llm.invoke([]) == "ok-m2"

    # After 3 rate-limit failures m1 is blacklisted; the 4th call skips it.
    assert built["m1"].calls == 3
    assert built["m2"].calls == 4


def test_fallback_skips_models_with_too_small_context(monkeypatch: pytest.MonkeyPatch):
    config = _base_config()
    config["agents"]["chat"] = "m1,m2"
    config["models"]["m1"]["context_size"] = 1024
    config["models"]["m2"]["context_size"] = 8192

    built: Dict[str, _DummyLLM] = {}

    def fake_build(self: LLMFactory, model_cfg: Dict[str, Any]) -> _DummyLLM:
        model_name = str(model_cfg["model"])
        llm = _DummyLLM(model_name=model_name)
        built[model_name] = llm
        return llm

    monkeypatch.setattr(LLMFactory, "_build", fake_build)

    factory = LLMFactory(config)
    llm = factory.get("chat")

    result = llm.invoke([HumanMessage(content="x" * 6000)])

    assert result == "ok-m2"
    assert built["m1"].calls == 0
    assert built["m2"].calls == 1


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


def test_agent_overrides_are_used_for_default_model_name():
    config = _base_config()
    config["models"]["fallback"] = {"model": "m-fallback", "provider": "test"}
    factory = LLMFactory(config)

    assert factory.default_model_name(
        "skill.browser",
        agent_overrides={"default": "fallback", "skills": {"browser": "m2,m1"}},
    ) == "m2"


def test_get_uses_agent_overrides_for_resolution(monkeypatch: pytest.MonkeyPatch):
    config = _base_config()

    built: Dict[str, _DummyLLM] = {}

    def fake_build(self: LLMFactory, model_cfg: Dict[str, Any]) -> _DummyLLM:
        model_name = str(model_cfg["model"])
        llm = _DummyLLM(model_name=model_name)
        built[model_name] = llm
        return llm

    monkeypatch.setattr(LLMFactory, "_build", fake_build)

    factory = LLMFactory(config)
    llm = factory.get("chat", agent_overrides={"chat": "m2,m1"})

    assert llm.invoke([]) == "ok-m2"


# ---------------------------------------------------------------------------
# max_tool_turns heuristic + per-model override
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model, expected", [
    # Frontier / cloud APIs by name
    ("openai/gpt-oss-120b", 40),
    ("gpt-4o", 40),
    ("claude-opus-4-7", 40),
    ("gemini-3-flash-preview:cloud", 40),
    ("deepseek-r1:14b", 40),  # name hint beats size
    # Local model size parsing
    ("qwen3.5:latest", 20),   # no size suffix → conservative middle ground
    ("qwen3:120b", 40),
    ("qwen3:70b", 40),
    ("qwen3:32b", 30),
    ("qwen3:14b", 22),
    ("qwen3:7b", 16),
    ("qwen3:4b", 12),
    ("qwen3:2b", 8),
    ("qwen3:0.6b", 8),
    # Gemma effective-param naming (gemma4:e4b)
    ("gemma4:e4b", 12),
    ("gemma4:26b", 22),
    # Edge cases
    ("", 20),
])
def test_estimate_max_tool_turns(model: str, expected: int):
    assert estimate_max_tool_turns(model) == expected


def test_max_tool_turns_explicit_config_overrides_heuristic():
    config = _base_config()
    config["models"]["fast"] = {"model": "qwen3:4b", "provider": "test", "max_tool_turns": 25}
    factory = LLMFactory(config)
    # Without override, 4b would give 12; explicit value must win.
    assert factory.max_tool_turns_for_model("fast") == 25


def test_max_tool_turns_falls_back_to_heuristic():
    config = _base_config()
    config["models"]["fast"] = {"model": "qwen3:4b", "provider": "test"}
    factory = LLMFactory(config)
    assert factory.max_tool_turns_for_model("fast") == 12


def test_max_tool_turns_works_for_raw_model_string():
    config = _base_config()
    factory = LLMFactory(config)
    # Unknown config key, falls through get_model_config → heuristic on the raw name.
    assert factory.max_tool_turns_for_model("qwen3:14b") == 22


def test_max_tool_turns_ignores_invalid_explicit_value():
    config = _base_config()
    config["models"]["m1"] = {"model": "qwen3:7b", "provider": "test", "max_tool_turns": 0}
    factory = LLMFactory(config)
    # Zero/negative is treated as "not set" — fall back to heuristic.
    assert factory.max_tool_turns_for_model("m1") == 16


# ---------------------------------------------------------------------------
# context_size heuristic + per-model override
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model, expected", [
    # Frontier APIs by name
    ("claude-opus-4-7", 200_000),
    ("gpt-4o", 200_000),
    ("gemini-3-flash-preview:cloud", 200_000),
    ("openai/gpt-oss-120b", 128_000),
    ("deepseek-r1:14b", 128_000),
    # Local families
    ("qwen3.5:latest", 32_768),
    ("qwen3:14b", 32_768),
    ("qwen3:4b", 32_768),
    ("gemma4:e4b", 8_192),
    ("gemma3:12b", 8_192),
    ("llama3.1:latest", 8_192),
    ("phi4:latest", 16_384),
    # Size-based fallback (no family match)
    ("unknown-model:70b", 32_768),
    ("unknown-model:14b", 16_384),
    ("unknown-model:7b", 8_192),
    ("unknown-model:3b", 4_096),
    ("unknown-model:1b", 2_048),
    # No size, no family
    ("totally-unknown", 8_192),
    ("", 8_192),
])
def test_estimate_context_size(model: str, expected: int):
    assert estimate_context_size(model) == expected


def test_context_size_explicit_config_overrides_heuristic():
    config = _base_config()
    config["models"]["big"] = {"model": "qwen3.5:latest", "provider": "test", "context_size": 65536}
    factory = LLMFactory(config)
    assert factory.context_size_for_model("big") == 65536


def test_context_size_num_ctx_alias_accepted():
    config = _base_config()
    config["models"]["legacy"] = {"model": "qwen3:7b", "provider": "test", "num_ctx": 4096}
    factory = LLMFactory(config)
    assert factory.context_size_for_model("legacy") == 4096


def test_context_size_falls_back_to_heuristic():
    config = _base_config()
    config["models"]["q"] = {"model": "qwen3:14b", "provider": "test"}
    factory = LLMFactory(config)
    assert factory.context_size_for_model("q") == 32_768


def test_context_size_in_cache_key_triggers_rebuild():
    config = _base_config()
    config["models"]["m1"] = {"model": "qwen3:7b", "provider": "test"}
    factory = LLMFactory(config)
    key_default = factory._cache_key(config["models"]["m1"])
    key_explicit = factory._cache_key({**config["models"]["m1"], "context_size": 16384})
    assert key_default != key_explicit


# ---------------------------------------------------------------------------
# summary_threshold_tokens
# ---------------------------------------------------------------------------

def test_summary_threshold_default_fraction():
    config = _base_config()
    config["models"]["m1"] = {"model": "qwen3:14b", "provider": "test"}
    factory = LLMFactory(config)
    # qwen3:14b → ctx 32768, default 0.6 → 19660
    assert factory.summary_threshold_tokens("m1") == int(32_768 * 0.6)


def test_summary_threshold_explicit_tokens_overrides_fraction():
    config = _base_config()
    config["models"]["m1"] = {
        "model": "claude-opus-4-7", "provider": "test",
        "summarize_at_tokens": 8000,
        "summarize_at_fraction": 0.9,  # would imply 180K — absolute wins
    }
    factory = LLMFactory(config)
    assert factory.summary_threshold_tokens("m1") == 8000


def test_summary_threshold_custom_fraction():
    config = _base_config()
    config["models"]["m1"] = {
        "model": "qwen3.5:latest", "provider": "test",
        "summarize_at_fraction": 0.4,
    }
    factory = LLMFactory(config)
    # qwen3.5 → ctx 32768, 0.4 → 13107
    assert factory.summary_threshold_tokens("m1") == int(32_768 * 0.4)


def test_summary_threshold_caps_fraction_at_0_95():
    config = _base_config()
    config["models"]["m1"] = {
        "model": "unknown-model:7b", "provider": "test",
        "summarize_at_fraction": 1.5,
    }
    factory = LLMFactory(config)
    # unknown-model:7b → size fallback ctx 8192; 1.5 capped to 0.95 → 7782
    assert factory.summary_threshold_tokens("m1") == int(8192 * 0.95)


def test_summary_threshold_invalid_fraction_falls_back_to_default():
    config = _base_config()
    config["models"]["m1"] = {
        "model": "unknown-model:7b", "provider": "test",
        "summarize_at_fraction": -0.5,
    }
    factory = LLMFactory(config)
    assert factory.summary_threshold_tokens("m1") == int(8192 * 0.6)


# ---------------------------------------------------------------------------
# Context-window API probe (pawlia.context_probe + LLMFactory integration)
# ---------------------------------------------------------------------------

from pawlia import context_probe  # noqa: E402


def test_probe_parses_groq_models_list():
    body = {"data": [
        {"id": "other-model", "context_window": 4096},
        {"id": "openai/gpt-oss-120b", "context_window": 131072},
    ]}
    assert context_probe._from_models_list(body, "openai/gpt-oss-120b") == 131072


def test_probe_parses_openrouter_context_length():
    body = {"data": [{"id": "google/gemini-3-flash-preview", "context_length": 1048576}]}
    assert context_probe._from_models_list(body, "google/gemini-3-flash-preview") == 1048576


def test_probe_parses_ollama_show():
    body = {"model_info": {"qwen35.context_length": 262144, "qwen35.block_count": 64}}
    assert context_probe._from_ollama_show(body) == 262144


def test_probe_returns_none_when_field_absent():
    body = {"data": [{"id": "glm-5", "object": "model", "owned_by": "z-ai"}]}
    assert context_probe._from_models_list(body, "glm-5") is None


def test_probe_groq_via_models_endpoint(monkeypatch: pytest.MonkeyPatch):
    calls = {}

    def fake_http(url, headers, payload=None):
        calls["url"] = url
        return {"data": [{"id": "gpt-oss", "context_window": 131072}]}

    monkeypatch.setattr(context_probe, "_http_json", fake_http)
    ctx = context_probe.probe_context_window(
        {"apiBase": "https://api.groq.com/openai/v1", "apiKey": "k"}, "gpt-oss"
    )
    assert ctx == 131072
    assert calls["url"].endswith("/models")


def test_probe_ollama_prefers_api_show(monkeypatch: pytest.MonkeyPatch):
    seen = []

    def fake_http(url, headers, payload=None):
        seen.append(url)
        if url.endswith("/api/show"):
            return {"model_info": {"llama.context_length": 32768}}
        raise AssertionError("should not fall through to /models")

    monkeypatch.setattr(context_probe, "_http_json", fake_http)
    ctx = context_probe.probe_context_window(
        {"apiBase": "http://192.168.0.5:11434/v1", "apiKey": "ollama"}, "llama3.1:8b"
    )
    assert ctx == 32768
    assert seen[0].endswith("/api/show")


def test_probe_swallows_network_errors(monkeypatch: pytest.MonkeyPatch):
    def boom(url, headers, payload=None):
        raise OSError("connection refused")

    monkeypatch.setattr(context_probe, "_http_json", boom)
    assert context_probe.probe_context_window(
        {"apiBase": "https://api.example/v1", "apiKey": "k"}, "m"
    ) is None


def test_context_size_uses_probe_over_heuristic(monkeypatch: pytest.MonkeyPatch):
    config = _base_config()
    config["context-probe"] = {"enabled": True}
    config["models"]["glm"] = {"model": "glm-5-turbo", "provider": "test"}

    # Heuristic would return 8192 for an unknown model; the probe wins.
    assert estimate_context_size("glm-5-turbo") == 8192
    monkeypatch.setattr(
        "pawlia.llm.probe_context_window", lambda cfg, mid: 131072
    )
    factory = LLMFactory(config)
    assert factory.context_size_for_model("glm") == 131072


def test_explicit_context_size_beats_probe(monkeypatch: pytest.MonkeyPatch):
    config = _base_config()
    config["context-probe"] = {"enabled": True}
    config["models"]["glm"] = {"model": "glm-5-turbo", "provider": "test", "context_size": 65536}

    def fail(cfg, mid):  # probe must not even be consulted
        raise AssertionError("explicit context_size should short-circuit the probe")

    monkeypatch.setattr("pawlia.llm.probe_context_window", fail)
    factory = LLMFactory(config)
    assert factory.context_size_for_model("glm") == 65536


def test_probe_result_is_cached(monkeypatch: pytest.MonkeyPatch):
    config = _base_config()
    config["context-probe"] = {"enabled": True}
    config["models"]["glm"] = {"model": "glm-5-turbo", "provider": "test"}

    calls = {"n": 0}

    def counting_probe(cfg, mid):
        calls["n"] += 1
        return 131072

    monkeypatch.setattr("pawlia.llm.probe_context_window", counting_probe)
    factory = LLMFactory(config)
    factory.context_size_for_model("glm")
    factory.context_size_for_model("glm")
    assert calls["n"] == 1


def test_probe_disabled_falls_back_to_heuristic(monkeypatch: pytest.MonkeyPatch):
    config = _base_config()  # probe disabled by default in _base_config
    config["models"]["glm"] = {"model": "glm-5-turbo", "provider": "test"}

    def fail(cfg, mid):
        raise AssertionError("probe must not run when disabled")

    monkeypatch.setattr("pawlia.llm.probe_context_window", fail)
    factory = LLMFactory(config)
    assert factory.context_size_for_model("glm") == 8192
