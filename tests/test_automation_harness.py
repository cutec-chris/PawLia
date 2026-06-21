"""The automation harness scripts import: param parsing, output gating, and
endpoint resolution from the app config. ``llm_call`` itself is not network-tested
here — only the config→endpoint mapping it relies on."""

import json

import pytest

from pawlia import automation_harness as h


# ---- get_params -----------------------------------------------------------
def test_get_params_returns_dict(monkeypatch):
    monkeypatch.setenv("AUTOMATION_PARAMS", json.dumps({"city": "Berlin"}))
    assert h.get_params() == {"city": "Berlin"}


def test_get_params_empty_when_unset(monkeypatch):
    monkeypatch.delenv("AUTOMATION_PARAMS", raising=False)
    assert h.get_params() == {}


@pytest.mark.parametrize("raw", ["not json", "[1,2,3]", "42"])
def test_get_params_tolerates_garbage(monkeypatch, raw):
    monkeypatch.setenv("AUTOMATION_PARAMS", raw)
    assert h.get_params() == {}


# ---- emit / silent --------------------------------------------------------
def test_emit_prints_real_text(capsys):
    h.emit("hello")
    assert capsys.readouterr().out.strip() == "hello"


@pytest.mark.parametrize("blank", ["", "   ", "\n\t"])
def test_emit_suppresses_blank(capsys, blank):
    h.emit(blank)
    assert capsys.readouterr().out == ""


def test_silent_prints_nothing(capsys):
    h.silent()
    assert capsys.readouterr().out == ""


def test_log_goes_to_stderr(capsys):
    h.log("diag")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "diag" in captured.err


# ---- endpoint resolution --------------------------------------------------
CFG = {
    "providers": {
        "vllm": {"apiBase": "http://host:11434/v1", "apiKey": "none", "timeout": 360},
        "groq": {"apiBase": "https://api.groq.com/openai/v1", "apiKey": "gsk_x"},
    },
    "models": {
        "fast": {"model": "qwen3:4b", "provider": "vllm"},
        "gpt-oss": {"model": "openai/gpt-oss-120b", "provider": "groq"},
    },
    "agents": {"default": "gpt-oss"},
}


def test_resolve_endpoint_follows_default_agent():
    ep = h._resolve_endpoint(CFG, None)
    assert ep["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert ep["model"] == "openai/gpt-oss-120b"
    assert ep["api_key"] == "gsk_x"


def test_resolve_endpoint_honours_explicit_model_key():
    ep = h._resolve_endpoint(CFG, "fast")
    assert ep["url"] == "http://host:11434/v1/chat/completions"
    assert ep["model"] == "qwen3:4b"
    assert ep["timeout"] == 360


def test_resolve_endpoint_chat_agent_wins_over_default():
    cfg = dict(CFG, agents={"chat": "fast", "default": "gpt-oss"})
    assert h._resolve_endpoint(cfg, None)["model"] == "qwen3:4b"


def test_resolve_endpoint_raises_without_providers():
    with pytest.raises(RuntimeError, match="no LLM provider"):
        h._resolve_endpoint({"models": {}, "agents": {}}, None)
