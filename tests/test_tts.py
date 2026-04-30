"""Tests for TTS configuration helpers."""

from pawlia import tts


def test_piper_voice_dir_prefers_env(monkeypatch):
    monkeypatch.setenv("PAWLIA_PIPER_DIR", "/voices/env")

    assert tts._piper_voice_dir({"voice_dir": "/voices/config"}) == "/voices/env"


def test_piper_voice_dir_uses_config_value(monkeypatch):
    monkeypatch.delenv("PAWLIA_PIPER_DIR", raising=False)
    monkeypatch.delenv("PIPER_VOICE_DIR", raising=False)

    assert tts._piper_voice_dir({"voice_dir": "/voices/config"}) == "/voices/config"
