"""Tests for the config skill helper script."""

import importlib.util
from pathlib import Path


def _load_config_skill():
    path = Path(__file__).resolve().parents[1] / "skills" / "config" / "scripts" / "config.py"
    spec = importlib.util.spec_from_file_location("config_skill_script", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_list_piper_voices_uses_env_dir(monkeypatch):
    mod = _load_config_skill()

    monkeypatch.setenv("PAWLIA_PIPER_DIR", "/voices/env")
    monkeypatch.setattr(mod, "_find_config", lambda: None)
    env_dir = mod.os.path.abspath("/voices/env")
    monkeypatch.setattr(mod.os.path, "isdir", lambda path: path == env_dir)

    def fake_glob(pattern):
        if pattern == mod.os.path.join(env_dir, "*.onnx"):
            return [mod.os.path.join(env_dir, "de_DE-thorsten-low.onnx")]
        return []

    import glob
    monkeypatch.setattr(glob, "glob", fake_glob)

    assert mod._list_piper_voices() == ["de_DE-thorsten-low"]


def test_list_piper_voices_uses_configured_model_dir(monkeypatch):
    mod = _load_config_skill()

    monkeypatch.delenv("PAWLIA_PIPER_DIR", raising=False)
    monkeypatch.delenv("PIPER_VOICE_DIR", raising=False)
    monkeypatch.setattr(mod, "_find_config", lambda: "/app/config.yaml")
    monkeypatch.setattr(
        mod,
        "_read",
        lambda path: {"tts": {"provider": "piper", "piper": {"model": "/voices/config/de_DE-kerstin-low.onnx"}}},
    )
    model_dir = mod.os.path.abspath("/voices/config")
    monkeypatch.setattr(mod.os.path, "isdir", lambda path: path == model_dir)

    def fake_glob(pattern):
        if pattern == mod.os.path.join(model_dir, "*.onnx"):
            return [mod.os.path.join(model_dir, "de_DE-ramona-low.onnx")]
        return []

    import glob
    monkeypatch.setattr(glob, "glob", fake_glob)

    assert mod._list_piper_voices() == ["de_DE-ramona-low"]
