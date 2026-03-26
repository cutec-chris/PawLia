"""Tests for pawlia.config."""

import json
import os
import tempfile

from pawlia.config import load_config


def test_load_config_from_explicit_path():
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({"providers": {"test": {}}}, f)
        f.flush()
        path = f.name

    try:
        cfg = load_config(path)
        assert cfg["providers"]["test"] == {}
    finally:
        os.unlink(path)


def test_load_config_missing_returns_empty(tmp_path, monkeypatch):
    """When no config file exists anywhere, return empty dict."""
    monkeypatch.chdir(tmp_path)
    import pawlia.config as cfg_mod
    # Prevent the project-root fallback from finding the real config.json
    monkeypatch.setattr(cfg_mod, "__file__", str(tmp_path / "pawlia" / "config.py"))
    cfg = load_config(os.path.join(str(tmp_path), "nonexistent.json"))
    assert cfg == {}


def test_load_config_from_project_root():
    """load_config() should return a dict from config.sample.yaml or empty dict."""
    cfg = load_config()
    # Returns a dict (possibly empty if no config.yaml exists, only config.sample.yaml)
    assert isinstance(cfg, dict)
