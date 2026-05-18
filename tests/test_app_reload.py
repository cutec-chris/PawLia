from unittest.mock import MagicMock

from pawlia.app import App
from pawlia.interfaces.common import handle_reload_command


def _config(session_dir: str = "/tmp/session", model_name: str = "m1") -> dict:
    return {
        "session_dir": session_dir,
        "providers": {
            "test": {
                "apiBase": "http://example.test/v1",
                "apiKey": "x",
            }
        },
        "models": {
            "chat": {"model": model_name, "provider": "test"},
        },
        "agents": {
            "default": "chat",
            "chat": "chat",
            "vision": "chat",
        },
    }


def test_app_reload_rebuilds_runtime_and_preserves_scheduler_callbacks(monkeypatch, tmp_path):
    discovered = [
        {"alpha": MagicMock(name="alpha")},
        {"beta": MagicMock(name="beta")},
    ]

    def fake_discover(*args, **kwargs):
        return discovered.pop(0)

    monkeypatch.setattr("pawlia.app.SkillLoader.discover", fake_discover)

    app = App(_config(session_dir=str(tmp_path / "session_a")), config_path=str(tmp_path / "config.yaml"))
    old_llm = app.llm
    old_tools = app.tools

    async def _notify(user_id: str, message: str) -> None:
        return None

    app.scheduler.register(_notify)
    monkeypatch.setattr("pawlia.app.load_config", lambda path: _config(session_dir=str(tmp_path / "session_a"), model_name="m2"))

    result = app.reload()

    assert app.llm is not old_llm
    assert app.tools is not old_tools
    assert app.scheduler._callbacks == [_notify]
    assert app.scheduler._config["models"]["chat"]["model"] == "m2"
    assert sorted(app.skills.keys()) == ["beta"]
    assert result["model_count"] == 1
    assert result["warnings"] == []


def test_handle_reload_command_reports_restart_only_settings(monkeypatch, tmp_path):
    monkeypatch.setattr("pawlia.app.SkillLoader.discover", lambda *args, **kwargs: {})

    app = App(_config(session_dir=str(tmp_path / "session_a")), config_path=str(tmp_path / "config.yaml"))
    monkeypatch.setattr("pawlia.app.load_config", lambda path: _config(session_dir=str(tmp_path / "session_b")))

    result = handle_reload_command(app)

    assert "Konfiguration neu geladen" in result.message
    assert "session_dir changed" in result.message
    assert "Prozess neu starten" in result.message


def test_app_normalizes_relative_session_dir(monkeypatch, tmp_path):
    monkeypatch.setattr("pawlia.app.SkillLoader.discover", lambda *args, **kwargs: {})

    app = App(_config(session_dir="session"), config_path=str(tmp_path / "config.yaml"))

    assert app.session_dir.endswith("/session")
    assert app.session_dir.startswith("/")
