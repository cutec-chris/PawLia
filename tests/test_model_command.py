from types import SimpleNamespace

from pawlia.interfaces.common import handle_model_command
from pawlia.memory import MemoryManager


def _app(tmp_path):
    from pawlia.llm import LLMFactory

    config = {
        "models": {
            "fast": {"model": "m-fast", "provider": "test"},
            "deep": {"model": "m-deep", "provider": "test"},
        },
        "providers": {
            "test": {"apiBase": "http://test", "apiKey": "test"},
        },
        "agents": {"default": "fast"},
    }
    return SimpleNamespace(
        config=config,
        llm=LLMFactory(config),
        memory=MemoryManager(str(tmp_path)),
    )


def test_model_one_arg_sets_chat_override(tmp_path):
    app = _app(tmp_path)

    result = handle_model_command(app, "u1", "fast")
    session = app.memory.load_session("u1")

    assert result.action == "set"
    assert result.path == "chat"
    assert result.model == "fast"
    assert app.memory.get_agent_override_value(session, "chat") == "fast"
    assert app.memory.get_agent_override_value(session, "default") is None


def test_model_two_args_sets_agent_override(tmp_path):
    app = _app(tmp_path)

    result = handle_model_command(app, "u1", "chat deep")
    session = app.memory.load_session("u1")

    assert result.action == "set"
    assert result.path == "chat"
    assert result.model == "deep"
    assert app.memory.get_agent_override_value(session, "chat") == "deep"
    assert app.memory.get_agent_override_value(session, "default") is None


def test_model_two_args_supports_nested_skill_path(tmp_path):
    app = _app(tmp_path)

    result = handle_model_command(app, "u1", "skills.browser deep")
    session = app.memory.load_session("u1")

    assert result.action == "set"
    assert result.path == "skills.browser"
    assert app.memory.get_agent_override_value(session, "skills.browser") == "deep"


def test_model_invalid_two_arg_path_is_rejected(tmp_path):
    app = _app(tmp_path)

    result = handle_model_command(app, "u1", "not.a.path deep")
    session = app.memory.load_session("u1")

    assert result.action == "invalid_path"
    assert result.path == "not.a.path"
    assert app.memory.get_agent_overrides(session) == {}


def test_model_one_arg_off_clears_chat_override(tmp_path):
    app = _app(tmp_path)
    handle_model_command(app, "u1", "fast")

    result = handle_model_command(app, "u1", "off")
    session = app.memory.load_session("u1")

    assert result.action == "cleared"
    assert result.path == "chat"
    assert app.memory.get_agent_override_value(session, "chat") is None


def test_model_show_resolves_global_chat_model_when_no_override(tmp_path):
    app = _app(tmp_path)

    result = handle_model_command(app, "u1", "")

    assert result.action == "show"
    assert result.path == "chat"
    assert result.model == "fast (global)"
