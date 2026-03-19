"""Shared utilities for PawLia interfaces."""

from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from pawlia.app import App


class AgentCache:
    """Per-user agent cache shared across interface handlers."""

    def __init__(self, app: "App"):
        self._app = app
        self._agents: Dict[str, Any] = {}

    def get(self, user_id: str, **kwargs) -> Any:
        if user_id not in self._agents:
            self._agents[user_id] = self._app.make_agent(user_id, **kwargs)
        return self._agents[user_id]

    def invalidate(self, user_id: str) -> None:
        """Remove cached agent so it gets recreated on next access."""
        self._agents.pop(user_id, None)


class ModelCommandResult:
    """Result of a /model command, ready for platform-specific formatting."""

    __slots__ = ("action", "model", "ctx_label", "invalidate_agent")

    def __init__(self, action: str, model: str, ctx_label: str, invalidate_agent: bool = False):
        self.action = action            # "show" or "set"
        self.model = model              # current or new model name
        self.ctx_label = ctx_label      # "Main", "Thread …", "Room", etc.
        self.invalidate_agent = invalidate_agent


def handle_model_command(
    app: "App",
    user_id: str,
    args: str,
    thread_id: Optional[str] = None,
    ctx_label: Optional[str] = None,
) -> ModelCommandResult:
    """Shared logic for /model and !model commands.

    Returns a ModelCommandResult describing what happened.
    The caller is responsible for formatting and sending the response,
    and for invalidating the agent cache if ``result.invalidate_agent``.
    """
    session = app.memory.load_session(user_id)
    if ctx_label is None:
        ctx_label = f"Thread {thread_id}" if thread_id else "Main"

    if not args.strip():
        if thread_id:
            current = app.memory.get_thread_model_override(session, thread_id) or "(default)"
        else:
            current = session.model_override or "(default)"
        return ModelCommandResult("show", current, ctx_label)

    new_model = args.strip()
    if thread_id:
        app.memory.set_thread_model_override(session, thread_id, new_model)
    else:
        app.memory.set_model_override(session, new_model)
    return ModelCommandResult("set", new_model, ctx_label, invalidate_agent=not thread_id)
