"""Shared utilities for PawLia interfaces."""

import base64
import re
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

class AgentCache:
    """Agent cache shared across interface handlers.

    Entries are keyed by ``cache_key`` (defaults to ``user_id``). When an
    interface needs per-thread isolation (e.g. Matrix threads) it passes a
    composite key like ``f"{user_id}#{thread_id}"`` so concurrent conversations
    don't share mutable agent state (callbacks, in-flight turn data). The
    underlying ``user_id`` still determines the Session/memory store, so
    thread-scoped memory keeps working across per-thread agents.
    """

    def __init__(self, app: "App"):
        self._app = app
        self._agents: Dict[str, Any] = {}

    def get(self, user_id: str, *, cache_key: Optional[str] = None, **kwargs) -> Any:
        key = cache_key or user_id
        if key not in self._agents:
            agent = self._app.make_agent(user_id, **kwargs)
            agent.on_model_change = lambda _model: self.invalidate(user_id)
            self._agents[key] = agent
        return self._agents[key]

    def invalidate(self, user_id: str) -> None:
        """Drop all cached agents whose cache_key starts with ``user_id``.

        Matches both bare ``user_id`` entries and composite keys like
        ``"{user_id}#{thread_id}"``, so a model-change for a user clears every
        per-thread agent belonging to that user.
        """
        prefix = f"{user_id}#"
        for k in [k for k in self._agents if k == user_id or k.startswith(prefix)]:
            self._agents.pop(k, None)

    def invalidate_all(self) -> None:
        """Drop every cached agent after a global app reload."""
        self._agents.clear()


class ModelCommandResult:
    """Result of a /model command, ready for platform-specific formatting."""

    __slots__ = ("action", "model", "ctx_label", "available", "invalidate_agent")

    def __init__(
        self,
        action: str,
        model: str,
        ctx_label: str,
        available: Optional[List[str]] = None,
        invalidate_agent: bool = False,
    ):
        self.action = action            # "show" | "set" | "cleared"
        self.model = model              # current or new model name (or "(default)")
        self.ctx_label = ctx_label      # "Main", "Thread …", "Room", etc.
        self.available = available or []  # available model keys from config.models
        self.invalidate_agent = invalidate_agent


class ReloadCommandResult:
    """Result of a /reload command."""

    __slots__ = ("message", "warnings")

    def __init__(self, message: str, warnings: Optional[List[str]] = None):
        self.message = message
        self.warnings = warnings or []


class AgentCommandResult:
    """Result of a /agent command."""

    __slots__ = ("action", "ctx_label", "path", "value", "available", "overrides", "invalidate_agent")

    def __init__(
        self,
        action: str,
        ctx_label: str,
        path: Optional[str] = None,
        value: Optional[str] = None,
        available: Optional[List[str]] = None,
        overrides: Optional[Dict[str, Any]] = None,
        invalidate_agent: bool = False,
    ):
        self.action = action
        self.ctx_label = ctx_label
        self.path = path
        self.value = value
        self.available = available or []
        self.overrides = overrides or {}
        self.invalidate_agent = invalidate_agent


_CLEAR_TOKENS = {"off", "none", "-", "default", "clear"}
_VALID_AGENT_PATHS = {"default", "defaults", "chat", "skill_runner", "vision", "compiler"}


def list_available_models(app: "App") -> List[str]:
    """Return all configured model keys from config.models."""
    models = app.config.get("models") or {}
    return sorted(k for k, v in models.items() if isinstance(v, dict))


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

    Use ``args`` = "off" / "none" / "-" / "default" to clear an override.
    """
    session = app.memory.load_session(user_id)
    if ctx_label is None:
        ctx_label = f"Thread {thread_id}" if thread_id else "Main"
    available = list_available_models(app)

    if not args.strip():
        current = app.memory.get_agent_override_value(session, "chat", thread_id=thread_id) or "(default)"
        return ModelCommandResult("show", current, ctx_label, available=available)

    new_model = args.strip()
    if new_model.lower() in _CLEAR_TOKENS:
        if thread_id:
            app.memory.set_thread_model_override(session, thread_id, None)
        else:
            app.memory.set_model_override(session, None)
        return ModelCommandResult(
            "cleared", "(default)", ctx_label,
            available=available, invalidate_agent=not thread_id,
        )

    if thread_id:
        app.memory.set_thread_model_override(session, thread_id, new_model)
    else:
        app.memory.set_model_override(session, new_model)
    return ModelCommandResult(
        "set", new_model, ctx_label,
        available=available, invalidate_agent=not thread_id,
    )


def _flatten_agent_overrides(overrides: Dict[str, Any], prefix: str = "") -> Dict[str, str]:
    flat: Dict[str, str] = {}
    for key, value in overrides.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten_agent_overrides(value, path))
        elif isinstance(value, str) and value.strip():
            flat[path] = value.strip()
    return flat


def _is_valid_agent_path(path: str) -> bool:
    if path in _VALID_AGENT_PATHS:
        return True
    if path.startswith("skills.") and len(path.split(".")) == 2:
        return True
    return False


def handle_agent_command(
    app: "App",
    user_id: str,
    args: str,
    thread_id: Optional[str] = None,
    ctx_label: Optional[str] = None,
) -> AgentCommandResult:
    """Shared logic for /agent commands using the `agents:` config shape."""
    session = app.memory.load_session(user_id)
    if ctx_label is None:
        ctx_label = f"Thread {thread_id}" if thread_id else "Main"
    available = list_available_models(app)

    stripped = args.strip()
    current = app.memory.get_thread_agent_overrides(session, thread_id) if thread_id else app.memory.get_agent_overrides(session)

    if not stripped:
        return AgentCommandResult(
            "show_all",
            ctx_label,
            available=available,
            overrides=current,
            invalidate_agent=False,
        )

    path, sep, value = stripped.partition(" ")
    path = path.strip()
    value = value.strip()
    if not _is_valid_agent_path(path):
        return AgentCommandResult(
            "invalid_path",
            ctx_label,
            path=path,
            available=available,
            overrides=current,
        )

    if not sep:
        return AgentCommandResult(
            "show_path",
            ctx_label,
            path=path,
            value=app.memory.get_agent_override_value(session, path, thread_id=thread_id) or "(default)",
            available=available,
            overrides=current,
        )

    if value.lower() in _CLEAR_TOKENS:
        app.memory.set_agent_override_value(session, path, None, thread_id=thread_id)
        return AgentCommandResult(
            "cleared",
            ctx_label,
            path=path,
            value="(default)",
            available=available,
            overrides=app.memory.get_thread_agent_overrides(session, thread_id) if thread_id else app.memory.get_agent_overrides(session),
            invalidate_agent=not thread_id,
        )

    app.memory.set_agent_override_value(session, path, value, thread_id=thread_id)
    return AgentCommandResult(
        "set",
        ctx_label,
        path=path,
        value=value,
        available=available,
        overrides=app.memory.get_thread_agent_overrides(session, thread_id) if thread_id else app.memory.get_agent_overrides(session),
        invalidate_agent=not thread_id,
    )


def format_agent_overrides(overrides: Dict[str, Any]) -> str:
    flat = _flatten_agent_overrides(overrides)
    if not flat:
        return "_(keine Overrides)_"
    return "\n".join(f"- `{path}` = `{value}`" for path, value in sorted(flat.items()))


def handle_reload_command(app: "App") -> ReloadCommandResult:
    """Reload config-driven app state and return a human-readable summary."""
    details = app.reload()
    config_label = details.get("config_path") or "auto-discovered config"
    skill_count = len(details.get("bundled_skills") or [])
    model_count = details.get("model_count", 0)
    warnings = list(details.get("warnings") or [])

    lines = [
        "✓ Konfiguration neu geladen.",
        f"**Config:** `{config_label}`",
        f"**Modelle:** {model_count}",
        f"**Bundled Skills:** {skill_count}",
    ]
    if warnings:
        lines.extend(f"_Hinweis: {warning}._" for warning in warnings)
        lines.append("_Für Ports, Tokens oder andere Interface-Listener-Einstellungen bitte den Prozess neu starten._")
    else:
        lines.append("_Falls du Ports, Tokens oder session_dir geändert hast, ist weiterhin ein Prozess-Neustart nötig._")
    return ReloadCommandResult("\n".join(lines), warnings=warnings)


def build_status(
    app: "App",
    user_id: str,
    agent: Any,
    thread_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Collect status information for the current session/thread.

    Returns a dict with all relevant fields. The caller formats it
    for the platform (plain text, HTML, markdown).
    """
    session = app.memory.load_session(user_id)

    effective_overrides = app.memory.effective_agent_overrides(session, thread_id)
    model_override = bool(effective_overrides)
    model_name = getattr(agent.llm, "model_name", None) or getattr(agent.llm, "model", "?")
    temperature = getattr(agent.llm, "temperature", None)
    backend = getattr(agent.llm, "backend", "pawlia")
    provider_name = getattr(agent.llm, "provider_name", None)
    # Context for thread or main
    if thread_id:
        exchanges = app.memory.get_thread_context(session, thread_id)
    else:
        exchanges = session.exchanges

    if hasattr(agent, "describe_backend"):
        meta = agent.describe_backend(thread_id, agent_type="chat")
        model_name = meta["selection"]
        backend = meta["backend"]
        provider_name = meta["provider_name"]
        temperature = meta.get("temperature")

    # Estimate context size (chars → rough token estimate at ~4 chars/token)
    context_chars = sum(len(e[0]) + len(e[1]) for e in exchanges)
    summary_chars = len(session.summary)
    estimated_tokens = (context_chars + summary_chars) // 4

    # Skills
    if hasattr(agent, "list_skills"):
        skills = agent.list_skills(thread_id)
    else:
        skills = sorted(agent.skills.keys()) if agent.skills else []

    # Idle time
    idle_seconds = (datetime.now() - session.last_activity).total_seconds()

    return {
        "user_id": user_id,
        "model": model_name,
        "model_override": model_override,
        "agent_overrides": _flatten_agent_overrides(effective_overrides),
        "backend": backend,
        "provider": provider_name,
        "temperature": temperature,
        "exchanges": len(exchanges),
        "context_chars": context_chars,
        "estimated_tokens": estimated_tokens,
        "has_summary": bool(session.summary.strip()),
        "summary_chars": summary_chars,
        "private": session.private if not thread_id else (thread_id in session.private_threads),
        "active_threads": len(session.thread_contexts),
        "skills": skills,
        "idle_seconds": int(idle_seconds),
        "thread_id": thread_id,
    }


def format_status(status: Dict[str, Any]) -> str:
    """Format status dict as markdown (single source of truth)."""
    lines: List[str] = []
    lines.append(f"**Model:** `{status['model']}`" + (" _(override)_" if status["model_override"] else ""))
    lines.append(f"**Backend:** `{status['backend']}`")
    if status.get("provider"):
        lines.append(f"**Provider:** `{status['provider']}`")
    if status["temperature"] is not None:
        lines.append(f"**Temp:** {status['temperature']}")
    if status.get("agent_overrides"):
        compact = ", ".join(
            f"`{path}={value}`" for path, value in sorted(status["agent_overrides"].items())
        )
        lines.append(f"**Agent Overrides:** {compact}")
    ctx = "Thread" if status["thread_id"] else "Session"
    lines.append(f"**Context:** {status['exchanges']} exchanges, ~{status['estimated_tokens']} tokens ({ctx})")
    if status["has_summary"]:
        lines.append(f"**Summary:** {status['summary_chars']} chars")
    lines.append(f"**Private:** {'yes' if status['private'] else 'no'}")
    lines.append(f"**Threads:** {status['active_threads']}")
    lines.append(f"**Skills:** {', '.join(status['skills']) or 'none'}")
    m, s = divmod(status["idle_seconds"], 60)
    lines.append(f"**Idle:** {m}m {s}s")
    return "\n".join(lines)


def md_to_text(text: str) -> str:
    """Convert simple markdown to plain text."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)    # bold
    text = re.sub(r"_(.+?)_", r"\1", text)           # italic
    text = re.sub(r"`([^`]+)`", r"\1", text)         # inline code
    return text


def preview_text(text: Optional[str], limit: int = 120) -> str:
    """Normalize text for single-line logs and truncate long output."""
    normalized = " ".join((text or "").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1] + "…"


def md_to_tg_html(text: str) -> str:
    """Convert markdown to Telegram-compatible HTML subset."""
    # Fenced code blocks
    text = re.sub(
        r"```(?:\w*)\n(.*?)```",
        lambda m: f"<pre>{m.group(1).rstrip()}</pre>",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)   # inline code
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)     # bold
    text = re.sub(r"__(.+?)__", r"<b>\1</b>", text)         # bold alt
    text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)         # italic
    text = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"<i>\1</i>", text)  # italic alt
    text = re.sub(r"~~(.+?)~~", r"<s>\1</s>", text)         # strikethrough
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)  # links
    return text

if TYPE_CHECKING:
    from pawlia.app import App

def format_private_toggle(active: bool) -> str:
    """Format the private-mode toggle response."""
    icon = "\U0001f512" if active else "\U0001f513"
    state = "aktiviert" if active else "deaktiviert"
    saving = "**nicht** " if active else ""
    return f"{icon} Private Mode {state} — Nachrichten werden {saving}gespeichert."

def format_bg_enqueue(message: str) -> str:
    """Format the background-task enqueue confirmation."""
    return f"⏳ Aufgabe in Warteschlange: **{message[:60]}**\nWird im Hintergrund verarbeitet wenn idle."

def bytes_to_data_uri(data: bytes, mimetype: str = "image/jpeg") -> str:
    """Convert raw image bytes to a base64 data-URI."""
    b64 = base64.b64encode(data).decode()
    return f"data:{mimetype};base64,{b64}"
