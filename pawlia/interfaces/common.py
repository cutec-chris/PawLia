"""Shared utilities for PawLia interfaces."""

import re
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

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

    # Model info
    model_override = session.model_override
    model_name = model_override or getattr(agent.llm, "model_name", None) or getattr(agent.llm, "model", "?")
    temperature = getattr(agent.llm, "temperature", None)
    # Context for thread or main
    if thread_id:
        exchanges = app.memory.get_thread_context(session, thread_id)
        thread_model = app.memory.get_thread_model_override(session, thread_id)
        if thread_model:
            model_name = thread_model
    else:
        exchanges = session.exchanges

    # Estimate context size (chars → rough token estimate at ~4 chars/token)
    context_chars = sum(len(e[0]) + len(e[1]) for e in exchanges)
    summary_chars = len(session.summary)
    estimated_tokens = (context_chars + summary_chars) // 4

    # Skills
    skills = sorted(agent.skills.keys()) if agent.skills else []

    # Idle time
    idle_seconds = (datetime.now() - session.last_activity).total_seconds()

    return {
        "user_id": user_id,
        "model": model_name,
        "model_override": model_override is not None,
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
    if status["temperature"] is not None:
        lines.append(f"**Temp:** {status['temperature']}")
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
