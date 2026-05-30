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

    __slots__ = ("action", "model", "ctx_label", "path", "available", "invalidate_agent", "chains")

    def __init__(
        self,
        action: str,
        model: str,
        ctx_label: str,
        path: str = "default",
        available: Optional[List[str]] = None,
        invalidate_agent: bool = False,
        chains: Optional[Dict[str, Any]] = None,
    ):
        self.action = action            # "show" | "set" | "cleared" | "invalid_path"
        self.model = model              # current or new model name (or "(default)")
        self.ctx_label = ctx_label      # "Main", "Thread …", "Room", etc.
        self.path = path                # agent selector path, e.g. "default" or "skills.browser"
        self.available = available or []  # available model keys from config.models
        self.invalidate_agent = invalidate_agent
        self.chains = chains or {}     # {agent_type: {chain: [...], source: "Global|Session-Override"}}


class ReloadCommandResult:
    """Result of a /reload command."""

    __slots__ = ("message", "warnings")

    def __init__(self, message: str, warnings: Optional[List[str]] = None):
        self.message = message
        self.warnings = warnings or []


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

    ``/model <model>`` sets the chat model selector.  ``/model <path>
    <model>`` sets a specific agent selector.
    """
    session = app.memory.load_session(user_id)
    if ctx_label is None:
        ctx_label = "Session"
    available = list_available_models(app)
    llm_factory = getattr(app, "llm", None)

    if not args.strip():
        if llm_factory is not None:
            overrides = app.memory.effective_agent_overrides(session)
            chains: Dict[str, Any] = {}

            # Chat chain
            chat_chain = llm_factory.get_fallback_chain("chat", overrides)
            chains["chat"] = {
                "chain": chat_chain,
                "source": _chain_source(overrides, "chat"),
            }

            # Skill runner chain
            sr_chain = llm_factory.get_fallback_chain("skill_runner", overrides)
            chains["skill_runner"] = {
                "chain": sr_chain,
                "source": _chain_source(overrides, "skill_runner"),
            }

            # Skill-specific overrides
            override_skills = overrides.get("skills", {})
            if isinstance(override_skills, dict):
                for skill_name in override_skills:
                    sk_chain = llm_factory.get_fallback_chain(f"skill.{skill_name}", overrides)
                    chains[f"skills.{skill_name}"] = {
                        "chain": sk_chain,
                        "source": "Session-Override",
                    }

            return ModelCommandResult("show", "", ctx_label, path="chat", available=available, chains=chains)

        # Fallback if no LLM factory available
        current = app.memory.get_agent_override_value(session, "chat")
        if current:
            return ModelCommandResult("show", current, ctx_label, path="chat", available=available)
        agents = app.config.get("agents") or {}
        effective = str(agents.get("chat") or agents.get("default") or agents.get("defaults") or "(unresolved)")
        effective = effective.split(",")[0].strip() if effective else "(unresolved)"
        return ModelCommandResult("show", f"{effective} (global)", ctx_label, path="chat", available=available)

    first, sep, rest = args.strip().partition(" ")
    if sep:
        path = first.strip()
        new_model = rest.strip()
    else:
        path = "chat"
        new_model = first.strip()

    if not _is_valid_agent_path(path):
        return ModelCommandResult(
            "invalid_path", new_model or "(default)", ctx_label,
            path=path, available=available, invalidate_agent=False,
        )

    if new_model.lower() in _CLEAR_TOKENS:
        app.memory.set_agent_override_value(session, path, None)
        return ModelCommandResult(
            "cleared", "(default)", ctx_label,
            path=path, available=available, invalidate_agent=True,
        )

    app.memory.set_agent_override_value(session, path, new_model)
    return ModelCommandResult(
        "set", new_model, ctx_label,
        path=path, available=available, invalidate_agent=True,
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


def _format_chain(chain: List[str], active: Optional[str]) -> str:
    """Format a model fallback chain, bolding the currently active model."""
    parts = []
    for m in chain:
        if active and m == active:
            parts.append(f"**`{m}`**")
        else:
            parts.append(f"`{m}`")
    return " → ".join(parts)


def _chain_source(overrides: Dict[str, Any], agent_type: str) -> str:
    """Return whether the chain comes from a session override or global config."""
    if agent_type.startswith("skill."):
        skill_name = agent_type[len("skill."):]
        skills = overrides.get("skills", {})
        if isinstance(skills, dict) and skill_name in skills:
            return "Session-Override"
        return "Global"
    if agent_type in overrides:
        return "Session-Override"
    return "Global"


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
    backend = getattr(agent.llm, "backend", "pawlia")
    provider_name = getattr(agent.llm, "provider_name", None)
    temperature = getattr(agent.llm, "temperature", None)

    # Context for thread or main
    if thread_id:
        exchanges = app.memory.get_thread_context(session, thread_id)
    else:
        exchanges = session.exchanges

    if hasattr(agent, "describe_backend"):
        meta = agent.describe_backend(thread_id, agent_type="chat")
        backend = meta["backend"]
        provider_name = meta["provider_name"]
        temperature = meta.get("temperature")

    # Detect active fallback model (if the LLM is a FallbackLLMWrapper)
    active_fallback = None
    for attr in ("llm", "bound_llm"):
        llm_obj = getattr(agent, attr, None)
        if llm_obj and hasattr(llm_obj, "active_label"):
            active_fallback = llm_obj.active_label
            break

    # Build model chains
    llm_factory = app.llm

    chat_chain = llm_factory.get_fallback_chain("chat", effective_overrides)
    chat_source = _chain_source(effective_overrides, "chat")

    skill_runner_chain = llm_factory.get_fallback_chain("skill_runner", effective_overrides)
    skill_runner_source = _chain_source(effective_overrides, "skill_runner")

    # Skill-specific chains (only if explicitly overridden)
    skill_chains: Dict[str, Dict[str, Any]] = {}
    if hasattr(agent, "list_skills"):
        skills = agent.list_skills(thread_id)
    else:
        skills = sorted(agent.skills.keys()) if agent.skills else []

    override_skills = effective_overrides.get("skills", {})
    for skill_name in skills:
        if isinstance(override_skills, dict) and skill_name in override_skills:
            chain = llm_factory.get_fallback_chain(f"skill.{skill_name}", effective_overrides)
            skill_chains[skill_name] = {
                "chain": chain,
                "source": "Session-Override",
            }

    # Estimate context size (chars → rough token estimate at ~4 chars/token)
    context_chars = sum(len(e[0]) + len(e[1]) for e in exchanges)
    summary_chars = len(session.summary)
    estimated_tokens = (context_chars + summary_chars) // 4

    # Idle time
    idle_seconds = (datetime.now() - session.last_activity).total_seconds()

    return {
        "user_id": user_id,
        "active_fallback": active_fallback,
        "chat_chain": chat_chain,
        "chat_source": chat_source,
        "skill_runner_chain": skill_runner_chain,
        "skill_runner_source": skill_runner_source,
        "skill_chains": skill_chains,
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

    active = status.get("active_fallback")

    # Chat chain
    chat_chain = _format_chain(status["chat_chain"], active)
    lines.append(f"**Chat** ({status['chat_source']}): {chat_chain}")

    # Skill runner chain
    sr_chain = _format_chain(status["skill_runner_chain"], active)
    lines.append(f"**Skills** ({status['skill_runner_source']}): {sr_chain}")

    # Skill-specific overrides
    for skill_name, info in status["skill_chains"].items():
        chain = _format_chain(info["chain"], active)
        lines.append(f"**Skills.{skill_name}** ({info['source']}): {chain}")

    lines.append("---")
    lines.append(f"**Backend:** `{status['backend']}`")
    if status.get("provider"):
        lines.append(f"**Provider:** `{status['provider']}`")
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


def format_model_chains(chains: Dict[str, Any]) -> str:
    """Format model fallback chains as markdown lines."""
    lines: List[str] = []
    for key, info in chains.items():
        label = key.replace("skills.", "Skills.")
        label = label[0].upper() + label[1:] if label else key
        chain = " → ".join(f"`{m}`" for m in info["chain"])
        lines.append(f"**{label}** ({info['source']}):\n{chain}")
    return "\n".join(lines)


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
