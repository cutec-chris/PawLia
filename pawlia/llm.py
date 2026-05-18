"""LLM Factory - creates and caches LangChain ChatOpenAI instances from config.

Config layout (YAML)::

    providers:
      ollama:
        apiBase: http://localhost:11434/v1
        apiKey: ollama
        timeout: 120
      groq:
        apiBase: https://api.groq.com/openai/v1
        apiKey: gsk_...

    models:
      fast:
        model: qwen3:4b
        provider: ollama
        temperature: 0.7
      smart:
        model: qwen3.5:latest
        provider: ollama
        temperature: 0.9
        think: true
      vision:
        model: qwen2.5vl:latest
        provider: ollama

    agents:
      default: smart       # fallback for any unspecified agent type
      chat: smart
      skill_runner: fast
      vision: vision
      skills:              # per-skill overrides
        searxng: fast
        browser: smart

Fallback chains
---------------
- ``get("chat")``          → agents.chat          → agents.default
- ``get("vision")``        → agents.vision        → agents.chat    → agents.default
- ``get("skill_runner")``  → agents.skill_runner  → agents.default
- ``get("skill.searxng")`` → agents.skills.searxng → agents.skill_runner → agents.default

``get_with_model(name)`` resolves a model by its key in ``models:``.  If the
name is not found there it is treated as a raw model string and the default
provider is used.
"""

import logging
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI


logger = logging.getLogger(__name__)

_CHARS_PER_TOKEN = 4
_CONTEXT_SKIP_RESERVE_TOKENS = 1024


# Heuristic: how many tool-call turns to grant the SkillRunner per model.
# Bigger / more capable models can sustain longer exploratory loops without
# losing the plot; tiny models tend to spiral and should be cut off earlier.
# Parses size hints like ":7b", ":120b", ":e4b", ":0.6b" from the identifier.
_SIZE_RE = re.compile(r":(?:e)?(\d+(?:\.\d+)?)\s*b\b", re.IGNORECASE)

# Frontier / cloud APIs get a generous budget regardless of size hint.
_CAPABLE_NAME_HINTS = (
    "gpt-oss", "gpt-4", "gpt-5", "o1-", "o3-",
    "claude", "claude-3", "claude-4",
    "gemini-", "deepseek-r1", "deepseek-v3",
    "llama-3.3", "llama-4",
)

_CONTEXT_ERROR_HINTS = (
    "context_length_exceeded",
    "prompt exceeds max length",
    "maximum context length",
    "maximum context",
    "context window",
    "please reduce the length of the messages or completion",
    "too many tokens",
)


def is_context_length_error(exc: BaseException) -> bool:
    """Return True when *exc* indicates a prompt/context window overflow."""
    text = str(exc).lower()
    return any(hint in text for hint in _CONTEXT_ERROR_HINTS)


def estimate_max_tool_turns(model_name: str) -> int:
    """Heuristic budget for SkillRunner tool-call loops, derived from model name.

    Used when neither the skill metadata nor the model config specifies
    ``max_tool_turns`` explicitly. The numbers are deliberately coarse —
    small models cap out earlier, frontier models get more room to explore.
    """
    name = (model_name or "").lower()

    if any(hint in name for hint in _CAPABLE_NAME_HINTS):
        return 40

    match = _SIZE_RE.search(model_name or "")
    if match is None:
        return 20  # unknown size — conservative middle ground

    try:
        size_b = float(match.group(1))
    except ValueError:
        return 20

    if size_b >= 70:
        return 40
    if size_b >= 30:
        return 30
    if size_b >= 14:
        return 22
    if size_b >= 7:
        return 16
    if size_b >= 3:
        return 12
    return 8


# Context window (num_ctx) heuristic. Default Ollama num_ctx is just 2048
# which silently truncates prompts — picking a sane per-model default avoids
# that footgun. Frontier APIs get their published windows; local models map
# to the typical native context for that family.
# Default fraction of the context window at which we trigger summarization.
# Leaves headroom for the summary call itself (system prompt, prior summary
# included as input, the conversation being summarized) and ~5 recent
# exchanges that survive untouched.
DEFAULT_SUMMARIZE_FRACTION = 0.6


_CTX_BY_FAMILY: List[Tuple[Tuple[str, ...], int]] = [
    # Tested in order — first match wins.
    (("claude", "gpt-4", "gpt-5", "gemini-", "o1-", "o3-"), 200_000),
    (("gpt-oss",), 128_000),
    (("deepseek-r1", "deepseek-v3"), 128_000),
    (("llama-4",), 1_000_000),
    (("llama-3.3",), 128_000),
    (("qwen3.5", "qwen3:", "qwen3-"), 32_768),
    (("gemma4", "gemma3", "gemma:"), 8_192),
    (("llama3.1", "llama3.2", "llama3:"), 8_192),
    (("phi4", "phi3"), 16_384),
]


def estimate_context_size(model_name: str) -> int:
    """Heuristic context-window (in tokens) for *model_name*.

    Used as ``num_ctx`` for Ollama-backed models and as a hint for
    client-side compaction logic. Explicit ``context_size`` in the model
    config overrides this.
    """
    name = (model_name or "").lower()
    if not name:
        return 8_192

    for prefixes, ctx in _CTX_BY_FAMILY:
        if any(p in name for p in prefixes):
            return ctx

    match = _SIZE_RE.search(model_name or "")
    if match is None:
        return 8_192

    try:
        size_b = float(match.group(1))
    except ValueError:
        return 8_192

    if size_b >= 70:
        return 32_768
    if size_b >= 14:
        return 16_384
    if size_b >= 7:
        return 8_192
    if size_b >= 3:
        return 4_096
    return 2_048


class _NoThinkWrapper:
    """Wraps an LLM and prepends /no_think to the system prompt."""

    def __init__(self, llm: Any):
        self._llm = llm

    def _inject(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        if messages and isinstance(messages[0], SystemMessage):
            messages = list(messages)
            messages[0] = SystemMessage(
                content="/no_think\n" + messages[0].content
            )
        else:
            messages = [SystemMessage(content="/no_think")] + list(messages)
        return messages

    async def ainvoke(self, messages: List[BaseMessage], **kwargs: Any) -> Any:
        return await self._llm.ainvoke(self._inject(messages), **kwargs)

    def invoke(self, messages: List[BaseMessage], **kwargs: Any) -> Any:
        return self._llm.invoke(self._inject(messages), **kwargs)

    def bind_tools(self, *args: Any, **kwargs: Any) -> "_NoThinkWrapper":
        return _NoThinkWrapper(self._llm.bind_tools(*args, **kwargs))

    async def astream(self, messages: List[BaseMessage], **kwargs: Any) -> Any:
        async for chunk in self._llm.astream(self._inject(messages), **kwargs):
            yield chunk

    def __getattr__(self, name: str) -> Any:
        return getattr(self._llm, name)


class _FallbackLLMWrapper:
    """Wrap multiple LLMs and fail over on runtime errors.

    Tracks failures across requests per model. After three failed requests,
    a model is skipped for 30 minutes before it is considered again.
    """

    _BLACKLIST_THRESHOLD = 3
    _BLACKLIST_COOLDOWN_SECONDS = 30 * 60

    def __init__(self, llms: List[Any], labels: List[str], context_sizes: List[int]):
        if not llms:
            raise ValueError("Fallback wrapper requires at least one LLM")
        self._llms = llms
        self._labels = labels
        self._context_sizes = context_sizes
        self._now = time.monotonic
        self._failures = [0 for _ in llms]
        self._blacklisted_until = [0.0 for _ in llms]
        self._last_errors: List[Optional[Exception]] = [None for _ in llms]

    @staticmethod
    def _estimate_tokens_for_messages(messages: List[BaseMessage]) -> int:
        total = 0
        for message in messages:
            content = message.content
            text = content if isinstance(content, str) else str(content)
            if text:
                total += (len(text) + _CHARS_PER_TOKEN - 1) // _CHARS_PER_TOKEN
        return total

    def _fits_context(self, idx: int, messages: List[BaseMessage]) -> bool:
        ctx = self._context_sizes[idx] if idx < len(self._context_sizes) else 0
        if ctx <= 0:
            return True
        budget = max(1, ctx - _CONTEXT_SKIP_RESERVE_TOKENS)
        return self._estimate_tokens_for_messages(messages) <= budget

    def _is_blacklisted(self, idx: int, now: float) -> bool:
        return self._blacklisted_until[idx] > now

    def _note_success(self, idx: int) -> None:
        self._failures[idx] = 0
        self._blacklisted_until[idx] = 0.0
        self._last_errors[idx] = None

    def _note_failure(self, idx: int, exc: Exception) -> None:
        self._last_errors[idx] = exc

        if is_context_length_error(exc):
            logger.warning(
                "LLM context limit: model '%s' rejected the prompt as too large",
                self._labels[idx],
            )
            return

        now = self._now()
        self._failures[idx] += 1
        if self._failures[idx] >= self._BLACKLIST_THRESHOLD:
            self._blacklisted_until[idx] = now + self._BLACKLIST_COOLDOWN_SECONDS
            self._failures[idx] = 0
            logger.warning(
                "LLM blacklist: model '%s' failed %d times across requests, skipping it for %d minutes",
                self._labels[idx],
                self._BLACKLIST_THRESHOLD,
                self._BLACKLIST_COOLDOWN_SECONDS // 60,
            )

    def _raise_if_all_blacklisted(self) -> None:
        now = self._now()
        active = [
            idx for idx in range(len(self._llms))
            if not self._is_blacklisted(idx, now)
        ]
        if active:
            return

        earliest_idx = min(
            range(len(self._llms)),
            key=lambda idx: self._blacklisted_until[idx],
        )
        remaining = max(0, int(self._blacklisted_until[earliest_idx] - now))
        last_exc = self._last_errors[earliest_idx]
        msg = (
            "All fallback models are temporarily blacklisted; "
            f"next retry for '{self._labels[earliest_idx]}' in about {remaining}s"
        )
        if last_exc is not None:
            raise RuntimeError(msg) from last_exc
        raise RuntimeError(msg)

    def invoke(self, messages: List[BaseMessage], **kwargs: Any) -> Any:
        last_exc: Optional[Exception] = None
        saw_context_error = False
        self._raise_if_all_blacklisted()
        for idx, llm in enumerate(self._llms):
            now = self._now()
            if self._is_blacklisted(idx, now):
                logger.info(
                    "LLM skip: model '%s' is temporarily blacklisted for %ds more",
                    self._labels[idx],
                    int(self._blacklisted_until[idx] - now),
                )
                continue
            if not self._fits_context(idx, messages):
                logger.warning(
                    "LLM skip: model '%s' context window too small for estimated prompt, trying next fallback",
                    self._labels[idx],
                )
                last_exc = RuntimeError(
                    f"Estimated prompt exceeds context window for model '{self._labels[idx]}'"
                )
                saw_context_error = True
                continue
            try:
                result = llm.invoke(messages, **kwargs)
                self._note_success(idx)
                return result
            except Exception as exc:
                last_exc = exc
                self._note_failure(idx, exc)
                saw_context_error = saw_context_error or is_context_length_error(exc)
            if idx < len(self._llms) - 1:
                if last_exc is not None and is_context_length_error(last_exc):
                    logger.warning(
                        "LLM fallback: model '%s' hit context limit (%s), trying '%s'",
                        self._labels[idx],
                        last_exc,
                        self._labels[idx + 1],
                    )
                else:
                    logger.warning(
                        "LLM fallback: model '%s' failed (%s), trying '%s'",
                        self._labels[idx],
                        last_exc,
                        self._labels[idx + 1],
                    )
            elif saw_context_error:
                logger.warning("LLM context limit: all fallback models rejected the prompt as too large")
        assert last_exc is not None
        raise last_exc

    async def ainvoke(self, messages: List[BaseMessage], **kwargs: Any) -> Any:
        last_exc: Optional[Exception] = None
        saw_context_error = False
        self._raise_if_all_blacklisted()
        for idx, llm in enumerate(self._llms):
            now = self._now()
            if self._is_blacklisted(idx, now):
                logger.info(
                    "LLM skip: model '%s' is temporarily blacklisted for %ds more",
                    self._labels[idx],
                    int(self._blacklisted_until[idx] - now),
                )
                continue
            if not self._fits_context(idx, messages):
                logger.warning(
                    "LLM skip: model '%s' context window too small for estimated prompt, trying next fallback",
                    self._labels[idx],
                )
                last_exc = RuntimeError(
                    f"Estimated prompt exceeds context window for model '{self._labels[idx]}'"
                )
                saw_context_error = True
                continue
            try:
                result = await llm.ainvoke(messages, **kwargs)
                self._note_success(idx)
                return result
            except Exception as exc:
                last_exc = exc
                self._note_failure(idx, exc)
                saw_context_error = saw_context_error or is_context_length_error(exc)
            if idx < len(self._llms) - 1:
                if last_exc is not None and is_context_length_error(last_exc):
                    logger.warning(
                        "LLM fallback: model '%s' hit context limit (%s), trying '%s'",
                        self._labels[idx],
                        last_exc,
                        self._labels[idx + 1],
                    )
                else:
                    logger.warning(
                        "LLM fallback: model '%s' failed (%s), trying '%s'",
                        self._labels[idx],
                        last_exc,
                        self._labels[idx + 1],
                    )
            elif saw_context_error:
                logger.warning("LLM context limit: all fallback models rejected the prompt as too large")
        assert last_exc is not None
        raise last_exc

    def bind_tools(self, *args: Any, **kwargs: Any) -> "_FallbackLLMWrapper":
        return _FallbackLLMWrapper(
            [llm.bind_tools(*args, **kwargs) for llm in self._llms],
            self._labels,
            self._context_sizes,
        )

    async def astream(self, messages: List[BaseMessage], **kwargs: Any) -> Any:
        last_exc: Optional[Exception] = None
        saw_context_error = False
        self._raise_if_all_blacklisted()
        for idx, llm in enumerate(self._llms):
            now = self._now()
            if self._is_blacklisted(idx, now):
                logger.info(
                    "LLM skip(stream): model '%s' is temporarily blacklisted for %ds more",
                    self._labels[idx],
                    int(self._blacklisted_until[idx] - now),
                )
                continue
            if not self._fits_context(idx, messages):
                logger.warning(
                    "LLM skip(stream): model '%s' context window too small for estimated prompt, trying next fallback",
                    self._labels[idx],
                )
                last_exc = RuntimeError(
                    f"Estimated prompt exceeds context window for model '{self._labels[idx]}'"
                )
                saw_context_error = True
                continue
            yielded_any = False
            try:
                async for chunk in llm.astream(messages, **kwargs):
                    yielded_any = True
                    yield chunk
                self._note_success(idx)
                return
            except Exception as exc:
                self._note_failure(idx, exc)
                if yielded_any:
                    raise
                last_exc = exc
                saw_context_error = saw_context_error or is_context_length_error(exc)
                if idx < len(self._llms) - 1:
                    if is_context_length_error(exc):
                        logger.warning(
                            "LLM fallback(stream): model '%s' hit context limit (%s), trying '%s'",
                            self._labels[idx],
                            exc,
                            self._labels[idx + 1],
                        )
                    else:
                        logger.warning(
                            "LLM fallback(stream): model '%s' failed (%s), trying '%s'",
                            self._labels[idx],
                            exc,
                            self._labels[idx + 1],
                        )
                elif saw_context_error:
                    logger.warning("LLM context limit: all fallback models rejected the prompt as too large")
        assert last_exc is not None
        raise last_exc

    def __getattr__(self, name: str) -> Any:
        return getattr(self._llms[0], name)


class LLMFactory:
    """Creates and caches LangChain LLM instances from config."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.providers: Dict[str, Dict[str, Any]] = config.get("providers", {})
        self.models: Dict[str, Dict[str, Any]] = config.get("models", {})
        self.agents_cfg: Dict[str, Any] = config.get("agents", {})
        self._cache: Dict[Tuple, Any] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, agent_type: str = "chat", agent_overrides: Optional[Dict[str, Any]] = None) -> Any:
        """Return a (cached) LLM for the given agent type."""
        model_cfgs = self._resolve_agent_candidates(
            agent_type,
            backend="pawlia",
            agent_overrides=agent_overrides,
        )
        if len(model_cfgs) == 1:
            return self._get_or_build_model(model_cfgs[0])

        key = ("fallback", tuple(self._cache_key(cfg) for cfg in model_cfgs))
        if key not in self._cache:
            llms = [self._get_or_build_model(cfg) for cfg in model_cfgs]
            labels = [str(cfg.get("model", "unknown")) for cfg in model_cfgs]
            context_sizes = [self.context_size_for_model(str(cfg.get("model", "unknown"))) for cfg in model_cfgs]
            self._cache[key] = _FallbackLLMWrapper(llms, labels, context_sizes)
        return self._cache[key]

    def resolve_model_name(self, name: str) -> str:
        """Resolve a config key (e.g. ``"fast"``) to its ``model`` value.

        If *name* is a known config key, returns the ``model`` field from that
        config.  Otherwise returns *name* unchanged.
        """
        cfg = self.models.get(name)
        if cfg and "model" in cfg:
            return cfg["model"]
        return name

    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Resolve *model_name* to its full model config without building an LLM."""
        if model_name in self.models:
            return dict(self.models[model_name])

        for _key, cfg in self.models.items():
            if cfg.get("model") == model_name:
                return dict(cfg)

        default = self._resolve_agent("default")
        return {**default, "model": model_name}

    def get_provider_name_for_model(self, model_name: str) -> str:
        model_cfg = self.get_model_config(model_name)
        return str(model_cfg.get("provider") or self._default_provider_name())

    def get_provider_config(self, provider_name: str) -> Dict[str, Any]:
        return dict(self._get_provider(provider_name))

    def get_backend_for_model(self, model_name: str) -> str:
        model_cfg = self.get_model_config(model_name)
        return self._provider_backend_from_cfg(model_cfg)

    def max_tool_turns_for_model(self, model_name: str) -> int:
        """Return per-model tool-call budget.

        Priority: explicit ``max_tool_turns`` in the model config →
        heuristic estimated from the model identifier. The caller (skill
        runner factory) further allows the skill itself to override this.
        """
        cfg = self.get_model_config(model_name)
        explicit = cfg.get("max_tool_turns")
        if isinstance(explicit, int) and explicit > 0:
            return explicit
        model_id = str(cfg.get("model") or model_name)
        return estimate_max_tool_turns(model_id)

    def context_size_for_model(self, model_name: str) -> int:
        """Return per-model context-window size (tokens).

        Priority: explicit ``context_size`` (or legacy ``num_ctx``) in the
        model config → heuristic from the model identifier.
        """
        cfg = self.get_model_config(model_name)
        for key in ("context_size", "num_ctx"):
            explicit = cfg.get(key)
            if isinstance(explicit, int) and explicit > 0:
                return explicit
        model_id = str(cfg.get("model") or model_name)
        return estimate_context_size(model_id)

    def summary_threshold_tokens(self, model_name: str) -> int:
        """Return the token threshold above which the conversation should
        be summarized.

        Priority:
        1. explicit ``summarize_at_tokens`` in the model config (absolute)
        2. ``summarize_at_fraction`` × context_size (per-model fraction)
        3. ``DEFAULT_SUMMARIZE_FRACTION`` × context_size

        The fraction default leaves headroom for the summary call itself
        plus a few user turns of recent exchanges.
        """
        cfg = self.get_model_config(model_name)

        explicit = cfg.get("summarize_at_tokens")
        if isinstance(explicit, int) and explicit > 0:
            return explicit

        fraction = cfg.get("summarize_at_fraction")
        if not isinstance(fraction, (int, float)) or fraction <= 0:
            fraction = DEFAULT_SUMMARIZE_FRACTION
        # Cap fraction at 0.95 so we never bump right up against the window.
        fraction = min(float(fraction), 0.95)

        ctx = self.context_size_for_model(model_name)
        return max(1, int(ctx * fraction))

    def default_model_name(
        self,
        agent_type: str = "chat",
        agent_overrides: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Return the first configured model selector for *agent_type*."""
        raw = self._resolve_agent_value_name(self._agent_value(agent_type, agent_overrides=agent_overrides))
        if raw:
            return raw

        fallback = self._fallback_agent(agent_type)
        if fallback:
            return self.default_model_name(fallback, agent_overrides=agent_overrides)

        if self.models:
            return next(iter(self.models))

        raise RuntimeError(
            f"Cannot resolve agent '{agent_type}': no models defined in config"
        )

    def get_with_model(self, model_name: str) -> Any:
        """Return a (cached) LLM by model name.

        *model_name* is first looked up in ``models:``.  If not found, a
        reverse lookup checks whether it matches the ``model`` field of any
        config entry.  If still unresolved, it is treated as a raw model
        identifier and the default provider is used.
        """
        model_cfg = self.get_model_config(model_name)
        backend = self._provider_backend_from_cfg(model_cfg)
        if backend != "pawlia":
            raise RuntimeError(
                f"Model '{model_name}' uses backend '{backend}' and cannot be built by LLMFactory"
            )
        key = self._cache_key(model_cfg)
        if key not in self._cache:
            self._cache[key] = self._build(model_cfg)
        return self._cache[key]

    # ------------------------------------------------------------------
    # Config resolution
    # ------------------------------------------------------------------

    def _resolve_agent(
        self,
        agent_type: str,
        agent_overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Resolve the model config for an agent type following fallback chains.

        Supports two config styles:

        *New* — models defined separately, agents reference them by key::

            models:
              fast: {model: qwen3:4b, provider: ollama}
            agents:
              default: fast
              chat: fast

        *Legacy* — inline model config inside each agent block::

            agents:
              defaults:           # note: plural accepted too
                model: qwen3:4b
                provider: ollama
              chat:
                model: qwen3.5:latest
        """
        candidates = self._resolve_agent_candidates(agent_type, agent_overrides=agent_overrides)
        return candidates[0]

    def _resolve_agent_candidates(
        self,
        agent_type: str,
        _visited: Optional[Set[str]] = None,
        backend: Optional[str] = None,
        agent_overrides: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Resolve one or more model configs for an agent type.

        Agent entries may contain comma-separated model references, e.g.
        ``chat: smart,fast``. Models are tried in that order.
        """
        visited = _visited or set()
        if agent_type in visited:
            return []
        visited = set(visited)
        visited.add(agent_type)

        value = self._agent_value(agent_type, agent_overrides=agent_overrides)
        resolved = self._resolve_agent_value(value)
        if backend is not None:
            resolved = [
                cfg for cfg in resolved
                if self._provider_backend_from_cfg(cfg) == backend
            ]
        if resolved:
            return resolved

        # Not found or unresolvable — walk up the fallback chain
        fallback = self._fallback_agent(agent_type)
        if fallback:
            fallback_resolved = self._resolve_agent_candidates(
                fallback,
                visited,
                backend=backend,
                agent_overrides=agent_overrides,
            )
            if fallback_resolved:
                return fallback_resolved

        # Last resort: use the first defined model in config
        if self.models:
            for cfg in self.models.values():
                if backend is None or self._provider_backend_from_cfg(cfg) == backend:
                    return [cfg]

        raise RuntimeError(
            f"Cannot resolve agent '{agent_type}': no models defined in config"
        )

    def _resolve_agent_value(self, value: Any) -> List[Dict[str, Any]]:
        """Resolve a raw agent config value into model config dicts."""
        if isinstance(value, dict):
            # Legacy: inline model config
            return [value]

        if not isinstance(value, str):
            return []

        parts = [p.strip() for p in value.split(",") if p.strip()]
        if not parts:
            return []

        configs: List[Dict[str, Any]] = []
        template = next(iter(self.models.values()), {}) if self.models else {}
        for part in parts:
            if part in self.models:
                configs.append(self.models[part])
                continue

            # Allow raw model names in agents config.
            raw_cfg = dict(template)
            raw_cfg["model"] = part
            raw_cfg.setdefault("provider", self._default_provider_name())
            raw_cfg.setdefault("temperature", 0.7)
            configs.append(raw_cfg)

        return configs

    def _agent_value(self, agent_type: str, agent_overrides: Optional[Dict[str, Any]] = None) -> Any:
        """Return the raw value assigned to an agent type (string key or inline dict)."""
        overrides = agent_overrides or {}

        # "default" accepts both "default" (new) and "defaults" (legacy plural)
        if agent_type == "default":
            if "default" in overrides:
                return overrides.get("default")
            if "defaults" in overrides:
                return overrides.get("defaults")
            return (
                self.agents_cfg.get("default")
                or self.agents_cfg.get("defaults")
            )

        if agent_type.startswith("skill."):
            skill_name = agent_type[len("skill."):]
            override_skills = overrides.get("skills", {})
            if isinstance(override_skills, dict) and skill_name in override_skills:
                return override_skills.get(skill_name)
            return self.agents_cfg.get("skills", {}).get(skill_name)

        if agent_type in overrides:
            return overrides.get(agent_type)
        return self.agents_cfg.get(agent_type)

    def _resolve_agent_value_name(self, value: Any) -> Optional[str]:
        """Return the first selector string for a raw agent config value."""
        if isinstance(value, str):
            parts = [p.strip() for p in value.split(",") if p.strip()]
            return parts[0] if parts else None
        if isinstance(value, dict):
            model = value.get("model")
            return str(model) if model else None
        return None

    def _fallback_agent(self, agent_type: str) -> Optional[str]:
        """Return the next agent type to try in the fallback chain."""
        if agent_type.startswith("skill."):
            return "skill_runner"
        if agent_type == "skill_runner":
            return "chat"
        if agent_type == "vision":
            return "chat"
        if agent_type == "chat":
            return "default"
        if agent_type == "compiler":
            return "default"
        return None

    # ------------------------------------------------------------------
    # Instance construction
    # ------------------------------------------------------------------

    def _build(self, model_cfg: Dict[str, Any]) -> Any:
        model = model_cfg.get("model", "llama3.1:latest")
        temperature = model_cfg.get("temperature", 0.7)
        provider_name = model_cfg.get("provider") or self._default_provider_name()
        provider_cfg = self._get_provider(provider_name)
        provider_backend = str(provider_cfg.get("backend") or "pawlia")

        if provider_backend != "pawlia":
            raise RuntimeError(
                f"Provider '{provider_name}' uses backend '{provider_backend}' and cannot be built by LLMFactory"
            )

        api_base = provider_cfg.get("apiBase", "").rstrip("/")
        api_key = provider_cfg.get("apiKey", "none")
        timeout = provider_cfg.get("timeout", 120)
        keep_alive = provider_cfg.get("keepAlive")

        logger.debug(
            "Creating LLM: model=%s provider=%s base=%s temp=%s",
            model, provider_name, api_base, temperature,
        )

        # Thinking / reasoning config
        think = model_cfg.get("think")  # true | false | int (token budget)

        if self._is_ollama(provider_name, api_base):
            ollama_base = api_base.removesuffix("/v1") or "http://localhost:11434"
            num_ctx = self._resolve_num_ctx(model_cfg, model)
            kwargs: Dict[str, Any] = dict(
                model=model, temperature=temperature, base_url=ollama_base,
                client_kwargs={"timeout": timeout},
            )
            if keep_alive is not None:
                kwargs["keep_alive"] = keep_alive
            if num_ctx is not None:
                kwargs["num_ctx"] = num_ctx
            return ChatOllama(**kwargs)

        extra_body: Dict[str, Any] = {}
        if isinstance(think, int):
            # Token budget for thinking
            extra_body["reasoning_format"] = "parsed"
            extra_body["reasoning_budget"] = think

        max_tokens = model_cfg.get("max_tokens")

        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            base_url=api_base,
            api_key=api_key,
            timeout=timeout,
            **({"max_tokens": max_tokens} if max_tokens else {}),
            **({"extra_body": extra_body} if extra_body else {}),
        )

        if think is False:
            llm = _NoThinkWrapper(llm)

        return llm

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def audio_model_info(self, model_or_agent: Optional[str] = None) -> Optional[Tuple[str, str]]:
        """Return ``(ollama_base_url, model_name)`` if the model supports native audio, else ``None``.

        *model_or_agent* can be:
        - a model config key (e.g. ``"gemma4"``)
        - a raw model name (e.g. ``"gemma4:e4b"``)
        - an agent type (e.g. ``"chat"``)
        - ``None`` → resolves ``"chat"`` agent type

        The model config must have ``audio_input: true`` to be eligible.
        """
        model_cfg = self._resolve_audio_cfg(model_or_agent)
        if model_cfg is None or not model_cfg.get("audio_input"):
            return None
        model = model_cfg.get("model", "")
        provider_name = model_cfg.get("provider") or self._default_provider_name()
        provider_cfg = self._get_provider(provider_name)
        api_base = provider_cfg.get("apiBase", "").rstrip("/")
        # Strip /v1 suffix — native Ollama endpoint lives at /api/chat
        ollama_base = api_base.removesuffix("/v1") or "http://localhost:11434"
        return ollama_base, model

    def _resolve_audio_cfg(self, model_or_agent: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Resolve a model_or_agent string to its model config dict."""
        if model_or_agent is None:
            return self._resolve_agent("chat")
        # Direct model config key
        if model_or_agent in self.models:
            return self.models[model_or_agent]
        # Reverse lookup by raw model name
        for _key, cfg in self.models.items():
            if cfg.get("model") == model_or_agent:
                return cfg
        # Agent type fallback
        try:
            return self._resolve_agent(model_or_agent)
        except RuntimeError:
            return None

    def _get_or_build_model(self, model_cfg: Dict[str, Any]) -> Any:
        key = self._cache_key(model_cfg)
        if key not in self._cache:
            self._cache[key] = self._build(model_cfg)
        return self._cache[key]

    def _cache_key(self, model_cfg: Dict[str, Any]) -> Tuple:
        provider_name = model_cfg.get("provider") or self._default_provider_name()
        provider_cfg = self._get_provider(provider_name)
        return (
            model_cfg.get("model", "llama3.1:latest"),
            provider_name,
            provider_cfg.get("apiBase", ""),
            provider_cfg.get("backend", "pawlia"),
            model_cfg.get("temperature", 0.7),
            model_cfg.get("think"),
            model_cfg.get("max_tokens"),
            model_cfg.get("context_size") or model_cfg.get("num_ctx"),
            provider_cfg.get("keepAlive"),
        )

    def _is_ollama(self, provider_name: str, api_base: str) -> bool:
        return "ollama" in provider_name.lower() or ":11434" in api_base

    def _resolve_num_ctx(self, model_cfg: Dict[str, Any], model: str) -> Optional[int]:
        """Resolve num_ctx for an Ollama-backed model.

        Explicit ``context_size`` / ``num_ctx`` in the model config wins;
        otherwise the size-based heuristic. Returns ``None`` only if the
        model_cfg has an explicit zero/negative, signalling "let Ollama's
        own default apply" — relevant when a user knowingly wants the
        2048-token default back.
        """
        for key in ("context_size", "num_ctx"):
            explicit = model_cfg.get(key)
            if isinstance(explicit, int):
                return explicit if explicit > 0 else None
        return estimate_context_size(model)

    def _get_provider(self, name: str) -> Dict[str, Any]:
        if name and name in self.providers:
            return self.providers[name]
        if self.providers:
            return next(iter(self.providers.values()))
        return {"apiBase": "http://localhost:11434/v1", "apiKey": "none"}

    def _provider_backend_from_cfg(self, model_cfg: Dict[str, Any]) -> str:
        provider_name = str(model_cfg.get("provider") or self._default_provider_name())
        provider_cfg = self._get_provider(provider_name)
        return str(provider_cfg.get("backend") or "pawlia")

    def _default_provider_name(self) -> str:
        if self.providers:
            return next(iter(self.providers))
        return "ollama"
