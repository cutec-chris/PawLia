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
from typing import Any, Dict, Optional, Tuple

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI


logger = logging.getLogger(__name__)


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

    def get(self, agent_type: str = "chat") -> Any:
        """Return a (cached) LLM for the given agent type."""
        model_cfg = self._resolve_agent(agent_type)
        key = self._cache_key(model_cfg)
        if key not in self._cache:
            self._cache[key] = self._build(model_cfg)
        return self._cache[key]

    def get_with_model(self, model_name: str) -> Any:
        """Return a (cached) LLM by model name.

        *model_name* is first looked up in ``models:``.  If not found it is
        treated as a raw model identifier and the default provider is used.
        """
        if model_name in self.models:
            model_cfg = self.models[model_name]
        else:
            # Raw model string — use default provider
            default = self._resolve_agent("default")
            model_cfg = {**default, "model": model_name}
        key = self._cache_key(model_cfg)
        if key not in self._cache:
            self._cache[key] = self._build(model_cfg)
        return self._cache[key]

    # ------------------------------------------------------------------
    # Config resolution
    # ------------------------------------------------------------------

    def _resolve_agent(self, agent_type: str) -> Dict[str, Any]:
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
        value = self._agent_value(agent_type)

        if isinstance(value, dict):
            # Legacy: inline model config
            return value

        if isinstance(value, str) and value in self.models:
            # New: named model key
            return self.models[value]

        # Not found or unresolvable — walk up the fallback chain
        fallback = self._fallback_agent(agent_type)
        if fallback:
            return self._resolve_agent(fallback)

        return {"model": "llama3.1:latest", "provider": self._default_provider_name()}

    def _agent_value(self, agent_type: str) -> Any:
        """Return the raw value assigned to an agent type (string key or inline dict)."""
        # "default" accepts both "default" (new) and "defaults" (legacy plural)
        if agent_type == "default":
            return (
                self.agents_cfg.get("default")
                or self.agents_cfg.get("defaults")
            )

        if agent_type.startswith("skill."):
            skill_name = agent_type[len("skill."):]
            return self.agents_cfg.get("skills", {}).get(skill_name)

        return self.agents_cfg.get(agent_type)

    def _fallback_agent(self, agent_type: str) -> Optional[str]:
        """Return the next agent type to try in the fallback chain."""
        if agent_type.startswith("skill."):
            return "skill_runner"
        if agent_type == "skill_runner":
            return "default"
        if agent_type == "vision":
            return "chat"
        if agent_type == "chat":
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

        api_base = provider_cfg.get("apiBase", "").rstrip("/")
        api_key = provider_cfg.get("apiKey", "none")
        timeout = provider_cfg.get("timeout", 120)
        keep_alive = provider_cfg.get("keepAlive")

        logger.debug(
            "Creating LLM: model=%s provider=%s base=%s temp=%s",
            model, provider_name, api_base, temperature,
        )

        if self._is_ollama(provider_name, api_base):
            ollama_base = api_base.removesuffix("/v1") or "http://localhost:11434"
            kwargs: Dict[str, Any] = dict(model=model, temperature=temperature, base_url=ollama_base)
            if keep_alive is not None:
                kwargs["keep_alive"] = keep_alive
            return ChatOllama(**kwargs)

        return ChatOpenAI(
            model=model,
            temperature=temperature,
            base_url=api_base,
            api_key=api_key,
            timeout=timeout,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _cache_key(self, model_cfg: Dict[str, Any]) -> Tuple:
        provider_name = model_cfg.get("provider") or self._default_provider_name()
        provider_cfg = self._get_provider(provider_name)
        return (
            model_cfg.get("model", "llama3.1:latest"),
            provider_cfg.get("apiBase", ""),
            model_cfg.get("temperature", 0.7),
        )

    def _is_ollama(self, provider_name: str, api_base: str) -> bool:
        return "ollama" in provider_name.lower() or ":11434" in api_base

    def _get_provider(self, name: str) -> Dict[str, Any]:
        if name and name in self.providers:
            return self.providers[name]
        if self.providers:
            return next(iter(self.providers.values()))
        return {"apiBase": "http://localhost:11434/v1", "apiKey": "none"}

    def _default_provider_name(self) -> str:
        if self.providers:
            return next(iter(self.providers))
        return "ollama"
