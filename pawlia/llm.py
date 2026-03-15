"""LLM Factory - creates and caches LangChain ChatOpenAI instances from config.

Supports multiple providers and per-agent / per-skill configuration.

Config layout (YAML)::

    providers:
      ollama:
        apiBase: http://localhost:11434/v1
        apiKey: ollama
        timeout: 120
      groq:
        apiBase: https://api.groq.com/openai/v1
        apiKey: gsk_...

    agents:
      defaults:
        model: qwen3:4b
        provider: ollama
        think: false
        temperature: 0.7

      chat:
        model: qwen3.5:latest
        think: true

      skill_runner:         # default LLM for all skills
        model: qwen3.5:latest

      vision:               # falls back to chat if not set
        model: qwen2.5vl:latest

      skills:               # per-skill overrides (fall back to skill_runner)
        searxng:
          model: qwen3:4b
          provider: groq
        browser:
          model: llama3.1:8b

Fallback chains
---------------
- ``get("chat")``          → agents.chat      → defaults
- ``get("vision")``        → agents.vision    → agents.chat    → defaults
- ``get("skill_runner")``  → agents.skill_runner               → defaults
- ``get("skill.searxng")`` → agents.skills.searxng → agents.skill_runner → defaults

LLMs with identical resolved config (model + provider + temperature) are reused.
"""

import logging
from typing import Any, Dict, Optional, Tuple

from langchain_openai import ChatOpenAI


logger = logging.getLogger(__name__)


class LLMFactory:
    """Creates and caches ChatOpenAI instances for different agent / skill types."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.providers: Dict[str, Dict[str, Any]] = config.get("providers", {})
        self._cache: Dict[Tuple, ChatOpenAI] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, agent_type: str = "chat") -> ChatOpenAI:
        """Return a (cached) ChatOpenAI instance for the given agent type.

        Instances with identical model + provider + temperature are shared.
        """
        merged = self._resolve(agent_type)
        key = self._cache_key(merged)
        if key not in self._cache:
            self._cache[key] = self._build(agent_type, merged)
        return self._cache[key]

    # ------------------------------------------------------------------
    # Config resolution
    # ------------------------------------------------------------------

    def _resolve(self, agent_type: str) -> Dict[str, Any]:
        """Merge agent-specific config on top of defaults following fallback chain."""
        agents_cfg = self.config.get("agents", {})
        defaults = agents_cfg.get("defaults", {})

        specific = self._specific_cfg(agent_type, agents_cfg)
        return {**defaults, **specific}

    def _specific_cfg(self, agent_type: str, agents_cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Return the most-specific config block for agent_type, following fallbacks."""
        # Per-skill lookup: "skill.searxng" → agents.skills.searxng → skill_runner
        if agent_type.startswith("skill."):
            skill_name = agent_type[len("skill."):]
            skill_cfg = agents_cfg.get("skills", {}).get(skill_name)
            if skill_cfg:
                return skill_cfg
            return agents_cfg.get("skill_runner", {})

        # vision → chat
        if agent_type == "vision":
            return agents_cfg.get("vision") or agents_cfg.get("chat", {})

        return agents_cfg.get(agent_type, {})

    # ------------------------------------------------------------------
    # Instance construction
    # ------------------------------------------------------------------

    def _build(self, agent_type: str, merged: Dict[str, Any]) -> ChatOpenAI:
        model = merged.get("model", "llama3.1:latest")
        temperature = merged.get("temperature", 0.7)
        provider_name = merged.get("provider") or self._default_provider_name()
        provider_cfg = self._get_provider(provider_name)

        api_base = provider_cfg.get("apiBase", "").rstrip("/")
        api_key = provider_cfg.get("apiKey", "none")
        timeout = provider_cfg.get("timeout", 120)
        keep_alive = provider_cfg.get("keepAlive")

        logger.debug(
            "Creating LLM for '%s': model=%s provider=%s base=%s temp=%s",
            agent_type, model, provider_name, api_base, temperature,
        )

        model_kwargs: Dict[str, Any] = {}
        if keep_alive is not None:
            # Ollama-specific; must go via extra_body — not a standard OpenAI kwarg
            model_kwargs["extra_body"] = {"keep_alive": keep_alive}

        return ChatOpenAI(
            model=model,
            temperature=temperature,
            base_url=api_base,
            api_key=api_key,
            timeout=timeout,
            model_kwargs=model_kwargs or None,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _cache_key(self, merged: Dict[str, Any]) -> Tuple:
        provider_name = merged.get("provider") or self._default_provider_name()
        provider_cfg = self._get_provider(provider_name)
        return (
            merged.get("model", "llama3.1:latest"),
            provider_cfg.get("apiBase", ""),
            merged.get("temperature", 0.7),
        )

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
