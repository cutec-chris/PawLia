"""FakeLLMFactory — a minimal LLMFactory stand-in for RouterAgent tests.

RouterAgent only asks the factory which *backend* a model uses (to choose the
local ChatAgent stack vs. an external Hermes server) and for model/provider
config. This fake answers those questions without any real provider wiring.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class FakeLLMFactory:
    def __init__(
        self,
        *,
        models: Optional[Dict[str, dict]] = None,
        providers: Optional[Dict[str, dict]] = None,
        default: str = "pawlia_model",
    ):
        self.models = models or {
            "pawlia_model": {"model": "m-local", "provider": "ollama", "temperature": 0.3},
            "hermes": {"model": "hermes-agent", "provider": "hermes_local"},
        }
        self.providers = providers or {
            "ollama": {"backend": "pawlia", "apiBase": "http://ollama.test/v1"},
            "hermes_local": {
                "backend": "hermes",
                "apiBase": "http://hermes.test/v1",
                "apiKey": "secret",
                "conversation_namespace": "pawlia",
            },
        }
        self._default = default

    def default_model_name(self, agent_type: str = "chat", agent_overrides=None) -> str:
        if agent_overrides and agent_overrides.get(agent_type):
            return agent_overrides[agent_type]
        return self._default

    def get_model_config(self, model_name: str) -> dict:
        return dict(self.models[model_name])

    def get_provider_name_for_model(self, model_name: str) -> str:
        return str(self.models[model_name]["provider"])

    def get_provider_config(self, provider_name: str) -> dict:
        return dict(self.providers[provider_name])

    def get_backend_for_model(self, model_name: str) -> str:
        provider = self.get_provider_name_for_model(model_name)
        return str(self.providers[provider]["backend"])
