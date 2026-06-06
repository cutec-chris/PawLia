"""Probe a provider's API for a model's real context-window size.

Name-based heuristics go stale the moment a provider ships a new model
(``glm-5``, ``gemini-3``, ``mimo-*`` …): the window is silently
under-estimated and PawLia then skips a perfectly capable model as
"context window too small".  Where the provider exposes the window over
its API we read it directly instead of guessing.

Signals (tried in order, first positive int wins):

1. **Ollama** native ``POST /api/show`` → ``model_info["<arch>.context_length"]``
   (only attempted for Ollama-looking bases — port 11434).
2. **OpenAI-compatible** ``GET /models`` → the matching model entry's
   ``context_window`` (Groq) or ``context_length`` (OpenRouter) field.

Providers that expose neither (e.g. zai, opencodezen) yield ``None`` so
the caller falls back to an explicit ``context_size`` config or the
name heuristic.  All network errors are swallowed and return ``None`` —
probing is best-effort and must never break model construction.
"""

import json
import logging
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_TIMEOUT_S = 4.0

# Per-model fields that carry the *input* context window, in priority order.
# ``max_tokens`` is deliberately excluded — it usually means max *output*.
_CTX_FIELDS = (
    "context_window",      # Groq
    "context_length",      # OpenRouter
    "max_context_length",
    "max_input_tokens",
)


def _http_json(url: str, headers: Dict[str, str], payload: Optional[bytes] = None) -> Any:
    req = urllib.request.Request(url, data=payload, headers=headers)
    if payload is not None and "Content-Type" not in headers:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _as_positive_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):  # bool is an int subclass — reject it
        return None
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, float) and value > 0:
        return int(value)
    return None


def _from_models_list(data: Any, model_id: str) -> Optional[int]:
    """Extract a context window for *model_id* from an OpenAI ``/models`` body."""
    entries = data.get("data") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        return None
    target = None
    for entry in entries:
        if isinstance(entry, dict) and entry.get("id") == model_id:
            target = entry
            break
    if target is None:
        return None
    for field in _CTX_FIELDS:
        ctx = _as_positive_int(target.get(field))
        if ctx is not None:
            return ctx
    # Some providers nest it under top_provider / architecture metadata.
    nested = target.get("top_provider")
    if isinstance(nested, dict):
        ctx = _as_positive_int(nested.get("context_length"))
        if ctx is not None:
            return ctx
    return None


def _from_ollama_show(info: Any) -> Optional[int]:
    """Extract context length from an Ollama ``/api/show`` body."""
    if not isinstance(info, dict):
        return None
    model_info = info.get("model_info")
    if isinstance(model_info, dict):
        for key, value in model_info.items():
            if key.endswith(".context_length") or key == "context_length":
                ctx = _as_positive_int(value)
                if ctx is not None:
                    return ctx
    return None


def _ollama_root(api_base: str) -> str:
    """Strip a trailing ``/v1`` so the native ``/api/*`` endpoints resolve."""
    if api_base.endswith("/v1"):
        return api_base[:-3]
    idx = api_base.find("/v1")
    return api_base[:idx] if idx != -1 else api_base


def probe_context_window(provider_cfg: Dict[str, Any], model_id: str) -> Optional[int]:
    """Return *model_id*'s context window per the provider API, or ``None``.

    Best-effort and side-effect-free: any failure (network, 404, missing
    field, unknown provider) returns ``None`` for the caller to fall back.
    """
    api_base = str(provider_cfg.get("apiBase") or "").rstrip("/")
    if not api_base or not model_id:
        return None
    api_key = str(provider_cfg.get("apiKey") or "")
    headers = {"Accept": "application/json"}
    if api_key and api_key.lower() != "none":
        headers["Authorization"] = f"Bearer {api_key}"

    looks_like_ollama = "11434" in api_base

    # Ollama exposes the window only via the native /api/show endpoint.
    if looks_like_ollama:
        try:
            info = _http_json(
                f"{_ollama_root(api_base)}/api/show",
                headers,
                payload=json.dumps({"name": model_id}).encode("utf-8"),
            )
            ctx = _from_ollama_show(info)
            if ctx is not None:
                return ctx
        except (urllib.error.URLError, OSError, ValueError) as exc:
            logger.debug("context_probe: /api/show failed for %s: %s", model_id, exc)

    # OpenAI-compatible /models listing (Groq: context_window, OpenRouter: context_length).
    try:
        data = _http_json(f"{api_base}/models", headers)
        ctx = _from_models_list(data, model_id)
        if ctx is not None:
            return ctx
    except (urllib.error.URLError, OSError, ValueError) as exc:
        logger.debug("context_probe: /models failed for %s: %s", model_id, exc)

    return None
