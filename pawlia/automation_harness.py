"""Standard harness for automation job scripts.

Automation jobs are deterministic scripts the scheduler runs (see
``pawlia/automation.py`` ``JobRunner``). A script gates with its own logic and
**prints nothing** when there is nothing to report — the scheduler then stays
silent. When there *is* something to say, the script prints it (that text
becomes the notification), optionally curating it through the LLM first.

This module is the toolkit a job script imports:

    from pawlia.automation_harness import get_params, emit, silent, llm_call

It is deliberately dependency-free (standard library only) and synchronous, so
it works inside the short-lived, sandboxed subprocess the scheduler spawns
(``ScriptExecutor`` puts the package root on ``PYTHONPATH`` and exports
``PAWLIA_CONFIG_PATH`` so ``llm_call`` can find a provider).

Design rules (see the project's automation guidance):
  - The deterministic gate decides *whether* there is anything to report.
  - ``llm_call`` is a step *inside* the script, used sparingly — never the
    thing that decides whether to notify.
  - Failures must be loud: let exceptions propagate (non-zero exit) so the
    scheduler surfaces them.
"""

import json
import os
import re
import sys
import time
import urllib.request
from typing import Any, Dict, Optional

__all__ = [
    "get_params",
    "session_dir",
    "user_id",
    "emit",
    "silent",
    "log",
    "llm_call",
    "SILENT_SENTINEL",
]

# Kept in sync with pawlia.automation.SILENT_SENTINEL. A script may print this
# marker instead of nothing to signal "stay silent" — handy when emitting an
# empty string is awkward.
SILENT_SENTINEL = "PAWLIA_SILENT"


# ---------------------------------------------------------------------------
# Context / params
# ---------------------------------------------------------------------------

def get_params() -> Dict[str, Any]:
    """Return the job's ``params`` dict (from the ``AUTOMATION_PARAMS`` env).

    Tolerant: returns ``{}`` when unset or unparseable, so a script can rely on
    ``get_params().get("city", "Berlin")`` style access.
    """
    raw = os.environ.get("AUTOMATION_PARAMS", "")
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except (ValueError, TypeError):
        return {}


def session_dir() -> str:
    """Absolute path to this user's session dir (or '' if unknown)."""
    return os.environ.get("PAWLIA_SESSION_DIR", "")


def user_id() -> str:
    """The user id the job runs for (or '' if unknown)."""
    return os.environ.get("PAWLIA_USER_ID", "")


# ---------------------------------------------------------------------------
# Output gating
# ---------------------------------------------------------------------------

def emit(text: str) -> None:
    """Report ``text`` — it becomes the notification the user receives.

    Only call this when there is genuinely something to say; calling it with an
    empty/whitespace string is treated as silence.
    """
    if text and text.strip():
        sys.stdout.write(text if text.endswith("\n") else text + "\n")
        sys.stdout.flush()


def silent() -> None:
    """Say nothing. A no-op that documents intent at the call site."""
    return None


def log(message: str) -> None:
    """Write a diagnostic line to stderr (never sent to the user)."""
    sys.stderr.write(str(message).rstrip("\n") + "\n")
    sys.stderr.flush()


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

def _load_config() -> Dict[str, Any]:
    path = os.environ.get("PAWLIA_CONFIG_PATH", "")
    candidates = [path] if path else []
    candidates += [
        os.path.join(os.getcwd(), "config.yaml"),
        os.path.join(os.getcwd(), "config.yml"),
    ]
    for cand in candidates:
        if cand and os.path.isfile(cand):
            try:
                import yaml  # PyYAML is a project dependency
                with open(cand, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:  # pragma: no cover - defensive
                raise RuntimeError(f"could not read LLM config at {cand}: {e}")
    return {}


def _resolve_endpoint(cfg: Dict[str, Any], model: Optional[str]) -> Dict[str, Any]:
    """Resolve {url, model, api_key, timeout} from the app config.

    Mirrors how ``LLMFactory`` maps agent → model → provider: the requested
    ``model`` (or the chat/default agent) selects a ``models`` entry, whose
    ``provider`` selects a ``providers`` entry with ``apiBase``/``apiKey``.
    """
    providers: Dict[str, Any] = cfg.get("providers", {}) or {}
    models: Dict[str, Any] = cfg.get("models", {}) or {}
    agents: Dict[str, Any] = cfg.get("agents", {}) or {}

    if not providers:
        raise RuntimeError("no LLM provider configured (config has no 'providers')")

    # Pick the model config key.
    key = model
    if not key:
        key = agents.get("chat") or agents.get("default")
    if not key and models:
        key = next(iter(models))

    # ``key`` may be a models-table key or a raw model name passed directly.
    model_cfg = models.get(key, {"model": key}) if key else {}
    model_name = model_cfg.get("model") or key
    if not model_name:
        raise RuntimeError("no LLM model configured (config has no 'models')")

    provider_name = model_cfg.get("provider") or next(iter(providers))
    provider_cfg = providers.get(provider_name) or next(iter(providers.values()))

    api_base = str(provider_cfg.get("apiBase", "")).rstrip("/")
    if not api_base:
        raise RuntimeError(f"provider '{provider_name}' has no apiBase")
    api_key = provider_cfg.get("apiKey", "")
    timeout = int(provider_cfg.get("timeout", 120))

    return {
        "url": f"{api_base}/chat/completions",
        "model": model_name,
        "api_key": api_key,
        "timeout": timeout,
    }


def llm_call(
    prompt: str,
    system: Optional[str] = None,
    *,
    model: Optional[str] = None,
    retries: int = 3,
    timeout: Optional[int] = None,
    temperature: float = 0.3,
) -> str:
    """Make a single LLM call and return the stripped text response.

    Reuses the app's config (``PAWLIA_CONFIG_PATH``) to find an OpenAI-compatible
    endpoint. Retries transient failures (network errors, 429, 5xx) with
    exponential backoff. Raises on misconfiguration or after exhausting retries —
    a job that needs the LLM should fail loudly, not return junk.
    """
    cfg = _load_config()
    ep = _resolve_endpoint(cfg, model)

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": ep["model"],
        "messages": messages,
        "temperature": temperature,
    }
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json", "User-Agent": "pawlia-automation"}
    api_key = ep["api_key"]
    if api_key and str(api_key).lower() != "none":
        headers["Authorization"] = f"Bearer {api_key}"

    req_timeout = timeout if timeout is not None else ep["timeout"]

    last_err: Optional[Exception] = None
    for attempt in range(max(1, retries)):
        try:
            req = urllib.request.Request(ep["url"], data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=req_timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            content = (
                data.get("choices", [{}])[0].get("message", {}).get("content", "")
            )
            content = re.sub(r"<think.*?</think>", "", content, flags=re.DOTALL).strip()
            if content:
                return content
            last_err = RuntimeError("LLM returned empty content")
        except urllib.error.HTTPError as e:  # type: ignore[attr-defined]
            last_err = e
            # Retry only on rate-limit / server errors.
            if e.code not in (429, 500, 502, 503, 504):
                raise
        except Exception as e:
            last_err = e
        if attempt < retries - 1:
            time.sleep(min(2 ** attempt, 8))

    raise RuntimeError(f"LLM call failed after {retries} attempt(s): {last_err}")
