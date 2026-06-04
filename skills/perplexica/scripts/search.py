#!/usr/bin/env python3
"""Perplexica/Vane AI search CLI script. Outputs JSON results to stdout."""
import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
import io
import requests

# Force UTF-8 on Windows to avoid charmap encoding errors
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Identify ourselves on the search-API calls (not a browser-emulating fetch).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
try:
    from pawlia.utils import PAWLIA_USER_AGENT as _UA
except Exception:
    _UA = "PawLia/0.0.0+unknown"  # degraded path: pawlia not importable


VALID_FOCUS_MODES = {"webSearch", "academicSearch", "writingAssistant", "wolframAlphaSearch", "youtubeSearch", "redditSearch"}

FOCUS_TO_SOURCES = {
    "webSearch": ["web"],
    "academicSearch": ["academic"],
    "writingAssistant": ["web"],
    "wolframAlphaSearch": ["web"],
    "youtubeSearch": ["web"],
    "redditSearch": ["discussions"],
}


def _cache_path() -> str:
    """Return the cache file path inside the session directory."""
    session_dir = os.environ.get("PAWLIA_SESSION_DIR", "")
    user_id = os.environ.get("PAWLIA_USER_ID", "")
    if session_dir and user_id:
        return os.path.join(session_dir, user_id, ".perplexica.json")
    elif session_dir:
        return os.path.join(session_dir, ".perplexica.json")
    else:
        return os.path.join(os.path.expanduser("~"), ".cache", "pawlia", "perplexica-model-cache.json")


def _cache_key(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def _load_cache(url: str) -> dict | None:
    """Load cached chatModel + embeddingModel for a URL."""
    try:
        path = _cache_path()
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            entry = data.get(_cache_key(url), {})
            if entry.get("chatModel") and entry.get("embeddingModel"):
                return entry
    except Exception:
        pass
    return None


def _save_cache(url: str, chat_model: dict, embedding_model: dict):
    """Save chatModel + embeddingModel pair for a URL."""
    if not chat_model or not embedding_model:
        return
    try:
        path = _cache_path()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {}
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        data[_cache_key(url)] = {"chatModel": chat_model, "embeddingModel": embedding_model}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def get_providers(url: str, timeout: int = 10) -> list[dict]:
    """Fetch available providers and models from Vane/Perplexica."""
    resp = requests.get(f"{url.rstrip('/')}/api/providers", timeout=timeout,
                        headers={"User-Agent": _UA})
    resp.raise_for_status()
    data = resp.json()
    return data.get("providers", [])


def find_provider_id(providers: list[dict], name: str) -> str | None:
    """Find a provider ID by name (case-insensitive)."""
    name_lower = name.lower()
    for p in providers:
        if p.get("name", "").lower() == name_lower:
            return p.get("id")
    return None


def find_model_key(models: list[dict], key_or_name: str) -> str | None:
    """Find a model key by key or display name (case-insensitive)."""
    key_lower = key_or_name.lower()
    for m in models:
        if m.get("key", "").lower() == key_lower or m.get("name", "").lower() == key_lower:
            return m.get("key")
    return None


def _pick_first_available(providers: list[dict], model_key: str) -> dict | None:
    """Pick the first available model from any provider."""
    for p in providers:
        models = p.get(model_key, [])
        if models:
            return {"providerId": p["id"], "key": models[0]["key"]}
    return None


def _is_chat_model(key: str) -> bool:
    """Filter out guard/safety/speech models that aren't meant for chat."""
    k = key.lower()
    return not any(tag in k for tag in ("safeguard", "prompt-guard", "whisper"))


def _model_quality_score(key: str) -> int:
    """Higher score = larger/better model, which we now prefer."""
    k = key.lower()
    if "120b" in k:
        return 7
    if "llama-4" in k:
        return 6
    if "70b" in k:
        return 5
    if "32b" in k:
        return 4
    if "compound" in k:
        return 3
    # Check speed/size suffixes before raw size tags so "8b-instant" scores 0 not 2
    if "instant" in k:
        return 0
    if "mini" in k or "turbo" in k:
        return 1
    if "8b" in k:
        return 2
    return 3


def resolve_chat_models(providers: list[dict], config: dict, cached: dict | None = None) -> list[dict]:
    """Build a prioritized list of chat model dicts to try."""
    chat_cfg = config.get("chat_model_provider") or config.get("chat_model_provider_name")
    chat_model = config.get("chat_model")

    result = []
    seen_keys = set()

    # 0. Cached model (fastest from previous run) always goes first
    if cached and cached.get("providerId") and cached.get("key"):
        result.append(dict(cached))
        seen_keys.add((cached["providerId"], cached["key"]))

    # 1. Explicitly configured model
    if chat_cfg and chat_model:
        provider_id = find_provider_id(providers, chat_cfg)
        if provider_id:
            for p in providers:
                if p.get("id") == provider_id:
                    matched_key = find_model_key(p.get("chatModels", []), chat_model)
                    if matched_key and (provider_id, matched_key) not in seen_keys:
                        result.append({"providerId": provider_id, "key": matched_key})
                        seen_keys.add((provider_id, matched_key))
                    break

    # 2. All remaining chat models from all providers, sorted by quality preference
    remaining = []
    for p in providers:
        pid = p.get("id")
        for m in p.get("chatModels", []):
            key = m.get("key")
            if key and _is_chat_model(key) and (pid, key) not in seen_keys:
                remaining.append({"providerId": pid, "key": key})
    remaining.sort(key=lambda m: _model_quality_score(m["key"]), reverse=True)
    result.extend(remaining)

    return result


def resolve_embedding_model(providers: list[dict], config: dict) -> dict:
    """Resolve embedding model from config or pick first available."""
    emb_cfg = config.get("embedding_model_provider") or config.get("embedding_model_provider_name")
    emb_model = config.get("embedding_model")

    if emb_cfg and emb_model:
        provider_id = find_provider_id(providers, emb_cfg)
        if provider_id:
            for p in providers:
                if p.get("id") == provider_id:
                    matched_key = find_model_key(p.get("embeddingModels", []), emb_model)
                    if matched_key:
                        return {"providerId": provider_id, "key": matched_key}

    fallback = _pick_first_available(providers, "embeddingModels")
    if fallback:
        return fallback
    raise RuntimeError("No embedding model available on the Vane/Perplexica instance.")


def _search_single(query: str, base_url: str, sources: list[str], focus_mode: str,
                   chat_model: dict, embedding_model: dict, timeout: int) -> dict:
    """Perform a single search request with a specific chat model."""
    payload: dict = {
        "query": query,
        "sources": sources,        # Vane API
        "focusMode": focus_mode,   # standard Perplexica API fallback
        "optimizationMode": "speed",
        "history": [],
        "stream": False,
        "chatModel": chat_model,
    }
    if embedding_model:
        payload["embeddingModel"] = embedding_model

    resp = requests.post(f"{base_url}/api/search", json=payload, timeout=timeout,
                         headers={"User-Agent": _UA})
    resp.raise_for_status()
    data = resp.json()

    answer = data.get("message", "")
    raw_sources = data.get("sources", [])
    sources_out = []
    for s in raw_sources:
        meta = s.get("metadata", {})
        sources_out.append({
            "title": meta.get("title", ""),
            "url": meta.get("url", ""),
            "snippet": s.get("content", s.get("pageContent", ""))[:300],
        })

    # Detect unhelpful/empty results so the LLM can react appropriately
    result_quality = "good"
    quality_hints = []
    if not answer or len(answer.strip()) < 20:
        result_quality = "empty"
        quality_hints.append("Answer is empty or very short — no information found.")
    elif any(phrase in answer.lower() for phrase in (
        "leider konnte ich keine",
        "konnte leider keine",
        "keine spezifischen informationen",
        "no specific information",
        "could not find",
        "couldn't find",
        "i was unable",
        "i could not",
        "no results found",
        "no information available",
        "nicht gefunden",
    )):
        result_quality = "no_results"
        quality_hints.append("Search returned no useful results for this query.")

    if not sources_out:
        quality_hints.append("No sources were returned.")
    elif len(sources_out) == 1 and not sources_out[0]["url"]:
        quality_hints.append("Only one source with no URL — likely a generic response.")

    result = {
        "answer": answer,
        "sources": sources_out,
    }
    if quality_hints:
        result["quality"] = result_quality
        result["hints"] = quality_hints

    model_label = f"{chat_model.get('providerId', '?')}/{chat_model.get('key', '?')}"
    result["model_used"] = model_label
    return result


def search(query: str, url: str, focus_mode: str = "webSearch", timeout: int = 60,
           chat_models: list[dict] | None = None, embedding_model: dict | None = None) -> dict:
    base_url = url.rstrip("/")
    if focus_mode not in VALID_FOCUS_MODES:
        focus_mode = "webSearch"

    sources = FOCUS_TO_SOURCES.get(focus_mode, ["web"])
    models_to_try = chat_models or []
    if not models_to_try:
        raise RuntimeError("No chat models available to try.")

    errors = []

    def _worker(chat_model: dict):
        try:
            return _search_single(query, base_url, sources, focus_mode, chat_model, embedding_model, timeout)
        except Exception as e:
            label = f"{chat_model.get('providerId', '?')}/{chat_model.get('key', '?')}"
            return {"_error": True, "_label": label, "_exc": str(e)}

    # Cap concurrency: no need to hammer the server with more than 5 parallel probes.
    max_workers = min(len(models_to_try), 5)
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    futures = {executor.submit(_worker, m): m for m in models_to_try}
    try:
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result.get("_error"):
                errors.append(f"{result['_label']}: {result['_exc']}")
                continue
            # Only cache a model that returned a useful answer (quality key absent = good)
            if "quality" not in result:
                _save_cache(url, futures[future], embedding_model)
            return result
    finally:
        executor.shutdown(wait=False)

    raise RuntimeError(f"All chat models failed. Errors: {'; '.join(errors)}")


def main():
    parser = argparse.ArgumentParser(description="Perplexica/Vane AI web search")
    parser.add_argument("--query", required=True, help="Search query")
    parser.add_argument("--url", help="Vane/Perplexica instance URL; defaults to skill-config.perplexica.url")
    parser.add_argument("--focus", default="webSearch", help="Focus mode (webSearch, academicSearch, youtubeSearch, redditSearch)")
    parser.add_argument("--timeout", type=int, default=60, help="Request timeout in seconds")
    args = parser.parse_args()

    try:
        url = args.url
        config = {}
        if not url:
            raw_config = os.environ.get("PAWLIA_SKILL_CONFIG", "{}")
            try:
                config = json.loads(raw_config)
                url = config.get("url", "")
            except json.JSONDecodeError:
                url = ""
        if not url:
            raise ValueError("Missing Perplexica/Vane URL. Set skill-config.perplexica.url or pass --url.")

        cache = _load_cache(url)
        if cache:
            # Fast path: we already know both models, skip the /api/providers round-trip
            chat_models = [cache["chatModel"]]
            embedding_model = cache["embeddingModel"]
            try:
                result = search(args.query, url, args.focus, args.timeout, chat_models, embedding_model)
                print(json.dumps(result, ensure_ascii=False, indent=2))
                return
            except (RuntimeError, requests.RequestException, OSError):
                # Cached model failed (network/API error), fall through to discovery
                pass

        # Discovery path: fetch providers, then probe models
        providers = get_providers(url, timeout=10)
        if not providers:
            raise RuntimeError("Could not fetch providers from the Vane/Perplexica instance.")

        # Don't re-insert the just-failed cached model at position 0
        chat_models = resolve_chat_models(providers, config, cached=None)
        if not chat_models:
            raise RuntimeError("No chat models available on the Vane/Perplexica instance.")

        embedding_model = resolve_embedding_model(providers, config)

        result = search(args.query, url, args.focus, args.timeout, chat_models, embedding_model)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
