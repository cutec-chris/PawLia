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
    """Return the cache file path inside the session directory (like other configs)."""
    session_dir = os.environ.get("PAWLIA_SESSION_DIR", "")
    user_id = os.environ.get("PAWLIA_USER_ID", "")
    if session_dir and user_id:
        cache_dir = os.path.join(session_dir, user_id, ".cache")
    elif session_dir:
        cache_dir = os.path.join(session_dir, ".cache")
    else:
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "pawlia")
    return os.path.join(cache_dir, "perplexica-model-cache.json")


def _cache_key(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def _load_cached_model(url: str) -> dict | None:
    cache_file = _cache_path()
    try:
        if os.path.exists(cache_file):
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get(_cache_key(url))
    except Exception:
        pass
    return None


def _save_cached_model(url: str, model: dict):
    cache_file = _cache_path()
    try:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        data = {}
        if os.path.exists(cache_file):
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        data[_cache_key(url)] = model
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def get_providers(url: str, timeout: int = 10) -> list[dict]:
    """Fetch available providers and models from Vane/Perplexica."""
    resp = requests.get(f"{url.rstrip('/')}/api/providers", timeout=timeout)
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


def _model_speed_score(key: str) -> int:
    """Lower score = faster/better priority for simple queries."""
    k = key.lower()
    if "instant" in k:
        return 0
    if "mini" in k or "turbo" in k:
        return 1
    if "8b" in k:
        return 2
    if "32b" in k or "70b" in k:
        return 3
    if "compound" in k:
        return 5
    if "whisper" in k:
        return 10
    return 4


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

    # 2. All remaining chat models from all providers, sorted by speed preference
    remaining = []
    for p in providers:
        pid = p.get("id")
        for m in p.get("chatModels", []):
            key = m.get("key")
            if key and (pid, key) not in seen_keys:
                remaining.append({"providerId": pid, "key": key})
    remaining.sort(key=lambda m: _model_speed_score(m["key"]))
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


def _search_single(query: str, base_url: str, sources: list[str], chat_model: dict,
                   embedding_model: dict, timeout: int) -> dict:
    """Perform a single search request with a specific chat model."""
    payload: dict = {
        "query": query,
        "sources": sources,
        "optimizationMode": "speed",
        "history": [],
        "stream": False,
        "chatModel": chat_model,
    }
    if embedding_model:
        payload["embeddingModel"] = embedding_model

    resp = requests.post(f"{base_url}/api/search", json=payload, timeout=timeout)
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
           chat_models: list[dict] | None = None, embedding_model: dict | None = None,
           cached_model: dict | None = None) -> dict:
    base_url = url.rstrip("/")
    if focus_mode not in VALID_FOCUS_MODES:
        focus_mode = "webSearch"

    sources = FOCUS_TO_SOURCES.get(focus_mode, ["web"])
    models_to_try = chat_models or []
    if not models_to_try:
        raise RuntimeError("No chat models available to try.")

    # Only probe the top 5 fastest models so we don't queue 16 requests.
    models_to_try = models_to_try[:5]

    # If we have a cached model, try it first alone (fast path)
    if cached_model and cached_model in models_to_try:
        try:
            result = _search_single(query, base_url, sources, cached_model, embedding_model, timeout)
            _save_cached_model(url, cached_model)
            return result
        except Exception:
            # Cached model failed, fall through to parallel sweep
            pass

    errors = []

    def _worker(chat_model: dict):
        try:
            return _search_single(query, base_url, sources, chat_model, embedding_model, timeout)
        except Exception as e:
            label = f"{chat_model.get('providerId', '?')}/{chat_model.get('key', '?')}"
            return {"_error": True, "_label": label, "_exc": str(e)}

    # Fire the remaining candidates all in parallel so the fastest one wins
    # immediately without waiting for a slow model to free up a worker slot.
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(models_to_try)) as executor:
        futures = {executor.submit(_worker, m): m for m in models_to_try}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result.get("_error"):
                errors.append(f"{result['_label']}: {result['_exc']}")
                continue
            # Remember the working model for next time
            _save_cached_model(url, futures[future])
            return result

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

        providers = get_providers(url, timeout=10)
        if not providers:
            raise RuntimeError("Could not fetch providers from the Vane/Perplexica instance.")

        cached = _load_cached_model(url)
        chat_models = resolve_chat_models(providers, config, cached=cached)
        if not chat_models:
            raise RuntimeError("No chat models available on the Vane/Perplexica instance.")

        embedding_model = resolve_embedding_model(providers, config)

        result = search(args.query, url, args.focus, args.timeout, chat_models, embedding_model, cached_model=cached)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
