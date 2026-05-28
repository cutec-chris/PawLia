#!/usr/bin/env python3
"""Perplexica/Vane AI search CLI script. Outputs JSON results to stdout."""
import argparse
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


def resolve_models(url: str, config: dict, timeout: int = 10) -> tuple[dict, dict]:
    """Resolve chatModel and embeddingModel from config + provider list.

    Falls back to the first available model if not explicitly configured.
    """
    providers = get_providers(url, timeout)
    if not providers:
        raise RuntimeError("Could not fetch providers from the Vane/Perplexica instance.")

    chat_cfg = config.get("chat_model_provider") or config.get("chat_model_provider_name")
    chat_model = config.get("chat_model")
    emb_cfg = config.get("embedding_model_provider") or config.get("embedding_model_provider_name")
    emb_model = config.get("embedding_model")

    chat_model_obj = None
    embedding_model_obj = None

    if chat_cfg and chat_model:
        provider_id = find_provider_id(providers, chat_cfg)
        if provider_id:
            for p in providers:
                if p.get("id") == provider_id:
                    matched_key = find_model_key(p.get("chatModels", []), chat_model)
                    if matched_key:
                        chat_model_obj = {"providerId": provider_id, "key": matched_key}
                    break
    if not chat_model_obj:
        chat_model_obj = _pick_first_available(providers, "chatModels")

    if emb_cfg and emb_model:
        provider_id = find_provider_id(providers, emb_cfg)
        if provider_id:
            for p in providers:
                if p.get("id") == provider_id:
                    matched_key = find_model_key(p.get("embeddingModels", []), emb_model)
                    if matched_key:
                        embedding_model_obj = {"providerId": provider_id, "key": matched_key}
                    break
    if not embedding_model_obj:
        embedding_model_obj = _pick_first_available(providers, "embeddingModels")

    if not chat_model_obj:
        raise RuntimeError("No chat model available on the Vane/Perplexica instance.")
    if not embedding_model_obj:
        raise RuntimeError("No embedding model available on the Vane/Perplexica instance.")

    return chat_model_obj, embedding_model_obj


def search(query: str, url: str, focus_mode: str = "webSearch", timeout: int = 60,
           chat_model: dict | None = None, embedding_model: dict | None = None) -> dict:
    base_url = url.rstrip("/")
    if focus_mode not in VALID_FOCUS_MODES:
        focus_mode = "webSearch"

    sources = FOCUS_TO_SOURCES.get(focus_mode, ["web"])

    payload: dict = {
        "query": query,
        "sources": sources,
        "optimizationMode": "balanced",
        "history": [],
        "stream": False,
    }

    if chat_model:
        payload["chatModel"] = chat_model
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

    return result


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

        chat_model, embedding_model = resolve_models(url, config, timeout=10)
        result = search(args.query, url, args.focus, args.timeout, chat_model, embedding_model)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
