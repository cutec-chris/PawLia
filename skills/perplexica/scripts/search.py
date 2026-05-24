#!/usr/bin/env python3
"""Perplexica AI search CLI script. Outputs JSON results to stdout."""
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


def search(query: str, url: str, focus_mode: str = "webSearch", timeout: int = 60) -> dict:
    base_url = url.rstrip("/")
    if focus_mode not in VALID_FOCUS_MODES:
        focus_mode = "webSearch"
    payload = {
        "query": query,
        "focusMode": focus_mode,
        "optimizationMode": "balanced",
        "history": [],
    }
    resp = requests.post(f"{base_url}/api/search", json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    answer = data.get("message", "")
    sources = []
    for s in data.get("sources", []):
        meta = s.get("metadata", {})
        sources.append({
            "title": meta.get("title", ""),
            "url": meta.get("url", ""),
            "snippet": s.get("pageContent", "")[:300],
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

    if not sources:
        quality_hints.append("No sources were returned.")
    elif len(sources) == 1 and not sources[0]["url"]:
        quality_hints.append("Only one source with no URL — likely a generic response.")

    result = {
        "answer": answer,
        "sources": sources,
    }
    if quality_hints:
        result["quality"] = result_quality
        result["hints"] = quality_hints

    return result


def main():
    parser = argparse.ArgumentParser(description="Perplexica AI web search")
    parser.add_argument("--query", required=True, help="Search query")
    parser.add_argument("--url", help="Perplexica instance URL; defaults to skill-config.perplexica.url")
    parser.add_argument("--focus", default="webSearch", help="Focus mode (webSearch, academicSearch, youtubeSearch, redditSearch)")
    parser.add_argument("--timeout", type=int, default=60, help="Request timeout in seconds")
    args = parser.parse_args()

    try:
        url = args.url
        if not url:
            raw_config = os.environ.get("PAWLIA_SKILL_CONFIG", "{}")
            try:
                url = json.loads(raw_config).get("url", "")
            except json.JSONDecodeError:
                url = ""
        if not url:
            raise ValueError("Missing Perplexica URL. Set skill-config.perplexica.url or pass --url.")

        result = search(args.query, url, args.focus, args.timeout)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
