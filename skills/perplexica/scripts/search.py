#!/usr/bin/env python3
"""Perplexica AI search CLI script. Outputs JSON results to stdout."""
import argparse
import json
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

    sources = []
    for s in data.get("sources", []):
        meta = s.get("metadata", {})
        sources.append({
            "title": meta.get("title", ""),
            "url": meta.get("url", ""),
            "snippet": s.get("pageContent", "")[:300],
        })

    return {
        "answer": data.get("message", ""),
        "sources": sources,
    }


def main():
    parser = argparse.ArgumentParser(description="Perplexica AI web search")
    parser.add_argument("--query", required=True, help="Search query")
    parser.add_argument("--url", required=True, help="Perplexica instance URL")
    parser.add_argument("--focus", default="webSearch", help="Focus mode (webSearch, academicSearch, youtubeSearch, redditSearch)")
    parser.add_argument("--timeout", type=int, default=60, help="Request timeout in seconds")
    args = parser.parse_args()

    try:
        result = search(args.query, args.url, args.focus, args.timeout)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
