#!/usr/bin/env python3
"""SearXNG search CLI script. Outputs JSON results to stdout."""
import argparse
import json
import os
import sys
import io
import requests

# Force UTF-8 on Windows to avoid charmap encoding errors
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


def search(query: str, limit: int, url: str, timeout: int = 30) -> list:
    base_url = url.rstrip("/search").rstrip("/")
    params = {
        "q": query,
        "format": "json",
        "pageno": 1,
        "safesearch": 0,
        "categories": "general",
    }
    resp = requests.get(f"{base_url}/search", params=params, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    results = []
    for r in data.get("results", [])[:limit]:
        results.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": r.get("content", ""),
        })
    return results


def main():
    parser = argparse.ArgumentParser(description="SearXNG web search")
    parser.add_argument("--query", required=True, help="Search query")
    parser.add_argument("--limit", type=int, default=5, help="Max results")
    parser.add_argument("--url", help="SearXNG instance URL; defaults to skill-config.searxng.url")
    parser.add_argument("--timeout", type=int, default=30, help="Request timeout in seconds")
    args = parser.parse_args()

    try:
        skill_config = {}
        raw_config = os.environ.get("PAWLIA_SKILL_CONFIG", "{}")
        try:
            skill_config = json.loads(raw_config)
        except json.JSONDecodeError:
            skill_config = {}

        url = args.url or skill_config.get("url", "")
        timeout = args.timeout
        if "timeout" in skill_config and "--timeout" not in sys.argv:
            timeout = int(skill_config["timeout"])
        if not url:
            raise ValueError("Missing SearXNG URL. Set skill-config.searxng.url or pass --url.")

        results = search(args.query, args.limit, url, timeout)
        print(json.dumps(results, ensure_ascii=False, indent=2))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
