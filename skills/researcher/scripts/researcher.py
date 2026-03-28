#!/usr/bin/env python3
"""Researcher skill — manage RAG-backed research projects.

Usage:
    researcher.py <user_id> create <name> <description>
    researcher.py <user_id> list
    researcher.py <user_id> add <project> <url> [depth]
    researcher.py <user_id> query <project> <question>
    researcher.py <user_id> delete <project>
    researcher.py <user_id> rename <old_name> <new_name>
"""

import asyncio
import hashlib
import json
import logging
import os
import pathlib
import shutil
import sys
import urllib.parse
import re

import bs4
import html2text
import requests
import trafilatura
import yaml

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------

_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_SKILL_DIR = _SCRIPT_DIR.parent
_PROJECT_ROOT = _SKILL_DIR.parent.parent  # thalia/
_SESSION_DIR = pathlib.Path(os.environ["PAWLIA_SESSION_DIR"]) if "PAWLIA_SESSION_DIR" in os.environ else _PROJECT_ROOT / "session"

sys.path.insert(0, str(_PROJECT_ROOT))

USER_AGENT = "pawlia-researcher/1.0"


def _load_skill_config() -> dict:
    """Load researcher config from config.yaml -> skill-config.researcher."""
    for candidate in (
        _PROJECT_ROOT / "config.yaml",
        _PROJECT_ROOT / "config.yml",
    ):
        if candidate.is_file():
            with open(candidate, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            skill_config_root = cfg.get("skill-config") or {}
            return skill_config_root.get("researcher", {})
    return {}


CFG = _load_skill_config()

# ---------------------------------------------------------------------------
# Per-project backend cache
# ---------------------------------------------------------------------------

_backends: dict[str, "RagBackend"] = {}


async def _get_backend(project_path: pathlib.Path):
    from pawlia.rag_backend import create_backend

    key = str(project_path)
    if key not in _backends:
        index_path = project_path / ".rag_index"
        index_path.mkdir(exist_ok=True)
        _backends[key] = create_backend(
            str(index_path),
            CFG,
            think=False,
        )
    return _backends[key]


# ---------------------------------------------------------------------------
# Content extraction
# ---------------------------------------------------------------------------

def _get_video_id(url: str):
    parsed = urllib.parse.urlparse(url)
    if parsed.query:
        qs = urllib.parse.parse_qs(parsed.query)
        if "v" in qs:
            return qs["v"][0]
    if parsed.netloc in ("youtu.be", "www.youtu.be"):
        return parsed.path.strip("/")
    m = re.match(r"^/live/([a-zA-Z0-9_-]{11})", parsed.path)
    if m:
        return m.group(1)
    return None


async def _youtube_to_markdown(url: str) -> str:
    from youtube_transcript_api import YouTubeTranscriptApi
    video_id = _get_video_id(url)
    def fetch():
        api = YouTubeTranscriptApi()
        return api.fetch(video_id, languages=["de", "en"], preserve_formatting=True)
    transcript = await asyncio.to_thread(fetch)
    md = f"# YouTube Transcript: {url}\n\n"
    md += "\n".join(s.text for s in transcript)
    return md


async def _pdf_to_markdown(path: pathlib.Path) -> str:
    import pdfminer.high_level
    return await asyncio.to_thread(pdfminer.high_level.extract_text, path)


def _extract_links(html: str, base_url: str) -> list[str]:
    soup = bs4.BeautifulSoup(html, "html.parser")
    base_domain = urllib.parse.urlparse(base_url).netloc
    links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        next_url = urllib.parse.urljoin(base_url, href)
        parsed = urllib.parse.urlparse(next_url)
        if parsed.netloc != base_domain:
            continue
        if any(x in next_url for x in ["#", "?", "login", "signup", "privacy", "terms", "contact"]):
            continue
        if not parsed.scheme.startswith("http"):
            continue
        links.add(next_url)
    return list(links)


async def _scrape_and_index(project_path: pathlib.Path, url: str) -> dict:
    """Scrape a single URL, convert to markdown, index via RAG backend."""
    headers = {"User-Agent": USER_AGENT}
    url_hash = hashlib.sha1(url.encode()).hexdigest()
    filename = project_path / f"{url_hash}.md"

    video_id = _get_video_id(url)

    if video_id:
        markdown_text = await _youtube_to_markdown(url)
        version = video_id
    else:
        def head():
            return requests.head(url, headers=headers, timeout=10, allow_redirects=True)
        head_resp = await asyncio.to_thread(head)
        if head_resp.status_code >= 400:
            return {"status": "error", "message": f"HTTP {head_resp.status_code}"}

        size = head_resp.headers.get("Content-Length", "unknown")
        last_mod = head_resp.headers.get("Last-Modified")
        version = f"size{size}"
        if last_mod:
            version += f"--{last_mod.replace(' ', '_')}"
        content_type = head_resp.headers.get("Content-Type", "")

        if filename.exists():
            current = await asyncio.to_thread(filename.read_text, encoding="utf-8")
            if current.startswith(f"# Version: {version}"):
                return {"status": "skipped", "message": "already indexed"}

        def get():
            return requests.get(url, headers=headers, timeout=30)
        resp = await asyncio.to_thread(get)
        if resp.status_code != 200:
            return {"status": "error", "message": f"HTTP {resp.status_code}"}

        if "application/pdf" in content_type or url.lower().endswith(".pdf"):
            pdf_path = project_path / f"{url_hash}.pdf"
            await asyncio.to_thread(pdf_path.write_bytes, resp.content)
            markdown_text = await _pdf_to_markdown(pdf_path)
        elif "text/" in content_type:
            extracted = trafilatura.extract(
                resp.text,
                include_formatting=True, include_links=True,
                include_tables=True, include_images=False,
                output_format="html",
            )
            h2t = html2text.HTML2Text()
            h2t.ignore_links = False
            h2t.body_width = 0
            markdown_text = h2t.handle(extracted) if extracted else resp.text
        else:
            return {"status": "error", "message": f"unsupported content type: {content_type}"}

    markdown_text = f"# Version: {version}\n# URL: {url}\n\n{markdown_text}"

    backend = await _get_backend(project_path)
    await backend.insert(markdown_text, url)
    await backend.wait_for_indexed(url, timeout=60, poll_interval=1.0)

    await asyncio.to_thread(filename.write_text, markdown_text, encoding="utf-8")
    return {"status": "ok", "file": str(filename), "version": version}


async def _scrape_recursive(project_path: pathlib.Path, base_url: str, max_depth: int = 1):
    visited = set()
    queue = [(base_url, 0)]
    results = []
    while queue:
        url, depth = queue.pop(0)
        if url in visited or depth > max_depth:
            continue
        visited.add(url)
        print(f"[depth={depth}] {url}", file=sys.stderr)
        try:
            resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=10)
            if resp.status_code == 429:
                await asyncio.sleep(60)
                queue.insert(0, (url, depth))
                visited.discard(url)
                continue
            if resp.status_code != 200 or "text/html" not in resp.headers.get("Content-Type", ""):
                continue
        except Exception as e:
            print(f"Error fetching {url}: {e}", file=sys.stderr)
            continue

        if depth < max_depth:
            for link in _extract_links(resp.text, url):
                if link not in visited:
                    queue.append((link, depth + 1))

        try:
            result = await _scrape_and_index(project_path, url)
            results.append({"url": url, **result})
        except Exception as e:
            results.append({"url": url, "status": "error", "message": str(e)})

    return results


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

async def cmd_create(user_dir: pathlib.Path, name: str, description: str):
    path = user_dir / name
    if path.exists():
        print(json.dumps({"error": f"Project '{name}' already exists"}))
        sys.exit(1)
    path.mkdir(parents=True)
    (path / "README.md").write_text(f"# {name}\n\n{description}\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "name": name}))


async def cmd_list(user_dir: pathlib.Path):
    if not user_dir.exists():
        print(json.dumps({"projects": []}))
        return
    projects = []
    for p in sorted(user_dir.iterdir()):
        if p.is_dir():
            readme = p / "README.md"
            desc = ""
            if readme.exists():
                lines = readme.read_text(encoding="utf-8").splitlines()
                desc = lines[2].strip() if len(lines) > 2 else ""
            doc_count = len(list(p.glob("*.md"))) - (1 if readme.exists() else 0)
            projects.append({"name": p.name, "description": desc, "documents": doc_count})
    print(json.dumps({"projects": projects}, ensure_ascii=False))


async def cmd_add(user_dir: pathlib.Path, project: str, url: str, depth: int = 1):
    path = user_dir / project
    if not path.exists():
        print(json.dumps({"error": f"Project '{project}' not found"}))
        sys.exit(1)

    if depth > 1:
        results = await _scrape_recursive(path, url, depth)
        print(json.dumps({"status": "ok", "results": results}, ensure_ascii=False))
    else:
        result = await _scrape_and_index(path, url)
        print(json.dumps(result, ensure_ascii=False))


async def cmd_query(user_dir: pathlib.Path, project: str, question: str):
    path = user_dir / project
    if not path.exists():
        print(json.dumps({"error": f"Project '{project}' not found"}))
        sys.exit(1)

    backend = await _get_backend(path)
    result = await backend.query(question)
    print(json.dumps({"result": result}, ensure_ascii=False))


async def cmd_delete(user_dir: pathlib.Path, project: str):
    path = user_dir / project
    if not path.exists():
        print(json.dumps({"error": f"Project '{project}' not found"}))
        sys.exit(1)
    _backends.pop(str(path), None)
    shutil.rmtree(path)
    print(json.dumps({"status": "ok", "message": f"Project '{project}' deleted"}))


async def cmd_rename(user_dir: pathlib.Path, old_name: str, new_name: str):
    old_path = user_dir / old_name
    new_path = user_dir / new_name
    if not old_path.exists():
        print(json.dumps({"error": f"Project '{old_name}' not found"}))
        sys.exit(1)
    if new_path.exists():
        print(json.dumps({"error": f"Project '{new_name}' already exists"}))
        sys.exit(1)
    old_path.rename(new_path)
    print(json.dumps({"status": "ok", "message": f"Renamed '{old_name}' to '{new_name}'"}))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    env_user_id = os.environ.get("PAWLIA_USER_ID")

    if env_user_id and len(sys.argv) >= 2 and sys.argv[1] in (
        "create", "list", "add", "query", "delete", "rename"
    ):
        user_id = env_user_id
        command = sys.argv[1]
        args = sys.argv[2:]
    elif len(sys.argv) >= 3:
        user_id = sys.argv[1]
        command = sys.argv[2]
        args = sys.argv[3:]
    else:
        print("Usage: researcher.py [<user_id>] <command> [args...]", file=sys.stderr)
        print("       (user_id can be set via PAWLIA_USER_ID env var)", file=sys.stderr)
        sys.exit(1)

    user_dir = _SESSION_DIR / user_id / "researches"
    user_dir.mkdir(parents=True, exist_ok=True)

    if command == "create":
        if len(args) < 2:
            print("Usage: researcher.py <user_id> create <name> <description>", file=sys.stderr)
            sys.exit(1)
        await cmd_create(user_dir, args[0], " ".join(args[1:]))
    elif command == "list":
        await cmd_list(user_dir)
    elif command == "add":
        if len(args) < 2:
            print("Usage: researcher.py <user_id> add <project> <url> [depth]", file=sys.stderr)
            sys.exit(1)
        depth = int(args[2]) if len(args) > 2 else 1
        await cmd_add(user_dir, args[0], args[1], depth)
    elif command == "query":
        if len(args) < 2:
            print("Usage: researcher.py <user_id> query <project> <question>", file=sys.stderr)
            sys.exit(1)
        await cmd_query(user_dir, args[0], " ".join(args[1:]))
    elif command == "delete":
        if len(args) < 1:
            print("Usage: researcher.py <user_id> delete <project>", file=sys.stderr)
            sys.exit(1)
        await cmd_delete(user_dir, args[0])
    elif command == "rename":
        if len(args) < 2:
            print("Usage: researcher.py <user_id> rename <old> <new>", file=sys.stderr)
            sys.exit(1)
        await cmd_rename(user_dir, args[0], args[1])
    else:
        print(f"Unknown command: {command}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
