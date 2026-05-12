#!/usr/bin/env python3
"""Researcher skill — scrape and search research projects in the workspace.

Files land in workspace/research/{project}/ — no RAG backend, no DreamWiki.
The DreamWiki is fed exclusively by conversations; research insights flow in
organically when the user discusses their findings.

Usage:
    researcher.py create <name> <description>
    researcher.py list
    researcher.py add <project> <url> [depth]
    researcher.py query <project> <question>
    researcher.py delete <project>
    researcher.py rename <old_name> <new_name>

    (user_id via PAWLIA_USER_ID env var; legacy positional arg also supported)
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
_PROJECT_ROOT = _SKILL_DIR.parent.parent
_SESSION_DIR = (
    pathlib.Path(os.environ["PAWLIA_SESSION_DIR"])
    if "PAWLIA_SESSION_DIR" in os.environ
    else _PROJECT_ROOT / "session"
)

sys.path.insert(0, str(_PROJECT_ROOT))

USER_AGENT = "pawlia-researcher/1.0"

# Boilerplate filters for _scrape_and_save.
# Pages matching these are not worth keeping: the LLM downstream just sees
# noise and may draw wrong conclusions from incidental words.
_MIN_CONTENT_CHARS = 300
_CAPTCHA_MARKERS = (
    "please wait for verification",
    "checking your browser",
    "verifying you are human",
    "just a moment",
    "enable javascript",
    "are you a robot",
    "cloudflare ray id",
)


def _load_skill_config() -> dict:
    raw = os.environ.get("PAWLIA_SKILL_CONFIG")
    if raw:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    for candidate in (_PROJECT_ROOT / "config.yaml", _PROJECT_ROOT / "config.yml"):
        if candidate.is_file():
            with open(candidate, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            return (cfg.get("skill-config") or {}).get("researcher", {})
    return {}


CFG = _load_skill_config()

# ---------------------------------------------------------------------------
# Embed-based search (optional — falls back to keyword if no embedding config)
# ---------------------------------------------------------------------------

_CHUNK_SIZE = 800
_TOP_K = 6
_MIN_SCORE = 0.1


def _chunk_text(text: str) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0
    for para in paragraphs:
        if buf_len + len(para) > _CHUNK_SIZE and buf:
            chunks.append("\n\n".join(buf))
            buf = [buf[-1]]
            buf_len = len(buf[0])
        buf.append(para)
        buf_len += len(para)
    if buf:
        chunks.append("\n\n".join(buf))
    return chunks or [text[:_CHUNK_SIZE]]


async def _embed(texts: list[str]) -> list[list[float]]:
    """Embed texts via Ollama /api/embed or OpenAI-compatible /embeddings."""
    import urllib.request as _urllib
    cfg = CFG
    provider = cfg.get("embedding_provider", "")
    model = cfg.get("embedding_model", "")
    host = cfg.get("embedding_host", "http://localhost:11434")
    timeout = int(cfg.get("rag_embedding_timeout", 120))

    if not provider or not model:
        raise RuntimeError("no embedding config")

    if provider == "ollama":
        url = f"{host.rstrip('/')}/api/embed"
        payload = json.dumps({"model": model, "input": texts}).encode()
        req = _urllib.Request(url, data=payload,
                              headers={"Content-Type": "application/json"},
                              method="POST")
        def _do():
            with _urllib.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode())["embeddings"]
    else:
        base = cfg.get("embedding_base_url", host)
        api_key = cfg.get("embedding_api_key", "")
        url = f"{base.rstrip('/')}/embeddings"
        payload = json.dumps({"model": model, "input": texts}).encode()
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        req = _urllib.Request(url, data=payload, headers=headers, method="POST")
        def _do():
            with _urllib.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
                return [item["embedding"] for item in data["data"]]

    from pawlia.utils import run_sync_in_thread
    return await run_sync_in_thread(_do)


async def _build_index(project_path: pathlib.Path) -> tuple[object, list[dict]] | None:
    """Build or load a numpy embed index for all .md files in the project.

    Returns (vectors_np_array, metadata_list) or None if no embedding config.
    Index is rebuilt whenever any source .md is newer than the stored index.
    """
    try:
        import numpy as np
    except ImportError:
        return None

    index_dir = project_path / ".index"
    vec_path = index_dir / "vectors.npy"
    meta_path = index_dir / "chunks.json"

    md_files = sorted(
        f for f in project_path.glob("*.md") if f.name != "README.md"
    )
    if not md_files:
        return None

    newest_md = max(f.stat().st_mtime for f in md_files)
    index_mtime = meta_path.stat().st_mtime if meta_path.exists() else 0

    if index_mtime >= newest_md and vec_path.exists() and meta_path.exists():
        vectors = np.load(str(vec_path))
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        return vectors, meta

    # Rebuild
    all_chunks: list[str] = []
    all_meta: list[dict] = []
    for md_file in md_files:
        text = md_file.read_text(encoding="utf-8")
        for i, chunk in enumerate(_chunk_text(text)):
            all_chunks.append(chunk)
            all_meta.append({"file": md_file.name, "chunk_idx": i})

    try:
        embeddings = await _embed(all_chunks)
    except RuntimeError:
        return None

    vectors = np.array(embeddings, dtype="float32")
    if np.isnan(vectors).any():
        vectors = np.nan_to_num(vectors, nan=0.0)

    index_dir.mkdir(exist_ok=True)
    np.save(str(vec_path), vectors)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            [{"text": c, **m} for c, m in zip(all_chunks, all_meta)],
            f, ensure_ascii=False,
        )

    return vectors, [{"text": c, **m} for c, m in zip(all_chunks, all_meta)]


async def _search_embed(project_path: pathlib.Path, question: str) -> str | None:
    """Semantic search using embeddings. Returns None if no embedding config."""
    import numpy as np

    result = await _build_index(project_path)
    if result is None:
        return None

    vectors, meta = result
    try:
        q_vecs = await _embed([question])
    except RuntimeError:
        return None

    q_vec = np.array(q_vecs[0], dtype="float32")
    norms = np.linalg.norm(vectors, axis=1)
    q_norm = np.linalg.norm(q_vec)
    scores = np.zeros(len(meta))
    valid = norms > 0
    if valid.any() and q_norm > 0:
        scores[valid] = (vectors[valid] @ q_vec) / (norms[valid] * q_norm)

    top_idx = np.argsort(scores)[::-1][:_TOP_K]
    parts = [meta[i]["text"] for i in top_idx if scores[i] >= _MIN_SCORE]
    if not parts:
        return "Keine relevanten Informationen gefunden."
    return "\n\n---\n\n".join(parts)


_STOP_WORDS = frozenset(
    "der die das und oder ein eine ist war hat haben was wie wer wo wann "
    "warum ich du wir sie er es mit von zu für auf in an bei nach über "
    "unter vor hinter zwischen nicht auch noch schon nur aber denn wenn "
    "dass weil als ob the a an and or is was are were has have what how "
    "who where when why i you we they he she it with from to for on at "
    "by about do did not also".split()
)


def _search_keyword(project_path: pathlib.Path, question: str) -> str:
    query_words = set(re.split(r"\W+", question.lower())) - _STOP_WORDS - {""}
    scored: list[tuple[int, str]] = []
    for md_file in sorted(project_path.glob("*.md")):
        if md_file.name == "README.md":
            continue
        content = md_file.read_text(encoding="utf-8")
        score = sum(1 for w in query_words if w in content.lower())
        scored.append((score, content))
    scored.sort(key=lambda x: x[0], reverse=True)
    results = [c for s, c in scored if s > 0][:_TOP_K]
    if not results:
        return "Keine relevanten Informationen gefunden."
    return "\n\n---\n\n".join(r[:3000] for r in results)


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


async def _scrape_and_save(project_path: pathlib.Path, url: str) -> dict:
    """Scrape a URL, convert to markdown, save to workspace. No RAG backend."""
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
                return {"status": "skipped", "message": "already saved"}

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
            if not extracted:
                return {"status": "skipped", "reason": "no extractable content"}
            h2t = html2text.HTML2Text()
            h2t.ignore_links = False
            h2t.body_width = 0
            markdown_text = h2t.handle(extracted)

            # Drop boilerplate pages: captchas, JS-walls, language pickers, etc.
            lower = markdown_text.lower()
            for marker in _CAPTCHA_MARKERS:
                if marker in lower:
                    return {"status": "skipped", "reason": f"blocked page ({marker})"}
            plain_len = len(re.sub(r"\s+", " ", markdown_text).strip())
            if plain_len < _MIN_CONTENT_CHARS:
                return {
                    "status": "skipped",
                    "reason": f"content too short ({plain_len} chars)",
                }
        else:
            return {"status": "error", "message": f"unsupported content type: {content_type}"}

    markdown_text = f"# Version: {version}\n# URL: {url}\n\n{markdown_text}"
    await asyncio.to_thread(filename.write_text, markdown_text, encoding="utf-8")

    # Invalidate embed index so next query rebuilds it
    index_dir = project_path / ".index"
    if index_dir.exists():
        shutil.rmtree(index_dir)

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
            result = await _scrape_and_save(project_path, url)
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
    print(json.dumps({"status": "ok", "name": name, "path": str(path)}))


async def cmd_list(user_dir: pathlib.Path):
    if not user_dir.exists():
        print(json.dumps({"projects": []}))
        return
    projects = []
    for p in sorted(user_dir.iterdir()):
        if not p.is_dir():
            continue
        readme = p / "README.md"
        desc = ""
        if readme.exists():
            lines = readme.read_text(encoding="utf-8").splitlines()
            desc = lines[2].strip() if len(lines) > 2 else ""
        doc_count = len([f for f in p.glob("*.md") if f.name != "README.md"])
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
        result = await _scrape_and_save(path, url)
        print(json.dumps(result, ensure_ascii=False))


async def cmd_query(user_dir: pathlib.Path, project: str, question: str):
    path = user_dir / project
    if not path.exists():
        print(json.dumps({"error": f"Project '{project}' not found"}))
        sys.exit(1)

    result = await _search_embed(path, question)
    if result is None:
        result = _search_keyword(path, question)

    print(json.dumps({"result": result}, ensure_ascii=False))


async def cmd_delete(user_dir: pathlib.Path, project: str):
    path = user_dir / project
    if not path.exists():
        print(json.dumps({"error": f"Project '{project}' not found"}))
        sys.exit(1)
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

    user_dir = _SESSION_DIR / user_id / "workspace" / "research"
    user_dir.mkdir(parents=True, exist_ok=True)

    if command == "create":
        if len(args) < 2:
            print("Usage: researcher.py create <name> <description>", file=sys.stderr)
            sys.exit(1)
        await cmd_create(user_dir, args[0], " ".join(args[1:]))
    elif command == "list":
        await cmd_list(user_dir)
    elif command == "add":
        if len(args) < 2:
            print("Usage: researcher.py add <project> <url> [depth]", file=sys.stderr)
            sys.exit(1)
        depth = int(args[2]) if len(args) > 2 else 1
        await cmd_add(user_dir, args[0], args[1], depth)
    elif command == "query":
        if len(args) < 2:
            print("Usage: researcher.py query <project> <question>", file=sys.stderr)
            sys.exit(1)
        await cmd_query(user_dir, args[0], " ".join(args[1:]))
    elif command == "delete":
        if len(args) < 1:
            print("Usage: researcher.py delete <project>", file=sys.stderr)
            sys.exit(1)
        await cmd_delete(user_dir, args[0])
    elif command == "rename":
        if len(args) < 2:
            print("Usage: researcher.py rename <old> <new>", file=sys.stderr)
            sys.exit(1)
        await cmd_rename(user_dir, args[0], args[1])
    else:
        print(f"Unknown command: {command}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
