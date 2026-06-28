"""Shared utility functions used across PawLia modules."""

import asyncio
import json
import logging
import os
import re
import threading
import unicodedata
import uuid
from difflib import SequenceMatcher
from typing import Any, Callable, Dict, List, Optional, TypeVar

import yaml

logger = logging.getLogger(__name__)
_T = TypeVar("_T")


# ---------------------------------------------------------------------------
# User-Agent handling
# ---------------------------------------------------------------------------
# Fixed UA for requests where PawLia identifies itself *as itself*: LLM /
# embedding / provider calls, search-API calls, transcription, internal
# services. Not configurable — some providers (e.g. Groq behind Cloudflare)
# 403 the default "Python-urllib" UA with error 1010, so we always send our own.
# Derived from the single source of truth pawlia.__version__ — never hardcode
# the version here. See agents.md › "Versioning & Releases (git-flow)".
from pawlia import __version__ as _VERSION
PAWLIA_USER_AGENT = f"PawLia/{_VERSION}"

# Default UA for browser-*emulating* web fetches (browser skill, researcher
# page scraping). Configurable via the top-level ``web_user_agent`` config key
# so a deploy can rotate the emulated browser. LLM/API calls never use this.
DEFAULT_WEB_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)

_web_ua_cache: Optional[str] = None


def _read_config_web_user_agent() -> str:
    """Read the top-level ``web_user_agent`` from the config file, or ''."""
    candidates: List[str] = []
    cfg_path = os.environ.get("PAWLIA_CONFIG_PATH")
    if cfg_path:
        candidates.append(cfg_path)
    candidates += ["config.yaml", "config.yml"]
    for path in candidates:
        try:
            if path and os.path.isfile(path):
                with open(path, encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                ua = cfg.get("web_user_agent")
                if isinstance(ua, str) and ua.strip():
                    return ua.strip()
        except Exception:
            continue
    return ""


def web_user_agent() -> str:
    """User-Agent for browser-emulating web fetches.

    Resolution order (first hit wins):
      1. ``$PAWLIA_WEB_USER_AGENT``
      2. top-level ``web_user_agent`` in the config file
         (``$PAWLIA_CONFIG_PATH``, else ``./config.yaml`` / ``./config.yml``)
      3. :data:`DEFAULT_WEB_USER_AGENT`

    Cached after first resolution. Tests can reset by setting
    ``pawlia.utils._web_ua_cache = None``.
    """
    global _web_ua_cache
    if _web_ua_cache is not None:
        return _web_ua_cache
    ua = os.environ.get("PAWLIA_WEB_USER_AGENT", "").strip()
    if not ua:
        ua = _read_config_web_user_agent()
    _web_ua_cache = ua or DEFAULT_WEB_USER_AGENT
    return _web_ua_cache


def _raise_invalid_dir(path: str) -> None:
    if os.path.islink(path):
        target = os.readlink(path)
        raise NotADirectoryError(
            f"{path} exists as a symlink but is not a usable directory. "
            f"Target inside current runtime: {target}"
        )
    raise NotADirectoryError(f"{path} exists but is not a directory")


# ---------------------------------------------------------------------------
# YAML frontmatter parsing
# ---------------------------------------------------------------------------

def parse_frontmatter(path: str) -> Optional[Dict[str, Any]]:
    """Parse YAML frontmatter from a Markdown file (e.g. SKILL.md).

    Returns the parsed dict, or None if no valid frontmatter is found.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    except OSError as e:
        logger.error("Cannot read %s: %s", path, e)
        return None

    lines = content.split("\n")
    frontmatter_lines: List[str] = []
    in_frontmatter = False

    for line in lines:
        if line.strip() == "---":
            if in_frontmatter:
                break
            in_frontmatter = True
            continue
        if in_frontmatter:
            frontmatter_lines.append(line)

    if not frontmatter_lines:
        return None

    try:
        return yaml.safe_load("\n".join(frontmatter_lines))
    except yaml.YAMLError as e:
        logger.error("Error parsing YAML in %s: %s", path, e)
        return None


# ---------------------------------------------------------------------------
# Skill directory discovery
# ---------------------------------------------------------------------------

def collect_skill_dirs(skills_dir: str) -> List[str]:
    """Collect skill directories: direct children + skills/user/*.

    Returns a list of absolute paths to directories that contain a SKILL.md.
    """
    candidates: List[str] = []
    if not os.path.isdir(skills_dir):
        return candidates

    for entry in os.listdir(skills_dir):
        entry_path = os.path.join(skills_dir, entry)
        if os.path.isdir(entry_path) and os.path.isfile(os.path.join(entry_path, "SKILL.md")):
            candidates.append(entry_path)

    user_dir = os.path.join(skills_dir, "user")
    if os.path.isdir(user_dir):
        for entry in os.listdir(user_dir):
            entry_path = os.path.join(user_dir, entry)
            if os.path.isdir(entry_path) and os.path.isfile(os.path.join(entry_path, "SKILL.md")):
                candidates.append(entry_path)

    return candidates


# ---------------------------------------------------------------------------
# JSON persistence helpers
# ---------------------------------------------------------------------------

def load_json(path: str) -> list:
    """Load a JSON array from *path*.  Returns [] on missing file or error."""
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.error("Failed to load %s: %s", path, e)
        return []


def save_json(path: str, data: list) -> None:
    """Write a JSON array to *path*, creating parent directories as needed."""
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def ensure_dir(path: str) -> str:
    """Ensure *path* exists as a directory, accepting symlinks to directories."""
    if os.path.isdir(path):
        return path

    if os.path.lexists(path):
        _raise_invalid_dir(path)

    try:
        os.makedirs(path, exist_ok=True)
    except FileExistsError:
        if os.path.isdir(path):
            return path
        _raise_invalid_dir(path)

    if not os.path.isdir(path):
        _raise_invalid_dir(path)

    return path


async def run_sync_in_thread(fn: Callable[..., _T], /, *args: Any, **kwargs: Any) -> _T:
    """Run a blocking callable in a plain thread and await the result.

    We avoid ``asyncio.to_thread`` / ``run_in_executor`` here because they hang
    in the current sandboxed Python 3.14 environment, while regular
    ``threading.Thread`` still works reliably.
    """
    loop = asyncio.get_running_loop()
    fut: asyncio.Future[_T] = loop.create_future()

    def _worker() -> None:
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:
            loop.call_soon_threadsafe(fut.set_exception, exc)
        else:
            loop.call_soon_threadsafe(fut.set_result, result)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    return await fut


# ---------------------------------------------------------------------------
# Script resolution
# ---------------------------------------------------------------------------

# ── Shared stop words (DE + EN) for keyword/query matching ──
_STOP_WORDS = frozenset(
    "der die das und oder ein eine ist war hat haben was wie wer wo wann "
    "warum ich du wir sie er es mit von zu für auf in an bei nach über "
    "unter vor hinter zwischen nicht auch noch schon nur aber denn wenn "
    "dass weil als ob the a an and or is was are were has have what how "
    "who where when why i you we they he she it with from to for on at "
    "by about do did not also".split()
)


SANITIZE_RE = re.compile(r'[<>:"|?*]')

# Characters that are *invalid* in a filename — used for detection
_INVALID_FILENAME_RE = re.compile(r'[<>:"|?*]')


def sanitize_filename(name: str) -> str:
    """Remove characters that are invalid on Windows/Android filesystems.

    ``<> : " | ? *`` are forbidden on FAT32/NTFS/exFAT (Android's emulated
    storage/sdcard) and cause ``git checkout`` to fail silently.
    Only the *basename* is sanitised — path separators (``/``) are preserved.
    """
    parent, leaf = os.path.split(name)
    leaf = unicodedata.normalize("NFC", leaf)
    leaf = SANITIZE_RE.sub("", leaf)
    leaf = leaf.strip(". ")
    clean = os.path.join(parent, leaf) if parent else leaf
    return clean or f"file-{uuid.uuid4().hex[:8]}"


def sanitize_workspace(workspace_dir: str) -> None:
    """Rename files/dirs with invalid characters (``<>:"|?*``) to safe names.

    Walks the workspace bottom-up so directories are processed after their
    children, avoiding stale parent references during rename.
    """
    if not os.path.isdir(workspace_dir):
        return
    entries: list[str] = []
    for root, dirs, files in os.walk(workspace_dir, topdown=False):
        for name in files + dirs:
            if _INVALID_FILENAME_RE.search(name):
                entries.append(os.path.join(root, name))
    if not entries:
        return
    logger = logging.getLogger("pawlia.utils")
    for path in sorted(entries, key=len, reverse=True):
        parent = os.path.dirname(path)
        name = os.path.basename(path)
        safe = sanitize_filename(name)
        if safe == name:
            continue
        safe_path = os.path.join(parent, safe)
        # Collision: append uuid suffix
        if os.path.exists(safe_path):
            base, ext = os.path.splitext(safe)
            safe_path = os.path.join(
                parent, f"{base}-{uuid.uuid4().hex[:6]}{ext}"
            )
        try:
            os.rename(path, safe_path)
            logger.warning("Renamed problematic file: %s -> %s", path, safe_path)
        except OSError as e:
            logger.error("Failed to rename %s: %s", path, e)


def slugify(name: str) -> str:
    """Convert a topic name to a filesystem-safe slug."""
    slug = name.lower().strip()
    for old, new in (("ä", "ae"), ("ö", "oe"), ("ü", "ue"), ("ß", "ss")):
        slug = slug.replace(old, new)
    slug = unicodedata.normalize("NFKD", slug)
    slug = slug.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", slug).strip("-")
    return slug[:80] or "misc"


def find_similar_slug(
    new_slug: str, existing_slugs: list[str], threshold: float = 0.7
) -> Optional[str]:
    """Return the most similar existing slug if similarity >= threshold."""
    best_slug: Optional[str] = None
    best_ratio = 0.0
    for existing in existing_slugs:
        ratio = SequenceMatcher(None, new_slug, existing).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_slug = existing
    return best_slug if best_ratio >= threshold else None


async def rag_llm_call(cfg: dict, system_prompt: str, user_prompt: str,
                      json_mode: bool | str = False) -> str:
    """Make a single LLM call for RAG backends and return the stripped content.

    json_mode asks the provider for JSON output. Accepts:
      - False (default): no JSON hint.
      - True / "object": expect a JSON object.
      - "array": expect a top-level JSON array.

    Ollama gets ``format: "json"`` for any truthy value (it does not constrain
    the top-level type). OpenAI-compatible providers only get
    ``response_format: {type: "json_object"}`` for object mode — that mode
    *forbids* a top-level array, so for "array" we send no response_format and
    rely on the prompt + tolerant parser. This is a hint, not a guarantee — the
    parser still tolerates prose-wrapped JSON so non-conforming models keep working.
    """
    import urllib.request

    provider = cfg.get("rag_provider", cfg.get("embedding_provider", "ollama"))
    model = cfg.get("rag_model", "qwen3.5:latest")
    host = cfg.get("embedding_host", "http://localhost:11434")

    if provider == "ollama":
        url = f"{host.rstrip('/')}/api/chat"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {
                "num_ctx": int(cfg.get("rag_numctx", 4096)),
                "temperature": 0.1,
            },
        }
        if json_mode:
            payload["format"] = "json"
    else:
        base = cfg.get("rag_base_url", cfg.get("embedding_base_url", host))
        url = f"{base.rstrip('/')}/chat/completions"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
        }
        # json_object mode forbids a top-level array, so only enable it when an
        # object is expected. For "array" we rely on the prompt + parser.
        if json_mode is True or json_mode == "object":
            payload["response_format"] = {"type": "json_object"}

    body = json.dumps(payload).encode()
    headers: dict[str, str] = {
        "Content-Type": "application/json",
        "User-Agent": PAWLIA_USER_AGENT,
    }
    if provider != "ollama":
        api_key = cfg.get("rag_api_key", cfg.get("embedding_api_key", ""))
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(url, data=body, headers=headers, method="POST")

    def _do():
        timeout = int(cfg.get("rag_timeout", 600))
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())

    result = await run_sync_in_thread(_do)

    if provider == "ollama":
        content = result.get("message", {}).get("content", "")
    else:
        content = (
            result.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )

    return re.sub(r"<think.*?</think>", "", content, flags=re.DOTALL).strip()


def resolve_script(session_dir: str, user_id: str, script: str) -> str:
    """Resolve a script name to an absolute path.

    Search order (first hit wins):
      1. ``workspace/skills/scripts/<script>`` — primary, where skill-creator
         writes automation scripts.
      2. ``workspace/.scripts/<script>`` — legacy, kept for older jobs.
      3. ``automations/<script>`` — pre-skills legacy location.

    Returns the primary path even if no file exists, so the caller surfaces
    a clear "not found" instead of silently picking a wrong directory.
    """
    base = os.path.join(session_dir, user_id)
    candidates = [
        os.path.join(base, "workspace", "skills", "scripts", script),
        os.path.join(base, "workspace", ".scripts", script),
        os.path.join(base, "automations", script),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return candidates[0]
