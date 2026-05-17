"""Shared utility functions used across PawLia modules."""

import asyncio
import json
import logging
import os
import threading
from typing import Any, Callable, Dict, List, Optional, TypeVar

import yaml

logger = logging.getLogger(__name__)
_T = TypeVar("_T")


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

def resolve_script(session_dir: str, user_id: str, script: str) -> str:
    """Resolve a script name to an absolute path.

    Checks workspace/.scripts/ first (LLM-visible), then automations/ (legacy).
    """
    workspace_scripts = os.path.join(session_dir, user_id, "workspace", ".scripts", script)
    if os.path.isfile(workspace_scripts):
        return workspace_scripts
    return os.path.join(session_dir, user_id, "automations", script)
