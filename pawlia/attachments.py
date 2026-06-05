"""Per-user attachment store: incoming files/images → workspace/Downloads/.

All three chat interfaces (Matrix, Telegram, Discord) call
:func:`save_incoming` when they receive a file or image. Saved files are
tracked in ``<session>/<user_id>/downloads_index.json`` (kept outside the
workspace so it does not pollute the workspace git repo) so the LLM can
find them again later via the ``files`` skill and re-attach them to a
reply using the :class:`pawlia.tools.attach_file.AttachFileTool` direct
tool.
"""

import json
import mimetypes
import os
import time
import uuid
from dataclasses import asdict, dataclass
from typing import List, Optional


DOWNLOADS_SUBDIR = "Downloads"
INDEX_FILENAME = "downloads_index.json"
_INDEX_MAX_ENTRIES = 500


@dataclass
class AttachmentMeta:
    saved_as: str
    original_name: str
    mimetype: str
    source: str
    received_at: float
    size: int

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AttachmentMeta":
        return cls(
            saved_as=str(d.get("saved_as", "")),
            original_name=str(d.get("original_name", "")),
            mimetype=str(d.get("mimetype", "application/octet-stream")),
            source=str(d.get("source", "")),
            received_at=float(d.get("received_at", 0) or 0),
            size=int(d.get("size", 0) or 0),
        )


def workspace_dir(session_dir: str, user_id: str) -> str:
    return os.path.join(session_dir, user_id, "workspace")


def downloads_dir(session_dir: str, user_id: str) -> str:
    return os.path.join(workspace_dir(session_dir, user_id), DOWNLOADS_SUBDIR)


def index_path(session_dir: str, user_id: str) -> str:
    return os.path.join(session_dir, user_id, INDEX_FILENAME)


def _safe_filename(name: str) -> str:
    name = os.path.basename((name or "").strip())
    if not name or name in (".", ".."):
        name = f"file-{uuid.uuid4().hex[:8]}"
    name = name.replace("..", "_")
    return name[:200]


def _collision_safe_path(target_dir: str, desired: str) -> str:
    base, ext = os.path.splitext(desired)
    candidate = os.path.join(target_dir, desired)
    n = 0
    while os.path.exists(candidate):
        n += 1
        candidate = os.path.join(target_dir, f"{base}-{n}{ext}")
    return candidate


def save_incoming(
    *,
    session_dir: str,
    user_id: str,
    data: bytes,
    filename: str,
    source: str,
    mimetype: Optional[str] = None,
    max_bytes: int = 26214400,
) -> Optional[AttachmentMeta]:
    """Persist an incoming file/image to ``<workspace>/Downloads/`` and index it.

    Returns the :class:`AttachmentMeta` on success, or ``None`` if the payload
    is empty, exceeds ``max_bytes``, or could not be written.
    """
    if not data:
        return None
    if max_bytes and len(data) > max_bytes:
        return None

    safe = _safe_filename(filename)
    guessed = mimetype or mimetypes.guess_type(safe)[0] or "application/octet-stream"

    ddir = downloads_dir(session_dir, user_id)
    os.makedirs(ddir, exist_ok=True)

    target = _collision_safe_path(ddir, safe)
    saved_as = os.path.basename(target)
    try:
        with open(target, "wb") as f:
            f.write(data)
    except OSError:
        return None

    meta = AttachmentMeta(
        saved_as=saved_as,
        original_name=filename or saved_as,
        mimetype=guessed,
        source=source,
        received_at=time.time(),
        size=len(data),
    )
    _append_index(index_path(session_dir, user_id), meta)
    return meta


def _append_index(path: str, meta: AttachmentMeta) -> None:
    try:
        entries = list_index_file(path)
    except Exception:
        entries = []
    entries.append(meta.to_dict())
    if len(entries) > _INDEX_MAX_ENTRIES:
        entries = entries[-_INDEX_MAX_ENTRIES:]
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except OSError:
        pass


def list_index_file(path: str) -> list:
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def list_for_user(session_dir: str, user_id: str) -> List[AttachmentMeta]:
    return [AttachmentMeta.from_dict(e) for e in list_index_file(index_path(session_dir, user_id))]
