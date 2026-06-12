"""Per-user attachment store: incoming files/images → workspace/Downloads/.

All three chat interfaces (Matrix, Telegram, Discord) call
:func:`save_incoming` when they receive a file or image. Each saved file gets
a sidecar markdown file next to it (``<saved_as>.md``) holding its metadata
plus a textual description (vision description for images, extracted text for
PDFs/documents). Because the sidecar is plain markdown inside the workspace,
the ``files`` skill finds it via grep/search/read — and the LLM can re-attach
the original file to a reply with
:class:`pawlia.tools.attach_file.AttachFileTool`.

The sidecars replace the former ``downloads_index.json``: the metadata now
lives as searchable workspace content instead of an opaque JSON blob.
"""

import json
import mimetypes
import os
import time
import uuid
from dataclasses import asdict, dataclass
from typing import List, Optional


DOWNLOADS_SUBDIR = "Downloads"
SIDECAR_SUFFIX = ".md"


@dataclass
class AttachmentMeta:
    saved_as: str
    original_name: str
    mimetype: str
    source: str
    received_at: float
    size: int
    description: str = ""

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
            description=str(d.get("description", "") or ""),
        )


def workspace_dir(session_dir: str, user_id: str) -> str:
    return os.path.join(session_dir, user_id, "workspace")


def downloads_dir(session_dir: str, user_id: str) -> str:
    return os.path.join(workspace_dir(session_dir, user_id), DOWNLOADS_SUBDIR)


def sidecar_path(file_path: str) -> str:
    """Path of the markdown sidecar that describes *file_path*."""
    return file_path + SIDECAR_SUFFIX


def _safe_filename(name: str) -> str:
    name = os.path.basename((name or "").strip())
    if not name or name in (".", ".."):
        name = f"file-{uuid.uuid4().hex[:8]}"
    from pawlia.utils import sanitize_filename
    name = sanitize_filename(name)
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
    description: Optional[str] = None,
    max_bytes: int = 26214400,
) -> Optional[AttachmentMeta]:
    """Persist an incoming file/image to ``<workspace>/Downloads/`` + a sidecar.

    *description* (optional) is a textual representation of the attachment —
    a vision description for images, extracted markdown/text for documents —
    written into the sidecar so the file becomes findable by content.

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
        description=(description or "").strip(),
    )
    _write_sidecar(target, meta)
    return meta


def _write_sidecar(file_path: str, meta: AttachmentMeta) -> None:
    """Write/overwrite the markdown sidecar describing *file_path*."""
    fm_keys = ("original_name", "saved_as", "mimetype", "source", "received_at", "size")
    md = meta.to_dict()
    lines = ["---"]
    for key in fm_keys:
        lines.append(f"{key}: {json.dumps(md[key], ensure_ascii=False)}")
    lines.append("---")
    lines.append("")
    lines.append(f"# Anhang: {meta.original_name}")
    lines.append("")
    lines.append(meta.description or "_(keine Beschreibung verfügbar)_")
    lines.append("")
    try:
        with open(sidecar_path(file_path), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    except OSError:
        pass


def _parse_sidecar(path: str) -> Optional[AttachmentMeta]:
    """Parse a sidecar markdown file back into :class:`AttachmentMeta`."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return None
    if not text.startswith("---"):
        return None
    parts = text.split("---", 2)
    if len(parts) < 3:
        return None
    front, body = parts[1], parts[2]
    data: dict = {}
    for line in front.splitlines():
        if ":" not in line:
            continue
        key, _, raw = line.partition(":")
        key = key.strip()
        if key not in AttachmentMeta.__dataclass_fields__:
            continue
        try:
            data[key] = json.loads(raw.strip())
        except (json.JSONDecodeError, ValueError):
            data[key] = raw.strip()
    if "saved_as" not in data:
        return None
    # The body after the "# Anhang:" heading is the description.
    desc = body.strip()
    if desc.startswith("# Anhang:"):
        desc = desc.split("\n", 1)[1].strip() if "\n" in desc else ""
    if desc == "_(keine Beschreibung verfügbar)_":
        desc = ""
    data["description"] = desc
    return AttachmentMeta.from_dict(data)


def list_for_user(session_dir: str, user_id: str) -> List[AttachmentMeta]:
    """All received attachments for *user_id*, oldest first (by received_at).

    Reconstructed from the sidecar markdown files in ``Downloads/`` — there is
    no separate index file anymore.
    """
    ddir = downloads_dir(session_dir, user_id)
    if not os.path.isdir(ddir):
        return []
    metas: List[AttachmentMeta] = []
    for name in os.listdir(ddir):
        if not name.endswith(SIDECAR_SUFFIX):
            continue
        meta = _parse_sidecar(os.path.join(ddir, name))
        if meta:
            metas.append(meta)
    metas.sort(key=lambda m: m.received_at)
    return metas
