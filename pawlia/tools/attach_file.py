"""AttachFileTool — re-attach a saved file/image to the next reply.

The tool reads a local file (typically from ``<workspace>/Downloads/``) and
queues its bytes on the agent's ``pending_attachments`` list. The chat
interface drains that list after ``agent.run()`` returns and ships each
entry to the originating channel (Matrix / Telegram / Discord).

The path must be inside the user's workspace or one of the explicitly
configured ``attachments.extra_allowed_roots`` — symlink resolution is
applied so a malicious workspace symlink cannot escape the sandbox.
"""

import mimetypes
import os
from typing import TYPE_CHECKING, Any, Dict, Optional

from pawlia.tools.base import Tool

if TYPE_CHECKING:
    from pawlia.agents.chat import ChatAgent


class AttachFileTool(Tool):
    """Attach a local file to the assistant's next reply."""

    name = "attach_file"
    description = (
        "Attach a local file (image, PDF, document, etc.) to the assistant's "
        "next reply. The recipient will see the file inline. The path must "
        "be inside the user's workspace (e.g. 'Downloads/regenradar.gif') "
        "or an absolute path under one of the configured allowed roots. "
        "Use this when the user asks to 'send the file' or to attach a "
        "previously received file (e.g. a rain radar GIF you just saved to "
        "Downloads/). Calling the tool multiple times attaches multiple "
        "files in a single reply."
    )
    trust = "internal"

    def parameters(self) -> Dict[str, Any]:
        return {
            "path": {
                "type": "string",
                "description": (
                    "Path to the file. Either relative to the workspace root "
                    "(e.g. 'Downloads/foo.png') or an absolute path inside an "
                    "allowed root."
                ),
                "minLength": 1,
            },
            "mimetype": {
                "type": "string",
                "description": (
                    "Optional MIME type override. If omitted, the type is "
                    "guessed from the filename extension."
                ),
            },
            "caption": {
                "type": "string",
                "description": "Optional caption shown alongside the attachment.",
            },
        }

    def required_parameters(self):
        return ["path"]

    def execute(
        self, args: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> Any:
        raw_path = (args.get("path") or "").strip()
        if not raw_path:
            return "Error: 'path' is required."

        explicit_mime = (args.get("mimetype") or "").strip() or None
        caption = (args.get("caption") or "").strip() or None

        session_dir = (context or {}).get("session_dir", "")
        if not session_dir:
            return "Error: session_dir missing from tool context."

        workspace = os.path.realpath(
            os.path.join(session_dir, (context.get("user_id") or ""), "workspace")
        )

        extra_roots = (context or {}).get("attachment_extra_roots") or []
        allowed_roots = [workspace] + [
            os.path.realpath(p) for p in extra_roots if isinstance(p, str) and p
        ]
        # Anything inside <workspace>/Downloads/ is always allowed (the default
        # landing spot for received files).  Also allow the entire workspace so
        # the LLM can attach generated artefacts from skills.
        downloads = os.path.realpath(os.path.join(workspace, "Downloads"))
        if downloads not in allowed_roots:
            allowed_roots.append(downloads)
        # /tmp is the sanctioned scratch space for generated, throwaway
        # artefacts (e.g. a rain-radar PNG) — skills write them there instead
        # of cluttering the user's workspace, so attach_file must accept it.
        tmp_root = os.path.realpath("/tmp")
        if tmp_root not in allowed_roots:
            allowed_roots.append(tmp_root)

        max_bytes = int((context or {}).get("max_outgoing_bytes") or 26214400)

        target = self._resolve_path(raw_path, allowed_roots, downloads, workspace)
        if target is None:
            return (
                f"Error: path '{raw_path}' is outside allowed roots "
                f"(workspace and configured extra roots)."
            )

        if not os.path.isfile(target):
            return f"Error: file not found: {raw_path}"

        try:
            size = os.path.getsize(target)
        except OSError as e:
            return f"Error: cannot stat file: {e}"

        if size > max_bytes:
            mb = max_bytes / (1024 * 1024)
            return f"Error: file too large ({size} bytes > {mb:.0f} MB cap)."

        try:
            with open(target, "rb") as f:
                data = f.read()
        except OSError as e:
            return f"Error: cannot read file: {e}"

        mimetype = explicit_mime or mimetypes.guess_type(target)[0] or "application/octet-stream"
        filename = os.path.basename(target)

        agent = (context or {}).get("agent")
        if agent is None or not hasattr(agent, "pending_attachments"):
            return "Error: agent context unavailable; cannot queue attachment."

        agent.pending_attachments.append({
            "data": data,
            "mimetype": mimetype,
            "filename": filename,
            "caption": caption,
            "size": size,
        })

        kb = max(size / 1024, 1) if size else 0
        note = f" ({caption})" if caption else ""
        return f"📎 {filename} queued for the next reply — {mimetype}, {kb:.0f} KB{note}."

    @staticmethod
    def _resolve_path(
        raw_path: str,
        allowed_roots: list,
        downloads: str,
        workspace: str,
    ) -> Optional[str]:
        """Resolve ``raw_path`` against allowed roots, preventing escape.

        Relative paths are interpreted as relative to the user's workspace
        (e.g. ``Downloads/rain.gif`` → ``<workspace>/Downloads/rain.gif``).
        Absolute paths are accepted only if their realpath lives under one
        of the allowed roots.
        """
        if os.path.isabs(raw_path):
            real = os.path.realpath(raw_path)
            for root in allowed_roots:
                if real == root or real.startswith(root + os.sep):
                    return real
            return None

        # Relative: try joining with each allowed root in order. The
        # workspace is the canonical root for relative paths; the
        # Downloads subdir and any extras are tried as fallbacks so that
        # absolute-relative ambiguity (e.g. "Downloads/foo") still works.
        for root in (workspace, downloads, *allowed_roots):
            real_root = os.path.realpath(root)
            real = os.path.realpath(os.path.join(real_root, raw_path))
            if real.startswith(real_root + os.sep) or real == real_root:
                return real
        return None
