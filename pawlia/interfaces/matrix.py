"""Matrix interface for PawLia using matrix-nio.

Config (in config.yaml under "interfaces.matrix"):

    matrix:
      homeserver: https://matrix.org
      user_id: "@yourbot:matrix.org"
      password: YOUR_PASSWORD
      # access_token: OR_USE_THIS_INSTEAD_OF_PASSWORD
            # stun_servers:            # transport-specific STUN/TURN endpoints
            #   - stun:stun.l.google.com:19302

        voip:
            # silence_threshold: 0.018
            # silence_seconds: 1.5
            # min_speech_seconds: 0.4
            # min_active_speech_ratio: 0.12
            # min_consecutive_speech_frames: 8
            # call_inactivity_seconds: 180
            # agc_window_seconds: 15.0
            # agc_target_rms: 0.10
            # agc_max_gain: 12.0
            # agc_smoothing: 0.15
"""

import asyncio
import base64
import io
import json
import logging
import os
import tempfile
import time
from typing import TYPE_CHECKING, Dict, List, Optional

import markdown
import yaml
from nio import (
    AsyncClient,
    CallCandidatesEvent,
    CallHangupEvent,
    CallInviteEvent,
    DownloadResponse,
    InviteMemberEvent,
    KeysQueryResponse,
    LoginResponse,
    MegolmEvent,
    MatrixRoom,
    RoomMessageAudio,
    RoomMessageFile,
    RoomMessageImage,
    RoomMessageText,
    SyncResponse,
    UnknownEvent,
)

if TYPE_CHECKING:
    from pawlia.app import App

logger = logging.getLogger("pawlia.interfaces.matrix")


_md = markdown.Markdown(extensions=["fenced_code", "nl2br", "tables"])
_MAX_MATRIX_FILE_TEXT_BYTES = 128 * 1024
_TEXT_FILE_EXTENSIONS = {
    ".ics", ".txt", ".md", ".csv", ".json", ".yaml", ".yml", ".xml", ".html", ".htm", ".log",
}
_MARKITDOWN_FILE_EXTENSIONS = {
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".rtf", ".odt", ".ods", ".odp",
}
_MARKITDOWN_MIMETYPES = {
    "application/pdf",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.ms-powerpoint",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/rtf",
    "application/vnd.oasis.opendocument.text",
    "application/vnd.oasis.opendocument.spreadsheet",
    "application/vnd.oasis.opendocument.presentation",
}
_TEXT_MIMETYPE_PREFIXES = ("text/",)
_TEXT_MIMETYPES = {
    "application/ics",
    "application/json",
    "application/xml",
    "application/yaml",
    "application/x-yaml",
}


def _cancel_pending_tasks(tasks, exclude=None) -> int:
    """Cancel every not-done task in *tasks* except *exclude*.

    Returns the number actually cancelled. ``exclude`` is the //stop command's
    own task, which must survive so it can report back.
    """
    live = [t for t in tasks if t is not exclude and not t.done()]
    for t in live:
        t.cancel()
    return len(live)


def _make_content(text: str) -> dict:
    """Build a Matrix m.text content dict with rendered markdown."""
    _md.reset()
    return {
        "msgtype": "m.text",
        "body": text,
        "format": "org.matrix.custom.html",
        "formatted_body": _md.convert(text),
    }


def _add_mentions(content: dict, mentions: List[tuple]) -> dict:
    """Prepend user pills and attach intentional mentions to a content dict.

    ``mentions`` is a list of ``(user_id, display_name)`` tuples. This makes the
    message ping the given users so it notifies even clients set to
    "mentions and keywords only" (via the modern ``m.mentions`` field and a
    matrix.to pill in the formatted body for older push-rule matching).
    """
    if not mentions:
        return content
    pills = " ".join(
        f'<a href="https://matrix.to/#/{uid}">{name or uid}</a>'
        for uid, name in mentions
    )
    plain = " ".join(name or uid for uid, name in mentions)
    content["body"] = f"{plain}: {content.get('body', '')}"
    content["formatted_body"] = f"{pills}: {content.get('formatted_body', '')}"
    content["m.mentions"] = {"user_ids": [uid for uid, _ in mentions]}
    return content


def _matrix_file_info(event: RoomMessageFile) -> tuple[str, str]:
    """Return filename and mimetype from a Matrix file message."""
    content = (getattr(event, "source", None) or {}).get("content", {}) or {}
    info = content.get("info") or {}
    filename = content.get("filename") or getattr(event, "body", "") or "attachment"
    mimetype = info.get("mimetype") or "application/octet-stream"
    return filename, mimetype


def _matrix_msgtype_for(mimetype: str) -> str:
    """Map a MIME type to the corresponding Matrix ``m.room.message`` msgtype."""
    mt = (mimetype or "").lower()
    if mt.startswith("image/"):
        return "m.image"
    if mt.startswith("video/"):
        return "m.video"
    if mt.startswith("audio/"):
        return "m.audio"
    return "m.file"


def _is_text_matrix_file(filename: str, mimetype: str) -> bool:
    lower_name = filename.lower()
    lower_mime = mimetype.lower().split(";", 1)[0].strip()
    if lower_mime.startswith(_TEXT_MIMETYPE_PREFIXES) or lower_mime in _TEXT_MIMETYPES:
        return True
    return any(lower_name.endswith(ext) for ext in _TEXT_FILE_EXTENSIONS)


def _is_markitdown_matrix_file(filename: str, mimetype: str) -> bool:
    lower_name = filename.lower()
    lower_mime = mimetype.lower().split(";", 1)[0].strip()
    if lower_mime in _MARKITDOWN_MIMETYPES:
        return True
    return any(lower_name.endswith(ext) for ext in _MARKITDOWN_FILE_EXTENSIONS)


def _decode_matrix_file_text(data: bytes) -> tuple[str, bool]:
    """Decode downloaded Matrix file bytes for prompt use.

    Returns ``(text, truncated)``.
    """
    truncated = len(data) > _MAX_MATRIX_FILE_TEXT_BYTES
    limited = data[:_MAX_MATRIX_FILE_TEXT_BYTES]
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return limited.decode(encoding), truncated
        except UnicodeDecodeError:
            continue
    return limited.decode("utf-8", errors="replace"), truncated


def _convert_matrix_file_markdown(data: bytes, filename: str) -> tuple[Optional[str], Optional[str]]:
    """Convert a binary attachment to Markdown via optional MarkItDown.

    Returns ``(markdown, error)``. ``error`` is user-facing and short.
    """
    try:
        from markitdown import MarkItDown  # type: ignore
    except Exception:
        return None, "MarkItDown ist nicht installiert."

    suffix = os.path.splitext(filename)[1]
    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(data)
            temp_path = f.name
        converter = MarkItDown()
        convert_local = getattr(converter, "convert_local", None)
        result = convert_local(temp_path) if convert_local else converter.convert(temp_path)
        text = getattr(result, "text_content", None) or str(result)
        return text.strip(), None
    except Exception as exc:
        logger.warning("Matrix: MarkItDown conversion failed for %s: %s", filename, exc)
        return None, f"MarkItDown konnte die Datei nicht konvertieren: {exc}"
    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except OSError:
                pass


_GREY = "#888888"


def _grey(html: str) -> str:
    return f'<font color="{_GREY}"><small>{html}</small></font>'


def _status_edit(event_id: str, new_body: str, new_html: str) -> dict:
    """Build an m.replace (edit) event.

    The outer body is plain text only — some clients (FluffyChat) render the
    outer body as a fallback even when they also apply the edit, causing the
    HTML-styled outer formatted_body to appear as an empty/invisible message.
    Keeping the outer body as plain text ensures a visible fallback.
    m.new_content carries the full HTML for clients (Element) that apply it.
    """
    new_content = {
        "msgtype": "m.text",
        "body": new_body,
        "format": "org.matrix.custom.html",
        "formatted_body": new_html,
    }
    return {
        "msgtype": "m.text",
        "body": f"* {new_body}",
        "m.new_content": new_content,
        "m.relates_to": {"rel_type": "m.replace", "event_id": event_id},
    }


def _make_status(skill_name: str, query: str) -> dict:
    short_q = (query[:60] + "…") if len(query) > 60 else query
    body = f"⚙ {skill_name}: {short_q}"
    html = _grey(f"⚙ <b>{skill_name}</b>: {short_q}")
    return {"msgtype": "m.text", "body": body, "format": "org.matrix.custom.html", "formatted_body": html}


def _make_status_step(event_id: str, skill_name: str, step: int, step_text: str, initial_query: str) -> dict:
    short = (step_text[:100] + "…") if len(step_text) > 100 else step_text
    short_q = (initial_query[:60] + "…") if len(initial_query) > 60 else initial_query
    body = f"⚙ {skill_name}: {short_q}\n[{step}] {short}"
    html = _grey(f"⚙ <b>{skill_name}</b>: {short_q}") + "<br>" + _grey(f"[{step}] <code>{short}</code>")
    return _status_edit(event_id, body, html)


def _make_status_done(event_id: str, skill_name: str, steps: int, initial_query: str, result: str = "") -> dict:
    short_q = (initial_query[:60] + "…") if len(initial_query) > 60 else initial_query
    # Extract a short summary from the result (first line or first 120 chars)
    summary = ""
    if result:
        # Strip trust header if present
        clean = result.lstrip()
        if clean.startswith("[Report from"):
            # Skip header lines until we find actual content
            for line in clean.splitlines():
                if (line.strip()
                        and not line.startswith("[")
                        and not line.startswith("---")
                        and not line.lower().startswith("trust:")):
                    clean = line
                    break
        first_line = clean.splitlines()[0].strip() if clean.splitlines() else ""
        summary_text = first_line if first_line else clean[:120]
        summary = (summary_text[:120] + "…") if len(summary_text) > 120 else summary_text
    if summary:
        body = f"✓ {skill_name}: {short_q}\n({steps} Schritte) — {summary}"
        html = _grey(f"✓ <b>{skill_name}</b>: {short_q}") + "<br>" + _grey(f"({steps} Schritte) — {summary}")
    else:
        body = f"✓ {skill_name}: {short_q}\n({steps} Schritte)"
        html = _grey(f"✓ <b>{skill_name}</b>: {short_q}") + "<br>" + _grey(f"({steps} Schritte)")
    return _status_edit(event_id, body, html)


def _save_allowed_users(app: "App", users: List[str]) -> None:
    """Write allowed_users back to config.yaml under interfaces.matrix."""
    config_path = app.config_path
    if not config_path or not os.path.isfile(config_path):
        logger.warning("Matrix: cannot save allowed_users — config_path unknown")
        return
    try:
        with open(config_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        raw.setdefault("interfaces", {}).setdefault("matrix", {})["allowed_users"] = users
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(raw, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        logger.info("Matrix: saved allowed_users=%s to %s", users, config_path)
    except Exception as exc:
        logger.error("Matrix: failed to save allowed_users: %s", exc)


def _resolve_thread_root(
    source: dict,
    known_thread_events: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """Resolve the thread root from a Matrix event payload.

    Preferred path is a proper ``m.thread`` relation. As a fallback, map a
    plain reply back to its thread root when we have already seen the replied-to
    event inside a known thread.
    """
    if not isinstance(source, dict):
        return None

    content = source.get("content", {})
    if not isinstance(content, dict):
        return None

    relates_to = content.get("m.relates_to", {})
    if not isinstance(relates_to, dict):
        return None

    if relates_to.get("rel_type") == "m.thread":
        thread_id = relates_to.get("event_id")
        if isinstance(thread_id, str) and thread_id:
            return thread_id

    reply_meta = relates_to.get("m.in_reply_to", {})
    if not isinstance(reply_meta, dict):
        reply_meta = {}

    reply_to = reply_meta.get("event_id")
    if isinstance(reply_to, str) and reply_to and known_thread_events:
        return known_thread_events.get(reply_to)

    return None


async def start_matrix(app: "App", cfg: Dict) -> None:
    """Connect to Matrix and start handling messages.

    ``cfg`` is the ``interfaces.matrix`` section of config.yaml.
    """
    homeserver: str = cfg["homeserver"]
    user_id: str = cfg["user_id"]
    password: Optional[str] = cfg.get("password")
    access_token: Optional[str] = cfg.get("access_token")
    allowed_users: Optional[List[str]] = cfg.get("allowed_users")

    # E2EE: persistent crypto store so keys survive restarts
    store_path = os.path.join(app.session_dir, "nio_store")
    os.makedirs(store_path, exist_ok=True)

    # Persistent bot session: if we already logged in once, reuse device_id
    # and access_token so restarts don't create a new device every time
    # (which accumulates zombie bot devices that break E2EE sends).
    session_path = os.path.join(store_path, "session.json")
    saved_session: Dict[str, str] = {}
    if os.path.isfile(session_path):
        try:
            import json
            with open(session_path, encoding="utf-8") as f:
                saved = json.load(f)
            if saved.get("user_id") == user_id and saved.get("homeserver") == homeserver:
                saved_session = saved
        except Exception as e:
            logger.warning("Matrix: could not read saved session: %s", e)

    saved_device_id = saved_session.get("device_id")
    saved_token = saved_session.get("access_token")
    # Only pre-seed the device_id when we intend to resume the saved session.
    # If the user set an explicit access_token in config, it may belong to a
    # different device — don't pin that one.
    resume_device_id = None if access_token else saved_device_id

    # Try to use SqliteStore for E2EE; fall back to plain client if deps missing
    try:
        # The nio E2EE store uses peewee/sqlite, not sqlalchemy.
        # Keep those SQL traces out of --debug so app-level debugging stays usable.
        logging.getLogger("peewee").setLevel(logging.WARNING)
        logging.getLogger("nio.store").setLevel(logging.WARNING)
        logging.getLogger("sqlite3").setLevel(logging.WARNING)
        from nio import ClientConfig
        from nio.store import SqliteStore
        client_config = ClientConfig(store=SqliteStore, store_name="")
        client = AsyncClient(
            homeserver, user_id,
            device_id=resume_device_id,
            store_path=store_path, config=client_config,
        )
        e2ee = True
    except ImportError:
        logger.warning("Matrix: E2EE unavailable (install matrix-nio[e2e])")
        client = AsyncClient(homeserver, user_id, device_id=resume_device_id)
        e2ee = False

    def _save_session() -> None:
        try:
            import json
            with open(session_path, "w", encoding="utf-8") as f:
                json.dump({
                    "user_id": user_id,
                    "homeserver": homeserver,
                    "device_id": client.device_id,
                    "access_token": client.access_token,
                }, f)
        except Exception as e:
            logger.warning("Matrix: could not save session: %s", e)

    # Authenticate
    if access_token:
        # Token explicitly configured — trust it over any saved session
        client.access_token = access_token
        client.user_id = user_id
        if e2ee:
            client.load_store()
        logger.info("Matrix: using access_token for %s", user_id)
    elif saved_token and saved_device_id:
        # Reuse previously persisted session
        client.access_token = saved_token
        client.user_id = user_id
        if e2ee:
            client.load_store()
        # Verify the token is still valid with a cheap whoami call
        try:
            from nio import WhoamiResponse
            whoami = await client.whoami()
            if isinstance(whoami, WhoamiResponse):
                logger.info("Matrix: resumed session as %s (device %s)", user_id, saved_device_id)
            else:
                logger.warning("Matrix: saved token invalid (%s), re-logging in", whoami)
                client.access_token = None
        except Exception as e:
            logger.warning("Matrix: whoami failed (%s), re-logging in", e)
            client.access_token = None

        if not client.access_token:
            if not password:
                logger.error("Matrix: saved session expired and no password to re-login")
                await client.close()
                return
            resp = await client.login(password, device_name="PawLia")
            if isinstance(resp, LoginResponse):
                logger.info("Matrix: re-logged in as %s (device %s)", user_id, resp.device_id)
                _save_session()
            else:
                logger.error("Matrix: login failed: %s", resp)
                await client.close()
                return
    elif password:
        resp = await client.login(password, device_name="PawLia")
        if isinstance(resp, LoginResponse):
            logger.info("Matrix: logged in as %s (device %s)", user_id, resp.device_id)
            _save_session()
        else:
            logger.error("Matrix: login failed: %s", resp)
            await client.close()
            return
    else:
        logger.error("Matrix: no password or access_token configured")
        await client.close()
        return

    # E2EE setup
    if e2ee:
        if client.should_upload_keys:
            await client.keys_upload()
            logger.info("Matrix: E2EE keys uploaded")
        # Trust all devices automatically so we can decrypt messages
        if client.store:
            client.store.blacklist_on_unverified = False

    from pawlia.interfaces.common import (
        AgentCache, build_status, format_status, handle_model_command,
        list_available_models, preview_text, format_private_toggle,
        format_bg_enqueue, bytes_to_data_uri, handle_reload_command,
    )

    # One agent per Matrix room (shared context for everyone in the room)
    agent_cache = AgentCache(app)
    thread_events: Dict[str, str] = {}        # event_id → thread_root_id
    thread_members: Dict[str, List[str]] = {} # thread_root_id → [event_ids]
    # Running agent turns, keyed by room#thread, so //stop can cancel the work
    # the *commissioning* thread kicked off (a long skill run otherwise has no
    # off switch).
    running_turns: Dict[str, set] = {}

    def _turn_key(room_id: str, thread_id: Optional[str]) -> str:
        return f"{room_id}#{thread_id or ''}"

    def _register_turn(key: str, task: "asyncio.Task") -> None:
        running_turns.setdefault(key, set()).add(task)
        task.add_done_callback(lambda t: running_turns.get(key, set()).discard(t))

    def _cancel_turns(tasks: "List[asyncio.Task]") -> int:
        return _cancel_pending_tasks(tasks, exclude=asyncio.current_task())

    def get_agent(room_id: str, thread_id: Optional[str] = None):
        """Return the agent for ``(room, thread)``. Per-thread cache prevents
        concurrent turns in different threads from clobbering each other's
        callbacks (on_interim, on_skill_*) on a shared instance."""
        user_id = f"mx_{room_id}"
        key = f"{user_id}#{thread_id}" if thread_id else user_id
        return agent_cache.get(user_id, cache_key=key)

    def _remember_thread_event(event_id: Optional[str], thread_root_id: Optional[str]) -> None:
        if not event_id or not thread_root_id:
            return
        thread_events[thread_root_id] = thread_root_id
        thread_events[event_id] = thread_root_id
        thread_members.setdefault(thread_root_id, [])
        if event_id not in thread_members[thread_root_id]:
            thread_members[thread_root_id].append(event_id)
        if thread_root_id not in thread_members[thread_root_id]:
            thread_members[thread_root_id].insert(0, thread_root_id)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _download_image(mxc_url: str, mimetype: str = "image/png") -> Optional[str]:
        """Download a Matrix mxc:// image and return a base64 data-URI."""
        resp = await client.download(mxc_url)
        if not isinstance(resp, DownloadResponse):
            logger.warning("Matrix: failed to download image: %s", resp)
            return None
        return bytes_to_data_uri(resp.body, resp.content_type or mimetype)

    async def _save_incoming_bytes(
        user_id: str, data: bytes, filename: str, mimetype: str,
        description: Optional[str] = None,
    ) -> Optional[str]:
        """Persist an incoming file/image to the user's Downloads/ folder.

        *description* (vision description / extracted text) is written into the
        markdown sidecar so the file is findable by content via the files skill.
        Returns the relative ``Downloads/<name>`` path on success, else ``None``.

        Failures are logged but never raised — saving is a best-effort side
        effect and must not break message handling.
        """
        try:
            from pawlia import attachments

            max_bytes = int(
                (app.config.get("attachments") or {}).get("max_incoming_bytes")
                or 26214400
            )
            meta = attachments.save_incoming(
                session_dir=app.session_dir,
                user_id=user_id,
                data=data,
                filename=filename,
                source="matrix",
                mimetype=mimetype,
                description=description,
                max_bytes=max_bytes,
            )
            if meta is None:
                logger.info(
                    "Matrix: dropped incoming file (empty or too large): %s (%d bytes)",
                    filename, len(data),
                )
                return None
            return f"{attachments.DOWNLOADS_SUBDIR}/{meta.saved_as}"
        except Exception as exc:
            logger.warning("Matrix: failed to save incoming file %s: %s", filename, exc)
            return None

    async def _describe_incoming_image(data_uri: str) -> str:
        """Best-effort text description of an incoming image for its sidecar.

        Uses the first vision-capable model in the ``vision`` fallback chain.
        Returns "" if no vision model is available or describing fails.
        """
        factory = getattr(app, "llm", None)
        session_dir = getattr(app, "session_dir", None)
        if not factory or not session_dir:
            return ""
        try:
            from pawlia import vision_probe

            chain = factory.get_fallback_chain("vision")
            describer_name = None
            for name in chain or []:
                try:
                    if await vision_probe.resolve_supports_images(factory, session_dir, name):
                        describer_name = name
                        break
                except Exception:
                    continue
            if not describer_name:
                return ""
            describer = factory.get_with_model(describer_name)
            desc = await vision_probe.describe_image(describer, data_uri)
            return (desc or "").strip()
        except Exception as exc:
            logger.debug("Matrix: could not describe incoming image: %s", exc)
            return ""

    def _attachment_kind(mimetype: str) -> str:
        """German article+noun for an attachment, e.g. 'ein Bild' / 'ein PDF'."""
        mt = (mimetype or "").lower()
        if mt.startswith("image/"):
            return "ein Bild"
        if mt == "application/pdf" or mt.endswith("/pdf"):
            return "ein PDF"
        return "eine Datei"

    def _attachment_note(kind: str, saved_rel: Optional[str], description: str) -> str:
        """Synthetic 'user sent an attachment' message used in chat AND calls.

        Frames the attachment as a user event with a wikilink to its sidecar so
        the model reacts naturally (looks like a direct reaction to the file)
        and can read the full content/description on demand via the files skill.
        """
        if saved_rel:
            head = (
                f"Der User hat {kind} gesendet — Datei: `{saved_rel}`, "
                f"Inhalt/Beschreibung: [[{saved_rel}.md]]."
            )
        else:
            head = f"Der User hat {kind} gesendet (konnte nicht gespeichert werden)."
        snippet = " ".join((description or "").split())
        if snippet:
            if len(snippet) > 1500:
                snippet = snippet[:1500] + " …"
            return f"{head}\n\nBeschreibung:\n{snippet}"
        return head

    async def _send_attachment(
        room_id: str, att: dict, thread_root: Optional[str] = None
    ) -> None:
        """Upload + send one queued attachment to Matrix.

        ``att`` is a ``pending_attachments`` entry: ``{data, mimetype, filename, caption}``.
        Maps MIME → Matrix msgtype (image/video/audio/file) and uploads the
        bytes via the nio content repository before sending the message.

        ``thread_root`` is the thread's root event id when the conversation runs
        in a thread — the attachment must carry the same ``m.relates_to`` block
        as the text reply, otherwise it lands in the room timeline instead of
        the thread and the user never sees it there.
        """
        data: bytes = att.get("data") or b""
        mimetype: str = att.get("mimetype") or "application/octet-stream"
        filename: str = att.get("filename") or "attachment"
        caption: Optional[str] = att.get("caption")

        msgtype = _matrix_msgtype_for(mimetype)

        try:
            resp, _ = await client.upload(
                io.BytesIO(data),
                content_type=mimetype,
                filename=filename,
                encrypt=False,
                filesize=len(data),
            )
        except Exception as exc:
            logger.error("Matrix: upload failed for %s: %s", filename, exc)
            return
        if not hasattr(resp, "content_uri"):
            logger.error("Matrix: upload error response: %s", resp)
            return

        content: Dict[str, object] = {
            "msgtype": msgtype,
            "body": caption or filename,
            "url": resp.content_uri,
            "info": {"mimetype": mimetype, "size": len(data)},
        }
        if caption:
            content["filename"] = filename
        if thread_root:
            content["m.relates_to"] = {
                "rel_type": "m.thread",
                "event_id": thread_root,
                "is_falling_back": False,
                "m.in_reply_to": {"event_id": thread_root},
            }

        try:
            resp = await client.room_send(
                room_id=room_id,
                message_type="m.room.message",
                content=content,
                ignore_unverified_devices=True,
            )
            if thread_root:
                _remember_thread_event(getattr(resp, "event_id", None), thread_root)
        except Exception as exc:
            logger.error("Matrix: send attachment failed: %s", exc)

    async def _send_text(room_id: str, text: str) -> Optional[str]:
        try:
            resp = await client.room_send(
                room_id=room_id,
                message_type="m.room.message",
                content=_make_content(text),
                ignore_unverified_devices=True,
            )
            return getattr(resp, "event_id", None)
        except Exception as e:
            logger.error("Matrix: send_text failed for %s: %s", room_id, e)
            return None

    async def _send_thread_reply(room_id: str, root_event_id: str, text: str) -> None:
        """Send a message as a Matrix thread reply rooted at root_event_id."""
        content = _make_content(text)
        content["m.relates_to"] = {
            "rel_type": "m.thread",
            "event_id": root_event_id,
            "is_falling_back": False,
            "m.in_reply_to": {"event_id": root_event_id},
        }
        try:
            resp = await client.room_send(
                room_id=room_id,
                message_type="m.room.message",
                content=content,
                ignore_unverified_devices=True,
            )
            _remember_thread_event(getattr(resp, "event_id", None), root_event_id)
        except Exception as e:
            logger.error("Matrix: send_thread_reply failed for %s: %s", room_id, e)

    def _get_thread_id(event: RoomMessageText) -> Optional[str]:
        """Return the thread root event_id for direct or inferred thread replies."""
        thread_id = _resolve_thread_root(getattr(event, "source", None), thread_events)
        logger.debug(
            "Matrix: resolved thread root for %s -> %s",
            getattr(event, "event_id", None),
            thread_id,
        )
        return thread_id

    async def _handle_model_cmd(
        room: MatrixRoom, session_id: str, args: str, thread_id: Optional[str]
    ) -> None:
        """Handle '//model [model]' or '//model [path] [model]'."""
        ctx_label = "Room-Session"
        result = handle_model_command(app, session_id, args, thread_id=thread_id, ctx_label=ctx_label)

        if result.invalidate_agent:
            agent_cache.invalidate(session_id)
            logger.info("Matrix: model changed for %s -> %s", session_id, result.model)
        elif result.action == "set":
            logger.info("Matrix: model changed for %s thread %s -> %s", session_id, thread_id and thread_id[:8], result.model)

        async def _reply(text: str) -> None:
            if thread_id:
                await _send_thread_reply(room.room_id, thread_id, text)
            else:
                await _send_text(room.room_id, text)

        avail = ", ".join(f"`{m}`" for m in result.available) or "_(keine konfiguriert)_"
        if result.action == "show" and result.chains:
            lines: List[str] = []
            for key, info in result.chains.items():
                label = key.replace("skills.", "Skills.")
                label = label[0].upper() + label[1:] if label else key
                chain = " → ".join(f"`{m}`" for m in info["chain"])
                lines.append(f"**{label}** ({info['source']}):\n{chain}")
            lines.append(f"\n**Verfügbar:** {avail}")
            lines.append("_Session-Chatmodell setzen: `//model <modell>` — Agent setzen: `//model <pfad> <modell>` — Löschen: `//model <pfad> off`_")
            await _reply("\n".join(lines))
        elif result.action == "show":
            await _reply(
                f"**Aktives Chat-Modell** [{result.ctx_label}]: `{result.model}`\n"
                f"**Verfügbar:** {avail}\n"
                f"_Session-Chatmodell setzen: `//model <modell>` — Agent setzen: `//model <pfad> <modell>` — Löschen: `//model <pfad> off`_"
            )
        elif result.action == "invalid_path":
            await _reply("Ungültiger Model-Pfad. Erlaubt: `default`, `chat`, `skill_runner`, `vision`, `compiler`, `skills.<name>`.")
        elif result.action == "cleared":
            await _reply(f"✓ Model-Override `{result.path}` für **{result.ctx_label}** entfernt.")
        else:
            await _reply(f"✓ Model-Override `{result.path}` für **{result.ctx_label}** auf `{result.model}` gesetzt.")

    async def _handle(
        room: MatrixRoom,
        text: str,
        images: Optional[List[str]] = None,
        thread_id: Optional[str] = None,
    ) -> None:
        """Shared handler for text and image messages."""
        session_id = f"mx_{room.room_id}"
        app.scheduler.touch_activity(session_id)
        ctx = f" [thread {thread_id[:8]}…]" if thread_id else ""
        logger.info("Matrix: message in %s%s: %s (images=%d)", room.room_id, ctx, text[:80], len(images or []))

        # Commands (// or / — Element strips one / from //)
        if _cmd(text, "status") is not None:
            agent = get_agent(room.room_id, thread_id)
            status = build_status(app, session_id, agent, thread_id=thread_id)
            text_out = format_status(status)
            if thread_id:
                await _send_thread_reply(room.room_id, thread_id, text_out)
            else:
                await _send_text(room.room_id, text_out)
            return

        if _cmd(text, "private") is not None:
            if not thread_id:
                await _send_text(room.room_id, "_//private funktioniert nur in Threads._")
                return
            session = app.memory.load_session(session_id)
            active = app.memory.toggle_private_thread(session, thread_id)
            await _send_text(room.room_id, format_private_toggle(active))
            return

        if _cmd(text, "reload") is not None:
            result = handle_reload_command(app)
            agent_cache.invalidate_all()
            logger.info("Matrix: app config reloaded")
            if thread_id:
                await _send_thread_reply(room.room_id, thread_id, result.message)
            else:
                await _send_text(room.room_id, result.message)
            return

        stop_args = _cmd(text, "stop")
        if stop_args is None:
            stop_args = _cmd(text, "cancel")
        if stop_args is not None:
            stop_all = stop_args.strip().lower() in ("all", "alle", "*")
            if stop_all:
                all_tasks = [t for tasks in running_turns.values() for t in tasks]
                n = _cancel_turns(all_tasks)
                reply = (
                    f"_🛑 {n} laufende Aufgabe{'n' if n != 1 else ''} überall abgebrochen._"
                    if n else "_Es läuft gerade nichts, was ich abbrechen könnte._"
                )
            else:
                key = _turn_key(room.room_id, thread_id)
                n = _cancel_turns(list(running_turns.get(key, set())))
                reply = (
                    f"_🛑 Abgebrochen ({n} laufende Aufgabe{'n' if n != 1 else ''} in diesem Thread)._"
                    if n else
                    "_In diesem Thread läuft gerade nichts. Mit `//stop all` brichst du alles ab._"
                )
            if thread_id:
                await _send_thread_reply(room.room_id, thread_id, reply)
            else:
                await _send_text(room.room_id, reply)
            return

        if _cmd(text, "stopall") is not None:
            n = _cancel_turns([t for tasks in running_turns.values() for t in tasks])
            reply = (
                f"_🛑 {n} laufende Aufgabe{'n' if n != 1 else ''} überall abgebrochen._"
                if n else "_Es läuft gerade nichts, was ich abbrechen könnte._"
            )
            if thread_id:
                await _send_thread_reply(room.room_id, thread_id, reply)
            else:
                await _send_text(room.room_id, reply)
            return

        model_args = _cmd(text, "model")
        if model_args is not None:
            await _handle_model_cmd(room, session_id, model_args, thread_id)
            return

        if _cmd(text, "clear") is not None:
            if not thread_id:
                await _send_text(room.room_id, "_//clear funktioniert nur in Threads._")
                return
            # Only delete messages IN the thread, not the thread root itself
            event_ids = [eid for eid in thread_members.get(thread_id, []) if eid != thread_id]
            if not event_ids:
                await _send_thread_reply(room.room_id, thread_id, "_Keine Nachrichten zum Löschen gefunden._")
                return
            count = 0
            for eid in list(event_ids):
                try:
                    await client.room_redact(room.room_id, eid)
                    count += 1
                except Exception as e:
                    logger.warning("Matrix: failed to redact %s: %s", eid, e)
            # Keep only the root in the tracker
            thread_members[thread_id] = [thread_id]
            for eid in event_ids:
                thread_events.pop(eid, None)
            logger.info("Matrix: cleared %d messages in thread %s", count, thread_id[:12])
            return

        bg_args = _cmd(text, "background")
        if bg_args is not None:
            if not bg_args:
                await _send_text(room.room_id, "_Verwendung: //background <Nachricht>_")
                return
            app.scheduler.bg_tasks.enqueue(session_id, bg_args)
            await _send_text(room.room_id, format_bg_enqueue(bg_args))
            return

        async def _send(text: str) -> Optional[str]:
            if thread_id:
                await _send_thread_reply(room.room_id, thread_id, text)
                return None
            else:
                return await _send_text(room.room_id, text)

        # Keep typing notification alive while agent works
        typing_stop = asyncio.Event()
        async def _typing_keepalive() -> None:
            while not typing_stop.is_set():
                try:
                    await asyncio.wait_for(typing_stop.wait(), timeout=2.5)
                except asyncio.TimeoutError:
                    pass
                if not typing_stop.is_set():
                    try:
                        await client.room_typing(room.room_id, typing_state=True)
                    except Exception:
                        pass

        # Make this turn cancellable from its own thread via //stop.
        current_turn = asyncio.current_task()
        turn_key = _turn_key(room.room_id, thread_id)
        if current_turn is not None:
            _register_turn(turn_key, current_turn)

        try:
            await client.room_typing(room.room_id, typing_state=True)
            typing_task = asyncio.ensure_future(_typing_keepalive())

            agent = get_agent(room.room_id, thread_id)

            status_event_id: Optional[str] = None
            step_count = 0
            current_skill: Optional[str] = None
            initial_query: Optional[str] = None

            async def _on_interim(interim_text: str) -> None:
                await _send(interim_text)
                await client.room_typing(room.room_id, typing_state=True)

            async def _on_skill_start(skill_name: str, query: str) -> None:
                nonlocal status_event_id, step_count, current_skill, initial_query
                current_skill = skill_name
                initial_query = query
                step_count = 0
                await client.room_typing(room.room_id, typing_state=True)
                content = _make_status(skill_name, query)
                if thread_id:
                    content["m.relates_to"] = {
                        "rel_type": "m.thread",
                        "event_id": thread_id,
                        "is_falling_back": False,
                        "m.in_reply_to": {"event_id": thread_id},
                    }
                resp = await client.room_send(
                    room_id=room.room_id,
                    message_type="m.room.message",
                    content=content,
                    ignore_unverified_devices=True,
                )
                status_event_id = getattr(resp, "event_id", None)
                _remember_thread_event(status_event_id, thread_id)

            async def _on_skill_step(step_text: str) -> None:
                nonlocal step_count
                step_count += 1
                await client.room_typing(room.room_id, typing_state=True)
                if status_event_id and current_skill:
                    await client.room_send(
                        room_id=room.room_id,
                        message_type="m.room.message",
                        content=_make_status_step(status_event_id, current_skill, step_count, step_text, initial_query or ""),
                        ignore_unverified_devices=True,
                    )

            async def _on_skill_done(skill_name: str, result: str = "") -> None:
                await client.room_typing(room.room_id, typing_state=True)
                if status_event_id:
                    await client.room_send(
                        room_id=room.room_id,
                        message_type="m.room.message",
                        content=_make_status_done(status_event_id, skill_name, step_count, initial_query or "", result),
                        ignore_unverified_devices=True,
                    )

            agent.on_interim = _on_interim
            response = await agent.run(
                text, images=images or None, thread_id=thread_id,
                on_skill_start=_on_skill_start,
                on_skill_step=_on_skill_step,
                on_skill_done=_on_skill_done,
            )

            typing_stop.set()
            typing_task.cancel()
            await client.room_typing(room.room_id, typing_state=False)
            logger.info("Matrix: response in %s%s: %s", room.room_id, ctx, preview_text(response))
            sent_event_id = await _send(response)
            # Drain any attachments queued by direct tools (e.g. attach_file)
            # and ship them as separate Matrix messages after the text reply.
            # Pass thread_id so the attachment lands in the same thread as the
            # reply, not loose in the room timeline.
            for att in getattr(agent, "pending_attachments", []) or []:
                await _send_attachment(room.room_id, att, thread_id)
            # Pre-seed so if user creates a thread from this response, context is preserved on restart
            if not thread_id and sent_event_id:
                session = app.memory.load_session(session_id)
                app.memory.seed_thread_context(session, sent_event_id, response)
        except asyncio.CancelledError:
            # //stop cancelled this turn. Clean up the typing indicator and
            # acknowledge in-thread; re-raise so the task ends as cancelled.
            typing_stop.set()
            try:
                typing_task.cancel()
            except NameError:
                pass
            try:
                await client.room_typing(room.room_id, typing_state=False)
            except Exception:
                pass
            logger.info("Matrix: turn cancelled in %s%s", room.room_id, ctx)
            try:
                note = "_🛑 Abgebrochen._"
                if thread_id:
                    await _send_thread_reply(room.room_id, thread_id, note)
                else:
                    await _send_text(room.room_id, note)
            except Exception:
                pass
            raise
        except Exception as e:
            typing_stop.set()
            try:
                typing_task.cancel()
            except NameError:
                pass
            logger.error("Matrix: error processing message: %s", e)
            try:
                await client.room_typing(room.room_id, typing_state=False)
            except Exception:
                pass
            session = app.memory.load_session(session_id)
            # Persist the turn even though the agent errored: agent.run() raised
            # before reaching its own _persist, so without this the inbound
            # message/attachment is lost from the daily log and vanishes on the
            # next restart (the original "PDF never recorded" failure mode).
            if text:
                try:
                    error_note = f"[Fehler bei der Verarbeitung: {e}]"
                    if thread_id:
                        app.memory.append_thread_exchange(session, thread_id, text, error_note)
                    else:
                        app.memory.append_exchange(session, text, error_note)
                except Exception as persist_exc:
                    logger.warning("Matrix: could not persist errored turn: %s", persist_exc)
            override = app.memory.get_agent_override_value(session, "chat", thread_id=thread_id)
            hint = ""
            if override:
                avail = ", ".join(f"`{m}`" for m in list_available_models(app))
                hint = (
                    f"\n\n_Aktiver Model-Override: `{override}`. "
                    f"Wechseln mit `//model chat <modell>` oder löschen mit `//model chat off`._"
                    + (f"\n_Verfügbar: {avail}_" if avail else "")
                )
            await _send(f"Fehler: {e}{hint}")

    # ------------------------------------------------------------------
    # Call manager (VoIP)
    # ------------------------------------------------------------------

    from pawlia.interfaces.matrix_call import CallManager

    call_manager = CallManager(
        client=client,
        app=app,
        cfg=cfg,
        send_text_cb=_send_text,
        send_thread_reply_cb=_send_thread_reply,
        get_agent_cb=get_agent,
    )

    if not call_manager.available():
        logger.warning(
            "Matrix: aiortc not installed — VoIP calls will be rejected. "
            "Install with: pip install aiortc"
        )

    # Element X / MatrixRTC calls (LiveKit SFU) — parallel to the classic path.
    from pawlia.interfaces.matrixrtc_call import MatrixRTCManager

    rtc_manager = MatrixRTCManager(
        client=client,
        app=app,
        cfg=cfg,
        send_text_cb=_send_text,
        send_thread_reply_cb=_send_thread_reply,
        get_agent_cb=get_agent,
    )
    if rtc_manager.available():
        logger.info("Matrix: Element X (MatrixRTC/LiveKit) calls enabled")

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _cmd(text: str, command: str) -> Optional[str]:
        """Check if *text* is a ``//command`` (or ``/command`` — Element strips one ``/``).

        Returns the arguments after the command, or ``None`` if no match.
        """
        for prefix in (f"//{command}", f"/{command}"):
            if text == prefix:
                return ""
            if text.startswith(prefix) and text[len(prefix)] in (" ", "\t", "\n"):
                return text[len(prefix):].strip()
        return None

    always_thread: bool = cfg.get("always_thread", True)

    def _auto_thread(event_id: str, thread_id: Optional[str]) -> Optional[str]:
        """Apply always_thread: if no thread yet, use the event itself as root."""
        if thread_id:
            return thread_id
        return event_id if always_thread else None

    async def _on_message_task(room: MatrixRoom, event: RoomMessageText) -> None:
        text = event.body.strip()

        thread_args = _cmd(text, "thread")
        if thread_args is not None:
            if not thread_args:
                await _send_text(room.room_id, "_Verwendung: //thread <Nachricht>_")
                return
            _remember_thread_event(event.event_id, event.event_id)
            await _handle(room, thread_args, thread_id=event.event_id)
            return

        thread_id = _auto_thread(event.event_id, _get_thread_id(event))
        _remember_thread_event(event.event_id, thread_id)
        await _handle(room, text, thread_id=thread_id)

    async def on_message(room: MatrixRoom, event: RoomMessageText) -> None:
        if event.sender == client.user_id:
            return
        if not event.body.strip():
            return
        _spawn(_on_message_task(room, event))

    async def _on_image_task(room: MatrixRoom, event: RoomMessageImage) -> None:
        mxc_url = event.url
        if not mxc_url:
            return
        mimetype = getattr(event, "mimetype", "image/png") or "image/png"
        data_uri = await _download_image(mxc_url, mimetype)
        if not data_uri:
            return
        caption = event.body if event.body and event.body != "image" else ""
        thread_id = _resolve_thread_root(getattr(event, "source", None), thread_events)
        thread_id = _auto_thread(event.event_id, thread_id)
        _remember_thread_event(event.event_id, thread_id)
        # Describe the image once (vision model) so the sidecar markdown is
        # searchable and the description can be reused for an active call.
        description = await _describe_incoming_image(data_uri)

        # Persist a copy under the user's Downloads/ so the LLM can re-attach
        # the image to a later reply via the attach_file tool.
        saved_rel: Optional[str] = None
        try:
            prefix, b64 = data_uri.split(",", 1)
            img_bytes = base64.b64decode(b64)
            filename = caption.strip() or f"matrix-image-{int(time.time())}.{mimetype.split('/', 1)[-1]}"
            if "." not in filename:
                filename = f"{filename}.{mimetype.split('/', 1)[-1]}"
            saved_rel = await _save_incoming_bytes(
                f"mx_{room.room_id}", img_bytes, filename, mimetype, description=description,
            )
        except Exception as exc:
            logger.debug("Matrix: could not persist incoming image: %s", exc)

        # The image content now lives in the sidecar markdown; the model gets a
        # synthetic "user sent an image" note (link + description) and reads the
        # sidecar on demand — no inline image, same representation in chat/call.
        note = _attachment_note(_attachment_kind(mimetype), saved_rel, description)
        if caption:
            note += f"\n\nText des Users dazu: {caption}"

        # If a call is live in this room, file it silently into the call context
        # instead of running a separate in-thread chat turn.
        call_session = call_manager.active_session_for_room(room.room_id)
        if call_session is not None:
            call_session.register_inbound_attachment(note)
            logger.info(
                "Matrix: image during call %s routed to voice agent (%s)",
                call_session.call_id[:8], saved_rel or "unsaved",
            )
            return

        await _handle(room, note, thread_id=thread_id)

    async def on_image(room: MatrixRoom, event: RoomMessageImage) -> None:
        if event.sender == client.user_id:
            return
        if not event.url:
            return
        _spawn(_on_image_task(room, event))

    async def _on_audio_task(room: MatrixRoom, event: RoomMessageAudio) -> None:
        """Handle voice messages: download → transcribe → agent."""
        mxc_url = event.url
        if not mxc_url:
            return

        logger.info("Matrix: voice message in %s from %s", room.room_id, event.sender)

        resp = await client.download(mxc_url)
        if not isinstance(resp, DownloadResponse):
            logger.warning("Matrix: failed to download audio: %s", resp)
            return

        mime = resp.content_type or "audio/ogg"

        from pawlia.transcription import transcribe

        # Resolve the active model (respects session + thread overrides)
        session_id = f"mx_{room.room_id}"
        session = app.memory.load_session(session_id)
        thread_id_pre = _resolve_thread_root(getattr(event, "source", None), thread_events)
        active_model = app.llm.default_model_name(
            "chat",
            agent_overrides=app.memory.effective_agent_overrides(session, thread_id_pre),
        )
        audio_info = app.llm.audio_model_info(active_model or "chat")
        if audio_info:
            from pawlia.transcription import transcribe_via_model
            text = await transcribe_via_model(resp.body, audio_info[0], audio_info[1], mime=mime)
        else:
            text = await transcribe(resp.body, app.config, mime=mime)
        if not text:
            logger.warning("Matrix: transcription returned nothing for %s", event.body)
            await _send_text(room.room_id, "*(Sprachnachricht konnte nicht transkribiert werden)*")
            return

        logger.info("Matrix: voice message transcribed: %s", text[:120])
        thread_id = _resolve_thread_root(getattr(event, "source", None), thread_events)
        thread_id = _auto_thread(event.event_id, thread_id)
        _remember_thread_event(event.event_id, thread_id)
        # Show transcription in UI
        if thread_id:
            await _send_thread_reply(room.room_id, thread_id, f"🎙️ *{text}*")
        else:
            await _send_text(room.room_id, f"🎙️ *{text}*")
        # Route through normal handler (prefixed so agent knows it was voice)
        await _handle(room, f"[Sprachnachricht]: {text}", thread_id=thread_id)

    async def on_audio(room: MatrixRoom, event: RoomMessageAudio) -> None:
        if event.sender == client.user_id:
            return
        if not event.url:
            return
        _spawn(_on_audio_task(room, event))

    async def _on_file_task(room: MatrixRoom, event: RoomMessageFile) -> None:
        """Handle Matrix file messages by downloading text-like files for the agent."""
        mxc_url = event.url
        if not mxc_url:
            return

        filename, mimetype = _matrix_file_info(event)
        logger.info(
            "Matrix: file message in %s from %s: %s (%s)",
            room.room_id,
            event.sender,
            filename,
            mimetype,
        )

        thread_id = _resolve_thread_root(getattr(event, "source", None), thread_events)
        thread_id = _auto_thread(event.event_id, thread_id)
        _remember_thread_event(event.event_id, thread_id)

        can_decode_text = _is_text_matrix_file(filename, mimetype)
        can_convert_markdown = _is_markitdown_matrix_file(filename, mimetype)

        resp = await client.download(mxc_url)
        if not isinstance(resp, DownloadResponse):
            logger.warning("Matrix: failed to download file %s: %s", filename, resp)
            return
        actual_mimetype = resp.content_type or mimetype

        # Extract a textual representation — stored as the sidecar description
        # so the file becomes findable by content; the model reads it on demand.
        description = ""
        if can_decode_text:
            text, truncated = _decode_matrix_file_text(resp.body)
            description = text + ("\n\n[Hinweis: Inhalt wegen Größe gekürzt.]" if truncated else "")
        elif can_convert_markdown:
            text, error = _convert_matrix_file_markdown(resp.body, filename)
            description = text if not error else ""

        # Always persist a copy (with the extracted text in the sidecar) so the
        # user can ask the bot to re-send or re-read the file later.
        saved_rel = await _save_incoming_bytes(
            f"mx_{room.room_id}", resp.body, filename, actual_mimetype, description=description,
        )

        note = _attachment_note(_attachment_kind(actual_mimetype), saved_rel, description)

        # If a call is live in this room, file it silently into the call context
        # instead of running a separate in-thread chat turn.
        call_session = call_manager.active_session_for_room(room.room_id)
        if call_session is not None:
            call_session.register_inbound_attachment(note)
            logger.info(
                "Matrix: file during call %s routed to voice agent (%s)",
                call_session.call_id[:8], saved_rel or "unsaved",
            )
            return

        await _handle(room, note, thread_id=thread_id)

    async def on_file(room: MatrixRoom, event: RoomMessageFile) -> None:
        if event.sender == client.user_id:
            return
        if not event.url:
            return
        _spawn(_on_file_task(room, event))

    async def on_call_invite(room: MatrixRoom, event: CallInviteEvent) -> None:
        if event.sender == client.user_id:
            return
        logger.info("Matrix: call invite in %s from %s", room.room_id, event.sender)
        await call_manager.on_invite(room, event)

    async def on_call_candidates(room: MatrixRoom, event: CallCandidatesEvent) -> None:
        if event.sender == client.user_id:
            return
        await call_manager.on_candidates(room, event)

    async def on_call_hangup(room: MatrixRoom, event: CallHangupEvent) -> None:
        if event.sender == client.user_id:
            return
        await call_manager.on_hangup(room, event)

    async def on_unknown_event(room: MatrixRoom, event: UnknownEvent) -> None:
        # MatrixRTC (Element X) membership state events arrive as UnknownEvent;
        # the manager filters by type and ignores everything else.
        await rtc_manager.on_member_event(room, event)

    # ------------------------------------------------------------------
    # Auto-join on invite (with pairing)
    # ------------------------------------------------------------------

    async def on_invite(room: MatrixRoom, event: InviteMemberEvent) -> None:
        if event.state_key != client.user_id:
            return

        sender = event.sender

        nonlocal allowed_users
        if allowed_users is not None and sender not in allowed_users:
            logger.info("Matrix: ignoring invite from %s (not in allowed_users)", sender)
            return

        resp = await client.join(room.room_id)
        if hasattr(resp, "room_id"):
            logger.info("Matrix: joined %s (invited by %s)", room.room_id, sender)
        else:
            logger.error("Matrix: failed to join %s: %s", room.room_id, resp)
            return

        # Pairing: first invite ever → persist sender as allowed_users
        if allowed_users is None:
            allowed_users = [sender]
            cfg["allowed_users"] = allowed_users
            _save_allowed_users(app, allowed_users)
            logger.info("Matrix: paired with %s (saved to config)", sender)

    # ------------------------------------------------------------------
    # E2EE: handle undecryptable messages
    # ------------------------------------------------------------------

    async def on_megolm(room: MatrixRoom, event: MegolmEvent) -> None:
        logger.warning(
            "Matrix: could not decrypt message in %s from %s (session %s)",
            room.room_id, event.sender, event.session_id,
        )

    def _trust_allowed_devices() -> int:
        """TOFU: auto-verify every device of users in allowed_users.

        Trust anchor is the ``allowed_users`` config (set via the pairing
        flow or explicitly).  Any device belonging to those users gets
        verified so we can send encrypted messages to them.  Our own other
        sessions are always trusted — otherwise encrypted sends to rooms
        where the bot has a second device fail with OlmUnverifiedDeviceError.
        Returns the number of devices newly verified in this pass.
        """
        if not client.device_store:
            return 0
        trust_users = list(allowed_users or [])
        if client.user_id and client.user_id not in trust_users:
            trust_users.append(client.user_id)
        if not trust_users:
            return 0
        verified = 0
        for user in trust_users:
            for device in client.device_store.active_user_devices(user):
                if not device.verified:
                    client.verify_device(device)
                    verified += 1
        return verified

    # ------------------------------------------------------------------
    # Scheduler callback for proactive notifications
    # ------------------------------------------------------------------

    def _room_mentions(room_id: str) -> List[tuple]:
        """Human members of a room (everyone but the bot), for @-mentioning."""
        room = client.rooms.get(room_id)
        if room is None:
            return []
        mentions = []
        for uid in room.users:
            if uid == client.user_id:
                continue
            try:
                name = room.user_name(uid)
            except Exception:
                name = None
            mentions.append((uid, name))
        return mentions

    async def _matrix_notify(session_id: str, message: str) -> None:
        # session_id for matrix agents is "mx_{room_id}"
        if not session_id.startswith("mx_"):
            return
        room_id = session_id[3:]
        # Proactive notifications go to the main room timeline and ping the
        # user so they notify even with "mentions and keywords only".
        content = _add_mentions(_make_content(message), _room_mentions(room_id))
        event_id = None
        try:
            resp = await client.room_send(
                room_id=room_id,
                message_type="m.room.message",
                content=content,
                ignore_unverified_devices=True,
            )
            event_id = getattr(resp, "event_id", None)
        except Exception as e:
            logger.error("Matrix: notify send failed for %s: %s", room_id, e)
        if event_id:
            session = app.memory.load_session(session_id)
            app.memory.seed_thread_context(session, event_id, message)

    app.scheduler.register(_matrix_notify)

    _active_tasks: set[asyncio.Task] = set()

    def _spawn(coro) -> asyncio.Task:
        task = asyncio.create_task(coro)
        _active_tasks.add(task)
        task.add_done_callback(_active_tasks.discard)
        return task

    # ------------------------------------------------------------------
    # Sync loop
    # ------------------------------------------------------------------

    logger.info("Matrix: starting sync loop...")
    try:
        # Initial sync to get a since-token and skip old messages (no callbacks yet)
        await client.sync(timeout=0)

        # E2EE: fetch device keys for all joined rooms so we can decrypt
        if client.store:
            for room_id in client.rooms:
                room_obj = client.rooms[room_id]
                if room_obj.encrypted:
                    user_ids = [u for u in room_obj.users if u != client.user_id]
                    if user_ids:
                        await client.keys_query()
                        logger.info("Matrix: E2EE keys queried for %d joined rooms", len(client.rooms))
                        break
            verified = _trust_allowed_devices()
            if verified:
                logger.info("Matrix: auto-verified %d device(s) from allowed_users", verified)

        client.add_event_callback(on_message, RoomMessageText)
        client.add_event_callback(on_image, RoomMessageImage)
        client.add_event_callback(on_audio, RoomMessageAudio)
        client.add_event_callback(on_file, RoomMessageFile)
        client.add_event_callback(on_call_invite, CallInviteEvent)
        client.add_event_callback(on_call_candidates, CallCandidatesEvent)
        client.add_event_callback(on_call_hangup, CallHangupEvent)
        client.add_event_callback(on_unknown_event, UnknownEvent)
        client.add_event_callback(on_invite, InviteMemberEvent)
        client.add_event_callback(on_megolm, MegolmEvent)

        async def _on_sync(response: SyncResponse) -> None:
            # Auto-verify any new devices of allowed users (TOFU)
            verified = _trust_allowed_devices()
            if verified:
                logger.info("Matrix: auto-verified %d new device(s) from allowed_users", verified)

        async def _on_keys_query(response: KeysQueryResponse) -> None:
            # Devices show up here (e.g. during a call's olm session negotiation)
            # before the next /sync delivers them — verify them immediately so
            # the subsequent encrypted send doesn't crash on the new device.
            verified = _trust_allowed_devices()
            if verified:
                logger.info("Matrix: auto-verified %d device(s) after keys_query", verified)

        client.add_response_callback(_on_sync, SyncResponse)
        client.add_response_callback(_on_keys_query, KeysQueryResponse)

        await client.sync_forever(timeout=30000)
    except asyncio.CancelledError:
        logger.info("Matrix: sync cancelled")
    finally:
        for task in list(_active_tasks):
            task.cancel()
        if _active_tasks:
            await asyncio.gather(*_active_tasks, return_exceptions=True)
        await client.close()
        logger.info("Matrix: disconnected")
