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
            # silence_threshold: 0.02
            # silence_seconds: 1.5
            # min_speech_seconds: 0.4
            # min_active_speech_ratio: 0.12
            # min_consecutive_speech_frames: 8
            # call_inactivity_seconds: 180
"""

import asyncio
import json
import logging
import os
from typing import TYPE_CHECKING, Dict, List, Optional

import markdown
from nio import (
    AsyncClient,
    CallCandidatesEvent,
    CallHangupEvent,
    CallInviteEvent,
    DownloadResponse,
    LoginResponse,
    MatrixRoom,
    RoomMessageAudio,
    RoomMessageImage,
    RoomMessageText,
)

if TYPE_CHECKING:
    from pawlia.app import App

logger = logging.getLogger("pawlia.interfaces.matrix")


_md = markdown.Markdown(extensions=["fenced_code", "nl2br", "tables"])


def _make_content(text: str) -> dict:
    """Build a Matrix m.text content dict with rendered markdown."""
    _md.reset()
    return {
        "msgtype": "m.text",
        "body": text,
        "format": "org.matrix.custom.html",
        "formatted_body": _md.convert(text),
    }


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


def _make_status_step(event_id: str, skill_name: str, step: int, step_text: str) -> dict:
    short = (step_text[:100] + "…") if len(step_text) > 100 else step_text
    body = f"⚙ {skill_name} [{step}]: {short}"
    html = _grey(f"⚙ <b>{skill_name}</b> [{step}]: <code>{short}</code>")
    return _status_edit(event_id, body, html)


def _make_status_done(event_id: str, skill_name: str, steps: int) -> dict:
    body = f"✓ {skill_name} ({steps} Schritte)"
    html = _grey(f"✓ <b>{skill_name}</b> ({steps} Schritte)")
    return _status_edit(event_id, body, html)


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

    client = AsyncClient(homeserver, user_id)

    # Authenticate
    if access_token:
        client.access_token = access_token
        client.user_id = user_id
        logger.info("Matrix: using access_token for %s", user_id)
    elif password:
        resp = await client.login(password)
        if isinstance(resp, LoginResponse):
            logger.info("Matrix: logged in as %s", user_id)
        else:
            logger.error("Matrix: login failed: %s", resp)
            await client.close()
            return
    else:
        logger.error("Matrix: no password or access_token configured")
        await client.close()
        return

    from pawlia.interfaces.common import (
        AgentCache, build_status, format_status, handle_model_command,
        preview_text, format_private_toggle,
        format_bg_enqueue, bytes_to_data_uri,
    )

    # One agent per Matrix room (shared context for everyone in the room)
    agent_cache = AgentCache(app)
    thread_events: Dict[str, str] = {}        # event_id → thread_root_id
    thread_members: Dict[str, List[str]] = {} # thread_root_id → [event_ids]

    def get_agent(room_id: str):
        return agent_cache.get(f"mx_{room_id}")

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

    async def _send_text(room_id: str, text: str) -> None:
        try:
            await client.room_send(
                room_id=room_id,
                message_type="m.room.message",
                content=_make_content(text),
            )
        except Exception as e:
            logger.error("Matrix: send_text failed for %s: %s", room_id, e)

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
        """Handle '//model [name]' — show or change the model for this context."""
        ctx_label = f"Thread `{thread_id[:8]}…`" if thread_id else "Room"
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

        if result.action == "show":
            await _reply(f"**Aktives Modell** [{result.ctx_label}]: `{result.model}`")
        else:
            await _reply(f"✓ Modell für **{result.ctx_label}** auf `{result.model}` gesetzt.")

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
            agent = get_agent(room.room_id)
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

        async def _send(text: str) -> None:
            if thread_id:
                await _send_thread_reply(room.room_id, thread_id, text)
            else:
                await _send_text(room.room_id, text)

        try:
            await client.room_typing(room.room_id, typing_state=True)

            agent = get_agent(room.room_id)

            status_event_id: Optional[str] = None
            step_count = 0
            current_skill: Optional[str] = None

            async def _on_interim(interim_text: str) -> None:
                await _send(interim_text)

            async def _on_skill_start(skill_name: str, query: str) -> None:
                nonlocal status_event_id, step_count, current_skill
                current_skill = skill_name
                step_count = 0
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
                )
                status_event_id = getattr(resp, "event_id", None)
                _remember_thread_event(status_event_id, thread_id)

            async def _on_skill_step(step_text: str) -> None:
                nonlocal step_count
                step_count += 1
                if status_event_id and current_skill:
                    await client.room_send(
                        room_id=room.room_id,
                        message_type="m.room.message",
                        content=_make_status_step(status_event_id, current_skill, step_count, step_text),
                    )

            async def _on_skill_done(skill_name: str) -> None:
                if status_event_id:
                    await client.room_send(
                        room_id=room.room_id,
                        message_type="m.room.message",
                        content=_make_status_done(status_event_id, skill_name, step_count),
                    )

            agent.on_interim = _on_interim
            response = await agent.run(
                text, images=images or None, thread_id=thread_id,
                on_skill_start=_on_skill_start,
                on_skill_step=_on_skill_step,
                on_skill_done=_on_skill_done,
            )

            await client.room_typing(room.room_id, typing_state=False)
            logger.info("Matrix: response in %s%s: %s", room.room_id, ctx, preview_text(response))
            await _send(response)
        except Exception as e:
            logger.error("Matrix: error processing message: %s", e)
            try:
                await client.room_typing(room.room_id, typing_state=False)
            except Exception:
                pass
            await _send(f"Fehler: {e}")

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
        await _handle(room, caption, images=[data_uri], thread_id=thread_id)

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
        active_model = (
            (app.memory.get_thread_model_override(session, thread_id_pre) if thread_id_pre else None)
            or session.model_override
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

    # ------------------------------------------------------------------
    # Scheduler callback for proactive notifications
    # ------------------------------------------------------------------

    async def _matrix_notify(session_id: str, message: str) -> None:
        # session_id for matrix agents is "mx_{room_id}"
        if not session_id.startswith("mx_"):
            return
        room_id = session_id[3:]
        await _send_text(room_id, message)

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

        client.add_event_callback(on_message, RoomMessageText)
        client.add_event_callback(on_image, RoomMessageImage)
        client.add_event_callback(on_audio, RoomMessageAudio)
        client.add_event_callback(on_call_invite, CallInviteEvent)
        client.add_event_callback(on_call_candidates, CallCandidatesEvent)
        client.add_event_callback(on_call_hangup, CallHangupEvent)

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
