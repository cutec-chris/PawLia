"""Matrix interface for PawLia using matrix-nio.

Config (in config.yaml under "interfaces.matrix"):

    matrix:
      homeserver: https://matrix.org
      user_id: "@yourbot:matrix.org"
      password: YOUR_PASSWORD
      # access_token: OR_USE_THIS_INSTEAD_OF_PASSWORD
      # stun_servers:            # for VoIP calls (optional)
      #   - stun:stun.l.google.com:19302
"""

import asyncio
import base64
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
    new_content = {"msgtype": "m.text", "body": new_body,
                   "format": "org.matrix.custom.html", "formatted_body": new_html}
    return {**new_content, "body": f"* {new_body}", "m.new_content": new_content,
            "m.relates_to": {"rel_type": "m.replace", "event_id": event_id}}


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

    # One agent per Matrix room (shared context for everyone in the room)
    agents: Dict[str, object] = {}  # room_id -> agent

    def get_agent(room_id: str):
        if room_id not in agents:
            agents[room_id] = app.make_agent(f"mx_{room_id}")
        return agents[room_id]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _download_image(mxc_url: str, mimetype: str = "image/png") -> Optional[str]:
        """Download a Matrix mxc:// image and return a base64 data-URI."""
        resp = await client.download(mxc_url)
        if not isinstance(resp, DownloadResponse):
            logger.warning("Matrix: failed to download image: %s", resp)
            return None
        b64 = base64.b64encode(resp.body).decode()
        mime = resp.content_type or mimetype
        return f"data:{mime};base64,{b64}"

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
            await client.room_send(
                room_id=room_id,
                message_type="m.room.message",
                content=content,
            )
        except Exception as e:
            logger.error("Matrix: send_thread_reply failed for %s: %s", room_id, e)

    def _get_thread_id(event: RoomMessageText) -> Optional[str]:
        """Return the thread root event_id if this message is a Matrix thread reply."""
        relates_to = event.source.get("content", {}).get("m.relates_to", {})
        if relates_to.get("rel_type") == "m.thread":
            return relates_to.get("event_id")
        return None

    async def _handle_model_command(
        room: MatrixRoom, session_id: str, args: str, thread_id: Optional[str]
    ) -> None:
        """Handle '!model [name]' — show or change the model for this context."""
        session = app.memory.load_session(session_id)
        ctx_label = f"Thread `{thread_id[:8]}…`" if thread_id else "Room"

        if not args.strip():
            if thread_id:
                current = app.memory.get_thread_model_override(session, thread_id) or "(default)"
            else:
                current = session.model_override or "(default)"
            await _send_text(room.room_id, f"**Aktives Modell** [{ctx_label}]: `{current}`")
            return

        new_model = args.strip()
        if thread_id:
            app.memory.set_thread_model_override(session, thread_id, new_model)
            logger.info("Matrix: model changed for %s thread %s -> %s", session_id, thread_id[:8], new_model)
        else:
            app.memory.set_model_override(session, new_model)
            agents.pop(session_id, None)  # recreate with new LLM on next message
            logger.info("Matrix: model changed for %s -> %s", session_id, new_model)

        await _send_text(room.room_id, f"✓ Modell für **{ctx_label}** auf `{new_model}` gesetzt.")

    async def _handle(
        room: MatrixRoom,
        text: str,
        images: Optional[List[str]] = None,
        thread_id: Optional[str] = None,
    ) -> None:
        """Shared handler for text and image messages."""
        session_id = f"mx_{room.room_id}"
        ctx = f" [thread {thread_id[:8]}…]" if thread_id else ""
        logger.info("Matrix: message in %s%s: %s (images=%d)", room.room_id, ctx, text[:80], len(images or []))

        # Commands
        if text.startswith("!private"):
            if not thread_id:
                await _send_text(room.room_id, "_!private funktioniert nur in Threads._")
                return
            session = app.memory.load_session(session_id)
            active = app.memory.toggle_private_thread(session, thread_id)
            icon = "🔒" if active else "🔓"
            state = "aktiviert" if active else "deaktiviert"
            await _send_text(room.room_id, f"{icon} Private Mode {state} — Nachrichten werden {'**nicht** ' if active else ''}gespeichert.")
            return

        if text.startswith("!model"):
            await _handle_model_command(room, session_id, text[len("!model"):], thread_id)
            return

        try:
            await client.room_typing(room.room_id, typing_state=True)

            agent = get_agent(room.room_id)

            status_event_id: Optional[str] = None
            step_count = 0
            current_skill: Optional[str] = None

            async def _on_interim(interim_text: str) -> None:
                await _send_text(room.room_id, interim_text)

            async def _on_skill_start(skill_name: str, query: str) -> None:
                nonlocal status_event_id, step_count, current_skill
                current_skill = skill_name
                step_count = 0
                resp = await client.room_send(
                    room_id=room.room_id,
                    message_type="m.room.message",
                    content=_make_status(skill_name, query),
                )
                status_event_id = getattr(resp, "event_id", None)

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
            agent.on_skill_start = _on_skill_start
            agent.on_skill_step = _on_skill_step
            agent.on_skill_done = _on_skill_done
            response = await agent.run(text, images=images or None, thread_id=thread_id)

            await client.room_typing(room.room_id, typing_state=False)
            await _send_text(room.room_id, response)
        except Exception as e:
            logger.error("Matrix: error processing message: %s", e)
            try:
                await client.room_typing(room.room_id, typing_state=False)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Call manager (VoIP)
    # ------------------------------------------------------------------

    from pawlia.interfaces.matrix_call import CallManager

    call_manager = CallManager(
        client=client,
        app=app,
        cfg=cfg,
        send_text_cb=_send_text,
    )

    if not call_manager.available():
        logger.warning(
            "Matrix: aiortc not installed — VoIP calls will be rejected. "
            "Install with: pip install aiortc"
        )

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    async def on_message(room: MatrixRoom, event: RoomMessageText) -> None:
        if event.sender == client.user_id:
            return
        text = event.body.strip()
        if not text:
            return

        if text.startswith("!thread"):
            message = text[len("!thread"):].strip()
            if not message:
                await _send_text(room.room_id, "_Verwendung: !thread <Nachricht>_")
                return
            thread_id = event.event_id
            session_id = f"mx_{room.room_id}"
            logger.info("Matrix: !thread in %s: %s", room.room_id, message[:80])
            try:
                await client.room_typing(room.room_id, typing_state=True)
                agent = get_agent(room.room_id)
                response = await agent.run(message, thread_id=thread_id)
                await client.room_typing(room.room_id, typing_state=False)
                await _send_thread_reply(room.room_id, thread_id, response)
            except Exception as e:
                logger.error("Matrix: !thread error: %s", e)
                try:
                    await client.room_typing(room.room_id, typing_state=False)
                except Exception:
                    pass
            return

        await _handle(room, text, thread_id=_get_thread_id(event))

    async def on_image(room: MatrixRoom, event: RoomMessageImage) -> None:
        if event.sender == client.user_id:
            return
        mxc_url = event.url
        if not mxc_url:
            return
        mimetype = getattr(event, "mimetype", "image/png") or "image/png"
        data_uri = await _download_image(mxc_url, mimetype)
        if not data_uri:
            return
        caption = event.body if event.body and event.body != "image" else ""
        thread_id = event.source.get("content", {}).get("m.relates_to", {})
        thread_id = thread_id.get("event_id") if thread_id.get("rel_type") == "m.thread" else None
        await _handle(room, caption, images=[data_uri], thread_id=thread_id)

    async def on_audio(room: MatrixRoom, event: RoomMessageAudio) -> None:
        """Handle voice messages: download → transcribe → agent."""
        if event.sender == client.user_id:
            return
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

        text = await transcribe(resp.body, app.config, mime=mime)
        if not text:
            logger.warning("Matrix: transcription returned nothing for %s", event.body)
            await _send_text(room.room_id, "*(Sprachnachricht konnte nicht transkribiert werden)*")
            return

        logger.info("Matrix: voice message transcribed: %s", text[:120])
        relates_to = event.source.get("content", {}).get("m.relates_to", {})
        thread_id = relates_to.get("event_id") if relates_to.get("rel_type") == "m.thread" else None
        # Route through normal handler (prefixed so agent knows it was voice)
        await _handle(room, f"[Sprachnachricht]: {text}", thread_id=thread_id)

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
        await client.close()
        logger.info("Matrix: disconnected")
