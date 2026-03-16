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
import logging
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


def _make_status(skill_name: str, query: str) -> dict:
    """Initial skill-status message."""
    short_q = (query[:60] + "…") if len(query) > 60 else query
    body = f"⚙️ {skill_name}: {short_q}"
    html = f"⚙️ <b>{skill_name}</b>: {short_q}"
    return {"msgtype": "m.text", "body": body, "format": "org.matrix.custom.html", "formatted_body": html}


def _make_status_edit(event_id: str, skill_name: str, step: int, step_text: str) -> dict:
    """Edit an existing status message to show current step."""
    short = (step_text[:100] + "…") if len(step_text) > 100 else step_text
    body = f"⚙️ {skill_name} · Schritt {step}: {short}"
    html = f"⚙️ <b>{skill_name}</b> · Schritt {step}: <code>{short}</code>"
    new_content = {"msgtype": "m.text", "body": body, "format": "org.matrix.custom.html", "formatted_body": html}
    return {**new_content, "body": f"* {body}", "m.new_content": new_content,
            "m.relates_to": {"rel_type": "m.replace", "event_id": event_id}}


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

    async def _handle(room: MatrixRoom, text: str, images: Optional[List[str]] = None) -> None:
        """Shared handler for text and image messages."""
        logger.info("Matrix: message in %s: %s (images=%d)", room.room_id, text[:80], len(images or []))

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
                        content=_make_status_edit(status_event_id, current_skill, step_count, step_text),
                    )

            agent.on_interim = _on_interim
            agent.on_skill_start = _on_skill_start
            agent.on_skill_step = _on_skill_step
            response = await agent.run(text, images=images or None)

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
        await _handle(room, text)

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
        await _handle(room, caption, images=[data_uri])

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
        # Route through normal handler (prefixed so agent knows it was voice)
        await _handle(room, f"[Sprachnachricht]: {text}")

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
        # Initial sync to skip old messages (no callbacks yet)
        await client.sync(timeout=0, full_state=True)

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
