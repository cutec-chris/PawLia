"""Matrix interface for PawLia using matrix-nio.

Config (in config.json under "interfaces.matrix"):
    {
      "homeserver": "https://matrix.org",
      "user_id": "@pawlia:matrix.org",
      "password": "...",          # either password ...
      "access_token": "..."       # ... or access_token
    }
"""

import asyncio
import base64
import logging
from typing import TYPE_CHECKING, Dict, List, Optional

import markdown
from nio import AsyncClient, DownloadResponse, LoginResponse, MatrixRoom, RoomMessageImage, RoomMessageText

if TYPE_CHECKING:
    from pawlia.app import App

logger = logging.getLogger("pawlia.interfaces.matrix")


_md = markdown.Markdown(extensions=["fenced_code", "nl2br", "tables"])


def _make_content(text: str) -> dict:
    """Build a Matrix m.notice content dict with rendered markdown."""
    _md.reset()
    return {
        "msgtype": "m.text",
        "body": text,
        "format": "org.matrix.custom.html",
        "formatted_body": _md.convert(text),
    }


async def start_matrix(app: "App", cfg: Dict) -> None:
    """Connect to Matrix and start handling messages.

    ``cfg`` is the ``interfaces.matrix`` section of config.json.
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

    async def _download_image(mxc_url: str, mimetype: str = "image/png") -> Optional[str]:
        """Download a Matrix mxc:// image and return a base64 data-URI."""
        resp = await client.download(mxc_url)
        if not isinstance(resp, DownloadResponse):
            logger.warning("Matrix: failed to download image: %s", resp)
            return None
        b64 = base64.b64encode(resp.body).decode()
        mime = resp.content_type or mimetype
        return f"data:{mime};base64,{b64}"

    async def _handle(room: MatrixRoom, text: str, images: Optional[List[str]] = None) -> None:
        """Shared handler for text and image messages."""
        logger.info("Matrix: message in %s: %s (images=%d)", room.room_id, text[:80], len(images or []))

        try:
            # Show typing indicator while processing
            await client.room_typing(room.room_id, typing_state=True)

            agent = get_agent(room.room_id)

            async def _on_interim(interim_text: str) -> None:
                await client.room_send(
                    room_id=room.room_id,
                    message_type="m.room.message",
                    content=_make_content(interim_text),
                )

            agent.on_interim = _on_interim
            response = await agent.run(text, images=images or None)

            # Stop typing indicator before sending the response
            await client.room_typing(room.room_id, typing_state=False)

            await client.room_send(
                room_id=room.room_id,
                message_type="m.room.message",
                content=_make_content(response),
            )
        except Exception as e:
            logger.error("Matrix: error processing message: %s", e)
            # Best-effort: stop typing on error
            try:
                await client.room_typing(room.room_id, typing_state=False)
            except Exception:
                pass

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

    # Register scheduler callback for proactive notifications
    async def _matrix_notify(session_id: str, message: str) -> None:
        # session_id for matrix agents is "mx_{room_id}"
        if not session_id.startswith("mx_"):
            return
        room_id = session_id[3:]
        try:
            await client.room_send(
                room_id=room_id,
                message_type="m.room.message",
                content=_make_content(message),
            )
        except Exception as e:
            logger.error("Matrix notify failed for %s: %s", room_id, e)

    app.scheduler.register(_matrix_notify)

    logger.info("Matrix: starting sync loop...")
    try:
        # Initial sync to skip old messages (no callback yet)
        await client.sync(timeout=0, full_state=True)
        # Now register callback and start continuous sync
        client.add_event_callback(on_message, RoomMessageText)
        client.add_event_callback(on_image, RoomMessageImage)
        await client.sync_forever(timeout=30000)
    except asyncio.CancelledError:
        logger.info("Matrix: sync cancelled")
    finally:
        await client.close()
        logger.info("Matrix: disconnected")
