"""MatrixRTC / Element Call (Element X) transport over a LiveKit SFU.

Parallel to ``matrix_call.py`` (classic ``m.call.*`` over aiortc).  Element X does
not use per-call SDP invites; instead:

  1. A participant publishes an ``org.matrix.msc3401.call.member`` *state* event
     declaring it has joined the call (and which LiveKit focus to use).
  2. To get media, a client exchanges a homeserver OpenID token at the
     ``lk-jwt-service`` (``/get_token``) for a LiveKit ``{url, jwt}``.
  3. The client connects to the LiveKit room with that JWT and publishes /
     subscribes audio.

This module implements the bot side of that flow.  The transport-agnostic call
logic (VAD/STT/agent/TTS) is reused from :mod:`pawlia.interfaces.call_core`; only
the media transport and the MatrixRTC signalling differ from the aiortc path.

Wire-protocol references (verbatim-sourced):
  - lk-jwt-service ``requests.go`` / ``helper.go`` (element-hq/lk-jwt-service)
  - matrix-js-sdk ``MatrixRTCSession.ts`` / ``CallMembership.ts`` (v34.5.0)
  - LiveKit Python SDK 1.1.x (livekit/python-sdks)
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from pawlia.interfaces.call_core import CallCore, CallTransport, TTSFrameBuffer

logger = logging.getLogger(__name__)

# Optional dependency: the LiveKit SDK ships only glibc wheels and lives in the
# voip container.  Guard the import like aiortc so the rest of the bot runs
# without it (calls are then declined rather than crashing).
try:
    import livekit.rtc as rtc  # type: ignore
    _LIVEKIT_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only where livekit is absent
    rtc = None  # type: ignore
    _LIVEKIT_AVAILABLE = False

try:
    import httpx  # type: ignore
    _HTTPX_AVAILABLE = True
except Exception:  # pragma: no cover
    httpx = None  # type: ignore
    _HTTPX_AVAILABLE = False


# Element Call membership state-event type (MSC3401 — what Element Call emits today).
RTC_MEMBER_EVENT_TYPE = "org.matrix.msc3401.call.member"
# A second, forward-looking type exists on matrix-js-sdk develop but is not broadly
# deployed yet; we also accept it for detection so we react to newer clients.
RTC_MEMBER_EVENT_TYPE_MSC4143 = "org.matrix.msc4143.rtc.member"
RTC_MEMBER_EVENT_TYPES = (RTC_MEMBER_EVENT_TYPE, RTC_MEMBER_EVENT_TYPE_MSC4143)

SAMPLE_RATE = 48000  # CallCore expects float32 mono 48 kHz


# ---------------------------------------------------------------------------
# Pure protocol helpers (unit-testable without livekit / a homeserver)
# ---------------------------------------------------------------------------

def make_membership_state_key(user_id: str, device_id: str) -> str:
    """State key for our own membership event: ``_${userId}_${deviceId}``.

    Mirrors matrix-js-sdk ``makeMembershipStateKey`` (the common case; special
    room versions msc3757/msc3779 drop the leading underscore — not handled here).
    """
    return f"_{user_id}_{device_id}"


def make_membership_content(
    device_id: str,
    focus_url: str,
    room_id: str,
    call_id: str = "",
) -> Dict[str, Any]:
    """Build our ``org.matrix.msc3401.call.member`` JOIN content.

    Shape from matrix-js-sdk ``makeMyMembership`` + ``LivekitFocus``.
    """
    return {
        "application": "m.call",
        "call_id": call_id,
        "scope": "m.room",
        "device_id": device_id,
        "focus_active": {"type": "livekit", "focus_selection": "oldest_membership"},
        "foci_preferred": [
            {
                "type": "livekit",
                "livekit_service_url": focus_url,
                "livekit_alias": room_id,
            }
        ],
    }


def empty_membership_content() -> Dict[str, Any]:
    """LEAVE = empty content to the same type/state-key."""
    return {}


def build_token_request(
    openid_token: Dict[str, Any],
    room_id: str,
    user_id: str,
    device_id: str,
    slot_id: str = "",
) -> Dict[str, Any]:
    """Body for ``POST /get_token`` (lk-jwt-service ``SFURequest``).

    ``openid_token`` is the homeserver OpenID response: it must carry
    ``access_token``, ``token_type``, ``matrix_server_name`` and ``expires_in``.
    The LiveKit room is derived server-side from ``(room_id, slot_id)``, so these
    must match the human client's pair for the bot to land in the same SFU room.
    """
    return {
        "room_id": room_id,
        "slot_id": slot_id,
        "openid_token": {
            "access_token": openid_token.get("access_token"),
            "token_type": openid_token.get("token_type", "Bearer"),
            "matrix_server_name": openid_token.get("matrix_server_name"),
            "expires_in": openid_token.get("expires_in", 3600),
        },
        "member": {
            "id": user_id,
            "claimed_user_id": user_id,
            "claimed_device_id": device_id,
        },
    }


def focus_url_from_member_content(content: Dict[str, Any]) -> Optional[str]:
    """Extract a LiveKit ``livekit_service_url`` from a member event's
    ``foci_preferred`` (the focus the initiating/oldest member proposes)."""
    foci = content.get("foci_preferred") or []
    for f in foci:
        if isinstance(f, dict) and f.get("type") == "livekit":
            url = f.get("livekit_service_url")
            if url:
                return url
    return None


def is_active_membership(content: Optional[Dict[str, Any]]) -> bool:
    """A non-empty member content with a call application = participant is in a call."""
    if not content:
        return False
    return bool(content.get("application") or content.get("foci_preferred"))


def _frame_to_float32_mono_48k(samples_int16: "np.ndarray", sample_rate: int,
                               num_channels: int) -> "np.ndarray":
    """Convert a LiveKit int16 PCM frame to float32 mono 48 kHz for CallCore."""
    pcm = samples_int16.astype(np.float32) / 32768.0
    if num_channels > 1:
        pcm = pcm.reshape(-1, num_channels).mean(axis=1)
    if sample_rate != SAMPLE_RATE and len(pcm) > 0:
        # Lightweight linear resample (avoids a scipy dependency); good enough for
        # speech, and LiveKit usually already delivers 48 kHz.
        n_out = int(round(len(pcm) * SAMPLE_RATE / sample_rate))
        if n_out > 0:
            x_old = np.linspace(0.0, 1.0, num=len(pcm), endpoint=False)
            x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
            pcm = np.interp(x_new, x_old, pcm).astype(np.float32)
    return pcm


async def fetch_livekit_credentials(
    focus_url: str,
    openid_token: Dict[str, Any],
    room_id: str,
    user_id: str,
    device_id: str,
    slot_id: str = "",
    timeout: float = 10.0,
) -> Optional[Dict[str, str]]:
    """Exchange a homeserver OpenID token for LiveKit ``{url, jwt}`` at the
    lk-jwt-service.  Tries the current ``/get_token`` endpoint, falling back to
    the legacy ``/sfu/get`` shape."""
    if not _HTTPX_AVAILABLE:
        logger.error("matrixrtc: httpx not installed — cannot reach lk-jwt-service")
        return None
    base = focus_url.rstrip("/")
    body = build_token_request(openid_token, room_id, user_id, device_id, slot_id)
    async with httpx.AsyncClient(timeout=timeout) as http:
        for path, payload in (
            ("/get_token", body),
            # Legacy fallback (pre-Matrix-2.0 lk-jwt-service).
            ("/sfu/get", {
                "room": room_id,
                "openid_token": body["openid_token"],
                "device_id": device_id,
            }),
        ):
            try:
                resp = await http.post(base + path, json=payload)
            except Exception as e:
                logger.warning("matrixrtc: %s request failed: %s", path, e)
                continue
            if resp.status_code == 404:
                continue  # endpoint not on this lk-jwt-service version; try next
            if resp.status_code >= 400:
                logger.warning("matrixrtc: %s → HTTP %s: %s",
                               path, resp.status_code, resp.text[:200])
                continue
            data = resp.json()
            url, jwt = data.get("url"), data.get("jwt")
            if url and jwt:
                logger.info("matrixrtc: obtained LiveKit creds via %s (url=%s)", path, url)
                return {"url": url, "jwt": jwt}
    return None


# ---------------------------------------------------------------------------
# LiveKitTransport — CallTransport over a LiveKit room
# ---------------------------------------------------------------------------

class LiveKitTransport(CallTransport):
    """Wraps a LiveKit ``rtc.Room`` as a CallTransport.

    Outgoing TTS is paced out of a shared :class:`TTSFrameBuffer` at real time;
    incoming participant audio is converted to float32 mono 48 kHz and pushed to
    ``on_incoming_pcm`` (→ ``CallCore.feed_pcm``).
    """

    FRAME_MS = 20  # 20 ms / 960 samples @ 48 kHz

    def __init__(self, call_id: str, recorder=None) -> None:
        self._call_id = call_id
        self._buf = TTSFrameBuffer(recorder=recorder)
        self._media_connected = asyncio.Event()
        self._room: Optional[Any] = None
        self._source: Optional[Any] = None
        self._publish_task: Optional[asyncio.Task] = None
        self._closed = False
        self._remote_audio_seen = False
        # Callbacks set by the session before start():
        self.on_incoming_pcm: Optional[Callable] = None
        self._on_ice_failed: Optional[Callable] = None  # async () -> None

    # -- CallTransport contract -------------------------------------------

    @property
    def media_connected(self) -> asyncio.Event:
        return self._media_connected

    @property
    def is_playing(self) -> bool:
        return self._buf.is_playing

    @property
    def is_tts_playing(self) -> bool:
        return self._buf.is_tts_playing

    @property
    def is_transport_finished(self) -> bool:
        return self._closed

    def enqueue_pcm_float32(self, pcm) -> None:
        self._buf.enqueue_pcm_float32(pcm)

    def interrupt(self) -> None:
        self._buf.interrupt()

    def stop_after_current_sentence(self) -> None:
        self._buf.stop_after_current_sentence()

    def start_hold(self) -> None:
        self._buf.start_hold()

    def stop_hold(self) -> None:
        self._buf.stop_hold()

    def set_hold_audio(self, pcm_int16) -> None:
        self._buf.set_hold_audio(pcm_int16)

    # -- lifecycle ---------------------------------------------------------

    async def start(self, url: str, jwt: str) -> bool:
        """Connect to the LiveKit room, publish our audio track, subscribe to
        remote audio.  Returns True on success."""
        if not _LIVEKIT_AVAILABLE:
            logger.error("matrixrtc: livekit not installed — cannot join call")
            return False

        self._room = rtc.Room()

        @self._room.on("track_subscribed")
        def _on_track_subscribed(track, publication, participant):  # noqa: ANN001
            if getattr(track, "kind", None) == rtc.TrackKind.KIND_AUDIO:
                logger.info("call %s: subscribed to audio from %s",
                            self._call_id[:8], getattr(participant, "identity", "?"))
                asyncio.create_task(self._read_remote_audio(track))

        @self._room.on("disconnected")
        def _on_disconnected(*_args):  # noqa: ANN002
            logger.info("call %s: LiveKit room disconnected", self._call_id[:8])
            self._closed = True
            self._media_connected.set()  # unblock waiters so they can fail fast
            if self._on_ice_failed is not None:
                asyncio.create_task(self._on_ice_failed())

        try:
            await self._room.connect(url, jwt, options=rtc.RoomOptions(auto_subscribe=True))
        except Exception as e:
            logger.error("call %s: LiveKit connect failed: %s", self._call_id[:8], e)
            self._closed = True
            return False

        # Publish our outgoing audio track.
        self._source = rtc.AudioSource(SAMPLE_RATE, 1)
        track = rtc.LocalAudioTrack.create_audio_track("pawlia-voice", self._source)
        options = rtc.TrackPublishOptions()
        options.source = rtc.TrackSource.SOURCE_MICROPHONE
        options.dtx = False  # keep sending frames during silence; avoids comfort-noise artefacts
        options.red = True   # redundant audio encoding mitigates packet loss
        await self._room.local_participant.publish_track(track, options)

        self._publish_task = asyncio.create_task(self._publish_loop())
        # Room connected + track published: media path is up.  (Greeting playback
        # is additionally gated on this event by CallCore.)
        self._media_connected.set()
        logger.info("call %s: joined LiveKit room, publishing audio", self._call_id[:8])
        return True

    async def _publish_loop(self) -> None:
        """Push 20 ms TTS/hold frames into the LiveKit AudioSource.

        capture_frame() back-pressures when the internal buffer is full, so it
        acts as its own real-time pacer — no explicit sleep needed.
        """
        samples_per_frame = TTSFrameBuffer.SAMPLES_PER_FRAME  # 960
        try:
            while not self._closed:
                frame_int16 = self._buf.next_frame_960()  # int16 mono 48 kHz
                try:
                    frame = rtc.AudioFrame(
                        data=frame_int16.tobytes(),
                        sample_rate=SAMPLE_RATE,
                        num_channels=1,
                        samples_per_channel=samples_per_frame,
                    )
                    await self._source.capture_frame(frame)
                except Exception as e:
                    logger.debug("call %s: capture_frame failed: %s", self._call_id[:8], e)
                    await asyncio.sleep(0.02)  # fallback pacing on error
        except asyncio.CancelledError:
            raise
        except Exception as e:  # pragma: no cover
            logger.warning("call %s: publish loop ended: %s", self._call_id[:8], e)

    async def _read_remote_audio(self, track) -> None:
        """Forward a remote participant's audio frames to the core pipeline.

        AudioStream resamples + downmixes internally (proper stateful resampler),
        so we ask it for 48 kHz mono 20 ms frames and skip our own conversion —
        per-frame np.interp would inject a discontinuity at every frame boundary.
        """
        if self.on_incoming_pcm is None:
            return
        try:
            stream = rtc.AudioStream(
                track, sample_rate=SAMPLE_RATE, num_channels=1, frame_size_ms=self.FRAME_MS)
            async for event in stream:
                frame = getattr(event, "frame", event)
                samples = np.frombuffer(frame.data, dtype=np.int16)
                pcm = _frame_to_float32_mono_48k(
                    samples, frame.sample_rate, frame.num_channels)
                if not self._remote_audio_seen:
                    self._remote_audio_seen = True
                    logger.info("call %s: first remote audio frame", self._call_id[:8])
                if len(pcm):
                    self.on_incoming_pcm(pcm)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("call %s: remote audio reader ended: %s", self._call_id[:8], e)

    async def close(self) -> None:
        self._closed = True
        if self._publish_task and not self._publish_task.done():
            self._publish_task.cancel()
        if self._room is not None:
            try:
                await self._room.disconnect()
            except Exception as e:
                logger.debug("call %s: room disconnect error: %s", self._call_id[:8], e)


# ---------------------------------------------------------------------------
# MatrixRTCSession — CallCore over a LiveKit transport
# ---------------------------------------------------------------------------

class MatrixRTCSession(CallCore):
    """A single Element X call.  Unlike :class:`CallSession` there is no SDP
    offer/answer; ``start`` joins the LiveKit room directly."""

    def __init__(self, *args, on_leave: Optional[Callable] = None,
                 on_announce: Optional[Callable] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._on_leave = on_leave    # async () -> None : redact our membership
        self._on_announce = on_announce  # async () -> None : post membership event

    async def start(self, focus_url: str, slot_id: str = "") -> bool:
        """Fetch LiveKit creds, join the room, start the audio/agent pipeline."""
        if not _LIVEKIT_AVAILABLE:
            logger.error("matrixrtc: livekit not installed — cannot accept call")
            return False

        openid = await self._get_openid_token()
        if not openid:
            return False

        creds = await fetch_livekit_credentials(
            focus_url=focus_url,
            openid_token=openid,
            room_id=self.room_id,
            user_id=self._client.user_id,
            device_id=getattr(self._client, "device_id", "") or "",
            slot_id=slot_id,
        )
        if not creds:
            logger.error("call %s: could not obtain LiveKit credentials", self.call_id[:8])
            return False

        transport = LiveKitTransport(call_id=self.call_id, recorder=self._recorder)
        transport.on_incoming_pcm = self.feed_pcm
        transport._on_ice_failed = self._on_transport_ice_failed

        hold_pcm = self._load_hold_audio()
        if hold_pcm is not None:
            transport.set_hold_audio(hold_pcm)

        # Wait for warmup before announcing or joining — Pawlia must not be
        # visible to the caller until the greeting is ready to play immediately.
        await self._run_preanswer_warmup()

        # Post membership only after warmup so Element X keeps showing "calling"
        # until Pawlia is actually ready to answer.
        if self._on_announce:
            await self._on_announce()

        self._transport = transport
        ok = await transport.start(creds["url"], creds["jwt"])
        if not ok:
            self._transport = None
            return False

        # No SDP answer to send; "answered" == joined.
        await self.mark_answer_sent()

        asyncio.create_task(self._audio_pipeline())
        asyncio.create_task(self._watchdog())
        asyncio.create_task(self._connect_timeout_watchdog())
        if not self._greeting_sent:
            self._greeting_task = asyncio.create_task(self._send_greeting_when_ready())
        asyncio.create_task(self._send_status("Anruf angenommen – Verbindung wird aufgebaut…"))
        logger.info("call %s accepted (Element X) in room %s", self.call_id[:8], self.room_id)
        return True

    async def _get_openid_token(self) -> Optional[Dict[str, Any]]:
        """Request a homeserver OpenID token (to auth to the lk-jwt-service).

        matrix-nio's get_openid_token omits the required JSON body, causing
        Synapse to return 400 M_NOT_JSON.  We use httpx directly instead.
        """
        if not _HTTPX_AVAILABLE:
            logger.error("call %s: httpx unavailable, cannot get OpenID token", self.call_id[:8])
            return None
        hs = self._client.homeserver.rstrip("/")
        uid = self._client.user_id
        token = self._client.access_token
        url = f"{hs}/_matrix/client/v3/user/{uid}/openid/request_token"
        try:
            async with httpx.AsyncClient(timeout=10) as hc:
                r = await hc.post(url, json={}, headers={"Authorization": f"Bearer {token}"})
            if r.status_code != 200:
                logger.error("call %s: OpenID token request failed %s: %s",
                             self.call_id[:8], r.status_code, r.text[:200])
                return None
            data = r.json()
        except Exception as e:
            logger.error("call %s: OpenID token request error: %s", self.call_id[:8], e)
            return None
        if not data.get("access_token"):
            logger.error("call %s: OpenID response missing access_token: %s",
                         self.call_id[:8], data)
            return None
        return {
            "access_token": data["access_token"],
            "token_type": data.get("token_type", "Bearer"),
            "matrix_server_name": data.get("matrix_server_name", uid.split(":", 1)[-1]),
            "expires_in": data.get("expires_in", 3600),
        }

    async def _send_hangup_event(self) -> None:
        """MatrixRTC has no m.call.hangup; leaving = redacting our membership."""
        if self._on_leave is not None:
            try:
                await self._on_leave()
            except Exception as e:
                logger.warning("call %s: leave (membership redact) failed: %s",
                               self.call_id[:8], e)


# ---------------------------------------------------------------------------
# MatrixRTCManager — detects calls via member state events, manages sessions
# ---------------------------------------------------------------------------

class MatrixRTCManager:
    """Tracks Element X calls for a Matrix bot instance, parallel to CallManager."""

    def __init__(
        self,
        client: "Any",
        app: "Any",
        cfg: Dict[str, Any],
        send_text_cb: Callable,
        send_thread_reply_cb: Callable,
        get_agent_cb: Callable,
    ) -> None:
        self._client = client
        self._app = app
        self._cfg = cfg
        self._send_text = send_text_cb
        self._send_thread_reply = send_thread_reply_cb
        self._get_agent = get_agent_cb
        self._sessions: Dict[str, MatrixRTCSession] = {}  # room_id -> session
        self._joining: set = set()  # room_ids with a join in flight
        rtc_cfg = cfg.get("matrixrtc", {}) if isinstance(cfg, dict) else {}
        self._enabled = bool(rtc_cfg.get("enabled", True))
        self._focus_url_override: Optional[str] = rtc_cfg.get("focus_url")
        self._wellknown_focus: Optional[str] = None

    def available(self) -> bool:
        return _LIVEKIT_AVAILABLE and self._enabled

    async def on_member_event(self, room: "Any", event: "Any") -> None:
        """Handle an ``org.matrix.msc3401.call.member`` state change.

        Registered for ``UnknownEvent``; filters on the event type.  A non-empty
        membership from another user in an allowed room triggers us to join.
        """
        if not self.available():
            return
        etype = getattr(event, "type", None)
        logger.debug("matrixrtc: unknown event type=%s sender=%s", etype, getattr(event, "sender", "?"))
        if etype not in RTC_MEMBER_EVENT_TYPES:
            return
        sender = getattr(event, "sender", None)
        if sender == self._client.user_id:
            return  # our own membership echo

        content = getattr(event, "content", None) or getattr(event, "source", {}).get("content")
        room_id = room.room_id
        logger.debug("matrixrtc: rtc member event content keys=%s", list((content or {}).keys()))

        if is_active_membership(content):
            if room_id in self._sessions or room_id in self._joining:
                return  # already in this call
            logger.info("matrixrtc: call detected in %s (member %s)", room_id, sender)
            focus_url = (
                self._focus_url_override
                or focus_url_from_member_content(content)
                or await self._discover_focus()
            )
            if not focus_url:
                logger.error("matrixrtc: no LiveKit focus available; cannot join %s", room_id)
                return
            slot_id = (content or {}).get("call_id", "") or ""
            await self._join(room, sender, focus_url, slot_id)
        else:
            # Membership cleared: if the other party left and we're the only one
            # remaining, hang up.
            session = self._sessions.get(room_id)
            if session and not session.finished:
                logger.info("matrixrtc: remote left %s, hanging up", room_id)
                await self._leave_session(room_id)

    async def _join(self, room: "Any", caller: str, focus_url: str, slot_id: str) -> None:
        room_id = room.room_id
        self._joining.add(room_id)
        try:
            call_id = uuid.uuid4().hex

            # Thread root so transcripts/responses land in a dedicated thread.
            call_thread_id: Optional[str] = None
            try:
                resp = await self._client.room_send(
                    room_id=room_id,
                    message_type="m.room.message",
                    content={"msgtype": "m.text",
                             "body": f"📞 Eingehender Element-X-Anruf von {caller}"},
                    ignore_unverified_devices=True,
                )
                call_thread_id = getattr(resp, "event_id", None)
            except Exception as e:
                logger.warning("matrixrtc: could not create thread root: %s", e)

            agent = self._get_agent(room_id, call_thread_id)
            _tid, _rid = call_thread_id, room_id

            async def _send_cb(text: str) -> None:
                if _tid:
                    await self._send_thread_reply(_rid, _tid, text)
                else:
                    await self._send_text(_rid, text)

            async def _on_leave() -> None:
                await self._redact_membership(room_id)

            _furl, _sid = focus_url, slot_id

            async def _on_announce() -> None:
                await self._post_membership(room_id, _furl, _sid)

            session = MatrixRTCSession(
                call_id=call_id,
                room_id=room_id,
                caller_id=caller,
                thread_id=call_thread_id or call_id,
                client=self._client,
                app=self._app,
                cfg=self._cfg,
                agent=agent,
                send_cb=_send_cb,
                on_leave=_on_leave,
                on_announce=_on_announce,
            )
            self._sessions[room_id] = session

            if hasattr(agent, "set_callbacks"):
                agent.set_callbacks(
                    on_fallback=lambda fm, tm: asyncio.create_task(
                        session._send_status(f"⚙ Fallback: {fm} → {tm}")
                    )
                )

            ok = await session.start(focus_url, slot_id)
            if not ok:
                self._sessions.pop(room_id, None)
                await self._redact_membership(room_id)
                logger.error("matrixrtc: failed to join call in %s", room_id)
        finally:
            self._joining.discard(room_id)

    async def _post_membership(self, room_id: str, focus_url: str, slot_id: str) -> None:
        device_id = getattr(self._client, "device_id", "") or ""
        content = make_membership_content(device_id, focus_url, room_id, call_id=slot_id)
        state_key = make_membership_state_key(self._client.user_id, device_id)
        try:
            await self._client.room_put_state(
                room_id, RTC_MEMBER_EVENT_TYPE, content, state_key=state_key)
        except Exception as e:
            logger.warning("matrixrtc: could not post membership in %s: %s", room_id, e)

    async def _redact_membership(self, room_id: str) -> None:
        device_id = getattr(self._client, "device_id", "") or ""
        state_key = make_membership_state_key(self._client.user_id, device_id)
        try:
            await self._client.room_put_state(
                room_id, RTC_MEMBER_EVENT_TYPE, empty_membership_content(),
                state_key=state_key)
        except Exception as e:
            logger.debug("matrixrtc: could not clear membership in %s: %s", room_id, e)

    async def _leave_session(self, room_id: str) -> None:
        session = self._sessions.pop(room_id, None)
        if session:
            await session.hangup()  # triggers _send_hangup_event → _on_leave redact

    async def _discover_focus(self) -> Optional[str]:
        """Read ``org.matrix.msc4143.rtc_foci`` from the homeserver .well-known."""
        if self._wellknown_focus:
            return self._wellknown_focus
        if not _HTTPX_AVAILABLE:
            return None
        base = (getattr(self._client, "homeserver", "") or "").rstrip("/")
        if not base:
            return None
        try:
            async with httpx.AsyncClient(timeout=10.0) as http:
                resp = await http.get(base + "/.well-known/matrix/client")
                data = resp.json()
        except Exception as e:
            logger.warning("matrixrtc: well-known fetch failed: %s", e)
            return None
        foci = data.get("org.matrix.msc4143.rtc_foci") or []
        for f in foci:
            if isinstance(f, dict) and f.get("type") == "livekit":
                url = f.get("livekit_service_url")
                if url:
                    self._wellknown_focus = url
                    return url
        return None

    def active_session_for_room(self, room_id: str) -> Optional["MatrixRTCSession"]:
        s = self._sessions.get(room_id)
        return s if s and not s.finished else None
