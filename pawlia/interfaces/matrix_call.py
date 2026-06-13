"""Matrix VoIP call handler for PawLia using aiortc (WebRTC).

Each incoming call gets its own :class:`CallSession` **with an isolated
thread context** (same isolation as ``//thread``).  All transcriptions and
responses are posted into a dedicated Matrix thread rooted at a
"📞 Eingehender Anruf" message.

Flow
----
1. ``m.call.invite`` arrives → thread-root message is sent → SDP answer
2. ICE candidates are exchanged via ``m.call.candidates``
3. Caller audio is received, silence-based VAD detects speech chunks
4. Each chunk is transcribed (STT) and streamed through the agent
5. The LLM response is **streamed sentence-by-sentence** — each sentence
   is synthesised (TTS) and enqueued for playback *immediately*, reducing
   perceived latency significantly compared to full-response TTS
6. While the agent is thinking, a configurable **hold audio** loop
   (default ``assets/keyboard.m4a``) is played to the caller and a
   Matrix typing indicator is kept alive
7. Call ends on ``m.call.hangup`` or prolonged inactivity

Dependencies: aiortc, av, numpy  (optional: edge-tts or piper for TTS)
"""

import asyncio
import fractions
import logging
import re
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

try:
    import numpy as np
    from aiortc import (  # type: ignore
        MediaStreamTrack,
        RTCIceCandidate,
        RTCPeerConnection,
        RTCSessionDescription,
    )
    from aiortc.mediastreams import MediaStreamError  # type: ignore
    from aiortc import RTCConfiguration, RTCIceServer  # type: ignore
    _AIORTC_AVAILABLE = True
except Exception as _e:
    import logging as _logging
    _logging.getLogger("pawlia.interfaces.matrix_call").warning("aiortc import failed: %s", _e)
    _AIORTC_AVAILABLE = False

from pawlia.interfaces.call_core import (
    CallCore, CallTransport, TTSFrameBuffer, SAMPLE_RATE, _STREAM_ENDED,
)

if TYPE_CHECKING:
    from nio import AsyncClient, MatrixRoom
    from pawlia.app import App

logger = logging.getLogger("pawlia.interfaces.matrix_call")


# ---------------------------------------------------------------------------
# SDP helpers (aiortc-specific)
# ---------------------------------------------------------------------------

def _strip_red_codec(sdp: str) -> str:
    """Remove RED codec (and CN) from an SDP offer.

    Element/Chrome may send RED-wrapped Opus (PT 63) which aiortc cannot
    decode, causing all received audio to be silence.
    """
    lines = sdp.splitlines()
    red_pts: set = set()
    for line in lines:
        m = re.match(r"a=rtpmap:(\d+)\s+red/", line, re.IGNORECASE)
        if m:
            red_pts.add(m.group(1))
    if not red_pts:
        return sdp
    out = []
    for line in lines:
        skip = False
        for pt in red_pts:
            if (line.startswith(f"a=rtpmap:{pt} ")
                    or line.startswith(f"a=fmtp:{pt} ")
                    or line.startswith(f"a=rtcp-fb:{pt} ")):
                skip = True
                break
        if skip:
            continue
        if line.startswith("m=audio "):
            for pt in red_pts:
                line = (line.replace(f" {pt} ", " ")
                            .replace(f" {pt}\r", "\r")
                            .replace(f" {pt}\n", "\n"))
                if line.endswith(f" {pt}"):
                    line = line[: -len(f" {pt}")]
        out.append(line)
    return "\n".join(out)


def _parse_sdp_candidates(sdp: str) -> List[Dict]:
    """Extract ICE candidates from a local SDP description."""
    candidates = []
    mid = None
    mline_index = -1
    for line in sdp.splitlines():
        if line.startswith("m="):
            mline_index += 1
            mid = None
        elif line.startswith("a=mid:"):
            mid = line[6:].strip()
        elif line.startswith("a=candidate:"):
            candidates.append({
                "sdpMid": mid or str(mline_index),
                "sdpMLineIndex": mline_index,
                "candidate": line[2:],
            })
    return candidates


# ---------------------------------------------------------------------------
# Outgoing audio track — delegates frame logic to TTSFrameBuffer
# ---------------------------------------------------------------------------

if _AIORTC_AVAILABLE:
    class _TTSAudioTrack(MediaStreamTrack):
        """aiortc AudioStreamTrack backed by TTSFrameBuffer."""

        kind = "audio"
        SAMPLE_RATE = SAMPLE_RATE
        SAMPLES_PER_FRAME = TTSFrameBuffer.SAMPLES_PER_FRAME

        def __init__(self, recorder=None) -> None:
            super().__init__()
            self._buf = TTSFrameBuffer(recorder=recorder)
            self._pts = 0
            self._time_base = fractions.Fraction(1, self.SAMPLE_RATE)
            self._start_time: Optional[float] = None

        # --- Delegated to TTSFrameBuffer ---
        @property
        def is_playing(self) -> bool:
            return self._buf.is_playing

        @property
        def is_tts_playing(self) -> bool:
            return self._buf.is_tts_playing

        def set_hold_audio(self, pcm_int16) -> None:
            self._buf.set_hold_audio(pcm_int16)

        def start_hold(self) -> None:
            self._buf.start_hold()

        def stop_hold(self) -> None:
            self._buf.stop_hold()

        def interrupt(self) -> None:
            self._buf.interrupt()

        def stop_after_current_sentence(self) -> None:
            self._buf.stop_after_current_sentence()

        def enqueue_pcm_float32(self, pcm) -> None:
            self._buf.enqueue_pcm_float32(pcm)

        async def recv(self):
            from av import AudioFrame  # type: ignore
            if self._start_time is None:
                self._start_time = time.monotonic()
            target = self._start_time + (self._pts / self.SAMPLE_RATE)
            delay = target - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)

            samples = self._buf.next_frame_960()

            frame = AudioFrame(format="s16", layout="mono", samples=self.SAMPLES_PER_FRAME)
            frame.planes[0].update(samples.tobytes())
            frame.sample_rate = self.SAMPLE_RATE
            frame.pts = self._pts
            frame.time_base = self._time_base
            self._pts += self.SAMPLES_PER_FRAME
            return frame


# ---------------------------------------------------------------------------
# AiortcTransport — CallTransport adapter wrapping RTCPeerConnection
# ---------------------------------------------------------------------------

class AiortcTransport(CallTransport):
    """Wraps aiortc RTCPeerConnection as a CallTransport."""

    def __init__(
        self,
        call_id: str,
        client: "AsyncClient",
        cfg: Dict[str, Any],
        recorder=None,
    ) -> None:
        self._call_id = call_id
        self._client = client
        self._cfg = cfg
        self._recorder = recorder
        self._pc: Optional["RTCPeerConnection"] = None
        self._tts_track: Optional["_TTSAudioTrack"] = None
        self._media_connected = asyncio.Event()
        self._ice_reconnect_task: Optional[asyncio.Task] = None
        self._exception_handler_installed = False
        self._original_exception_handler = None
        # Callbacks set by CallSession before start():
        self.on_incoming_pcm: Optional[Callable] = None
        self._on_ice_failed: Optional[Callable] = None  # async () -> None
        self._on_rtp_sample: Optional[Callable] = None  # async (recv, lost, jitter) -> None

    @property
    def media_connected(self) -> asyncio.Event:
        return self._media_connected

    @property
    def is_playing(self) -> bool:
        return bool(self._tts_track and self._tts_track.is_playing)

    @property
    def is_tts_playing(self) -> bool:
        return bool(self._tts_track and self._tts_track.is_tts_playing)

    @property
    def is_transport_finished(self) -> bool:
        return bool(self._pc and self._pc.connectionState in {"closed", "failed"})

    def enqueue_pcm_float32(self, pcm) -> None:
        if self._tts_track:
            self._tts_track.enqueue_pcm_float32(pcm)

    def interrupt(self) -> None:
        if self._tts_track:
            self._tts_track.interrupt()

    def stop_after_current_sentence(self) -> None:
        if self._tts_track:
            self._tts_track.stop_after_current_sentence()

    def start_hold(self) -> None:
        if self._tts_track:
            self._tts_track.start_hold()

    def stop_hold(self) -> None:
        if self._tts_track:
            self._tts_track.stop_hold()

    def set_hold_audio(self, pcm_int16) -> None:
        if self._tts_track:
            self._tts_track.set_hold_audio(pcm_int16)

    def add_candidate(self, candidate: Dict) -> None:
        pass  # async; use add_candidate_async instead

    async def add_candidate_async(self, c: Dict) -> None:
        if not c.get("candidate"):
            return
        try:
            parsed = self._parse_candidate_string(c["candidate"])
            if not parsed:
                return
            cand = RTCIceCandidate(
                sdpMid=c.get("sdpMid"),
                sdpMLineIndex=c.get("sdpMLineIndex"),
                **parsed,
            )
            await self._pc.addIceCandidate(cand)
        except Exception as e:
            logger.debug("call %s: could not add ICE candidate: %s", self._call_id[:8], e)

    async def start(self, sdp_offer: str) -> Optional[str]:
        """Set up PC, process SDP offer, return SDP answer or None on error."""
        if not _AIORTC_AVAILABLE:
            return None

        # Suppress per-call aioice TURN CHANNEL_BIND errors
        loop = asyncio.get_running_loop()
        self._original_exception_handler = loop.get_exception_handler()

        def _turn_aware_handler(loop, context):
            exc = context.get("exception")
            msg = str(exc) if exc else ""
            if "CHANNEL_BIND" in msg or "TransactionFailed" in msg:
                logger.debug("call %s: suppressed known aioice TURN error: %s",
                             self._call_id[:8], msg)
                return
            h = self._original_exception_handler
            if h:
                h(loop, context)
            else:
                loop.default_exception_handler(context)

        loop.set_exception_handler(_turn_aware_handler)
        self._exception_handler_installed = True

        for _name in ("aiortc", "aioice"):
            logging.getLogger(_name).setLevel(logging.ERROR)
        logging.getLogger("aiohttp").setLevel(logging.WARNING)

        ice_servers = await self._get_ice_servers()
        self._pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=ice_servers))
        self._tts_track = _TTSAudioTrack(recorder=self._recorder)
        self._pc.addTrack(self._tts_track)

        @self._pc.on("track")
        def on_track(track):
            logger.info("call %s: track received kind=%s", self._call_id[:8], track.kind)
            if track.kind == "audio":
                asyncio.create_task(self._read_audio_track(track))
            elif track.kind == "video":
                asyncio.create_task(self._drain_video_track(track))

        @self._pc.on("connectionstatechange")
        async def on_conn_state():
            state = self._pc.connectionState
            logger.info("call %s: connection state → %s", self._call_id[:8], state)
            if state == "connected":
                self._media_connected.set()

        @self._pc.on("iceconnectionstatechange")
        async def on_ice_state():
            state = self._pc.iceConnectionState
            logger.info("call %s: ICE state → %s", self._call_id[:8], state)
            if state == "connected":
                if self._ice_reconnect_task and not self._ice_reconnect_task.done():
                    self._ice_reconnect_task.cancel()
                    self._ice_reconnect_task = None
            elif state == "disconnected":
                if not self._ice_reconnect_task or self._ice_reconnect_task.done():
                    self._ice_reconnect_task = asyncio.create_task(
                        self._ice_reconnect_watchdog())
            elif state == "failed":
                asyncio.create_task(self._fire_ice_failed())
                self._media_connected.set()  # unblock any waiter so it can fail fast
            elif state == "closed":
                self._media_connected.set()

        _gathering_done = asyncio.Event()

        @self._pc.on("icegatheringstatechange")
        def on_gathering_state():
            state = self._pc.iceGatheringState
            logger.info("call %s: ICE gathering → %s", self._call_id[:8], state)
            if state == "complete":
                _gathering_done.set()

        sdp_offer = _strip_red_codec(sdp_offer)
        logger.debug("call %s: SDP offer (cleaned):\n%s", self._call_id[:8], sdp_offer)
        await self._pc.setRemoteDescription(RTCSessionDescription(sdp=sdp_offer, type="offer"))
        answer = await self._pc.createAnswer()
        await self._pc.setLocalDescription(answer)
        logger.debug("call %s: SDP answer:\n%s", self._call_id[:8], self._pc.localDescription.sdp)
        self._widen_jitter_buffers()

        # Start aiortc-specific background tasks
        asyncio.create_task(self._flush_local_candidates(_gathering_done))
        asyncio.create_task(self._log_receiver_stats())

        return self._pc.localDescription.sdp

    async def close(self) -> None:
        if self._pc:
            await self._pc.close()
        if self._exception_handler_installed:
            try:
                loop = asyncio.get_running_loop()
                loop.set_exception_handler(self._original_exception_handler)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Internal aiortc helpers
    # ------------------------------------------------------------------

    async def _get_ice_servers(self) -> List["RTCIceServer"]:
        servers = []
        try:
            import aiohttp
            from pawlia.utils import PAWLIA_USER_AGENT
            url = f"{self._client.homeserver}/_matrix/client/v3/voip/turnServer"
            headers = {"Authorization": f"Bearer {self._client.access_token}",
                       "User-Agent": PAWLIA_USER_AGENT}
            async with aiohttp.ClientSession() as s:
                async with s.get(url, headers=headers,
                                 timeout=aiohttp.ClientTimeout(total=5)) as r:
                    if r.status == 200:
                        data = await r.json()
                        uris = data.get("uris", [])
                        username = data.get("username", "")
                        password = data.get("password", "")
                        if uris:
                            servers.append(RTCIceServer(
                                urls=uris, username=username, credential=password))
                            logger.info("call %s: using %d TURN/STUN URIs from Synapse",
                                        self._call_id[:8], len(uris))
        except Exception as e:
            logger.warning("call %s: could not fetch TURN servers: %s", self._call_id[:8], e)
        for stun in self._cfg.get("stun_servers",
                                  [] if servers else ["stun:stun.l.google.com:19302"]):
            servers.append(RTCIceServer(urls=stun))
        return servers

    def _widen_jitter_buffers(self) -> None:
        from pawlia.audio.config import get_int_config
        capacity = getattr(self, "_jitter_buffer_capacity",
                           self._cfg.get("jitter_buffer_capacity", 32))
        if capacity <= 16 or not self._pc:
            return
        if capacity & (capacity - 1) != 0:
            logger.warning("call %s: jitter_buffer_capacity %d is not a power of two",
                           self._call_id[:8], capacity)
            return
        try:
            from aiortc.jitterbuffer import JitterBuffer  # type: ignore
        except Exception:
            return
        attr = "_RTCRtpReceiver__jitter_buffer"
        widened = 0
        for receiver in self._pc.getReceivers():
            track = getattr(receiver, "track", None)
            if track is not None and getattr(track, "kind", None) != "audio":
                continue
            old = getattr(receiver, attr, None)
            if old is None:
                continue
            prefetch = getattr(old, "_prefetch", 4)
            try:
                setattr(receiver, attr, JitterBuffer(capacity=capacity, prefetch=prefetch))
                widened += 1
            except Exception:
                pass
        if widened:
            logger.info("call %s: audio jitter buffer widened to %d packets (~%d ms)",
                        self._call_id[:8], capacity, capacity * 20)

    async def _read_audio_track(self, track) -> None:
        """Read aiortc audio frames, convert to float32, forward to core."""
        while True:
            try:
                frame = await asyncio.wait_for(track.recv(), timeout=5.0)
            except asyncio.TimeoutError:
                continue
            except MediaStreamError:
                if self.on_incoming_pcm is not None:
                    # Signal end-of-stream to the core pipeline
                    import asyncio as _asyncio
                    # We can't call feed_stream_ended directly; instead put sentinel
                    # via a dummy queue — but on_incoming_pcm is feed_pcm which expects ndarray.
                    # Use a workaround: the core's _incoming_q is filled by on_incoming_pcm,
                    # but _STREAM_ENDED must be put directly. Use _on_stream_ended if set.
                    pass
                if self._on_stream_ended is not None:
                    self._on_stream_ended()
                return
            except Exception:
                if self._on_stream_ended is not None:
                    self._on_stream_ended()
                return

            raw_bytes = bytes(frame.planes[0])
            n_channels = max(len(frame.layout.channels), 1)
            n_int16 = frame.samples * n_channels
            raw = np.frombuffer(raw_bytes, dtype=np.int16)[:n_int16]
            if n_channels > 1:
                pcm = raw.reshape(-1, n_channels).astype(np.float32).mean(axis=1) / 32768.0
            else:
                pcm = raw.astype(np.float32) / 32768.0

            if self.on_incoming_pcm is not None:
                self.on_incoming_pcm(pcm)

    _on_stream_ended: Optional[Callable] = None  # set by CallSession to session.feed_stream_ended

    async def _drain_video_track(self, track) -> None:
        try:
            while True:
                await track.recv()
        except Exception:
            pass

    async def _flush_local_candidates(self, done: asyncio.Event) -> None:
        try:
            await asyncio.wait_for(done.wait(), timeout=10.0)
        except asyncio.TimeoutError:
            logger.warning("call %s: ICE gathering timed out", self._call_id[:8])
        if not self._pc or not self._pc.localDescription:
            return
        sdp = self._pc.localDescription.sdp
        candidates = _parse_sdp_candidates(sdp)
        if not candidates:
            return
        await self._client.room_send(
            room_id=self._room_id,
            message_type="m.call.candidates",
            content={"call_id": self._call_id, "version": 0, "candidates": candidates},
            ignore_unverified_devices=True,
        )
        logger.info("call %s: sent %d local ICE candidates", self._call_id[:8], len(candidates))

    async def _log_receiver_stats(self) -> None:
        await asyncio.sleep(5)
        interval = 0
        while self._pc and self._pc.connectionState not in {"closed", "failed"}:
            interval += 1
            stats_level = logging.INFO if interval <= 15 else logging.DEBUG
            try:
                stats = await self._pc.getStats()
                for report in stats.values():
                    t = getattr(report, "type", "")
                    if t in ("inbound-rtp", "transport", "candidate-pair"):
                        logger.log(stats_level, "call %s: STATS [%s] %s",
                                   self._call_id[:8], t,
                                   {k: v for k, v in report.__dict__.items()
                                    if not k.startswith("_")})
                    if t == "inbound-rtp" and getattr(report, "kind", "audio") == "audio":
                        if self._on_rtp_sample:
                            await self._on_rtp_sample(
                                int(getattr(report, "packetsReceived", 0) or 0),
                                int(getattr(report, "packetsLost", 0) or 0),
                                float(getattr(report, "jitter", 0.0) or 0.0),
                            )
            except Exception as e:
                logger.debug("call %s: stats error: %s", self._call_id[:8], e)
            await asyncio.sleep(5)

    async def _ice_reconnect_watchdog(self) -> None:
        ICE_RECONNECT_TIMEOUT = 30
        logger.info("call %s: ICE disconnected — waiting %ds for recovery",
                    self._call_id[:8], ICE_RECONNECT_TIMEOUT)
        try:
            await asyncio.sleep(ICE_RECONNECT_TIMEOUT)
        except asyncio.CancelledError:
            logger.info("call %s: ICE recovered — reconnect watchdog cancelled",
                        self._call_id[:8])
            return
        if self._pc and self._pc.iceConnectionState == "disconnected":
            logger.warning("call %s: ICE did not recover after %ds — ending call",
                           self._call_id[:8], ICE_RECONNECT_TIMEOUT)
            await self._fire_ice_failed()

    async def _fire_ice_failed(self) -> None:
        if self._on_ice_failed:
            await self._on_ice_failed()

    @staticmethod
    def _parse_candidate_string(candidate_str: str) -> Optional[Dict]:
        s = candidate_str
        if s.startswith("candidate:"):
            s = s[len("candidate:"):]
        parts = s.split()
        if len(parts) < 8:
            return None
        result: Dict = {
            "foundation": parts[0],
            "component": int(parts[1]),
            "protocol": parts[2].lower(),
            "priority": int(parts[3]),
            "ip": parts[4],
            "port": int(parts[5]),
            "type": parts[7],
        }
        for i in range(8, len(parts) - 1, 2):
            if parts[i] == "raddr":
                result["relatedAddress"] = parts[i + 1]
            elif parts[i] == "rport":
                result["relatedPort"] = int(parts[i + 1])
        return result


# ---------------------------------------------------------------------------
# CallSession — aiortc-specific CallCore subclass
# ---------------------------------------------------------------------------

class CallSession(CallCore):
    """Manages a single active VoIP call over aiortc/WebRTC."""

    def __init__(
        self,
        call_id: str,
        room_id: str,
        caller_id: str,
        thread_id: str,
        client: "AsyncClient",
        app: "App",
        cfg: Dict[str, Any],
        agent: Any,
        send_cb: Callable,
    ) -> None:
        super().__init__(call_id, room_id, caller_id, thread_id, client, app, cfg, agent, send_cb)
        self._pending_candidates: List[Dict] = []

    async def start(self, sdp_offer: str) -> Optional[str]:
        """Accept the call. Returns SDP answer string, or None on error."""
        if not _AIORTC_AVAILABLE:
            logger.error("matrix_call: aiortc not installed — cannot accept call")
            return None

        transport = AiortcTransport(
            call_id=self.call_id,
            client=self._client,
            cfg=self._cfg,
            recorder=self._recorder,
        )
        # Store room_id for ICE candidate sending
        transport._room_id = self.room_id
        # Use configured jitter buffer capacity
        transport._jitter_buffer_capacity = self.JITTER_BUFFER_CAPACITY

        # Wire audio seams
        transport.on_incoming_pcm = self.feed_pcm
        transport._on_stream_ended = self.feed_stream_ended
        transport._on_ice_failed = self._on_transport_ice_failed
        transport._on_rtp_sample = self._handle_rtp_sample

        # Load hold audio
        hold_pcm = self._load_hold_audio()
        if hold_pcm is not None:
            transport.set_hold_audio(hold_pcm)
            logger.info("call %s: hold audio loaded (%d samples, %.1fs)",
                        self.call_id[:8], len(hold_pcm), len(hold_pcm) / SAMPLE_RATE)

        # Pre-answer warmup (LLM/STT warm-up before media is live)
        await self._run_preanswer_warmup()

        self._transport = transport
        sdp_answer = await transport.start(sdp_offer)
        if sdp_answer is None:
            self._transport = None
            return None

        # Add candidates that arrived before the offer was processed
        for c in self._pending_candidates:
            await transport.add_candidate_async(c)
        self._pending_candidates.clear()

        # Start the shared audio pipeline (reads from _incoming_q)
        asyncio.create_task(self._audio_pipeline())

        # Start core background tasks
        asyncio.create_task(self._watchdog())
        asyncio.create_task(self._connect_timeout_watchdog())
        if not self._greeting_sent:
            self._greeting_task = asyncio.create_task(self._send_greeting_when_ready())

        asyncio.create_task(self._send_status("Anruf angenommen – Verbindung wird aufgebaut…"))
        logger.info("call %s accepted in room %s", self.call_id[:8], self.room_id)
        return sdp_answer

    async def add_candidates(self, candidates: List[Dict]) -> None:
        """Handle ``m.call.candidates``."""
        transport = self._transport
        if isinstance(transport, AiortcTransport):
            for c in candidates:
                await transport.add_candidate_async(c)
        else:
            self._pending_candidates.extend(candidates)

    async def hangup(self) -> None:
        """Hang up — send m.call.hangup then clean up."""
        await super().hangup()

    async def _send_hangup_event(self) -> None:
        try:
            await self._client.room_send(
                room_id=self.room_id,
                message_type="m.call.hangup",
                content={"call_id": self.call_id, "version": 0},
                ignore_unverified_devices=True,
            )
        except Exception as e:
            logger.warning("call %s: hangup event failed: %s", self.call_id[:8], e)

    async def _handle_rtp_sample(self, recv: int, lost: int, jitter: float) -> None:
        msg = self._update_net_state(recv, lost, jitter)
        if msg:
            logger.info("call %s: network %s", self.call_id[:8],
                        "degraded" if self._net_degraded else "recovered")
            await self._send_status(msg)


# ---------------------------------------------------------------------------
# CallManager — tracks all active sessions
# ---------------------------------------------------------------------------

class CallManager:
    """Manages all active calls for a Matrix bot instance."""

    def __init__(
        self,
        client: "AsyncClient",
        app: "App",
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
        self._sessions: Dict[str, CallSession] = {}

    def available(self) -> bool:
        return _AIORTC_AVAILABLE

    def active_session_for_room(self, room_id: str) -> Optional["CallSession"]:
        for session in self._sessions.values():
            if session.room_id == room_id and not session.finished:
                return session
        return None

    async def on_invite(self, room: "MatrixRoom", event) -> None:
        """Handle ``m.call.invite``."""
        if not _AIORTC_AVAILABLE:
            logger.warning("matrix_call: aiortc not installed — rejecting call")
            await self._reject(room.room_id, event.call_id)
            await self._send_text(
                room.room_id,
                "Anruf erhalten, aber aiortc ist nicht installiert.",
            )
            return

        if event.expired:
            logger.info("call %s: invite expired, ignoring", event.call_id[:8])
            return

        existing = self._sessions.get(event.call_id)
        if existing:
            if existing.finished:
                logger.info("call %s: replacing finished session on duplicate invite",
                            event.call_id[:8])
                self._sessions.pop(event.call_id, None)
            else:
                logger.warning("call %s: duplicate invite, ignoring", event.call_id[:8])
                return

        sdp_offer = event.offer.get("sdp", "")
        if not sdp_offer:
            logger.error("call %s: no SDP in invite", event.call_id[:8])
            return

        call_thread_id: Optional[str] = None
        try:
            resp = await self._client.room_send(
                room_id=room.room_id,
                message_type="m.room.message",
                content={
                    "msgtype": "m.text",
                    "body": f"📞 Eingehender Anruf von {event.sender}",
                },
                ignore_unverified_devices=True,
            )
            call_thread_id = getattr(resp, "event_id", None)
            logger.info("call %s: thread root event_id=%s", event.call_id[:8], call_thread_id)
        except Exception as e:
            logger.warning("call %s: could not create thread root: %s", event.call_id[:8], e)

        agent = self._get_agent(room.room_id, call_thread_id)
        _tid = call_thread_id
        _rid = room.room_id

        async def _send_cb(text: str) -> None:
            if _tid:
                await self._send_thread_reply(_rid, _tid, text)
            else:
                await self._send_text(_rid, text)

        session = CallSession(
            call_id=event.call_id,
            room_id=room.room_id,
            caller_id=event.sender,
            thread_id=call_thread_id or event.call_id,
            client=self._client,
            app=self._app,
            cfg=self._cfg,
            agent=agent,
            send_cb=_send_cb,
        )
        self._sessions[event.call_id] = session

        if hasattr(agent, "set_callbacks"):
            agent.set_callbacks(
                on_fallback=lambda from_m, to_m: asyncio.create_task(
                    session._send_status(f"⚙ Fallback: {from_m} → {to_m}")
                )
            )

        sdp_answer = await session.start(sdp_offer)
        if sdp_answer is None:
            del self._sessions[event.call_id]
            await self._reject(room.room_id, event.call_id)
            return

        await self._client.room_send(
            room_id=room.room_id,
            message_type="m.call.answer",
            content={
                "call_id": event.call_id,
                "version": 0,
                "answer": {"type": "answer", "sdp": sdp_answer},
            },
            ignore_unverified_devices=True,
        )
        await session.mark_answer_sent()
        logger.info("call %s: answer sent", event.call_id[:8])

    async def on_candidates(self, room: "MatrixRoom", event) -> None:
        """Handle ``m.call.candidates``."""
        session = self._sessions.get(event.call_id)
        if session:
            await session.add_candidates(event.candidates)

    async def on_hangup(self, room: "MatrixRoom", event) -> None:
        """Handle ``m.call.hangup``."""
        session = self._sessions.pop(event.call_id, None)
        if session:
            await session.hangup()
        logger.info("call %s: remote hangup", event.call_id[:8])

    async def _reject(self, room_id: str, call_id: str) -> None:
        try:
            await self._client.room_send(
                room_id=room_id,
                message_type="m.call.hangup",
                content={"call_id": call_id, "version": 0},
                ignore_unverified_devices=True,
            )
        except Exception as e:
            logger.warning("could not send hangup for %s: %s", call_id[:8], e)
