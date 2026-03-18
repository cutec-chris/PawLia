"""Matrix VoIP call handler for PawLia using aiortc (WebRTC).

Each incoming call gets its own :class:`CallSession`. The session:

1. Accepts the SDP offer from ``m.call.invite``
2. Sends back ``m.call.answer``
3. Exchanges ICE candidates via ``m.call.candidates``
4. Receives caller audio, runs silence-based VAD, transcribes speech chunks
5. Passes transcription to the agent, optionally synthesises TTS and plays it back
6. Cleans up on ``m.call.hangup`` or timeout

Dependencies: aiortc, av, numpy  (optional: edge-tts or piper for TTS)
"""

import asyncio
import fractions
import logging
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

if TYPE_CHECKING:
    from nio import AsyncClient, MatrixRoom
    from pawlia.app import App

logger = logging.getLogger("pawlia.interfaces.matrix_call")

# ---------------------------------------------------------------------------
# Outgoing audio track (TTS playback)
# ---------------------------------------------------------------------------

if _AIORTC_AVAILABLE:
    class _TTSAudioTrack(MediaStreamTrack):
        """An aiortc AudioStreamTrack that streams TTS audio from a queue.

        While the queue is empty silence is transmitted so the WebRTC
        connection stays alive.
        """

        kind = "audio"
        SAMPLE_RATE = 48000
        SAMPLES_PER_FRAME = 960  # 20 ms @ 48 kHz

        def __init__(self) -> None:
            super().__init__()
            self._queue: asyncio.Queue[Optional[np.ndarray]] = asyncio.Queue()
            self._pts = 0
            self._time_base = fractions.Fraction(1, self.SAMPLE_RATE)

        async def recv(self):  # noqa: D401
            from av import AudioFrame  # type: ignore

            try:
                samples = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                samples = None

            if samples is None or len(samples) == 0:
                samples = np.zeros(self.SAMPLES_PER_FRAME, dtype=np.int16)
            else:
                samples = samples[:self.SAMPLES_PER_FRAME]
                if len(samples) < self.SAMPLES_PER_FRAME:
                    samples = np.pad(samples, (0, self.SAMPLES_PER_FRAME - len(samples)))

            frame = AudioFrame(format="s16", layout="mono", samples=self.SAMPLES_PER_FRAME)
            frame.planes[0].update(samples.tobytes())
            frame.sample_rate = self.SAMPLE_RATE
            frame.pts = self._pts
            frame.time_base = self._time_base
            self._pts += self.SAMPLES_PER_FRAME
            return frame

        def enqueue_pcm_float32(self, pcm: np.ndarray) -> None:
            """Enqueue float32 mono PCM for playback (chunks it into 20 ms frames)."""
            pcm_int16 = (np.clip(pcm, -1.0, 1.0) * 32767).astype(np.int16)
            for i in range(0, len(pcm_int16), self.SAMPLES_PER_FRAME):
                self._queue.put_nowait(pcm_int16[i : i + self.SAMPLES_PER_FRAME])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_red_codec(sdp: str) -> str:
    """Remove RED codec (and CN) from an SDP offer.

    Element/Chrome may send RED-wrapped Opus (PT 63) which aiortc cannot
    decode, causing all received audio to be silence.  By removing RED
    from the m= line and dropping its rtpmap/fmtp lines, the caller is
    forced to use plain Opus.
    """
    import re
    lines = sdp.splitlines()
    # Find RED payload type(s)
    red_pts: set = set()
    for line in lines:
        m = re.match(r"a=rtpmap:(\d+)\s+red/", line)
        if m:
            red_pts.add(m.group(1))
    if not red_pts:
        return sdp

    out = []
    for line in lines:
        # Drop rtpmap / fmtp / rtcp-fb lines for RED PTs
        skip = False
        for pt in red_pts:
            if line.startswith(f"a=rtpmap:{pt} ") or \
               line.startswith(f"a=fmtp:{pt} ") or \
               line.startswith(f"a=rtcp-fb:{pt} "):
                skip = True
                break
        if skip:
            continue
        # Remove RED PTs from the m= line
        if line.startswith("m=audio "):
            for pt in red_pts:
                line = line.replace(f" {pt} ", " ").replace(f" {pt}\r", "\r").replace(f" {pt}\n", "\n")
                if line.endswith(f" {pt}"):
                    line = line[: -len(f" {pt}")]
        out.append(line)
    return "\n".join(out)


def _parse_sdp_candidates(sdp: str) -> List[Dict]:
    """Extract ICE candidates from a local SDP description.

    Returns a list of dicts suitable for ``m.call.candidates``.
    """
    import re
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
                "candidate": line[2:],  # strip "a=" prefix → "candidate:..."
            })
    return candidates


# ---------------------------------------------------------------------------
# Per-call session
# ---------------------------------------------------------------------------

class CallSession:
    """Manages a single active VoIP call."""

    # Silence detection: RMS below this → silence
    SILENCE_THRESHOLD = 0.015
    # Seconds of silence that end a speech chunk
    SILENCE_SECONDS = 1.5
    # Minimum seconds of speech before we transcribe (filter short noise bursts)
    MIN_SPEECH_SECONDS = 0.4
    # Maximum call duration in seconds (auto-hangup)
    MAX_CALL_SECONDS = 600

    def __init__(
        self,
        call_id: str,
        room_id: str,
        caller_id: str,
        client: "AsyncClient",
        app: "App",
        cfg: Dict[str, Any],
        send_text_cb: Callable,
    ) -> None:
        self.call_id = call_id
        self.room_id = room_id
        self.caller_id = caller_id
        self._client = client
        self._app = app
        self._cfg = cfg
        self._send_text = send_text_cb  # async (room_id, text)

        self._pc: Optional["RTCPeerConnection"] = None
        self._tts_track: Optional["_TTSAudioTrack"] = None
        self._agent = app.make_agent(f"call_{room_id}")
        self._done = asyncio.Event()
        self._pending_candidates: List[Dict] = []
        self._speaking = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def _get_ice_servers(self) -> List["RTCIceServer"]:
        """Fetch TURN credentials from Synapse, fall back to config STUN servers."""
        servers = []
        try:
            import aiohttp
            url = f"{self._client.homeserver}/_matrix/client/v3/voip/turnServer"
            headers = {"Authorization": f"Bearer {self._client.access_token}"}
            async with aiohttp.ClientSession() as s:
                async with s.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=5)) as r:
                    if r.status == 200:
                        data = await r.json()
                        uris = data.get("uris", [])
                        username = data.get("username", "")
                        password = data.get("password", "")
                        if uris:
                            servers.append(RTCIceServer(urls=uris, username=username, credential=password))
                            logger.info("call %s: using %d TURN/STUN URIs from Synapse: %s",
                                        self.call_id[:8], len(uris), uris)
        except Exception as e:
            logger.warning("call %s: could not fetch TURN servers from Synapse: %s", self.call_id[:8], e)

        for stun in self._cfg.get("stun_servers", [] if servers else ["stun:stun.l.google.com:19302"]):
            servers.append(RTCIceServer(urls=stun))

        return servers

    async def start(self, sdp_offer: str) -> Optional[str]:
        """Accept the call. Returns SDP answer string, or None on error."""
        if not _AIORTC_AVAILABLE:
            logger.error("matrix_call: aiortc not installed — cannot accept call")
            return None

        # Temporarily enable verbose aiortc logging for DTLS/SRTP debugging
        for _name in ("aiortc", "aioice"):
            logging.getLogger(_name).setLevel(logging.DEBUG)

        ice_servers = await self._get_ice_servers()
        self._pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=ice_servers))
        self._tts_track = _TTSAudioTrack()
        self._pc.addTrack(self._tts_track)

        @self._pc.on("track")
        def on_track(track):
            logger.info("call %s: track received kind=%s", self.call_id[:8], track.kind)
            if track.kind == "audio":
                asyncio.ensure_future(self._audio_pipeline(track))

        @self._pc.on("connectionstatechange")
        async def on_conn_state():
            logger.info("call %s: connection state → %s",
                        self.call_id[:8], self._pc.connectionState)

        @self._pc.on("iceconnectionstatechange")
        async def on_ice_state():
            state = self._pc.iceConnectionState
            logger.info("call %s: ICE state → %s", self.call_id[:8], state)
            if state in ("failed", "closed"):
                self._done.set()

        _gathering_done = asyncio.Event()

        @self._pc.on("icegatheringstatechange")
        def on_gathering_state():
            state = self._pc.iceGatheringState
            logger.info("call %s: ICE gathering → %s", self.call_id[:8], state)
            if state == "complete":
                _gathering_done.set()

        # Strip RED codec from offer — Element may send RED-wrapped Opus
        # (PT 63) which aiortc silently drops, causing silence.
        sdp_offer = _strip_red_codec(sdp_offer)
        logger.info("call %s: SDP offer (cleaned):\n%s", self.call_id[:8], sdp_offer)
        await self._pc.setRemoteDescription(
            RTCSessionDescription(sdp=sdp_offer, type="offer")
        )

        # Add any candidates that arrived before the offer was processed
        for c in self._pending_candidates:
            await self._add_candidate(c)
        self._pending_candidates.clear()

        answer = await self._pc.createAnswer()
        await self._pc.setLocalDescription(answer)
        logger.info("call %s: SDP answer:\n%s", self.call_id[:8],
                    self._pc.localDescription.sdp)

        # Auto-hangup watchdog
        asyncio.ensure_future(self._watchdog())
        # Send our ICE candidates once gathering completes (parsed from local SDP)
        asyncio.ensure_future(self._flush_local_candidates(_gathering_done))

        logger.info("call %s accepted in room %s", self.call_id[:8], self.room_id)
        return self._pc.localDescription.sdp

    async def _flush_local_candidates(self, done: asyncio.Event) -> None:
        """Wait for ICE gathering then send candidates parsed from local SDP."""
        try:
            await asyncio.wait_for(done.wait(), timeout=10.0)
        except asyncio.TimeoutError:
            logger.warning("call %s: ICE gathering timed out", self.call_id[:8])

        if not self._pc or not self._pc.localDescription:
            return

        sdp = self._pc.localDescription.sdp
        logger.debug("call %s: local SDP:\n%s", self.call_id[:8], sdp)

        candidates = _parse_sdp_candidates(sdp)
        for c in candidates:
            logger.info("call %s: local candidate: %s", self.call_id[:8], c["candidate"])
        if not candidates:
            return

        await self._client.room_send(
            room_id=self.room_id,
            message_type="m.call.candidates",
            content={"call_id": self.call_id, "version": 0, "candidates": candidates},
        )
        logger.info("call %s: sent %d local ICE candidates", self.call_id[:8], len(candidates))

    async def add_candidates(self, candidates: List[Dict]) -> None:
        """Feed ICE candidates from ``m.call.candidates``."""
        for c in candidates:
            if self._pc and self._pc.remoteDescription:
                await self._add_candidate(c)
            else:
                self._pending_candidates.append(c)

    async def hangup(self) -> None:
        """Terminate the call."""
        self._done.set()
        if self._pc:
            await self._pc.close()
        logger.info("call %s hung up", self.call_id[:8])

    # ------------------------------------------------------------------
    # Internal: audio pipeline
    # ------------------------------------------------------------------

    async def _audio_pipeline(self, track) -> None:
        """Continuously read audio frames, detect speech, transcribe, respond."""
        SAMPLE_RATE = 48000
        fps = 50  # aiortc default: 20 ms frames
        silence_threshold = int(self.SILENCE_SECONDS * fps)
        min_speech_frames = int(self.MIN_SPEECH_SECONDS * fps)

        speech_buffer: List[np.ndarray] = []
        silence_count = 0

        logger.info("call %s: audio pipeline started", self.call_id[:8])
        frames_received = 0
        try:
            while not self._done.is_set():
                try:
                    frame = await asyncio.wait_for(track.recv(), timeout=5.0)
                except asyncio.TimeoutError:
                    if frames_received == 0:
                        logger.warning("call %s: no audio frames received yet", self.call_id[:8])
                    continue
                except MediaStreamError:
                    logger.warning("call %s: MediaStreamError after %d frames — track ended",
                                   self.call_id[:8], frames_received)
                    break

                frames_received += 1

                # Convert AudioFrame → float32 mono via PyAV reformat
                mono_frame = frame.reformat(format='s16', layout='mono')
                raw = np.frombuffer(bytes(mono_frame.planes[0]),
                                    dtype=np.int16)[:mono_frame.samples]
                pcm = raw.astype(np.float32) / 32768.0
                n_channels = len(frame.layout.channels)

                rms = float(np.sqrt(np.mean(pcm ** 2)))
                if frames_received <= 5:
                    nz_count = int(np.count_nonzero(raw))
                    logger.info("call %s: frame #%d fmt=%s pts=%s ch=%d "
                                "pcm_len=%d nz_samples=%d rms=%.4f "
                                "raw_first10=%s",
                                self.call_id[:8], frames_received,
                                frame.format.name, frame.pts, n_channels,
                                len(pcm), nz_count, rms,
                                raw[:10].tolist())
                elif frames_received % 50 == 0:
                    import hashlib
                    h = hashlib.md5(pcm.tobytes()).hexdigest()[:8]
                    logger.info("call %s: frame #%d rms=%.4f buf=%d silence=%d hash=%s",
                                self.call_id[:8], frames_received, rms,
                                len(speech_buffer), silence_count, h)

                if rms > self.SILENCE_THRESHOLD:
                    if not speech_buffer and silence_count == 0:
                        logger.info("call %s: speech started (rms=%.4f)",
                                    self.call_id[:8], rms)
                    speech_buffer.append(pcm)
                    silence_count = 0
                elif speech_buffer:
                    silence_count += 1
                    speech_buffer.append(pcm)  # keep trailing silence for context

                    if silence_count >= silence_threshold:
                        chunk = np.concatenate(speech_buffer)
                        duration = len(chunk) / SAMPLE_RATE
                        logger.info("call %s: speech ended — %.1fs, %d samples",
                                    self.call_id[:8], duration, len(chunk))
                        speech_buffer = []
                        silence_count = 0

                        if len(chunk) >= min_speech_frames * (SAMPLE_RATE // fps):
                            logger.info("call %s: sending chunk for transcription",
                                        self.call_id[:8])
                            asyncio.ensure_future(
                                self._process_speech(chunk, SAMPLE_RATE)
                            )
                        else:
                            logger.info("call %s: chunk too short (%.1fs), skipping",
                                        self.call_id[:8], duration)
        except Exception as e:
            logger.error("call %s: audio pipeline error: %s", self.call_id[:8], e)
        finally:
            self._done.set()
            logger.info("call %s: audio pipeline ended", self.call_id[:8])

    async def _process_speech(self, pcm: "np.ndarray", sample_rate: int) -> None:
        """Transcribe a speech chunk and query the agent."""
        try:
            from pawlia.transcription import transcribe_pcm
            from pawlia.tts import synthesize_pcm
        except ImportError as e:
            logger.error("call %s: missing dependency: %s", self.call_id[:8], e)
            return

        text = await transcribe_pcm(pcm, sample_rate, self._app.config)
        if not text:
            logger.info("call %s: empty transcription (no text returned)", self.call_id[:8])
            return

        logger.info("call %s: transcribed: %s", self.call_id[:8], text[:120])

        # Show typing / speaking indicator in the room
        try:
            await self._client.room_typing(self.room_id, typing_state=True)
        except Exception:
            pass

        try:
            response = await self._agent.run(text)
        except Exception as e:
            logger.error("call %s: agent error: %s", self.call_id[:8], e)
            return
        finally:
            try:
                await self._client.room_typing(self.room_id, typing_state=False)
            except Exception:
                pass

        # Always send text to the room as well (transcript + reply)
        await self._send_text(self.room_id, f"🎙️ *{text}*\n\n{response}")

        # TTS: synthesise and feed to outgoing audio track
        if self._tts_track and self._app.config.get("tts"):
            try:
                tts_pcm = await synthesize_pcm(response, self._app.config, sample_rate=48000)
                if tts_pcm is not None and len(tts_pcm):
                    self._tts_track.enqueue_pcm_float32(tts_pcm)
            except Exception as e:
                logger.warning("call %s: TTS failed: %s", self.call_id[:8], e)

    async def _watchdog(self) -> None:
        """Auto-hangup after MAX_CALL_SECONDS."""
        try:
            await asyncio.wait_for(self._done.wait(), timeout=self.MAX_CALL_SECONDS)
        except asyncio.TimeoutError:
            logger.info("call %s: max duration reached, hanging up", self.call_id[:8])
            await self.hangup()
            await self._send_hangup_event()

    @staticmethod
    def _parse_candidate_string(candidate_str: str) -> Optional[Dict]:
        """Parse an SDP candidate attribute string into field kwargs for RTCIceCandidate."""
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
            # parts[6] == "typ"
            "type": parts[7],
        }
        for i in range(8, len(parts) - 1, 2):
            if parts[i] == "raddr":
                result["relatedAddress"] = parts[i + 1]
            elif parts[i] == "rport":
                result["relatedPort"] = int(parts[i + 1])
        return result

    async def _add_candidate(self, c: Dict) -> None:
        if not c.get("candidate"):
            return  # end-of-candidates signal
        try:
            parsed = self._parse_candidate_string(c["candidate"])
            if not parsed:
                return
            candidate = RTCIceCandidate(
                sdpMid=c.get("sdpMid"),
                sdpMLineIndex=c.get("sdpMLineIndex"),
                **parsed,
            )
            await self._pc.addIceCandidate(candidate)
        except Exception as e:
            logger.debug("call %s: could not add ICE candidate: %s", self.call_id[:8], e)

    async def _send_hangup_event(self) -> None:
        try:
            await self._client.room_send(
                room_id=self.room_id,
                message_type="m.call.hangup",
                content={"call_id": self.call_id, "version": 0},
            )
        except Exception as e:
            logger.warning("call %s: hangup event failed: %s", self.call_id[:8], e)


# ---------------------------------------------------------------------------
# CallManager — tracks all active sessions in a Matrix interface
# ---------------------------------------------------------------------------

class CallManager:
    """Manages all active calls for a Matrix bot instance."""

    def __init__(self, client: "AsyncClient", app: "App", cfg: Dict[str, Any], send_text_cb: Callable) -> None:
        self._client = client
        self._app = app
        self._cfg = cfg
        self._send_text = send_text_cb
        self._sessions: Dict[str, CallSession] = {}  # call_id → session

    def available(self) -> bool:
        return _AIORTC_AVAILABLE

    async def on_invite(self, room: "MatrixRoom", event) -> None:
        """Handle ``m.call.invite``."""
        if not _AIORTC_AVAILABLE:
            logger.warning("matrix_call: aiortc not installed — rejecting call")
            await self._reject(room.room_id, event.call_id)
            await self._send_text(
                room.room_id,
                "Anruf erhalten, aber aiortc ist nicht installiert. "
                "Bitte `pip install aiortc` ausführen.",
            )
            return

        if event.expired:
            logger.info("call %s: invite expired, ignoring", event.call_id[:8])
            return

        if event.call_id in self._sessions:
            logger.warning("call %s: duplicate invite, ignoring", event.call_id[:8])
            return

        sdp_offer = event.offer.get("sdp", "")
        if not sdp_offer:
            logger.error("call %s: no SDP in invite", event.call_id[:8])
            return

        session = CallSession(
            call_id=event.call_id,
            room_id=room.room_id,
            caller_id=event.sender,
            client=self._client,
            app=self._app,
            cfg=self._cfg,
            send_text_cb=self._send_text,
        )
        self._sessions[event.call_id] = session

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
        )
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
            )
        except Exception as e:
            logger.warning("could not send hangup for %s: %s", call_id[:8], e)
