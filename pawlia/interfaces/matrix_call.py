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
7. Call ends on ``m.call.hangup`` or timeout

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
            self._start_time: Optional[float] = None
            # Hold audio: looping background sound while waiting for agent
            self._hold_pcm: Optional[np.ndarray] = None  # int16 mono @ 48 kHz
            self._hold_pos: int = 0
            self._hold_active: bool = False

        @property
        def is_playing(self) -> bool:
            """True while TTS or hold audio is playing."""
            return not self._queue.empty() or self._hold_active

        def set_hold_audio(self, pcm_int16: np.ndarray) -> None:
            """Set the hold audio loop (int16 mono PCM at 48 kHz)."""
            self._hold_pcm = pcm_int16
            self._hold_pos = 0

        def start_hold(self) -> None:
            """Start looping hold audio (until :meth:`stop_hold`)."""
            self._hold_active = True
            self._hold_pos = 0

        def stop_hold(self) -> None:
            """Stop hold audio playback."""
            self._hold_active = False

        async def recv(self):  # noqa: D401
            from av import AudioFrame  # type: ignore

            # Pace output at real-time (20 ms per frame)
            if self._start_time is None:
                self._start_time = time.monotonic()
            target = self._start_time + (self._pts / self.SAMPLE_RATE)
            delay = target - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)

            try:
                samples = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                samples = None

            if samples is None or len(samples) == 0:
                if (self._hold_active
                        and self._hold_pcm is not None
                        and len(self._hold_pcm) > 0):
                    # Loop hold audio
                    end = self._hold_pos + self.SAMPLES_PER_FRAME
                    if end <= len(self._hold_pcm):
                        samples = self._hold_pcm[self._hold_pos:end]
                    else:
                        tail = self._hold_pcm[self._hold_pos:]
                        head = self._hold_pcm[:self.SAMPLES_PER_FRAME - len(tail)]
                        samples = np.concatenate([tail, head])
                    self._hold_pos = (self._hold_pos + self.SAMPLES_PER_FRAME) % len(self._hold_pcm)
                else:
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
    SILENCE_THRESHOLD = 0.02
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
        thread_id: str,
        client: "AsyncClient",
        app: "App",
        cfg: Dict[str, Any],
        agent: Any,
        send_cb: Callable,
    ) -> None:
        self.call_id = call_id
        self.room_id = room_id
        self.caller_id = caller_id
        self.thread_id = thread_id
        self._client = client
        self._app = app
        self._cfg = cfg
        self._send_cb = send_cb  # async (text,) — already routed to the call thread

        self._pc: Optional["RTCPeerConnection"] = None
        self._tts_track: Optional["_TTSAudioTrack"] = None
        self._agent = agent
        self._done = asyncio.Event()
        self._pending_candidates: List[Dict] = []
        self._speaking = False
        self._ice_reconnect_task: Optional[asyncio.Task] = None

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

        for _name in ("aiortc", "aioice"):
            logging.getLogger(_name).setLevel(logging.WARNING)

        ice_servers = await self._get_ice_servers()
        self._pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=ice_servers))
        self._tts_track = _TTSAudioTrack()
        self._pc.addTrack(self._tts_track)

        @self._pc.on("track")
        def on_track(track):
            logger.info("call %s: track received kind=%s id=%s readyState=%s",
                        self.call_id[:8], track.kind,
                        getattr(track, "id", "?"), getattr(track, "readyState", "?"))
            if track.kind == "audio":
                # Log codec info from receivers
                for r in self._pc.getReceivers():
                    if r.track == track:
                        logger.debug("call %s: receiver params: %s",
                                     self.call_id[:8], getattr(r, "_track", None))
                asyncio.ensure_future(self._audio_pipeline(track))

        @self._pc.on("connectionstatechange")
        async def on_conn_state():
            logger.info("call %s: connection state → %s",
                        self.call_id[:8], self._pc.connectionState)

        @self._pc.on("iceconnectionstatechange")
        async def on_ice_state():
            state = self._pc.iceConnectionState
            logger.info("call %s: ICE state → %s", self.call_id[:8], state)
            if state == "connected":
                # Cancel reconnect watchdog if ICE recovered
                if self._ice_reconnect_task and not self._ice_reconnect_task.done():
                    self._ice_reconnect_task.cancel()
                    self._ice_reconnect_task = None
            elif state == "disconnected":
                if not self._ice_reconnect_task or self._ice_reconnect_task.done():
                    self._ice_reconnect_task = asyncio.ensure_future(
                        self._ice_reconnect_watchdog()
                    )
            elif state == "failed":
                asyncio.ensure_future(self._notify_disconnect())
                self._done.set()
            elif state == "closed":
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
        logger.debug("call %s: SDP offer (cleaned):\n%s", self.call_id[:8], sdp_offer)
        await self._pc.setRemoteDescription(
            RTCSessionDescription(sdp=sdp_offer, type="offer")
        )

        # Add any candidates that arrived before the offer was processed
        for c in self._pending_candidates:
            await self._add_candidate(c)
        self._pending_candidates.clear()

        answer = await self._pc.createAnswer()
        await self._pc.setLocalDescription(answer)
        logger.debug("call %s: SDP answer:\n%s", self.call_id[:8],
                     self._pc.localDescription.sdp)

        # Load hold audio (background sound while waiting for agent response)
        hold_pcm = self._load_hold_audio()
        if hold_pcm is not None:
            self._tts_track.set_hold_audio(hold_pcm)
            logger.info("call %s: hold audio loaded (%d samples, %.1fs)",
                        self.call_id[:8], len(hold_pcm), len(hold_pcm) / self._tts_track.SAMPLE_RATE)

        # Auto-hangup watchdog
        asyncio.ensure_future(self._watchdog())
        # Send our ICE candidates once gathering completes (parsed from local SDP)
        asyncio.ensure_future(self._flush_local_candidates(_gathering_done))
        # Periodic RTP receiver stats for diagnostics
        asyncio.ensure_future(self._log_receiver_stats())

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
            logger.debug("call %s: local candidate: %s", self.call_id[:8], c["candidate"])
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

                # Convert AudioFrame → float32 mono
                raw_bytes = bytes(frame.planes[0])
                n_channels = max(len(frame.layout.channels), 1)
                n_int16 = frame.samples * n_channels
                raw = np.frombuffer(raw_bytes, dtype=np.int16)[:n_int16]
                if n_channels > 1:
                    # Stereo → mono: average as float (no int16 truncation)
                    pcm = raw.reshape(-1, n_channels).astype(np.float32).mean(axis=1) / 32768.0
                else:
                    pcm = raw.astype(np.float32) / 32768.0

                rms = float(np.sqrt(np.mean(pcm ** 2)))
                if frames_received <= 5:
                    nz_count = int(np.count_nonzero(raw))
                    logger.debug("call %s: frame #%d fmt=%s pts=%s ch=%d "
                                 "pcm_len=%d nz_samples=%d rms=%.4f "
                                 "raw_first10=%s",
                                 self.call_id[:8], frames_received,
                                 frame.format.name, frame.pts, n_channels,
                                 len(pcm), nz_count, rms,
                                 raw[:10].tolist())
                elif frames_received % 50 == 0:
                    import hashlib
                    h = hashlib.md5(pcm.tobytes()).hexdigest()[:8]
                    logger.debug("call %s: frame #%d rms=%.4f buf=%d silence=%d hash=%s",
                                 self.call_id[:8], frames_received, rms,
                                 len(speech_buffer), silence_count, h)

                # Suppress echo: discard mic input while TTS is playing
                if self._tts_track and self._tts_track.is_playing:
                    if speech_buffer:
                        logger.debug("call %s: dropping speech buffer (TTS playing)",
                                     self.call_id[:8])
                        speech_buffer = []
                        silence_count = 0
                    continue

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
        """Transcribe a speech chunk, stream the agent response with sentence-by-sentence TTS."""
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

        # Send transcription immediately so it appears before the response
        await self._send_cb(f"🎙️ *{text}*")

        # Start hold audio while waiting for agent response
        if self._tts_track:
            self._tts_track.start_hold()

        # Keep typing indicator alive (Matrix times it out after ~30s)
        typing_task = asyncio.ensure_future(self._keep_typing())

        try:
            first_sentence_received = False
            call_prompt = self._agent.build_system_prompt(mode="call")

            async def _on_sentence(sentence: str) -> None:
                """Synthesize and enqueue one sentence for immediate TTS playback."""
                nonlocal first_sentence_received
                if not self._tts_track:
                    return
                # Stop hold audio as soon as first real TTS arrives
                if not first_sentence_received:
                    first_sentence_received = True
                    self._tts_track.stop_hold()
                try:
                    tts_pcm = await synthesize_pcm(sentence, self._app.config, sample_rate=48000)
                    if tts_pcm is not None and len(tts_pcm):
                        logger.info("call %s: TTS sentence (%d samples): %s",
                                    self.call_id[:8], len(tts_pcm), sentence[:60])
                        self._tts_track.enqueue_pcm_float32(tts_pcm)
                except Exception as e:
                    logger.warning("call %s: TTS sentence failed: %s", self.call_id[:8], e)

            response = await self._agent.run_streamed(
                text,
                system_prompt=call_prompt,
                thread_id=self.thread_id,
                on_sentence=_on_sentence,
            )
        except Exception as e:
            logger.error("call %s: agent error: %s", self.call_id[:8], e)
            return
        finally:
            typing_task.cancel()
            if self._tts_track:
                self._tts_track.stop_hold()
            try:
                await self._client.room_typing(self.room_id, typing_state=False)
            except Exception:
                pass

        await self._send_cb(response)

    def _load_hold_audio(self) -> Optional["np.ndarray"]:
        """Load hold audio from config and decode to int16 mono PCM at 48 kHz.

        Config: ``tts.hold_audio`` — explicit audio file path.
        Returns ``None`` if not configured or the file cannot be loaded.
        """
        import os
        path = self._app.config.get("tts", {}).get("hold_audio")
        if not path:
            path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "assets",
                "keyboard_mono.wav",
            )
        if not path:
            return None
        if not os.path.exists(path):
            logger.warning("call %s: hold audio file not found: %s", self.call_id[:8], path)
            return None
        try:
            import io
            import av  # type: ignore
            with open(path, "rb") as f:
                data = f.read()
            container = av.open(io.BytesIO(data))
            resampler = av.AudioResampler(format="fltp", layout="mono", rate=48000)
            chunks: List["np.ndarray"] = []
            for frame in container.decode(audio=0):
                for out in resampler.resample(frame):
                    arr = out.to_ndarray()
                    if arr.size:
                        chunks.append(arr[0].astype(np.float32, copy=False))
            for out in resampler.resample(None):
                arr = out.to_ndarray()
                if arr.size:
                    chunks.append(arr[0].astype(np.float32, copy=False))
            if not chunks:
                return None
            pcm = np.concatenate(chunks)
            volume = float(self._app.config.get("tts", {}).get("hold_audio_volume", 0.25))
            if volume != 1.0:
                pcm = np.clip(pcm * volume, -1.0, 1.0)
            return (pcm * 32767).astype(np.int16)
        except Exception as e:
            logger.warning("call %s: failed to load hold audio: %s", self.call_id[:8], e)
            return None

    async def _keep_typing(self) -> None:
        """Periodically refresh the Matrix typing indicator."""
        try:
            while True:
                try:
                    await self._client.room_typing(self.room_id, typing_state=True)
                except Exception:
                    pass
                await asyncio.sleep(15)
        except asyncio.CancelledError:
            pass

    async def _log_receiver_stats(self) -> None:
        """Periodically log RTP receiver stats to diagnose audio delivery."""
        await asyncio.sleep(5)  # wait for connection to establish
        for _ in range(15):  # log for ~75s max
            if self._done.is_set() or not self._pc:
                break
            try:
                stats = await self._pc.getStats()
                for report in stats.values():
                    t = getattr(report, "type", "")
                    if t in ("inbound-rtp", "transport", "candidate-pair"):
                        logger.info("call %s: STATS [%s] %s",
                                    self.call_id[:8], t,
                                    {k: v for k, v in report.__dict__.items()
                                     if not k.startswith("_")})
            except Exception as e:
                logger.debug("call %s: stats error: %s", self.call_id[:8], e)
            await asyncio.sleep(5)

    async def _ice_reconnect_watchdog(self) -> None:
        """Give ICE 30 s to recover from 'disconnected' before ending the call."""
        ICE_RECONNECT_TIMEOUT = 30
        logger.info("call %s: ICE disconnected — waiting %ds for recovery",
                    self.call_id[:8], ICE_RECONNECT_TIMEOUT)
        try:
            await asyncio.sleep(ICE_RECONNECT_TIMEOUT)
        except asyncio.CancelledError:
            logger.info("call %s: ICE recovered — reconnect watchdog cancelled", self.call_id[:8])
            return
        if self._pc and self._pc.iceConnectionState == "disconnected":
            logger.warning("call %s: ICE did not recover after %ds — ending call",
                           self.call_id[:8], ICE_RECONNECT_TIMEOUT)
            await self._notify_disconnect()
            self._done.set()

    async def _notify_disconnect(self) -> None:
        """Send a Matrix message when the connection drops unexpectedly."""
        try:
            await self._send_cb("📞 Verbindung unterbrochen")
        except Exception as e:
            logger.warning("call %s: could not send disconnect notification: %s",
                           self.call_id[:8], e)

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
        self._send_thread_reply = send_thread_reply_cb  # async (room_id, thread_id, text)
        self._get_agent = get_agent_cb                  # (room_id) -> agent
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

        # Create a thread-root message → its event_id becomes the call's thread_id.
        # All transcriptions and responses will be posted as replies into that thread.
        call_thread_id: Optional[str] = None
        try:
            resp = await self._client.room_send(
                room_id=room.room_id,
                message_type="m.room.message",
                content={
                    "msgtype": "m.text",
                    "body": f"📞 Eingehender Anruf von {event.sender}",
                },
            )
            call_thread_id = getattr(resp, "event_id", None)
            logger.info("call %s: thread root event_id=%s", event.call_id[:8], call_thread_id)
        except Exception as e:
            logger.warning("call %s: could not create thread root: %s", event.call_id[:8], e)

        agent = self._get_agent(room.room_id)

        # Build a send callback already bound to the call's thread
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
