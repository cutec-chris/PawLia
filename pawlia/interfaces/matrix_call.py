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

    async def start(self, sdp_offer: str) -> Optional[str]:
        """Accept the call. Returns SDP answer string, or None on error."""
        if not _AIORTC_AVAILABLE:
            logger.error("matrix_call: aiortc not installed — cannot accept call")
            return None

        self._pc = RTCPeerConnection()
        self._tts_track = _TTSAudioTrack()
        self._pc.addTrack(self._tts_track)

        @self._pc.on("track")
        def on_track(track):
            if track.kind == "audio":
                asyncio.ensure_future(self._audio_pipeline(track))

        @self._pc.on("iceconnectionstatechange")
        async def on_ice_state():
            state = self._pc.iceConnectionState
            logger.info("call %s: ICE state → %s", self.call_id[:8], state)
            if state in ("failed", "closed", "disconnected"):
                self._done.set()

        await self._pc.setRemoteDescription(
            RTCSessionDescription(sdp=sdp_offer, type="offer")
        )

        # Add any candidates that arrived before the offer was processed
        for c in self._pending_candidates:
            await self._add_candidate(c)
        self._pending_candidates.clear()

        answer = await self._pc.createAnswer()
        await self._pc.setLocalDescription(answer)

        # Auto-hangup watchdog
        asyncio.ensure_future(self._watchdog())

        logger.info("call %s accepted in room %s", self.call_id[:8], self.room_id)
        return self._pc.localDescription.sdp

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
        try:
            while not self._done.is_set():
                try:
                    frame = await asyncio.wait_for(track.recv(), timeout=5.0)
                except asyncio.TimeoutError:
                    continue
                except MediaStreamError:
                    break

                # Convert AudioFrame → float32 mono
                arr = frame.to_ndarray()  # (channels, samples) or (samples,)
                if arr.ndim > 1:
                    arr = arr.mean(axis=0)
                pcm = arr.astype(np.float32)
                if pcm.dtype == np.int16 or pcm.max() > 1.0:
                    pcm = pcm / 32768.0

                rms = float(np.sqrt(np.mean(pcm ** 2)))

                if rms > self.SILENCE_THRESHOLD:
                    speech_buffer.append(pcm)
                    silence_count = 0
                elif speech_buffer:
                    silence_count += 1
                    speech_buffer.append(pcm)  # keep trailing silence for context

                    if silence_count >= silence_threshold:
                        chunk = np.concatenate(speech_buffer)
                        speech_buffer = []
                        silence_count = 0

                        if len(chunk) >= min_speech_frames * (SAMPLE_RATE // fps):
                            asyncio.ensure_future(
                                self._process_speech(chunk, SAMPLE_RATE)
                            )
        except Exception as e:
            logger.error("call %s: audio pipeline error: %s", self.call_id[:8], e)
        finally:
            self._done.set()
            logger.info("call %s: audio pipeline ended", self.call_id[:8])

    async def _process_speech(self, pcm: "np.ndarray", sample_rate: int) -> None:
        """Transcribe a speech chunk and query the agent."""
        from pawlia.transcription import transcribe_pcm
        from pawlia.tts import synthesize_pcm

        text = await transcribe_pcm(pcm, sample_rate, self._app.config)
        if not text:
            logger.debug("call %s: empty transcription", self.call_id[:8])
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

    async def _add_candidate(self, c: Dict) -> None:
        if not c.get("candidate"):
            return  # end-of-candidates signal
        try:
            candidate = RTCIceCandidate(
                sdpMid=c.get("sdpMid"),
                sdpMLineIndex=c.get("sdpMLineIndex"),
                foundation=c.get("foundation", ""),
                component=int(c.get("component", 1)),
                priority=int(c.get("priority", 0)),
                host=c.get("ip", ""),
                type=c.get("type", "host"),
                port=int(c.get("port", 0)),
                protocol=c.get("protocol", "udp"),
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
