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
from collections import deque
import fractions
import logging
import os
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

try:
    import webrtcvad  # type: ignore
    _WEBRTCVAD_IMPORT_ERROR = None
except Exception as _e:
    webrtcvad = None  # type: ignore[assignment]
    _WEBRTCVAD_IMPORT_ERROR = _e

if TYPE_CHECKING:
    from nio import AsyncClient, MatrixRoom
    from pawlia.app import App

logger = logging.getLogger("pawlia.interfaces.matrix_call")

_INTERRUPT_KEYWORD_RE = re.compile(
    r"\b(?:halt|stop|stopp|wait|warte|warten|moment|sekunde|pause)\b",
    re.IGNORECASE,
)


def _build_webrtc_vad(mode: int):
    """Create a WebRTC VAD instance when the optional dependency is available."""
    if webrtcvad is None:
        return None
    try:
        return webrtcvad.Vad(mode)
    except Exception as e:
        logger.warning("matrix-call: could not initialize webrtcvad(mode=%s): %s", mode, e)
        return None

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
            self._queue: asyncio.Queue[Any] = asyncio.Queue()
            self._pts = 0
            self._time_base = fractions.Fraction(1, self.SAMPLE_RATE)
            self._start_time: Optional[float] = None
            self._next_sentence_id = 1
            self._current_sentence_id: Optional[int] = None
            # Hold audio: looping background sound while waiting for agent
            self._hold_pcm: Optional[np.ndarray] = None  # int16 mono @ 48 kHz
            self._hold_pos: int = 0
            self._hold_active: bool = False

        @property
        def is_playing(self) -> bool:
            """True while TTS or hold audio is playing."""
            return not self._queue.empty() or self._hold_active

        @property
        def is_tts_playing(self) -> bool:
            """True while spoken TTS audio is queued or mid-sentence."""
            return not self._queue.empty() or self._current_sentence_id is not None

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

        def interrupt(self) -> None:
            """Barge-in: clear all queued TTS audio and stop hold immediately."""
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            self._hold_active = False
            self._current_sentence_id = None

        def stop_after_current_sentence(self) -> None:
            """Barge-in: finish the sentence in progress and discard later TTS."""
            self._hold_active = False
            current_sid = self._current_sentence_id
            if current_sid is None:
                self.interrupt()
                return

            kept: List[Any] = []
            while not self._queue.empty():
                try:
                    item = self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if isinstance(item, tuple) and len(item) == 3 and item[1] == current_sid:
                    kept.append(item)
            for item in kept:
                self._queue.put_nowait(item)

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
                item = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                item = None

            if item is None:
                samples = None
            elif isinstance(item, tuple) and len(item) == 3:
                samples, sentence_id, is_last = item
                self._current_sentence_id = sentence_id
                if is_last:
                    self._current_sentence_id = None
            else:
                samples = item

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
            # Debug: Check if we received valid audio data
            if pcm is None or len(pcm) == 0:
                logger.warning("TTS: Received empty or None audio data")
                return

            # Ensure proper range and convert to int16
            pcm_normalized = np.clip(pcm, -1.0, 1.0)
            pcm_int16 = (pcm_normalized * 32767).astype(np.int16)

            # Debug: Log audio statistics
            logger.debug("TTS: Enqueuing audio - samples: %d, min: %.4f, max: %.4f, mean: %.4f",
                       len(pcm), float(np.min(pcm)), float(np.max(pcm)), float(np.mean(pcm)))

            sentence_id = self._next_sentence_id
            self._next_sentence_id += 1
            chunks = [
                pcm_int16[i : i + self.SAMPLES_PER_FRAME]
                for i in range(0, len(pcm_int16), self.SAMPLES_PER_FRAME)
            ]
            for index, chunk in enumerate(chunks):
                if len(chunk) > 0:
                    self._queue.put_nowait((chunk, sentence_id, index == len(chunks) - 1))


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
    """Manages a single active VoIP call.

    Adaptive silence detection
    --------------------------
    Rather than using a fixed RMS threshold to distinguish speech from silence,
    the pipeline maintains a rolling EMA of the background noise floor
    (_noise_floor) during periods when no speech buffer is active.  The
    frame-level gate (_is_speech_like_frame) then uses

        effective_threshold = max(SILENCE_THRESHOLD, _noise_floor × 2.0)

    so that steady background noise (e.g. road noise while cycling) falls
    below the effective threshold and counts as silence.  This lets
    silence_count accumulate even when the raw RMS never drops to zero,
    preventing speech chunks from growing indefinitely.

    Adaptive response delay
    -----------------------
    After a speech chunk is accepted and transcribed, the pipeline waits
    _compute_response_delay() seconds before generating a reply.  The delay
    scales with the duration of the last accepted chunk so that long monologues
    get a longer pause window (the caller may just be thinking), while short
    utterances get a quick response.  A small bonus is added when the noise
    floor is elevated, since background noise can mask the true end of speech.

        last_speech_duration  →  base delay
        < 6 s                    RESPONSE_DELAY_SECONDS (default 1.2 s)
        6 – 12 s                 max(base, 3.0 s)
        12 – 20 s                max(base, 4.0 s)
        > 20 s                   max(base, 5.0 s)
    """

    # Silence detection: RMS below this (after AGC) → silence
    SILENCE_THRESHOLD = 0.018
    # Seconds of silence that end a speech chunk
    SILENCE_SECONDS = 1.5
    # Minimum seconds of speech before we transcribe (filter short noise bursts)
    MIN_SPEECH_SECONDS = 0.4
    # Chunk-level guard: require enough active speech frames before STT
    MIN_ACTIVE_SPEECH_RATIO = 0.12
    MIN_CONSECUTIVE_SPEECH_FRAMES = 8
    MIN_SPEECH_BAND_RATIO = 0.35
    MAX_SPECTRAL_FLATNESS = 0.72
    MIN_SPEECH_LIKE_RATIO = 0.08
    MIN_CONSECUTIVE_SPEECHLIKE_FRAMES = 4
    # Require a short sustained return to speech before a pause is treated as
    # broken. This prevents single noisy frames from stretching a chunk for
    # many extra seconds.
    MIN_RESUME_SPEECH_FRAMES = 3
    # Keep a small lead-in before detected speech so STT does not miss word
    # onsets that arrive just before the VAD is confident enough to start.
    PRE_SPEECH_SECONDS = 0.4
    WEBRTC_VAD_ENABLED = True
    WEBRTC_VAD_MODE = 2
    WEBRTC_VAD_MIN_VOICED_RATIO = 0.12
    WEBRTC_VAD_MIN_CONSECUTIVE_FRAMES = 4
    # End calls when no speech chunk has been sent to STT for too long.
    CALL_INACTIVITY_SECONDS = 180
    WATCHDOG_POLL_SECONDS = 5.0
    # AGC: boost gain in windows where we expect the user to speak
    AGC_WINDOW_SECONDS = 15.0     # how long the AGC stays active
    AGC_TARGET_RMS = 0.10         # target RMS level for normalization
    AGC_MAX_GAIN = 12.0           # don't amplify more than this
    AGC_SMOOTHING = 0.15          # EMA alpha for gain updates (higher = faster)
    # Barge-in: buffer possible interruptions during TTS above this RMS and
    # only stop playback after the transcript looks meaningful.
    BARGEIN_RMS_THRESHOLD = 0.05
    BARGEIN_MIN_WORDS = 4
    BARGEIN_MIN_CHARS = 12
    # Pre-answer warmup: load STT and prepare the first greeting before Matrix
    # sees the call as answered so the caller does not hear a long cold start.
    PREANSWER_WARMUP_ENABLED = True
    PREANSWER_WARMUP_TIMEOUT_SECONDS = 25.0
    PREANSWER_STT_SILENCE_SECONDS = 0.4
    CONNECT_TIMEOUT_SECONDS = 45.0
    HANGUP_ON_MEDIA_END = True
    # Wait this long after the user's latest speech before replying.  This
    # lets callers tell a longer story without the agent jumping into every
    # pause that was only used for breathing or thinking.
    RESPONSE_DELAY_SECONDS = 1.2

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
        self._hungup = False
        self._pending_candidates: List[Dict] = []
        self._speaking = False
        self._ice_reconnect_task: Optional[asyncio.Task] = None
        self._last_activity_at = time.monotonic()
        self._last_user_speech_at = self._last_activity_at
        # AGC state
        self._agc_until: float = 0.0   # monotonic timestamp; AGC active while now < this
        self._agc_gain: float = 1.0    # current smoothed gain factor
        self._active_response_task: Optional[asyncio.Task] = None
        self._greeting_sent = False
        self._answer_sent = asyncio.Event()
        self._media_connected = asyncio.Event()
        self._prepared_greeting: Optional[tuple[str, List[np.ndarray]]] = None
        self._prepare_greeting_task: Optional[asyncio.Task] = None
        self._greeting_task: Optional[asyncio.Task] = None
        self._pending_response_task: Optional[asyncio.Task] = None
        self._pending_transcripts: List[str] = []
        # Adaptive silence detection: rolling EMA of background noise floor
        self._noise_floor: float = 0.01
        # Duration of the most recently accepted speech chunk (seconds)
        self._last_speech_duration: float = 0.0
        self._load_voip_audio_config()
        self._webrtc_vad = self._init_webrtc_vad()

    def _mark_activity(self) -> None:
        """Record user or bot activity to keep the call alive."""
        self._last_activity_at = time.monotonic()

    def _mark_user_speech_started(self) -> None:
        self._speaking = True
        self._last_user_speech_at = time.monotonic()
        self._mark_activity()

    def _mark_user_speech_ended(self) -> None:
        self._speaking = False
        self._last_user_speech_at = time.monotonic()
        self._mark_activity()

    def _bot_is_active(self) -> bool:
        """Return True while Pawlia is still speaking or generating audio."""
        if self._tts_track and self._tts_track.is_playing:
            return True
        task = self._active_response_task
        return bool(task and not task.done())

    def _activate_agc(self) -> None:
        """Open an AGC window for the next AGC_WINDOW_SECONDS."""
        self._agc_until = time.monotonic() + self.AGC_WINDOW_SECONDS

    @property
    def _agc_active(self) -> bool:
        return time.monotonic() < self._agc_until

    def _agc_rms(self, raw_rms: float) -> float:
        """Return the AGC-adjusted RMS for VAD decisions.

        When the AGC window is inactive, returns raw_rms unchanged.
        When active, tracks a smoothed gain factor that brings the signal
        toward AGC_TARGET_RMS and returns ``raw_rms * gain``.
        """
        if not self._agc_active:
            self._agc_gain = 1.0
            return raw_rms

        if raw_rms > 1e-6:
            ideal_gain = self.AGC_TARGET_RMS / raw_rms
            ideal_gain = min(ideal_gain, self.AGC_MAX_GAIN)
            alpha = self.AGC_SMOOTHING
            self._agc_gain = alpha * ideal_gain + (1 - alpha) * self._agc_gain

        return raw_rms * self._agc_gain

    def _voice_override(self) -> Optional[str]:
        """Return the user's persistent TTS voice override (if any)."""
        try:
            session = self._app.memory.load_session(f"mx_{self.room_id}")
            return session.voice_override
        except Exception:
            return None

    def _load_voip_audio_config(self) -> None:
        """Apply per-instance VAD/STT gating thresholds from shared VoIP config."""
        app_cfg = self._app.config if isinstance(self._app.config, dict) else {}
        voip_cfg = app_cfg.get("voip", {}) if isinstance(app_cfg, dict) else {}
        if not isinstance(voip_cfg, dict):
            logger.warning("call %s: ignoring non-dict voip config", self.call_id[:8])
            voip_cfg = {}

        self.SILENCE_THRESHOLD = self._get_float_config(
            voip_cfg,
            "silence_threshold",
            self.SILENCE_THRESHOLD,
            minimum=0.0,
        )
        self.SILENCE_SECONDS = self._get_float_config(
            voip_cfg,
            "silence_seconds",
            self.SILENCE_SECONDS,
            minimum=0.1,
        )
        self.MIN_SPEECH_SECONDS = self._get_float_config(
            voip_cfg,
            "min_speech_seconds",
            self.MIN_SPEECH_SECONDS,
            minimum=0.1,
        )
        self.MIN_ACTIVE_SPEECH_RATIO = self._get_float_config(
            voip_cfg,
            "min_active_speech_ratio",
            self.MIN_ACTIVE_SPEECH_RATIO,
            minimum=0.0,
            maximum=1.0,
        )
        self.MIN_CONSECUTIVE_SPEECH_FRAMES = self._get_int_config(
            voip_cfg,
            "min_consecutive_speech_frames",
            self.MIN_CONSECUTIVE_SPEECH_FRAMES,
            minimum=1,
        )
        self.MIN_SPEECH_BAND_RATIO = self._get_float_config(
            voip_cfg,
            "min_speech_band_ratio",
            self.MIN_SPEECH_BAND_RATIO,
            minimum=0.0,
            maximum=1.0,
        )
        self.MAX_SPECTRAL_FLATNESS = self._get_float_config(
            voip_cfg,
            "max_spectral_flatness",
            self.MAX_SPECTRAL_FLATNESS,
            minimum=0.0,
            maximum=1.0,
        )
        self.MIN_SPEECH_LIKE_RATIO = self._get_float_config(
            voip_cfg,
            "min_speech_like_ratio",
            self.MIN_SPEECH_LIKE_RATIO,
            minimum=0.0,
            maximum=1.0,
        )
        self.MIN_CONSECUTIVE_SPEECHLIKE_FRAMES = self._get_int_config(
            voip_cfg,
            "min_consecutive_speechlike_frames",
            self.MIN_CONSECUTIVE_SPEECHLIKE_FRAMES,
            minimum=1,
        )
        self.MIN_RESUME_SPEECH_FRAMES = self._get_int_config(
            voip_cfg,
            "min_resume_speech_frames",
            self.MIN_RESUME_SPEECH_FRAMES,
            minimum=1,
        )
        self.PRE_SPEECH_SECONDS = self._get_float_config(
            voip_cfg,
            "pre_speech_seconds",
            self.PRE_SPEECH_SECONDS,
            minimum=0.0,
        )
        self.WEBRTC_VAD_ENABLED = self._get_bool_config(
            voip_cfg,
            "webrtcvad_enabled",
            self.WEBRTC_VAD_ENABLED,
        )
        self.WEBRTC_VAD_MODE = self._get_int_config(
            voip_cfg,
            "webrtcvad_mode",
            self.WEBRTC_VAD_MODE,
            minimum=0,
            maximum=3,
        )
        self.WEBRTC_VAD_MIN_VOICED_RATIO = self._get_float_config(
            voip_cfg,
            "webrtcvad_min_voiced_ratio",
            self.WEBRTC_VAD_MIN_VOICED_RATIO,
            minimum=0.0,
            maximum=1.0,
        )
        self.WEBRTC_VAD_MIN_CONSECUTIVE_FRAMES = self._get_int_config(
            voip_cfg,
            "webrtcvad_min_consecutive_frames",
            self.WEBRTC_VAD_MIN_CONSECUTIVE_FRAMES,
            minimum=1,
        )
        self.CALL_INACTIVITY_SECONDS = self._get_int_config(
            voip_cfg,
            "call_inactivity_seconds",
            self.CALL_INACTIVITY_SECONDS,
            minimum=1,
        )
        self.AGC_WINDOW_SECONDS = self._get_float_config(
            voip_cfg,
            "agc_window_seconds",
            self.AGC_WINDOW_SECONDS,
            minimum=0.1,
        )
        self.AGC_TARGET_RMS = self._get_float_config(
            voip_cfg,
            "agc_target_rms",
            self.AGC_TARGET_RMS,
            minimum=0.001,
        )
        self.AGC_MAX_GAIN = self._get_float_config(
            voip_cfg,
            "agc_max_gain",
            self.AGC_MAX_GAIN,
            minimum=1.0,
        )
        self.AGC_SMOOTHING = self._get_float_config(
            voip_cfg,
            "agc_smoothing",
            self.AGC_SMOOTHING,
            minimum=0.001,
            maximum=1.0,
        )
        self.BARGEIN_RMS_THRESHOLD = self._get_float_config(
            voip_cfg,
            "bargein_rms_threshold",
            self.BARGEIN_RMS_THRESHOLD,
            minimum=0.0,
        )
        self.PREANSWER_WARMUP_ENABLED = self._get_bool_config(
            voip_cfg,
            "preanswer_warmup_enabled",
            self.PREANSWER_WARMUP_ENABLED,
        )
        self.PREANSWER_WARMUP_TIMEOUT_SECONDS = self._get_float_config(
            voip_cfg,
            "preanswer_warmup_timeout_seconds",
            self.PREANSWER_WARMUP_TIMEOUT_SECONDS,
            minimum=0.1,
        )
        self.PREANSWER_STT_SILENCE_SECONDS = self._get_float_config(
            voip_cfg,
            "preanswer_stt_silence_seconds",
            self.PREANSWER_STT_SILENCE_SECONDS,
            minimum=0.05,
        )
        self.RESPONSE_DELAY_SECONDS = self._get_float_config(
            voip_cfg,
            "response_delay_seconds",
            self.RESPONSE_DELAY_SECONDS,
            minimum=0.0,
        )
        self.CONNECT_TIMEOUT_SECONDS = self._get_float_config(
            voip_cfg,
            "connect_timeout_seconds",
            self.CONNECT_TIMEOUT_SECONDS,
            minimum=1.0,
        )
        self.HANGUP_ON_MEDIA_END = self._get_bool_config(
            voip_cfg,
            "hangup_on_media_end",
            self.HANGUP_ON_MEDIA_END,
        )

    def _init_webrtc_vad(self):
        """Initialize the optional WebRTC VAD instance from config."""
        if not self.WEBRTC_VAD_ENABLED:
            return None
        vad = _build_webrtc_vad(self.WEBRTC_VAD_MODE)
        if vad is None and _WEBRTCVAD_IMPORT_ERROR is not None:
            logger.info(
                "call %s: webrtcvad unavailable, continuing without it: %s",
                self.call_id[:8],
                _WEBRTCVAD_IMPORT_ERROR,
            )
        return vad

    def _get_float_config(
        self,
        cfg: Dict[str, Any],
        key: str,
        default: float,
        minimum: Optional[float] = None,
        maximum: Optional[float] = None,
    ) -> float:
        value = cfg.get(key, default)
        try:
            value = float(value)
        except (TypeError, ValueError):
            logger.warning(
                "call %s: invalid voip.%s=%r, using default %s",
                self.call_id[:8],
                key,
                value,
                default,
            )
            return default

        if minimum is not None and value < minimum:
            logger.warning(
                "call %s: voip.%s=%s below minimum %s, using default %s",
                self.call_id[:8],
                key,
                value,
                minimum,
                default,
            )
            return default
        if maximum is not None and value > maximum:
            logger.warning(
                "call %s: voip.%s=%s above maximum %s, using default %s",
                self.call_id[:8],
                key,
                value,
                maximum,
                default,
            )
            return default
        return value

    def _get_bool_config(
        self,
        cfg: Dict[str, Any],
        key: str,
        default: bool,
    ) -> bool:
        value = cfg.get(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
        logger.warning(
            "call %s: invalid boolean voip.%s=%r — using default %r",
            self.call_id[:8],
            key,
            value,
            default,
        )
        return default

    def _get_int_config(
        self,
        cfg: Dict[str, Any],
        key: str,
        default: int,
        minimum: Optional[int] = None,
        maximum: Optional[int] = None,
    ) -> int:
        value = cfg.get(key, default)
        try:
            value = int(value)
        except (TypeError, ValueError):
            logger.warning(
                "call %s: invalid voip.%s=%r, using default %s",
                self.call_id[:8],
                key,
                value,
                default,
            )
            return default

        if minimum is not None and value < minimum:
            logger.warning(
                "call %s: voip.%s=%s below minimum %s, using default %s",
                self.call_id[:8],
                key,
                value,
                minimum,
                default,
            )
            return default
        if maximum is not None and value > maximum:
            logger.warning(
                "call %s: voip.%s=%s above maximum %s, using default %s",
                self.call_id[:8],
                key,
                value,
                maximum,
                default,
            )
            return default
        return value

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
            elif track.kind == "video":
                # Drain decoded video frames so aiortc doesn't buffer them indefinitely.
                # Replace with frame-processing logic once video input to the model is implemented.
                asyncio.ensure_future(self._drain_video_track(track))

        @self._pc.on("connectionstatechange")
        async def on_conn_state():
            state = self._pc.connectionState
            logger.info("call %s: connection state → %s",
                        self.call_id[:8], state)
            if state == "connected":
                self._media_connected.set()

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

        await self._run_preanswer_warmup()

        # Auto-hangup watchdog
        asyncio.ensure_future(self._watchdog())
        asyncio.ensure_future(self._connect_timeout_watchdog())
        # Send our ICE candidates once gathering completes (parsed from local SDP)
        asyncio.ensure_future(self._flush_local_candidates(_gathering_done))
        # Periodic RTP receiver stats for diagnostics
        asyncio.ensure_future(self._log_receiver_stats())
        # Greet the caller once signaling and media are both ready.  Warmup may
        # prepare the LLM/TTS result ahead of time, but playback is gated here.
        if not self._greeting_sent:
            self._greeting_task = asyncio.ensure_future(self._send_greeting_when_ready())

        logger.info("call %s accepted in room %s", self.call_id[:8], self.room_id)
        return self._pc.localDescription.sdp

    async def _run_preanswer_warmup(self) -> None:
        """Warm STT and prepare greeting audio before the call is answered."""
        if not self.PREANSWER_WARMUP_ENABLED:
            return

        started = time.monotonic()
        stt_task = asyncio.create_task(self._warm_stt_with_silence())
        greeting_task = self._ensure_prepare_greeting_task()
        task = asyncio.gather(stt_task, asyncio.shield(greeting_task))
        try:
            await asyncio.wait_for(task, timeout=self.PREANSWER_WARMUP_TIMEOUT_SECONDS)
            logger.info(
                "call %s: pre-answer warmup finished in %.1fs",
                self.call_id[:8],
                time.monotonic() - started,
            )
        except asyncio.TimeoutError:
            stt_task.cancel()
            logger.warning(
                "call %s: pre-answer warmup timed out after %.1fs; answering anyway"
                " (greeting warmup continues in background)",
                self.call_id[:8],
                self.PREANSWER_WARMUP_TIMEOUT_SECONDS,
            )
        finally:
            if stt_task.done():
                try:
                    await stt_task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    pass

    def _ensure_prepare_greeting_task(self) -> asyncio.Task:
        """Start greeting warmup once and reuse it across answer/greeting flow."""
        task = self._prepare_greeting_task
        if task is None or task.done():
            task = asyncio.create_task(self._prepare_greeting())
            self._prepare_greeting_task = task
        return task

    async def _warm_stt_with_silence(self) -> None:
        """Send a short silent WAV through the active STT path to trigger loading."""
        try:
            sample_rate = 16000
            sample_count = max(1, int(sample_rate * self.PREANSWER_STT_SILENCE_SECONDS))
            silence = np.zeros(sample_count, dtype=np.float32)

            active_model = None
            active_override = getattr(self._agent, "_active_override_model", None)
            if callable(active_override):
                active_model = active_override(self.thread_id)
            llm_factory = getattr(self._app, "llm", None)
            audio_model_info = getattr(llm_factory, "audio_model_info", None)
            audio_info = (
                audio_model_info(active_model or "chat")
                if callable(audio_model_info)
                else None
            )
            if audio_info:
                from pawlia.transcription import transcribe_pcm_via_model
                await transcribe_pcm_via_model(
                    silence,
                    sample_rate,
                    audio_info[0],
                    audio_info[1],
                    prompt="This is a silent warmup audio clip. Return an empty response.",
                )
            else:
                from pawlia.transcription import transcribe_pcm
                await transcribe_pcm(silence, sample_rate, self._app.config)
            logger.info("call %s: STT warmup completed", self.call_id[:8])
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("call %s: STT warmup failed: %s", self.call_id[:8], e)

    async def mark_answer_sent(self) -> None:
        """Mark that Matrix accepted our SDP answer event for this call."""
        self._answer_sent.set()

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
            ignore_unverified_devices=True,
        )
        logger.info("call %s: sent %d local ICE candidates", self.call_id[:8], len(candidates))

    async def _prepare_greeting(self) -> None:
        """Generate greeting text and TTS audio without playing it yet."""
        if self._greeting_sent or self._prepared_greeting is not None:
            return

        try:
            from pawlia.tts import synthesize_pcm
        except ImportError:
            logger.debug("call %s: TTS not available, skipping greeting warmup", self.call_id[:8])
            return

        try:
            call_prompt = self._agent.build_system_prompt(mode="call", thread_id=self.thread_id)
            greeting_input = (
                "[SYSTEM: A voice call was just accepted. "
                "Greet the caller with a short, friendly greeting. "
                "Keep the established persona and preferred form of address from the profile/history. "
                "If speaking German and there is no explicit preference, use informal 'du', not formal 'Sie'. "
                "Keep it to one or two sentences.]"
            )
            prepared_pcm: List[np.ndarray] = []

            async def _on_sentence(sentence: str) -> None:
                try:
                    tts_pcm = await synthesize_pcm(
                        sentence, self._app.config, sample_rate=48000,
                        voice_override=self._voice_override(),
                    )
                    if tts_pcm is not None and len(tts_pcm):
                        logger.info(
                            "call %s: prepared greeting TTS (%d samples): %s",
                            self.call_id[:8], len(tts_pcm), sentence[:60],
                        )
                        prepared_pcm.append(tts_pcm)
                except Exception as e:
                    logger.warning("call %s: greeting TTS warmup failed: %s", self.call_id[:8], e)

            response = await self._agent.run_streamed(
                greeting_input,
                system_prompt=call_prompt,
                thread_id=self.thread_id,
                on_sentence=_on_sentence,
                allow_skills=False,
            )
            self._prepared_greeting = (response, prepared_pcm)
            logger.info("call %s: greeting prepared", self.call_id[:8])
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("call %s: greeting warmup failed: %s", self.call_id[:8], e)

    async def _send_greeting(self) -> None:
        """Generate and play a greeting via LLM + TTS when the call is accepted."""
        if self._greeting_sent:
            return

        task = self._prepare_greeting_task
        if self._prepared_greeting is None and task and not task.done():
            try:
                await task
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning("call %s: awaiting greeting warmup failed: %s", self.call_id[:8], e)

        if self._prepared_greeting is not None:
            response, prepared_pcm = self._prepared_greeting
            if self._tts_track:
                for tts_pcm in prepared_pcm:
                    self._tts_track.enqueue_pcm_float32(tts_pcm)
                if prepared_pcm:
                    self._tts_track.stop_hold()
            await self._send_cb(response)
            self._greeting_sent = True
            self._mark_activity()
            self._activate_agc()
            logger.info("call %s: prepared greeting sent", self.call_id[:8])
            return

        try:
            from pawlia.tts import synthesize_pcm
        except ImportError:
            logger.debug("call %s: TTS not available, skipping greeting", self.call_id[:8])
            return

        if not self._tts_track:
            return

        try:
            call_prompt = self._agent.build_system_prompt(mode="call", thread_id=self.thread_id)
            greeting_input = (
                "[SYSTEM: A voice call was just accepted. "
                "Greet the caller with a short, friendly greeting. "
                "Keep the established persona and preferred form of address from the profile/history. "
                "If speaking German and there is no explicit preference, use informal 'du', not formal 'Sie'. "
                "Keep it to one or two sentences.]"
            )

            async def _on_sentence(sentence: str) -> None:
                if not self._tts_track or self._done.is_set() or self._hungup:
                    return
                try:
                    tts_pcm = await synthesize_pcm(
                        sentence, self._app.config, sample_rate=48000,
                        voice_override=self._voice_override(),
                    )
                    if self._done.is_set() or self._hungup:
                        logger.info(
                            "call %s: greeting TTS dropped (%d samples) — call ended before playback: %s",
                            self.call_id[:8], len(tts_pcm) if tts_pcm is not None else 0, sentence[:60],
                        )
                        return
                    if tts_pcm is not None and len(tts_pcm):
                        logger.info(
                            "call %s: greeting TTS (%d samples): %s",
                            self.call_id[:8], len(tts_pcm), sentence[:60],
                        )
                        self._tts_track.enqueue_pcm_float32(tts_pcm)
                        self._tts_track.stop_hold()
                except Exception as e:
                    logger.warning("call %s: greeting TTS failed: %s", self.call_id[:8], e)

            response = await self._agent.run_streamed(
                greeting_input,
                system_prompt=call_prompt,
                thread_id=self.thread_id,
                on_sentence=_on_sentence,
                allow_skills=False,
            )
            if self._done.is_set() or self._hungup:
                logger.info("call %s: greeting completed after hangup — not posting to room", self.call_id[:8])
                return
            await self._send_cb(response)
            self._greeting_sent = True
            self._mark_activity()
            self._activate_agc()
            logger.info("call %s: greeting sent", self.call_id[:8])
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("call %s: greeting failed: %s", self.call_id[:8], e)

    async def _send_greeting_when_ready(self) -> None:
        """Send the greeting only after signaling and media are both ready."""
        if self._greeting_sent:
            return
        try:
            await self._answer_sent.wait()
            await self._media_connected.wait()
            if not self._done.is_set():
                await self._send_greeting()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("call %s: deferred greeting failed: %s", self.call_id[:8], e)

    async def add_candidates(self, candidates: List[Dict]) -> None:
        """Feed ICE candidates from ``m.call.candidates``."""
        for c in candidates:
            if self._pc and self._pc.remoteDescription:
                await self._add_candidate(c)
            else:
                self._pending_candidates.append(c)

    async def hangup(self) -> None:
        """Terminate the call. Idempotent: safe to call multiple times.

        The watchdog and the remote peer can both trigger a hangup for the same
        call; without this guard `pc.close()` runs twice and aioice leaks TURN
        CHANNEL_BIND tasks that then flood the log with `socket.send()`
        exceptions (seen 2026-04-17 around the pi hang).
        """
        if self._hungup:
            return
        self._hungup = True
        self._done.set()
        if self._greeting_task and not self._greeting_task.done():
            self._greeting_task.cancel()
        if self._prepare_greeting_task and not self._prepare_greeting_task.done():
            self._prepare_greeting_task.cancel()
        if self._pc:
            await self._pc.close()
        logger.info("call %s hung up", self.call_id[:8])

    @property
    def finished(self) -> bool:
        """True once this session cannot usefully accept more signaling."""
        if self._hungup or self._done.is_set():
            return True
        if self._pc and self._pc.connectionState in {"closed", "failed"}:
            return True
        return False

    # ------------------------------------------------------------------
    # Internal: audio pipeline
    # ------------------------------------------------------------------

    def _analyze_speech_chunk(
        self,
        pcm: "np.ndarray",
        sample_rate: int,
        fps: int,
    ) -> Dict[str, float]:
        """Summarize frame-level activity for a completed speech chunk."""
        frame_size = max(sample_rate // fps, 1)
        usable = len(pcm) // frame_size * frame_size
        if usable < frame_size:
            return {
                "frame_count": 0.0,
                "active_frames": 0.0,
                "active_ratio": 0.0,
                "longest_run": 0.0,
                "voiced_frames": 0.0,
                "voiced_ratio": 0.0,
                "voiced_run": 0.0,
                "speech_like_frames": 0.0,
                "speech_like_ratio": 0.0,
                "speech_like_run": 0.0,
                "median_band_ratio": 0.0,
                "median_flatness": 1.0,
                "p90_rms": 0.0,
            }

        framed = pcm[:usable].reshape(-1, frame_size)
        frame_rms = np.sqrt(np.mean(framed ** 2, axis=1))
        # Apply current AGC gain so chunk analysis matches frame-level VAD
        if self._agc_active:
            frame_rms = frame_rms * self._agc_gain
        active_mask = frame_rms > self.SILENCE_THRESHOLD

        longest_run = 0
        current_run = 0
        for is_active in active_mask:
            current_run = current_run + 1 if is_active else 0
            longest_run = max(longest_run, current_run)

        voiced_mask = np.zeros(len(frame_rms), dtype=bool)
        if self._webrtc_vad is not None:
            for idx, frame in enumerate(framed):
                frame_int16 = (np.clip(frame, -1.0, 1.0) * 32767.0).astype(np.int16)
                try:
                    voiced_mask[idx] = bool(self._webrtc_vad.is_speech(frame_int16.tobytes(), sample_rate))
                except Exception as e:
                    logger.debug("call %s: webrtcvad frame analysis failed: %s", self.call_id[:8], e)
                    voiced_mask = np.zeros(len(frame_rms), dtype=bool)
                    break

        voiced_run = 0
        current_voiced_run = 0
        for is_voiced in voiced_mask:
            current_voiced_run = current_voiced_run + 1 if is_voiced else 0
            voiced_run = max(voiced_run, current_voiced_run)

        window = np.hanning(frame_size).astype(np.float32)
        spectra = np.fft.rfft(framed * window[None, :], axis=1)
        power = np.abs(spectra) ** 2
        freqs = np.fft.rfftfreq(frame_size, d=1.0 / sample_rate)
        speech_band = (freqs >= 180.0) & (freqs <= 4000.0)
        total_power = np.maximum(np.sum(power, axis=1), 1e-9)
        band_ratio = np.sum(power[:, speech_band], axis=1) / total_power
        flatness = np.exp(np.mean(np.log(power + 1e-9), axis=1)) / np.maximum(np.mean(power + 1e-9, axis=1), 1e-9)
        speech_like_mask = active_mask & (band_ratio >= self.MIN_SPEECH_BAND_RATIO) & (flatness <= self.MAX_SPECTRAL_FLATNESS)

        speech_like_run = 0
        current_speech_like_run = 0
        for is_speech_like in speech_like_mask:
            current_speech_like_run = current_speech_like_run + 1 if is_speech_like else 0
            speech_like_run = max(speech_like_run, current_speech_like_run)

        return {
            "frame_count": float(len(frame_rms)),
            "active_frames": float(np.count_nonzero(active_mask)),
            "active_ratio": float(np.mean(active_mask)),
            "longest_run": float(longest_run),
            "voiced_frames": float(np.count_nonzero(voiced_mask)),
            "voiced_ratio": float(np.mean(voiced_mask)),
            "voiced_run": float(voiced_run),
            "speech_like_frames": float(np.count_nonzero(speech_like_mask)),
            "speech_like_ratio": float(np.mean(speech_like_mask)),
            "speech_like_run": float(speech_like_run),
            "median_band_ratio": float(np.median(band_ratio)),
            "median_flatness": float(np.median(flatness)),
            "p90_rms": float(np.percentile(frame_rms, 90)),
        }

    def _is_speech_like_frame(
        self,
        pcm: "np.ndarray",
        sample_rate: int,
        adjusted_rms: float,
    ) -> bool:
        """Return True when a live frame looks like speech, not just loud noise."""
        # Raise silence threshold dynamically when background noise is present.
        # noise_floor is an EMA of RMS during non-speech periods; a frame whose
        # RMS is less than 2× the measured floor is treated as silence so that
        # steady background noise (e.g. cycling) does not prevent silence_count
        # from accumulating.
        effective_threshold = max(self.SILENCE_THRESHOLD, self._noise_floor * 2.0)
        if adjusted_rms <= effective_threshold:
            return False

        if len(pcm) < 2:
            return False

        window = np.hanning(len(pcm)).astype(np.float32)
        spectrum = np.fft.rfft(pcm * window)
        power = np.abs(spectrum) ** 2
        total_power = float(np.sum(power))
        if total_power <= 1e-9:
            return False

        freqs = np.fft.rfftfreq(len(pcm), d=1.0 / sample_rate)
        speech_band = (freqs >= 180.0) & (freqs <= 4000.0)
        band_ratio = float(np.sum(power[speech_band]) / total_power)
        flatness = float(
            np.exp(np.mean(np.log(power + 1e-9)))
            / max(float(np.mean(power + 1e-9)), 1e-9)
        )
        return (
            band_ratio >= self.MIN_SPEECH_BAND_RATIO
            and flatness <= self.MAX_SPECTRAL_FLATNESS
        )

    def _should_transcribe_chunk(
        self,
        pcm: "np.ndarray",
        sample_rate: int,
        fps: int,
    ) -> bool:
        """Return True only when a chunk contains sustained speech-like activity."""
        stats = self._analyze_speech_chunk(pcm, sample_rate, fps)
        basic_match = (
            stats["active_ratio"] >= self.MIN_ACTIVE_SPEECH_RATIO
            and stats["longest_run"] >= self.MIN_CONSECUTIVE_SPEECH_FRAMES
            and stats["speech_like_ratio"] >= self.MIN_SPEECH_LIKE_RATIO
            and stats["speech_like_run"] >= self.MIN_CONSECUTIVE_SPEECHLIKE_FRAMES
        )
        if not basic_match:
            return False
        if self._webrtc_vad is None:
            return True
        return (
            stats["voiced_ratio"] >= self.WEBRTC_VAD_MIN_VOICED_RATIO
            and stats["voiced_run"] >= self.WEBRTC_VAD_MIN_CONSECUTIVE_FRAMES
        )

    def _resume_speech_after_pause(
        self,
        speech_like_frame: bool,
        silence_count: int,
        resume_speech_count: int,
    ) -> tuple[bool, int]:
        """Require a short sustained return before breaking an in-progress pause."""
        if not speech_like_frame or silence_count <= 0:
            return False, 0
        resume_speech_count += 1
        return resume_speech_count >= self.MIN_RESUME_SPEECH_FRAMES, resume_speech_count

    def _start_speech_buffer(
        self,
        pre_speech_buffer: "deque[np.ndarray]",
        pcm: "np.ndarray",
    ) -> List["np.ndarray"]:
        """Seed a new speech chunk with a small pre-roll before the trigger frame."""
        chunk = list(pre_speech_buffer)
        chunk.append(pcm)
        return chunk

    def _is_meaningful_interrupt(self, text: str) -> bool:
        """Return True when a transcript is strong enough to justify barge-in."""
        normalized = " ".join((text or "").strip().split())
        if not normalized:
            return False
        if _INTERRUPT_KEYWORD_RE.search(normalized):
            return True

        words = re.findall(r"\b\w+\b", normalized, flags=re.UNICODE)
        if len(words) >= self.BARGEIN_MIN_WORDS and len(normalized) >= self.BARGEIN_MIN_CHARS:
            return True
        if len(words) >= 3 and bool(re.search(r"[.!?…]\s*$", normalized)):
            return True
        return False

    def _track_response_task(self, task: asyncio.Task) -> None:
        """Remember the currently active speech-response task."""
        self._active_response_task = task

        def _clear(done_task: asyncio.Task) -> None:
            if self._active_response_task is done_task:
                self._active_response_task = None

        task.add_done_callback(_clear)

    def _compute_response_delay(self) -> float:
        """Return how long to wait before replying after the user goes quiet.

        Scales with the duration of the last accepted speech chunk: a longer
        monologue gets a longer pause window so brief thinking gaps are not
        mistaken for end-of-turn.  Also adds a small bonus when significant
        background noise is present so transient noise dips don't cut off early.
        """
        base = self.RESPONSE_DELAY_SECONDS
        dur = self._last_speech_duration
        if dur > 20.0:
            base = max(base, 5.0)
        elif dur > 12.0:
            base = max(base, 4.0)
        elif dur > 6.0:
            base = max(base, 3.0)
        # Small bonus when noise floor is elevated (background noise context)
        noise_ratio = self._noise_floor / max(self.SILENCE_THRESHOLD, 1e-4)
        if noise_ratio > 1.5:
            base += min((noise_ratio - 1.5) * 0.5, 1.5)
        return base

    async def _cancel_active_response(self) -> None:
        """Cancel any in-flight response generation for this call."""
        pending = self._pending_response_task
        current = asyncio.current_task()
        if pending and not pending.done() and pending is not current:
            pending.cancel()
            try:
                await pending
            except asyncio.CancelledError:
                logger.info("call %s: pending response cancelled", self.call_id[:8])
            except Exception as e:
                logger.debug("call %s: pending response cancel cleanup failed: %s", self.call_id[:8], e)

        task = self._active_response_task
        if not task or task.done() or task is current:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            logger.info("call %s: active response cancelled", self.call_id[:8])
        except Exception as e:
            logger.debug("call %s: active response cancel cleanup failed: %s", self.call_id[:8], e)

    async def _transcribe_speech(self, pcm: "np.ndarray", sample_rate: int) -> Optional[str]:
        """Transcribe raw speech audio to text."""
        try:
            from pawlia.transcription import transcribe_pcm
        except ImportError as e:
            logger.error("call %s: missing dependency: %s", self.call_id[:8], e)
            return None

        # Debug mode: save audio chunk to disk for inspection
        if logger.isEnabledFor(logging.DEBUG):
            try:
                import wave
                from datetime import datetime
                debug_dir = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                    "log", "debug_audio",
                )
                os.makedirs(debug_dir, exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                fname = f"{ts}_{self.call_id[:8]}.wav"
                fpath = os.path.join(debug_dir, fname)
                pcm_int16 = (np.clip(pcm, -1.0, 1.0) * 32767).astype(np.int16)
                with wave.open(fpath, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sample_rate)
                    wf.writeframes(pcm_int16.tobytes())
                logger.debug("call %s: debug audio saved to %s", self.call_id[:8], fpath)
            except Exception as e:
                logger.debug("call %s: could not save debug audio: %s", self.call_id[:8], e)

        # Use native-audio model for transcription if available (e.g. Gemma4),
        # otherwise fall back to Whisper-based STT
        active_model = self._agent._active_override_model(self.thread_id)
        audio_info = self._app.llm.audio_model_info(active_model or "chat")
        if audio_info:
            from pawlia.transcription import transcribe_pcm_via_model
            text = await transcribe_pcm_via_model(
                pcm, sample_rate, audio_info[0], audio_info[1]
            )
            if text:
                return text
            logger.info(
                "call %s: native audio transcription returned nothing; falling back to configured STT",
                self.call_id[:8],
            )
        return await transcribe_pcm(pcm, sample_rate, self._app.config)

    async def _respond_to_transcript(self, text: str, announce_transcript: bool = True) -> None:
        """Stream the agent response for an already transcribed utterance."""
        from pawlia.tts import synthesize_pcm

        if announce_transcript:
            logger.info("call %s: transcribed: %s", self.call_id[:8], text[:120])
            await self._send_cb(f"🎙️ *{text}*")

        # Start hold audio while waiting for agent response
        if self._tts_track:
            self._tts_track.start_hold()

        # Keep typing indicator alive (Matrix times it out after ~30s)
        typing_task = asyncio.ensure_future(self._keep_typing())

        try:
            first_sentence_received = False
            call_prompt = self._agent.build_system_prompt(mode="call", thread_id=self.thread_id)

            async def _on_sentence(sentence: str) -> None:
                """Synthesize and enqueue one sentence for immediate TTS playback."""
                nonlocal first_sentence_received
                if not self._tts_track:
                    return
                current_task = asyncio.current_task()
                if current_task and current_task.cancelling():
                    return
                try:
                    tts_pcm = await synthesize_pcm(
                        sentence, self._app.config, sample_rate=48000,
                        voice_override=self._voice_override(),
                    )
                    if current_task and current_task.cancelling():
                        return
                    if tts_pcm is not None and len(tts_pcm):
                        logger.info("call %s: TTS sentence (%d samples): %s",
                                    self.call_id[:8], len(tts_pcm), sentence[:60])
                        self._tts_track.enqueue_pcm_float32(tts_pcm)
                        # Stop hold audio only AFTER TTS is queued so there is
                        # no silence gap that could cause playback glitches.
                        if not first_sentence_received:
                            first_sentence_received = True
                            self._tts_track.stop_hold()
                except Exception as e:
                    logger.warning("call %s: TTS sentence failed: %s", self.call_id[:8], e)

            async def _on_skill_start(skill_name: str, query: str) -> None:
                short_q = (query[:60] + "…") if len(query) > 60 else query
                await self._send_cb(f"⚙ *{skill_name}*: {short_q}")
                # Restart hold audio while this skill is executing
                if self._tts_track:
                    self._tts_track.start_hold()

            async def _on_skill_done(skill_name: str) -> None:
                await self._send_cb(f"✓ *{skill_name}*")
                # Restart hold audio for the next skill or agent thinking phase
                if self._tts_track:
                    self._tts_track.start_hold()

            response = await self._agent.run_streamed(
                text,
                system_prompt=call_prompt,
                thread_id=self.thread_id,
                on_sentence=_on_sentence,
                on_skill_start=_on_skill_start,
                on_skill_done=_on_skill_done,
            )
        finally:
            typing_task.cancel()
            if self._tts_track:
                self._tts_track.stop_hold()
            self._activate_agc()
            try:
                await self._client.room_typing(self.room_id, typing_state=False)
            except Exception:
                pass

        await self._send_cb(response)

    async def _queue_transcript_response(self, text: str) -> None:
        """Show the transcript now, but reply only after the caller is quiet."""
        logger.info("call %s: transcribed: %s", self.call_id[:8], text[:120])
        await self._send_cb(f"🎙️ *{text}*")
        self._pending_transcripts.append(text)

        current = asyncio.current_task()
        pending = self._pending_response_task
        if pending and not pending.done() and pending is not current:
            pending.cancel()

        task = asyncio.create_task(self._delayed_pending_response())
        self._pending_response_task = task
        self._track_response_task(task)

    async def _delayed_pending_response(self) -> None:
        try:
            response_delay = self._compute_response_delay()
            while not self._done.is_set():
                idle_for = time.monotonic() - self._last_user_speech_at
                if not self._speaking and idle_for >= response_delay:
                    tts_playing = False
                    if self._tts_track:
                        tts_playing = bool(
                            getattr(
                                self._tts_track,
                                "is_tts_playing",
                                getattr(self._tts_track, "is_playing", False),
                            )
                        )
                    if tts_playing:
                        await asyncio.sleep(0.2)
                        continue
                    break
                await asyncio.sleep(0.2)

            if self._done.is_set() or not self._pending_transcripts:
                return

            text = "\n".join(self._pending_transcripts)
            self._pending_transcripts = []
            logger.info(
                "call %s: replying after %.1fs quiet to %d transcript chunk(s) "
                "(speech_dur=%.1fs noise_floor=%.4f)",
                self.call_id[:8],
                response_delay,
                len(text.splitlines()),
                self._last_speech_duration,
                self._noise_floor,
            )
            await self._respond_to_transcript(text, announce_transcript=False)
        finally:
            if self._pending_response_task is asyncio.current_task():
                self._pending_response_task = None

    async def _drain_video_track(self, track) -> None:
        """Discard incoming video frames so aiortc doesn't buffer them in memory."""
        try:
            while not self._done.is_set():
                await track.recv()
        except Exception:
            pass

    async def _audio_pipeline(self, track) -> None:
        """Continuously read audio frames, detect speech, transcribe, respond."""
        SAMPLE_RATE = 48000
        fps = 50  # aiortc default: 20 ms frames
        min_speech_frames = int(self.MIN_SPEECH_SECONDS * fps)
        pre_speech_frames = int(max(0.0, self.PRE_SPEECH_SECONDS) * fps)

        speech_buffer: List[np.ndarray] = []
        pre_speech_buffer: "deque[np.ndarray]" = deque(maxlen=max(pre_speech_frames, 1))
        silence_count = 0
        resume_speech_count = 0

        logger.info("call %s: audio pipeline started", self.call_id[:8])
        frames_received = 0
        media_ended = False
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
                    media_ended = True
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
                elif frames_received % 50 == 0 and logger.isEnabledFor(logging.DEBUG):
                    import hashlib
                    h = hashlib.md5(pcm.tobytes()).hexdigest()[:8]
                    logger.debug("call %s: frame #%d rms=%.4f nf=%.4f buf=%d silence=%d hash=%s",
                                 self.call_id[:8], frames_received, rms, self._noise_floor,
                                 len(speech_buffer), silence_count, h)

                # Apply AGC to RMS for VAD decision (raw PCM stays untouched)
                adjusted_rms = self._agc_rms(rms)

                if not speech_buffer and pre_speech_frames > 0:
                    pre_speech_buffer.append(pcm)

                # Update background noise floor while not accumulating speech.
                # EMA alpha=0.02 → ~50-frame (1 s) time constant, slow enough to
                # ignore transient spikes but fast enough to adapt to the cycling
                # noise floor within the first few seconds of a call.
                if not speech_buffer:
                    self._noise_floor = 0.02 * rms + 0.98 * self._noise_floor

                # silence_threshold in frames; kept at the configured value (minimum
                # 1.2 s).  Because _is_speech_like_frame now uses an adaptive
                # threshold that treats background noise as silence, silence_count
                # accumulates naturally without needing to reduce this value.
                silence_threshold = int(max(1.2, self.SILENCE_SECONDS) * fps)

                speech_like_frame = self._is_speech_like_frame(
                    pcm,
                    SAMPLE_RATE,
                    adjusted_rms,
                )

                # While TTS is playing, keep buffering possible interruptions, but
                # only stop playback after the resulting transcript is meaningful.
                if self._tts_track and self._tts_track.is_playing:
                    if (
                        rms >= max(self.BARGEIN_RMS_THRESHOLD, self.SILENCE_THRESHOLD)
                        and speech_like_frame
                    ):
                        if not speech_buffer and silence_count == 0:
                            logger.info(
                                "call %s: possible barge-in started (rms=%.4f)",
                                self.call_id[:8],
                                rms,
                            )
                            self._mark_user_speech_started()
                            speech_buffer = self._start_speech_buffer(pre_speech_buffer, pcm)
                        else:
                            speech_buffer.append(pcm)
                        resume_confirmed, resume_speech_count = self._resume_speech_after_pause(
                            speech_like_frame,
                            silence_count,
                            resume_speech_count,
                        )
                        if resume_confirmed:
                            silence_count = 0
                            resume_speech_count = 0
                        elif silence_count == 0:
                            resume_speech_count = 0
                    elif speech_buffer:
                        silence_count += 1
                        resume_speech_count = 0
                        speech_buffer.append(pcm)

                        if silence_count >= silence_threshold:
                            chunk = np.concatenate(speech_buffer)
                            duration = len(chunk) / SAMPLE_RATE
                            logger.info(
                                "call %s: possible barge-in ended — %.1fs, %d samples",
                                self.call_id[:8], duration, len(chunk),
                            )
                            speech_buffer = []
                            pre_speech_buffer.clear()
                            silence_count = 0
                            resume_speech_count = 0
                            self._mark_user_speech_ended()

                            if len(chunk) >= min_speech_frames * (SAMPLE_RATE // fps):
                                chunk_stats = self._analyze_speech_chunk(chunk, SAMPLE_RATE, fps)
                                if self._should_transcribe_chunk(chunk, SAMPLE_RATE, fps):
                                    logger.info(
                                        "call %s: transcribing barge-in candidate "
                                        "(active_ratio=%.2f longest_run=%d speech_like=%.2f voiced=%.2f p90_rms=%.4f)",
                                        self.call_id[:8],
                                        chunk_stats["active_ratio"],
                                        int(chunk_stats["longest_run"]),
                                        chunk_stats["speech_like_ratio"],
                                        chunk_stats["voiced_ratio"],
                                        chunk_stats["p90_rms"],
                                    )
                                    self._mark_activity()
                                    task = asyncio.create_task(
                                        self._process_speech(
                                            chunk,
                                            SAMPLE_RATE,
                                            interrupt_playback=True,
                                        )
                                    )
                                else:
                                    logger.info(
                                        "call %s: barge-in candidate looked like noise "
                                        "(active_ratio=%.2f longest_run=%d speech_like=%.2f voiced=%.2f p90_rms=%.4f)",
                                        self.call_id[:8],
                                        chunk_stats["active_ratio"],
                                        int(chunk_stats["longest_run"]),
                                        chunk_stats["speech_like_ratio"],
                                        chunk_stats["voiced_ratio"],
                                        chunk_stats["p90_rms"],
                                    )
                    continue
                if speech_like_frame:
                    if not speech_buffer and silence_count == 0:
                        logger.info("call %s: speech started (rms=%.4f)",
                                    self.call_id[:8], rms)
                        self._mark_user_speech_started()
                        speech_buffer = self._start_speech_buffer(pre_speech_buffer, pcm)
                    else:
                        speech_buffer.append(pcm)
                    resume_confirmed, resume_speech_count = self._resume_speech_after_pause(
                        speech_like_frame,
                        silence_count,
                        resume_speech_count,
                    )
                    if resume_confirmed:
                        silence_count = 0
                        resume_speech_count = 0
                    elif silence_count == 0:
                        resume_speech_count = 0
                elif speech_buffer:
                    silence_count += 1
                    resume_speech_count = 0
                    speech_buffer.append(pcm)  # keep trailing silence for context

                    if silence_count >= silence_threshold:
                        chunk = np.concatenate(speech_buffer)
                        duration = len(chunk) / SAMPLE_RATE
                        logger.info("call %s: speech ended — %.1fs, %d samples",
                                    self.call_id[:8], duration, len(chunk))
                        speech_buffer = []
                        pre_speech_buffer.clear()
                        silence_count = 0
                        resume_speech_count = 0
                        self._mark_user_speech_ended()

                        if len(chunk) >= min_speech_frames * (SAMPLE_RATE // fps):
                            chunk_stats = self._analyze_speech_chunk(chunk, SAMPLE_RATE, fps)
                            if self._should_transcribe_chunk(chunk, SAMPLE_RATE, fps):
                                self._last_speech_duration = duration
                                logger.info(
                                    "call %s: sending chunk for transcription "
                                    "(active_ratio=%.2f longest_run=%d speech_like=%.2f voiced=%.2f p90_rms=%.4f noise_floor=%.4f)",
                                    self.call_id[:8],
                                    chunk_stats["active_ratio"],
                                    int(chunk_stats["longest_run"]),
                                    chunk_stats["speech_like_ratio"],
                                    chunk_stats["voiced_ratio"],
                                    chunk_stats["p90_rms"],
                                    self._noise_floor,
                                )
                                self._mark_activity()
                                task = asyncio.create_task(
                                    self._process_speech(chunk, SAMPLE_RATE)
                                )
                                self._track_response_task(task)
                            else:
                                logger.info(
                                    "call %s: skipping chunk as background noise "
                                    "(active_ratio=%.2f longest_run=%d speech_like=%.2f voiced=%.2f p90_rms=%.4f)",
                                    self.call_id[:8],
                                    chunk_stats["active_ratio"],
                                    int(chunk_stats["longest_run"]),
                                    chunk_stats["speech_like_ratio"],
                                    chunk_stats["voiced_ratio"],
                                    chunk_stats["p90_rms"],
                                )
                        else:
                            logger.info("call %s: chunk too short (%.1fs), skipping",
                                        self.call_id[:8], duration)
        except Exception as e:
            logger.error("call %s: audio pipeline error: %s", self.call_id[:8], e)
        finally:
            self._done.set()
            logger.info("call %s: audio pipeline ended", self.call_id[:8])
            if media_ended and self.HANGUP_ON_MEDIA_END and not self._hungup:
                logger.info(
                    "call %s: media track ended; sending hangup to avoid stale client reconnect",
                    self.call_id[:8],
                )
                try:
                    await self._send_hangup_event()
                finally:
                    await self.hangup()
    async def _process_speech(
        self,
        pcm: "np.ndarray",
        sample_rate: int,
        interrupt_playback: bool = False,
    ) -> None:
        """Transcribe a speech chunk and optionally use it to barge into TTS."""
        started_hold = False
        if not interrupt_playback and self._tts_track:
            self._tts_track.start_hold()
            started_hold = True
        try:
            text = await self._transcribe_speech(pcm, sample_rate)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("call %s: transcription error: %s", self.call_id[:8], e)
            if started_hold and self._tts_track:
                self._tts_track.stop_hold()
            return

        if not text:
            logger.info("call %s: empty transcription (no text returned)", self.call_id[:8])
            if started_hold and self._tts_track:
                self._tts_track.stop_hold()
            return

        try:
            if interrupt_playback:
                if not self._is_meaningful_interrupt(text):
                    logger.info(
                        "call %s: ignoring non-meaningful barge-in transcript: %s",
                        self.call_id[:8],
                        text[:120],
                    )
                    return
                logger.info(
                    "call %s: meaningful barge-in transcript detected, interrupting playback: %s",
                    self.call_id[:8],
                    text[:120],
                )
                if self._tts_track:
                    self._tts_track.stop_after_current_sentence()
                await self._cancel_active_response()
                current_task = asyncio.current_task()
                if current_task:
                    self._track_response_task(current_task)

            await self._queue_transcript_response(text)
        except asyncio.CancelledError:
            if self._tts_track:
                self._tts_track.stop_hold()
            raise
        except Exception as e:
            logger.error("call %s: agent error: %s", self.call_id[:8], e)

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
            await self._send_hangup_event()
            await self.hangup()

    async def _notify_disconnect(self) -> None:
        """Send a Matrix message when the connection drops unexpectedly."""
        try:
            await self._send_cb("📞 Verbindung unterbrochen")
        except Exception as e:
            logger.warning("call %s: could not send disconnect notification: %s",
                           self.call_id[:8], e)

    async def _connect_timeout_watchdog(self) -> None:
        """End calls that never establish media after the SDP answer."""
        try:
            await self._answer_sent.wait()
            await asyncio.wait_for(
                self._media_connected.wait(),
                timeout=self.CONNECT_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            if self._done.is_set() or self._hungup:
                return
            state = self._pc.connectionState if self._pc else "unknown"
            ice_state = self._pc.iceConnectionState if self._pc else "unknown"
            logger.warning(
                "call %s: media did not connect within %.1fs "
                "(connection=%s ice=%s); ending call",
                self.call_id[:8],
                self.CONNECT_TIMEOUT_SECONDS,
                state,
                ice_state,
            )
            await self._notify_disconnect()
            await self._send_hangup_event()
            await self.hangup()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.debug("call %s: connect timeout watchdog error: %s", self.call_id[:8], e)

    async def _watchdog(self) -> None:
        """Auto-hangup after prolonged call inactivity."""
        while not self._done.is_set():
            # Do not count our own live response generation or playback as
            # inactivity. Otherwise long spoken replies can trip the timeout
            # mid-sentence and drop an otherwise healthy call.
            if self._bot_is_active():
                self._mark_activity()
                try:
                    await asyncio.wait_for(
                        self._done.wait(),
                        timeout=self.WATCHDOG_POLL_SECONDS,
                    )
                except asyncio.TimeoutError:
                    continue

            idle_for = time.monotonic() - self._last_activity_at
            remaining = self.CALL_INACTIVITY_SECONDS - idle_for
            if remaining <= 0:
                logger.info(
                    "call %s: inactive for %.1fs, hanging up",
                    self.call_id[:8],
                    idle_for,
                )
                await self.hangup()
                await self._send_hangup_event()
                return

            # Activate AGC when half the inactivity timeout has passed
            # — the user might be speaking softly and we're not hearing them
            half = self.CALL_INACTIVITY_SECONDS / 2
            if idle_for >= half and not self._agc_active:
                logger.info("call %s: half inactivity reached, activating AGC",
                            self.call_id[:8])
                self._activate_agc()

            try:
                await asyncio.wait_for(
                    self._done.wait(),
                    timeout=min(remaining, self.WATCHDOG_POLL_SECONDS),
                )
            except asyncio.TimeoutError:
                continue

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
                ignore_unverified_devices=True,
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

        existing = self._sessions.get(event.call_id)
        if existing:
            if existing.finished:
                logger.info(
                    "call %s: replacing finished session on duplicate invite",
                    event.call_id[:8],
                )
                self._sessions.pop(event.call_id, None)
            else:
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
                ignore_unverified_devices=True,
            )
            call_thread_id = getattr(resp, "event_id", None)
            logger.info("call %s: thread root event_id=%s", event.call_id[:8], call_thread_id)
        except Exception as e:
            logger.warning("call %s: could not create thread root: %s", event.call_id[:8], e)

        agent = self._get_agent(room.room_id, call_thread_id)

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
