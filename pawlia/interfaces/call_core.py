"""Transport-agnostic call session logic shared by aiortc and LiveKit transports.

Defines:
  - TTSFrameBuffer  — queue/hold/barge-in helper, frame emission is transport-specific
  - CallTransport   — abstract adapter interface (one impl per media transport)
  - CallCore        — VAD/AGC/STT/agent/TTS business logic; transport-independent
"""

import asyncio
from abc import ABC, abstractmethod
from collections import deque
import logging
import os
import re
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False

from pawlia.audio.agc import AGCController
from pawlia.audio.vad import SpeechDetector
from pawlia.audio.config import get_float_config, get_int_config, get_bool_config

if TYPE_CHECKING:
    from nio import AsyncClient, MatrixRoom
    from pawlia.app import App

logger = logging.getLogger("pawlia.interfaces.call_core")

SAMPLE_RATE = 48000

# Sentinel value pushed into _incoming_q by transport when audio stream ends.
_STREAM_ENDED: object = object()

_KEYWORD_INTERRUPT_RE = re.compile(
    r"\b(?:halt|stop|stopp|wait|warte|warten|moment|sekunde|pause|pawlia)\b",
    re.IGNORECASE,
)

_TTS_INTERNAL_RE = re.compile(
    r"^\s*("
    r"\[Earlier skill use"
    r"|\[Report from `"
    r"|\[internal context"
    r"|Trust: (INTERNAL|EXTERNAL)"
    r"|Raw outside data"
    r"|Treat with skepticism"
    r"|This information comes from the user.s own"
    r"|Cross-check with what you know"
    r"|when in conflict, follow this source"
    r"|---\s*$"
    r")",
    re.IGNORECASE,
)


def _for_tts(sentence: str) -> Optional[str]:
    if _TTS_INTERNAL_RE.search(sentence):
        return None
    return sentence


# ---------------------------------------------------------------------------
# TTSFrameBuffer — queue/hold/barge-in/pacing helper
# ---------------------------------------------------------------------------

class TTSFrameBuffer:
    """TTS queue + hold-audio + barge-in logic, without transport-specific framing.

    Both aiortc (_TTSAudioTrack) and LiveKit (_publish_task) call next_frame_960()
    to get the next 960-sample int16 frame; they handle their own pacing.
    """

    SAMPLES_PER_FRAME = 960  # 20 ms @ 48 kHz

    def __init__(self, recorder=None) -> None:
        self._queue: asyncio.Queue[Any] = asyncio.Queue()
        self._next_sentence_id = 1
        self._current_sentence_id: Optional[int] = None
        self._hold_pcm: Optional["np.ndarray"] = None
        self._hold_pos: int = 0
        self._hold_active: bool = False
        self._recorder = recorder

    @property
    def is_playing(self) -> bool:
        return not self._queue.empty() or self._hold_active

    @property
    def is_tts_playing(self) -> bool:
        return not self._queue.empty() or self._current_sentence_id is not None

    def set_hold_audio(self, pcm_int16: "np.ndarray") -> None:
        self._hold_pcm = pcm_int16
        self._hold_pos = 0

    def start_hold(self) -> None:
        if not self._hold_active:
            self._hold_pos = 0
        self._hold_active = True

    def stop_hold(self) -> None:
        self._hold_active = False

    def interrupt(self) -> None:
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._hold_active = False
        self._current_sentence_id = None

    def stop_after_current_sentence(self) -> None:
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

    def enqueue_pcm_float32(self, pcm: "np.ndarray") -> None:
        if pcm is None or len(pcm) == 0:
            logger.warning("TTS: Received empty or None audio data")
            return
        pcm_normalized = np.clip(pcm, -1.0, 1.0)
        pcm_int16 = (pcm_normalized * 32767).astype(np.int16)
        logger.debug("TTS: Enqueuing audio - samples: %d, min: %.4f, max: %.4f, mean: %.4f",
                     len(pcm), float(np.min(pcm)), float(np.max(pcm)), float(np.mean(pcm)))
        if self._recorder is not None:
            self._recorder.push_pawlia(pcm_normalized)
        sentence_id = self._next_sentence_id
        self._next_sentence_id += 1
        chunks = [
            pcm_int16[i: i + self.SAMPLES_PER_FRAME]
            for i in range(0, len(pcm_int16), self.SAMPLES_PER_FRAME)
        ]
        for index, chunk in enumerate(chunks):
            if len(chunk) > 0:
                self._queue.put_nowait((chunk, sentence_id, index == len(chunks) - 1))

    def next_frame_960(self) -> "np.ndarray":
        """Return the next 960-sample int16 frame; silence when nothing is queued."""
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
            if self._hold_active and self._hold_pcm is not None and len(self._hold_pcm) > 0:
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

        return samples


# ---------------------------------------------------------------------------
# CallTransport — abstract adapter interface
# ---------------------------------------------------------------------------

class CallTransport(ABC):
    """One implementation per media transport (aiortc, LiveKit, …).

    The core sets ``on_incoming_pcm`` before calling ``start()``.
    The transport calls it with float32 mono 48 kHz numpy arrays.
    """

    # Set by CallCore before start():
    on_incoming_pcm: Optional[Callable[["np.ndarray"], None]] = None

    @property
    @abstractmethod
    def media_connected(self) -> asyncio.Event:
        """Event set when media is flowing (ICE+DTLS complete or LK room joined)."""

    @abstractmethod
    async def start(self, *args, **kwargs) -> Any:
        """Set up the transport. Returns transport-specific result (e.g. SDP answer)."""

    @abstractmethod
    async def close(self) -> None:
        """Tear down the transport cleanly."""

    @abstractmethod
    def enqueue_pcm_float32(self, pcm: "np.ndarray") -> None:
        """Queue float32 PCM for outgoing TTS playback."""

    @abstractmethod
    def interrupt(self) -> None:
        """Barge-in: clear all queued TTS audio immediately."""

    @abstractmethod
    def stop_after_current_sentence(self) -> None:
        """Barge-in: finish current TTS sentence, discard the rest."""

    @abstractmethod
    def start_hold(self) -> None:
        """Begin looping hold audio."""

    @abstractmethod
    def stop_hold(self) -> None:
        """Stop hold audio."""

    @property
    @abstractmethod
    def is_playing(self) -> bool:
        """True while TTS or hold audio is playing."""

    @property
    @abstractmethod
    def is_tts_playing(self) -> bool:
        """True while spoken TTS audio is queued or mid-sentence."""

    def set_hold_audio(self, pcm_int16: "np.ndarray") -> None:
        """Set the hold-audio loop buffer. Default no-op."""

    @property
    def is_transport_finished(self) -> bool:
        """True when the underlying connection is closed/failed. Default False."""
        return False

    def add_candidate(self, candidate: Dict) -> None:
        """Add an ICE candidate (legacy WebRTC only). Default no-op."""


# ---------------------------------------------------------------------------
# CallCore — transport-agnostic call session logic
# ---------------------------------------------------------------------------

class CallCore:
    """Manages a single active VoIP call, independent of media transport.

    Subclasses provide a concrete transport by setting self._transport and
    calling super().__init__().  Audio arrives via feed_pcm() (called by the
    transport's on_incoming_pcm callback) and is processed by _audio_pipeline().
    """

    CALL_INACTIVITY_SECONDS = 180
    WATCHDOG_POLL_SECONDS = 5.0
    BARGEIN_RMS_THRESHOLD = 0.05
    PREANSWER_WARMUP_ENABLED = True
    PREANSWER_WARMUP_TIMEOUT_SECONDS = 25.0
    PREANSWER_STT_SILENCE_SECONDS = 0.4
    CONNECT_TIMEOUT_SECONDS = 45.0
    HANGUP_ON_MEDIA_END = True
    NET_WARN_LOSS_RATIO = 0.05
    NET_WARN_JITTER = 2000.0
    NET_DEGRADED_INTERVALS = 2
    NET_RECOVER_INTERVALS = 2
    NET_WARN_MESSAGE = (
        "📶 Verbindung gerade schlecht — ich verstehe dich evtl. nur "
        "bruchstückhaft."
    )
    NET_RECOVER_MESSAGE = "📶 Verbindung wieder stabil."
    JITTER_BUFFER_CAPACITY = 32
    RESPONSE_DELAY_SECONDS = 1.2
    # Semantic endpointing: when the reply timer elapses but the accumulated
    # transcript still looks like a mid-thought fragment (see
    # SpeechDetector.looks_like_incomplete_utterance), hold the response back a
    # little longer rather than reply to half a sentence. The extra grace grows
    # with how long the caller has already been speaking — a long, complex
    # monologue earns longer thinking pauses — but is hard-capped so a dangling
    # fragment can never stall the turn indefinitely. Set base 0 to disable.
    INCOMPLETE_GRACE_BASE = 2.0
    INCOMPLETE_GRACE_GROWTH = 0.15
    INCOMPLETE_GRACE_MAX = 8.0
    # While the agent is *thinking* (hold tone playing, not yet speaking),
    # speech that lacks an interrupt keyword is discarded by default — so a
    # side conversation (e.g. talking to someone else in the room) does not
    # get captured and fed into the next turn. Set True to instead queue such
    # speech for the next response ("nachreichen"-Modus).
    QUEUE_SPEECH_WHILE_THINKING = False

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
        self._send_cb = send_cb

        self._transport: Optional[CallTransport] = None
        self._incoming_q: asyncio.Queue[Any] = asyncio.Queue()
        self._agent = agent
        self._done = asyncio.Event()
        self._hungup = False
        self._speaking = False
        self._last_activity_at = time.monotonic()
        self._last_user_speech_at = self._last_activity_at
        self._active_response_task: Optional[asyncio.Task] = None
        self._greeting_sent = False
        self._answer_sent = asyncio.Event()
        self._prepared_greeting: Optional[tuple] = None
        self._prepare_greeting_task: Optional[asyncio.Task] = None
        # Set once the first greeting sentence has been synthesized to PCM.  The
        # transport waits on this before sending the SDP answer, so there is audio
        # ready to play the moment media connects (no silence after pickup).
        self._greeting_first_sentence_ready = asyncio.Event()
        # Greeting sentences are synthesized incrementally during warmup and
        # streamed to the transport as soon as media connects — playing sentence 1
        # immediately instead of waiting for the whole greeting to be ready.
        self._greeting_pcm: List["np.ndarray"] = []
        self._greeting_pcm_flushed = 0
        self._greeting_audio_started = False
        self._greeting_task: Optional[asyncio.Task] = None
        self._pending_response_task: Optional[asyncio.Task] = None
        self._pending_transcripts: List[str] = []
        self._pending_attachments: List[str] = []
        self._net_degraded = False
        self._net_bad_streak = 0
        self._net_good_streak = 0
        self._net_prev_recv: Optional[int] = None
        self._net_prev_lost: Optional[int] = None
        self._load_voip_audio_config()
        ctx = f"call {call_id[:8]}"
        voip_cfg = self._voip_cfg
        self._agc = AGCController(voip_cfg, context=ctx)
        self._speech_detector = SpeechDetector(voip_cfg, context=ctx)

        voip_rec_cfg = voip_cfg.get("recording", {}) if isinstance(voip_cfg, dict) else {}
        self._recorder = None
        _rec_enabled = voip_rec_cfg.get("enabled", False) or logger.isEnabledFor(logging.DEBUG)
        if _rec_enabled:
            from pawlia.audio.recorder import CallRecorder
            self._recorder = CallRecorder(
                call_id=call_id,
                record_dir=voip_rec_cfg.get("directory"),
                sample_rate=SAMPLE_RATE,
                compress_to_flac=voip_rec_cfg.get("compress_flac", True),
            )

    # ------------------------------------------------------------------
    # Audio seams
    # ------------------------------------------------------------------

    def feed_pcm(self, pcm: "np.ndarray") -> None:
        """Transport calls this with each float32 mono 48 kHz audio frame."""
        self._incoming_q.put_nowait(pcm)

    def feed_stream_ended(self) -> None:
        """Transport calls this when the incoming audio stream ends."""
        self._incoming_q.put_nowait(_STREAM_ENDED)

    # ------------------------------------------------------------------
    # Activity tracking
    # ------------------------------------------------------------------

    def _mark_activity(self) -> None:
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
        if self._transport and self._transport.is_playing:
            return True
        task = self._active_response_task
        return bool(task and not task.done())

    # ------------------------------------------------------------------
    # Status / helpers
    # ------------------------------------------------------------------

    async def _send_status(self, text: str) -> None:
        try:
            html = (
                '<font size="1" color="#888888" data-mx-color="#888888">'
                f"{text}"
                '</font>'
            )
            await self._send_cb(html)
        except Exception:
            pass

    def _voice_override(self) -> Optional[str]:
        try:
            session = self._app.memory.load_session(f"mx_{self.room_id}")
            return session.voice_override
        except Exception:
            return None

    def _load_voip_audio_config(self) -> None:
        app_cfg = self._app.config if isinstance(self._app.config, dict) else {}
        voip_cfg = app_cfg.get("voip", {}) if isinstance(app_cfg, dict) else {}
        if not isinstance(voip_cfg, dict):
            logger.warning("call %s: ignoring non-dict voip config", self.call_id[:8])
            voip_cfg = {}
        self._voip_cfg = voip_cfg
        ctx = f"call {self.call_id[:8]}"
        self.CALL_INACTIVITY_SECONDS = get_int_config(
            voip_cfg, "call_inactivity_seconds", self.CALL_INACTIVITY_SECONDS,
            context=ctx, minimum=1)
        self.BARGEIN_RMS_THRESHOLD = get_float_config(
            voip_cfg, "bargein_rms_threshold", self.BARGEIN_RMS_THRESHOLD,
            context=ctx, minimum=0.0)
        self.PREANSWER_WARMUP_ENABLED = get_bool_config(
            voip_cfg, "preanswer_warmup_enabled", self.PREANSWER_WARMUP_ENABLED,
            context=ctx)
        self.PREANSWER_WARMUP_TIMEOUT_SECONDS = get_float_config(
            voip_cfg, "preanswer_warmup_timeout_seconds",
            self.PREANSWER_WARMUP_TIMEOUT_SECONDS, context=ctx, minimum=0.1)
        self.PREANSWER_STT_SILENCE_SECONDS = get_float_config(
            voip_cfg, "preanswer_stt_silence_seconds",
            self.PREANSWER_STT_SILENCE_SECONDS, context=ctx, minimum=0.05)
        self.RESPONSE_DELAY_SECONDS = get_float_config(
            voip_cfg, "response_delay_seconds", self.RESPONSE_DELAY_SECONDS,
            context=ctx, minimum=0.0)
        self.INCOMPLETE_GRACE_BASE = get_float_config(
            voip_cfg, "incomplete_grace_base", self.INCOMPLETE_GRACE_BASE,
            context=ctx, minimum=0.0)
        self.INCOMPLETE_GRACE_GROWTH = get_float_config(
            voip_cfg, "incomplete_grace_growth", self.INCOMPLETE_GRACE_GROWTH,
            context=ctx, minimum=0.0)
        self.INCOMPLETE_GRACE_MAX = get_float_config(
            voip_cfg, "incomplete_grace_max", self.INCOMPLETE_GRACE_MAX,
            context=ctx, minimum=0.0)
        self.CONNECT_TIMEOUT_SECONDS = get_float_config(
            voip_cfg, "connect_timeout_seconds", self.CONNECT_TIMEOUT_SECONDS,
            context=ctx, minimum=1.0)
        self.HANGUP_ON_MEDIA_END = get_bool_config(
            voip_cfg, "hangup_on_media_end", self.HANGUP_ON_MEDIA_END, context=ctx)
        self.VAD_MAX_CHUNK_SECONDS = get_float_config(
            voip_cfg, "vad_max_chunk_seconds", self.VAD_MAX_CHUNK_SECONDS,
            context=ctx, minimum=0.0)
        self.JITTER_BUFFER_CAPACITY = get_int_config(
            voip_cfg, "jitter_buffer_capacity", self.JITTER_BUFFER_CAPACITY,
            context=ctx, minimum=2)
        self.QUEUE_SPEECH_WHILE_THINKING = get_bool_config(
            voip_cfg, "queue_speech_while_thinking",
            self.QUEUE_SPEECH_WHILE_THINKING, context=ctx)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def mark_answer_sent(self) -> None:
        """Mark that Matrix accepted our SDP/join answer for this call."""
        self._answer_sent.set()

    async def hangup(self) -> None:
        """Terminate the call. Idempotent."""
        if self._hungup:
            return
        self._hungup = True
        self._done.set()
        if self._greeting_task and not self._greeting_task.done():
            self._greeting_task.cancel()
        if self._prepare_greeting_task and not self._prepare_greeting_task.done():
            self._prepare_greeting_task.cancel()
        if self._transport:
            await self._transport.close()
        if self._recorder is not None:
            try:
                self._recorder.finish()
            except Exception as e:
                logger.debug("call %s: recording finish failed: %s", self.call_id[:8], e)
        await self._send_status("Telefonat beendet")
        logger.info("call %s hung up", self.call_id[:8])

    @property
    def finished(self) -> bool:
        if self._hungup or self._done.is_set():
            return True
        if self._transport and self._transport.is_transport_finished:
            return True
        return False

    # ------------------------------------------------------------------
    # Hook methods — subclasses override for protocol-specific signaling
    # ------------------------------------------------------------------

    async def _send_hangup_event(self) -> None:
        """Send a protocol-level hangup signal. Default no-op."""

    async def _notify_disconnect(self) -> None:
        try:
            await self._send_cb("📞 Verbindung unterbrochen")
        except Exception as e:
            logger.warning("call %s: could not send disconnect notification: %s",
                           self.call_id[:8], e)

    async def _on_transport_ice_failed(self) -> None:
        """Called by transport adapter when ICE fails beyond recovery."""
        await self._notify_disconnect()
        await self._send_hangup_event()
        await self.hangup()

    # ------------------------------------------------------------------
    # Greeting / warmup
    # ------------------------------------------------------------------

    async def _run_preanswer_warmup(self) -> None:
        """Block until the first greeting sentence is ready, then return.

        Answering only when the first sentence's TTS audio is prepared avoids the
        awkward silence after pickup (the greeting plays immediately once media
        connects), while keeping the answer fast enough that the caller's ICE
        timeout does not fire: we wait for the *first* sentence, not the whole
        greeting, and STT warmup + the remaining sentences continue in the
        background.  On timeout we answer anyway rather than dropping the call.
        """
        if not self.PREANSWER_WARMUP_ENABLED:
            return
        started = time.monotonic()
        # STT is only needed once the caller speaks (after the greeting), so warm
        # it in the background without blocking the answer.
        asyncio.create_task(self._warm_stt_with_silence())
        self._ensure_prepare_greeting_task()
        try:
            await asyncio.wait_for(
                self._greeting_first_sentence_ready.wait(),
                timeout=self.PREANSWER_WARMUP_TIMEOUT_SECONDS,
            )
            logger.info("call %s: first greeting sentence ready in %.1fs, answering",
                        self.call_id[:8], time.monotonic() - started)
        except asyncio.TimeoutError:
            logger.warning("call %s: greeting not ready after %.1fs; answering anyway"
                           " (greeting continues in background)",
                           self.call_id[:8], self.PREANSWER_WARMUP_TIMEOUT_SECONDS)

    def _ensure_prepare_greeting_task(self) -> asyncio.Task:
        task = self._prepare_greeting_task
        if task is None or task.done():
            task = asyncio.create_task(self._prepare_greeting())
            # Guarantee the pre-answer wait is released even if greeting prep
            # produced no audio (TTS unavailable, LLM error, empty response):
            # answer anyway rather than eating the full warmup timeout.
            task.add_done_callback(
                lambda _t: self._greeting_first_sentence_ready.set()
            )
            self._prepare_greeting_task = task
        return task

    async def _warm_stt_with_silence(self) -> None:
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
                    silence, sample_rate, audio_info[0], audio_info[1],
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

    def _flush_greeting_pcm(self) -> None:
        """Enqueue any greeting sentences synthesized so far, once media is up.

        Safe to call repeatedly from both ``_prepare_greeting`` (as each sentence
        becomes ready) and ``_send_greeting`` (once media connects).  Plays
        sentence 1 the instant media is connected instead of waiting for the
        whole greeting, and stops the hold tone on the first flushed sentence.
        """
        if self._greeting_sent:
            return
        media = getattr(self._transport, "media_connected", None)
        if not self._transport or media is None or not media.is_set():
            return
        while self._greeting_pcm_flushed < len(self._greeting_pcm):
            pcm = self._greeting_pcm[self._greeting_pcm_flushed]
            self._transport.enqueue_pcm_float32(pcm)
            self._greeting_pcm_flushed += 1
            if not self._greeting_audio_started:
                self._greeting_audio_started = True
                self._transport.stop_hold()

    async def _prepare_greeting(self) -> None:
        if self._greeting_sent or self._prepared_greeting is not None:
            return
        try:
            from pawlia.tts import synthesize_pcm
        except ImportError:
            logger.debug("call %s: TTS not available, skipping greeting warmup", self.call_id[:8])
            return
        try:
            call_prompt = self._agent.build_system_prompt(
                mode="call", thread_id=self.thread_id, extra_context=self._call_extra_context())
            greeting_input = (
                "[SYSTEM: A voice call was just accepted. "
                "Greet the caller with a short, friendly greeting. "
                "Keep the established persona and preferred form of address from the profile/history. "
                "If speaking German and there is no explicit preference, use informal 'du', not formal 'Sie'. "
                "Keep it to one or two sentences.]"
            )

            async def _on_sentence(sentence: str) -> None:
                sentence = _for_tts(sentence) or ""
                if not sentence:
                    return
                try:
                    tts_pcm = await synthesize_pcm(
                        sentence, self._app.config, sample_rate=48000,
                        voice_override=self._voice_override())
                    if tts_pcm is not None and len(tts_pcm):
                        logger.info("call %s: prepared greeting TTS (%d samples): %s",
                                    self.call_id[:8], len(tts_pcm), sentence[:60])
                        self._greeting_pcm.append(tts_pcm)
                        # Unblock the pre-answer wait as soon as the first sentence
                        # has audio: the answer can go out now and play without gaps.
                        self._greeting_first_sentence_ready.set()
                        # Stream this sentence to the transport immediately if media
                        # is already connected (no waiting for the full greeting).
                        self._flush_greeting_pcm()
                except Exception as e:
                    logger.warning("call %s: greeting TTS warmup failed: %s", self.call_id[:8], e)

            response = await self._agent.run_streamed(
                greeting_input,
                system_prompt=call_prompt,
                thread_id=self.thread_id,
                on_sentence=_on_sentence,
                allow_skills=False,
            )
            self._prepared_greeting = (response, self._greeting_pcm)
            logger.info("call %s: greeting prepared", self.call_id[:8])
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("call %s: greeting warmup failed: %s", self.call_id[:8], e)

    async def _send_greeting(self) -> None:
        if self._greeting_sent:
            return
        task = self._prepare_greeting_task

        # Start playing the greeting sentences that are already synthesized right
        # now (sentence 1 the moment media connects), then await the rest of the
        # warmup task instead of blocking on the whole greeting up front.
        self._flush_greeting_pcm()  # push whatever is ready now
        if task is not None and self._prepared_greeting is None and not task.done():
            try:
                await task
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning("call %s: awaiting greeting warmup failed: %s",
                               self.call_id[:8], e)
        self._flush_greeting_pcm()  # push any sentences finished while awaiting

        if self._prepared_greeting is not None:
            response, _ = self._prepared_greeting
            await self._send_cb(response)
            self._greeting_sent = True
            self._mark_activity()
            self._agc.activate()
            await self._send_status("Telefonat verbunden")
            logger.info("call %s: prepared greeting sent", self.call_id[:8])
            return

        # A warmup greeting was already (partially) played but its text response
        # never materialized — don't re-synthesize a fresh greeting on top of it.
        if self._greeting_audio_started:
            self._greeting_sent = True
            self._mark_activity()
            self._agc.activate()
            return

        try:
            from pawlia.tts import synthesize_pcm
        except ImportError:
            logger.debug("call %s: TTS not available, skipping greeting", self.call_id[:8])
            return

        if not self._transport:
            return

        try:
            call_prompt = self._agent.build_system_prompt(
                mode="call", thread_id=self.thread_id, extra_context=self._call_extra_context())
            greeting_input = (
                "[SYSTEM: A voice call was just accepted. "
                "Greet the caller with a short, friendly greeting. "
                "Keep the established persona and preferred form of address from the profile/history. "
                "If speaking German and there is no explicit preference, use informal 'du', not formal 'Sie'. "
                "Keep it to one or two sentences.]"
            )

            async def _on_sentence(sentence: str) -> None:
                if not self._transport or self._done.is_set() or self._hungup:
                    return
                sentence = _for_tts(sentence) or ""
                if not sentence:
                    return
                try:
                    tts_pcm = await synthesize_pcm(
                        sentence, self._app.config, sample_rate=48000,
                        voice_override=self._voice_override())
                    if self._done.is_set() or self._hungup:
                        return
                    if tts_pcm is not None and len(tts_pcm):
                        logger.info("call %s: greeting TTS (%d samples): %s",
                                    self.call_id[:8], len(tts_pcm), sentence[:60])
                        self._transport.enqueue_pcm_float32(tts_pcm)
                        self._transport.stop_hold()
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
                logger.info("call %s: greeting completed after hangup", self.call_id[:8])
                return
            await self._send_cb(response)
            self._greeting_sent = True
            self._mark_activity()
            self._agc.activate()
            logger.info("call %s: greeting sent", self.call_id[:8])
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("call %s: greeting failed: %s", self.call_id[:8], e)

    async def _send_greeting_when_ready(self) -> None:
        if self._greeting_sent:
            return
        try:
            await self._answer_sent.wait()
            await self._transport.media_connected.wait()
            if not self._done.is_set():
                await self._send_greeting()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("call %s: deferred greeting failed: %s", self.call_id[:8], e)

    # ------------------------------------------------------------------
    # Response generation
    # ------------------------------------------------------------------

    def _track_response_task(self, task: asyncio.Task) -> None:
        self._active_response_task = task

        def _clear(done_task: asyncio.Task) -> None:
            if self._active_response_task is done_task:
                self._active_response_task = None

        task.add_done_callback(_clear)

    def _compute_response_delay(self) -> float:
        base = self.RESPONSE_DELAY_SECONDS
        silence_trail = max(1.2, self._speech_detector.SILENCE_SECONDS)
        dur = max(0.0, self._speech_detector.last_speech_duration - silence_trail)
        if dur > 20.0:
            base = max(base, 5.0)
        elif dur > 12.0:
            base = max(base, 4.0)
        elif dur > 6.0:
            base = max(base, 3.0)
        noise_ratio = self._speech_detector.noise_floor / max(
            self._speech_detector.SILENCE_THRESHOLD, 1e-4)
        if noise_ratio > 1.5:
            base += min((noise_ratio - 1.5) * 0.5, 1.5)
        return base

    def _incomplete_grace_seconds(self) -> float:
        """Extra wait granted when the pending transcript looks unfinished.

        Grows with how long the caller has already been speaking — a long,
        complex sentence earns longer thinking pauses — but is hard-capped at
        INCOMPLETE_GRACE_MAX so a dangling fragment can never stall the turn
        forever. Returns 0 when disabled (INCOMPLETE_GRACE_BASE == 0)."""
        if self.INCOMPLETE_GRACE_BASE <= 0.0:
            return 0.0
        dur = max(0.0, self._speech_detector.last_speech_duration)
        grace = self.INCOMPLETE_GRACE_BASE + dur * self.INCOMPLETE_GRACE_GROWTH
        return min(grace, self.INCOMPLETE_GRACE_MAX)

    async def _cancel_active_response(self) -> None:
        pending = self._pending_response_task
        current = asyncio.current_task()
        if pending and not pending.done() and pending is not current:
            pending.cancel()
            try:
                await pending
            except asyncio.CancelledError:
                logger.info("call %s: pending response cancelled", self.call_id[:8])
            except Exception as e:
                logger.debug("call %s: pending response cancel cleanup: %s", self.call_id[:8], e)
        task = self._active_response_task
        if not task or task.done() or task is current:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            logger.info("call %s: active response cancelled", self.call_id[:8])
        except Exception as e:
            logger.debug("call %s: active response cancel cleanup: %s", self.call_id[:8], e)

    async def _transcribe_speech(self, pcm: "np.ndarray", sample_rate: int) -> Optional[str]:
        try:
            from pawlia.transcription import transcribe_pcm
        except ImportError as e:
            logger.error("call %s: missing dependency: %s", self.call_id[:8], e)
            return None

        # Peak-normalize to prevent hard-clipping when AGC has over-boosted the signal.
        # Clipped audio causes STT to produce hallucinations/garbage.
        max_amp = float(np.max(np.abs(pcm))) if len(pcm) else 0.0
        if max_amp > 1.0:
            logger.debug(
                "call %s: normalizing transcription PCM (max_amp=%.2f)", self.call_id[:8], max_amp
            )
            pcm = pcm / max_amp

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
                fpath = os.path.join(debug_dir, f"{ts}_{self.call_id[:8]}.wav")
                pcm_int16 = (np.clip(pcm, -1.0, 1.0) * 32767).astype(np.int16)
                with wave.open(fpath, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sample_rate)
                    wf.writeframes(pcm_int16.tobytes())
                try:
                    import shutil, subprocess
                    if shutil.which("flac"):
                        flac_path = fpath.rsplit(".", 1)[0] + ".flac"
                        subprocess.run(["flac", "--best", "--silent", "-o", flac_path, fpath],
                                       capture_output=True, timeout=10)
                        if os.path.exists(flac_path):
                            os.remove(fpath)
                except Exception:
                    pass
                logger.debug("call %s: debug audio saved", self.call_id[:8])
            except Exception as e:
                logger.debug("call %s: could not save debug audio: %s", self.call_id[:8], e)

        active_model = self._agent._active_override_model(self.thread_id)
        audio_info = self._app.llm.audio_model_info(active_model or "chat")
        if audio_info:
            from pawlia.transcription import transcribe_pcm_via_model
            text = await transcribe_pcm_via_model(pcm, sample_rate, audio_info[0], audio_info[1])
            if text:
                return text
            logger.info("call %s: native audio transcription returned nothing; falling back",
                        self.call_id[:8])
        return await transcribe_pcm(pcm, sample_rate, self._app.config)

    async def _respond_to_transcript(self, text: str, announce_transcript: bool = True) -> None:
        from pawlia.tts import synthesize_pcm
        if announce_transcript:
            logger.info("call %s: transcribed: %s", self.call_id[:8], text)
            await self._send_cb(f"🎙️ *{text}*")
        if self._transport:
            self._transport.start_hold()
        typing_task = asyncio.create_task(self._keep_typing())
        try:
            first_sentence_received = False
            call_prompt = self._agent.build_system_prompt(
                mode="call", thread_id=self.thread_id, extra_context=self._call_extra_context())

            async def _on_sentence(sentence: str) -> None:
                nonlocal first_sentence_received
                if not self._transport:
                    return
                current_task = asyncio.current_task()
                if current_task and current_task.cancelling():
                    return
                sentence = _for_tts(sentence) or ""
                if not sentence:
                    return
                try:
                    tts_pcm = await synthesize_pcm(
                        sentence, self._app.config, sample_rate=48000,
                        voice_override=self._voice_override())
                    if current_task and current_task.cancelling():
                        return
                    if tts_pcm is not None and len(tts_pcm):
                        logger.info("call %s: TTS sentence (%d samples): %s",
                                    self.call_id[:8], len(tts_pcm), sentence[:60])
                        self._transport.enqueue_pcm_float32(tts_pcm)
                        if not first_sentence_received:
                            first_sentence_received = True
                            self._transport.stop_hold()
                except Exception as e:
                    logger.warning("call %s: TTS sentence failed: %s", self.call_id[:8], e)

            async def _on_skill_start(skill_name: str, query: str) -> None:
                short_q = (query[:60] + "…") if len(query) > 60 else query
                await self._send_cb(f"⚙ *{skill_name}*: {short_q}")
                if self._transport:
                    self._transport.start_hold()

            async def _on_skill_done(skill_name: str, result: str = "") -> None:
                await self._send_cb(f"✓ *{skill_name}*")
                if self._transport:
                    self._transport.start_hold()

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
            if self._transport:
                self._transport.stop_hold()
            self._agc.activate()
            try:
                await self._client.room_typing(self.room_id, typing_state=False)
            except Exception:
                pass
        await self._send_cb(response)

    async def _send_discarded_transcript(self, text: str) -> None:
        logger.debug("call %s: non-keyword speech during playback, discarded: %s",
                     self.call_id[:8], text)
        await self._send_cb(f"~~🎙️ *{text}*~~ *(verworfen)*")

    async def _queue_transcript_response(self, text: str) -> None:
        logger.info("call %s: transcribed: %s", self.call_id[:8], text)
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
            max_incomplete_grace = self._incomplete_grace_seconds()
            waited_incomplete = 0.0
            while not self._done.is_set():
                idle_for = time.monotonic() - self._last_user_speech_at
                if self._speaking or idle_for < response_delay:
                    # Caller resumed (or never paused long enough) — reset the
                    # incomplete-grace budget so a fresh fragment gets the full
                    # allowance again rather than inheriting a spent one.
                    waited_incomplete = 0.0
                if not self._speaking and idle_for >= response_delay:
                    tts_playing = False
                    if self._transport:
                        tts_playing = bool(self._transport.is_tts_playing)
                    if tts_playing:
                        await asyncio.sleep(0.2)
                        continue
                    # Semantic endpointing: the reply timer elapsed, but if the
                    # accumulated transcript still trails off mid-thought, hold
                    # back a little longer instead of replying to half a
                    # sentence. Bounded by max_incomplete_grace so a dangling
                    # fragment can never stall the turn indefinitely.
                    if waited_incomplete < max_incomplete_grace:
                        pending_text = " ".join(self._pending_transcripts)
                        if self._speech_detector.looks_like_incomplete_utterance(pending_text):
                            logger.info(
                                "call %s: transcript looks unfinished, holding "
                                "(%.1f/%.1fs grace): %s",
                                self.call_id[:8], waited_incomplete,
                                max_incomplete_grace, pending_text[-40:])
                            await asyncio.sleep(0.3)
                            waited_incomplete += 0.3
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
                self.call_id[:8], response_delay, len(text.splitlines()),
                self._speech_detector.last_speech_duration, self._speech_detector.noise_floor,
            )
            await self._respond_to_transcript(text, announce_transcript=False)
            for _ in range(5):
                if self._done.is_set() or not self._pending_transcripts:
                    break
                followup = "\n".join(self._pending_transcripts)
                self._pending_transcripts = []
                await self._respond_to_transcript(followup, announce_transcript=False)
        finally:
            if self._pending_response_task is asyncio.current_task():
                self._pending_response_task = None

    # ------------------------------------------------------------------
    # Speech finalization
    # ------------------------------------------------------------------

    def _finalize_speech_chunk(
        self,
        chunk_parts: List["np.ndarray"],
        sample_rate: int,
        fps: int,
        min_speech_frames: int,
        is_barge_in: bool = False,
    ) -> Optional[asyncio.Task]:
        chunk = np.concatenate(chunk_parts)
        duration = len(chunk) / sample_rate
        label = "possible barge-in" if is_barge_in else "speech"
        logger.info("call %s: %s ended — %.1fs, %d samples",
                    self.call_id[:8], label, duration, len(chunk))
        frame_size = sample_rate // fps
        if len(chunk) < min_speech_frames * frame_size:
            if not is_barge_in:
                logger.info("call %s: chunk too short (%.1fs), skipping",
                            self.call_id[:8], duration)
            return None
        chunk_stats = self._speech_detector.analyze_chunk(
            chunk, sample_rate, fps, agc_gain=self._agc.gain, agc_active=self._agc.active)
        if not self._speech_detector.should_transcribe(
            chunk, sample_rate, fps, agc_gain=self._agc.gain, agc_active=self._agc.active):
            noise_label = ("barge-in candidate looked like noise" if is_barge_in
                           else "skipping chunk as background noise")
            logger.info(
                "call %s: %s (active_ratio=%.2f longest_run=%d "
                "speech_like=%.2f voiced=%.2f p90_rms=%.4f)",
                self.call_id[:8], noise_label,
                chunk_stats["active_ratio"], int(chunk_stats["longest_run"]),
                chunk_stats["speech_like_ratio"], chunk_stats["voiced_ratio"],
                chunk_stats["p90_rms"])
            return None
        transcribe_label = ("transcribing barge-in candidate" if is_barge_in
                            else "sending chunk for transcription")
        logger.info(
            "call %s: %s (active_ratio=%.2f longest_run=%d "
            "speech_like=%.2f voiced=%.2f p90_rms=%.4f noise_floor=%.4f)",
            self.call_id[:8], transcribe_label,
            chunk_stats["active_ratio"], int(chunk_stats["longest_run"]),
            chunk_stats["speech_like_ratio"], chunk_stats["voiced_ratio"],
            chunk_stats["p90_rms"], self._speech_detector.noise_floor)
        if not is_barge_in:
            self._speech_detector.last_speech_duration = duration
        self._mark_activity()
        task = asyncio.create_task(
            self._process_speech(chunk, sample_rate, interrupt_playback=is_barge_in))
        if not is_barge_in:
            self._track_response_task(task)
        return task

    VAD_SETTLE_FRAMES: int = 40
    VAD_SETTLE_EMA_ALPHA: float = 0.08
    VAD_MAX_CHUNK_SECONDS: float = 15.0

    # ------------------------------------------------------------------
    # Audio pipeline
    # ------------------------------------------------------------------

    async def _audio_pipeline(self) -> None:
        """Continuously read audio from _incoming_q, detect speech, transcribe, respond."""
        fps = 50
        min_speech_frames = int(self._speech_detector.MIN_SPEECH_SECONDS * fps)
        pre_speech_frames = int(max(0.0, self._speech_detector.PRE_SPEECH_SECONDS) * fps)

        speech_buffer: List["np.ndarray"] = []
        pre_speech_buffer: "deque[np.ndarray]" = deque(maxlen=max(pre_speech_frames, 1))
        silence_count = 0
        resume_speech_count = 0
        speech_buffer_start_frame = 0
        speech_ref = 0.0
        # The hard max-chunk cap is resolved per-frame from the live noise floor
        # (effective_max_chunk_seconds) so it can shorten in loud environments.

        logger.info("call %s: audio pipeline started", self.call_id[:8])
        frames_received = 0
        media_ended = False
        try:
            while not self._done.is_set():
                try:
                    item = await asyncio.wait_for(self._incoming_q.get(), timeout=5.0)
                except asyncio.TimeoutError:
                    if frames_received == 0:
                        logger.warning("call %s: no audio frames received yet", self.call_id[:8])
                    continue

                if item is _STREAM_ENDED:
                    logger.warning("call %s: audio stream ended after %d frames",
                                   self.call_id[:8], frames_received)
                    media_ended = True
                    break

                pcm: "np.ndarray" = item
                frames_received += 1

                if self._recorder is not None:
                    self._recorder.push(pcm)

                rms = float(np.sqrt(np.mean(pcm ** 2)))
                if frames_received <= 5:
                    pcm_int16 = (np.clip(pcm, -1.0, 1.0) * 32768).astype(np.int16)
                    nz_count = int(np.count_nonzero(pcm_int16))
                    logger.debug("call %s: frame #%d pcm_len=%d nz_samples=%d rms=%.4f",
                                 self.call_id[:8], frames_received, len(pcm), nz_count, rms)
                elif frames_received % 50 == 0 and logger.isEnabledFor(logging.DEBUG):
                    import hashlib
                    h = hashlib.md5(pcm.tobytes()).hexdigest()[:8]
                    logger.debug("call %s: frame #%d rms=%.4f nf=%.4f buf=%d silence=%d hash=%s",
                                 self.call_id[:8], frames_received, rms,
                                 self._speech_detector.noise_floor,
                                 len(speech_buffer), silence_count, h)

                adjusted_rms = self._agc.adjust_rms(rms, self._bot_is_active())
                pcm = pcm * min(self._agc.gain, 4.0)

                if not speech_buffer and pre_speech_frames > 0:
                    pre_speech_buffer.append(pcm)

                if frames_received <= self.VAD_SETTLE_FRAMES:
                    if not bool(speech_buffer):
                        settle_alpha = self.VAD_SETTLE_EMA_ALPHA
                        nf = self._speech_detector.noise_floor
                        self._speech_detector._noise_floor = (
                            settle_alpha * rms + (1.0 - settle_alpha) * nf)
                    speech_like_frame = False
                else:
                    self._speech_detector.update_noise_floor(adjusted_rms, during_speech=bool(speech_buffer))
                    speech_like_frame = self._speech_detector.is_speech_like_frame(
                        pcm, SAMPLE_RATE, adjusted_rms)
                    pause_ratio = self._speech_detector.effective_pause_ratio()
                    if (speech_like_frame and pause_ratio > 0.0
                            and speech_ref > 0.0
                            and adjusted_rms < speech_ref * pause_ratio):
                        speech_like_frame = False
                    if speech_like_frame:
                        speech_ref = (
                            0.2 * adjusted_rms + 0.8 * speech_ref
                            if speech_ref > 0.0 else adjusted_rms)

                # The longer this utterance has already run, the longer a
                # mid-thought pause we tolerate before closing the chunk — so a
                # long sentence is not split across STT calls (short replies
                # still close at the SILENCE_SECONDS base).
                spoken_seconds = (
                    (frames_received - speech_buffer_start_frame) / fps
                    if speech_buffer else 0.0)
                # In a persistently loud environment the patient endpointing
                # (adaptive growth, gentle relative pause, long max-chunk cap)
                # stops working — noise fills the gaps so the silence counter
                # never advances and only the cap closes the chunk. Couple the
                # endpoint to the noise floor: aggressive when loud, patient when
                # quiet (so long sentences with thinking pauses are not chopped).
                high_noise = self._speech_detector.noise_is_high
                adaptive_silence = self._speech_detector.adaptive_silence_seconds(
                    spoken_seconds, high_noise=high_noise)
                silence_threshold = int(adaptive_silence * fps)
                effective_max_chunk_seconds = self._speech_detector.effective_max_chunk_seconds(
                    self.VAD_MAX_CHUNK_SECONDS)
                effective_max_chunk_frames = (
                    int(effective_max_chunk_seconds * fps)
                    if effective_max_chunk_seconds > 0 else 0)
                nf_ratio = self._speech_detector.noise_floor / max(
                    self._speech_detector.SILENCE_THRESHOLD, 1e-6)
                # As the tolerated pause grows, require a more sustained return to
                # count as resumed speech — so a brief wind gust can't keep the
                # longer pause open (it would otherwise hold the chunk to the cap).
                effective_resume_frames = min(
                    20, max(
                        self._speech_detector.MIN_RESUME_SPEECH_FRAMES,
                        int(self._speech_detector.MIN_RESUME_SPEECH_FRAMES * nf_ratio),
                        self._speech_detector.resume_frames_for_silence(adaptive_silence)))

                if (effective_max_chunk_frames > 0 and speech_buffer
                        and (frames_received - speech_buffer_start_frame) >= effective_max_chunk_frames):
                    logger.info("call %s: max chunk duration reached (%.0fs%s), force-flushing",
                                self.call_id[:8], effective_max_chunk_seconds,
                                ", high-noise" if high_noise else "")
                    task = self._finalize_speech_chunk(
                        speech_buffer, SAMPLE_RATE, fps, min_speech_frames, is_barge_in=False)
                    if task is not None and self._transport:
                        self._transport.start_hold()
                    speech_buffer = []
                    pre_speech_buffer.clear()
                    silence_count = 0
                    resume_speech_count = 0
                    speech_buffer_start_frame = 0
                    speech_ref = 0.0
                    self._mark_user_speech_ended()

                if self._transport and self._transport.is_playing:
                    if (rms >= max(self.BARGEIN_RMS_THRESHOLD, self._speech_detector.SILENCE_THRESHOLD)
                            and speech_like_frame):
                        if not speech_buffer and silence_count == 0:
                            logger.info("call %s: possible barge-in started (rms=%.4f)",
                                        self.call_id[:8], rms)
                            self._mark_user_speech_started()
                            speech_buffer = SpeechDetector.start_buffer(pre_speech_buffer, pcm)
                            speech_buffer_start_frame = frames_received
                        else:
                            speech_buffer.append(pcm)
                        resume_confirmed, resume_speech_count = self._speech_detector.resume_after_pause(
                            speech_like_frame, silence_count, resume_speech_count,
                            min_frames=effective_resume_frames)
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
                            task = self._finalize_speech_chunk(
                                speech_buffer, SAMPLE_RATE, fps, min_speech_frames, is_barge_in=True)
                            speech_buffer = []
                            pre_speech_buffer.clear()
                            silence_count = 0
                            resume_speech_count = 0
                            speech_buffer_start_frame = 0
                            speech_ref = 0.0
                            self._mark_user_speech_ended()
                    continue

                if speech_like_frame:
                    if not speech_buffer and silence_count == 0:
                        self._agc.activate()
                        logger.info("call %s: speech started (rms=%.4f)", self.call_id[:8], rms)
                        self._mark_user_speech_started()
                        speech_buffer = SpeechDetector.start_buffer(pre_speech_buffer, pcm)
                        speech_buffer_start_frame = frames_received
                    else:
                        speech_buffer.append(pcm)
                    resume_confirmed, resume_speech_count = self._speech_detector.resume_after_pause(
                        speech_like_frame, silence_count, resume_speech_count,
                        min_frames=effective_resume_frames)
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
                        task = self._finalize_speech_chunk(
                            speech_buffer, SAMPLE_RATE, fps, min_speech_frames, is_barge_in=False)
                        if task is not None and self._transport:
                            self._transport.start_hold()
                        speech_buffer = []
                        pre_speech_buffer.clear()
                        silence_count = 0
                        resume_speech_count = 0
                        speech_buffer_start_frame = 0
                        speech_ref = 0.0
                        self._mark_user_speech_ended()
        except Exception as e:
            logger.error("call %s: audio pipeline error: %s", self.call_id[:8], e)
        finally:
            self._done.set()
            logger.info("call %s: audio pipeline ended", self.call_id[:8])
            if media_ended and self.HANGUP_ON_MEDIA_END and not self._hungup:
                logger.info("call %s: media track ended; sending hangup", self.call_id[:8])
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
        try:
            text = await self._transcribe_speech(pcm, sample_rate)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("call %s: transcription error: %s", self.call_id[:8], e)
            if not interrupt_playback and self._transport:
                self._transport.stop_hold()
            return

        if not text:
            logger.info("call %s: empty transcription", self.call_id[:8])
            if not interrupt_playback and self._transport:
                self._transport.stop_hold()
            return

        if self._speech_detector.looks_like_stt_hallucination(text):
            logger.info("call %s: ignoring likely STT hallucination: %s", self.call_id[:8], text)
            if not interrupt_playback and self._transport:
                self._transport.stop_hold()
            return

        try:
            if interrupt_playback:
                if _KEYWORD_INTERRUPT_RE.search(text):
                    logger.info("call %s: keyword barge-in detected: %s", self.call_id[:8], text)
                    if self._transport:
                        self._transport.stop_after_current_sentence()
                    await self._cancel_active_response()
                    current_task = asyncio.current_task()
                    if current_task:
                        self._track_response_task(current_task)
                    await self._queue_transcript_response(text)
                elif self._transport and self._transport.is_tts_playing:
                    await self._send_discarded_transcript(text)
                elif self.QUEUE_SPEECH_WHILE_THINKING:
                    logger.info("call %s: speech during hold (bot thinking), queueing: %s",
                                self.call_id[:8], text)
                    await self._send_cb(f"🎙️ *{text}*")
                    self._pending_transcripts.append(text)
                else:
                    # Bot is thinking (hold tone, not speaking). Without an
                    # interrupt keyword this is treated as side conversation and
                    # discarded, so it isn't fed into the next turn.
                    await self._send_discarded_transcript(text)
                return

            if self._transport:
                self._transport.start_hold()
            await self._queue_transcript_response(text)
        except asyncio.CancelledError:
            if self._transport:
                self._transport.stop_hold()
            raise
        except Exception as e:
            logger.error("call %s: agent error: %s", self.call_id[:8], e)

    # ------------------------------------------------------------------
    # Hold audio / keep-typing / network quality
    # ------------------------------------------------------------------

    def _load_hold_audio(self) -> Optional["np.ndarray"]:
        path = self._app.config.get("tts", {}).get("hold_audio")
        if not path:
            path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "assets", "keyboard_mono.wav",
            )
        if not path or not os.path.exists(path):
            if path:
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
        try:
            while True:
                try:
                    await self._client.room_typing(self.room_id, typing_state=True)
                except Exception:
                    pass
                await asyncio.sleep(15)
        except asyncio.CancelledError:
            pass

    def _update_net_state(self, recv: int, lost: int, jitter: float) -> Optional[str]:
        prev_recv, prev_lost = self._net_prev_recv, self._net_prev_lost
        self._net_prev_recv, self._net_prev_lost = recv, lost
        if prev_recv is None or prev_lost is None:
            return None
        d_recv = max(recv - prev_recv, 0)
        d_lost = max(lost - prev_lost, 0)
        loss_interval = d_lost / max(d_recv + d_lost, 1)
        bad = loss_interval > self.NET_WARN_LOSS_RATIO or jitter > self.NET_WARN_JITTER
        if bad:
            self._net_bad_streak += 1
            self._net_good_streak = 0
        else:
            self._net_good_streak += 1
            self._net_bad_streak = 0
        if not self._net_degraded and self._net_bad_streak >= self.NET_DEGRADED_INTERVALS:
            self._net_degraded = True
            return self.NET_WARN_MESSAGE
        if self._net_degraded and self._net_good_streak >= self.NET_RECOVER_INTERVALS:
            self._net_degraded = False
            return self.NET_RECOVER_MESSAGE
        return None

    def _network_prompt_hint(self) -> str:
        if self._net_degraded:
            return (
                "Call network quality: poor right now — the audio link is "
                "choppy/lossy, so the caller's words may arrive garbled or cut "
                "off. If something sounds nonsensical, ask them to repeat "
                "instead of guessing."
            )
        return "Call network quality: good."

    def register_inbound_attachment(self, note: str) -> None:
        if note:
            self._pending_attachments.append(note)
            logger.info("call %s: inbound attachment registered for voice agent", self.call_id[:8])

    def _call_extra_context(self) -> str:
        parts = [self._network_prompt_hint()]
        if self._pending_attachments:
            parts.append(
                "Während dieses Gesprächs eingegangene Anhänge (nur erwähnen/lesen, "
                "wenn der Anrufer danach fragt):\n" + "\n".join(self._pending_attachments)
            )
        return "\n\n".join(p for p in parts if p)

    # ------------------------------------------------------------------
    # Watchdogs
    # ------------------------------------------------------------------

    async def _connect_timeout_watchdog(self) -> None:
        try:
            await self._answer_sent.wait()
            await asyncio.wait_for(
                self._transport.media_connected.wait(),
                timeout=self.CONNECT_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            if self._done.is_set() or self._hungup:
                return
            logger.warning("call %s: media did not connect within %.1fs; ending call",
                           self.call_id[:8], self.CONNECT_TIMEOUT_SECONDS)
            await self._notify_disconnect()
            await self._send_hangup_event()
            await self.hangup()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.debug("call %s: connect timeout watchdog error: %s", self.call_id[:8], e)

    async def _watchdog(self) -> None:
        while not self._done.is_set():
            if self._bot_is_active():
                self._mark_activity()
                try:
                    await asyncio.wait_for(self._done.wait(), timeout=self.WATCHDOG_POLL_SECONDS)
                except asyncio.TimeoutError:
                    continue
            idle_for = time.monotonic() - self._last_activity_at
            remaining = self.CALL_INACTIVITY_SECONDS - idle_for
            if remaining <= 0:
                logger.info("call %s: inactive for %.1fs, hanging up",
                            self.call_id[:8], idle_for)
                await self.hangup()
                await self._send_hangup_event()
                return
            half = self.CALL_INACTIVITY_SECONDS / 2
            if idle_for >= half and not self._agc.active:
                logger.info("call %s: half inactivity reached, activating AGC",
                            self.call_id[:8])
                self._agc.activate()
            try:
                await asyncio.wait_for(
                    self._done.wait(), timeout=min(remaining, self.WATCHDOG_POLL_SECONDS))
            except asyncio.TimeoutError:
                continue
