"""Voice Activity Detection for VoIP audio pipelines.

Provides frame-level and chunk-level speech analysis using RMS energy,
spectral features (band ratio, spectral flatness), and optional WebRTC VAD.
"""

import logging
import re
from typing import Any, Dict, List, Optional
from collections import deque

import numpy as np

from pawlia.audio.config import get_float_config, get_int_config, get_bool_config

logger = logging.getLogger("pawlia.audio.vad")

try:
    import webrtcvad  # type: ignore
    _WEBRTCVAD_IMPORT_ERROR = None
except Exception as _e:
    webrtcvad = None  # type: ignore[assignment]
    _WEBRTCVAD_IMPORT_ERROR = _e

_INTERRUPT_KEYWORD_RE = re.compile(
    r"\b(?:halt|stop|stopp|wait|warte|warten|moment|sekunde|pause)\b",
    re.IGNORECASE,
)

_STANDALONE_STT_HALLUCINATION_RE = re.compile(
    r"^(?:"
    r"(?:vielen\s+dank|danke(?:\s+schön)?)"
    r"|(?:tschüss|auf\s+wiedersehen)"
    r"|(?:untertitelung\s+des\s+zdf(?:,\s*\d{4})?)"
    r")\.?$",
    re.IGNORECASE,
)


def _build_webrtc_vad(mode: int):
    """Create a WebRTC VAD instance when the optional dependency is available."""
    if webrtcvad is None:
        return None
    try:
        return webrtcvad.Vad(mode)
    except Exception as e:
        logger.warning("could not initialize webrtcvad(mode=%s): %s", mode, e)
        return None


class SpeechDetector:
    """Configurable speech detection pipeline for VoIP audio.

    Adaptive silence detection
    --------------------------
    Rather than using a fixed RMS threshold to distinguish speech from silence,
    the pipeline maintains a rolling EMA of the background noise floor
    (``_noise_floor``) during periods when no speech buffer is active.  The
    frame-level gate (``is_speech_like_frame``) then uses

        effective_threshold = max(SILENCE_THRESHOLD, noise_floor * SILENCE_EMPHASIS_FACTOR)

    so that steady background noise (e.g. road noise while cycling) falls
    below the effective threshold and counts as silence.  This lets
    silence_count accumulate even when the raw RMS never drops to zero.
    """

    SILENCE_THRESHOLD: float = 0.018
    SILENCE_EMPHASIS_FACTOR: float = 2.0
    SILENCE_SECONDS: float = 1.8
    MIN_SPEECH_SECONDS: float = 0.4
    MIN_ACTIVE_SPEECH_RATIO: float = 0.12
    MIN_CONSECUTIVE_SPEECH_FRAMES: int = 8
    MIN_SPEECH_BAND_RATIO: float = 0.35
    MAX_SPECTRAL_FLATNESS: float = 0.72
    MIN_SPEECH_LIKE_RATIO: float = 0.08
    MIN_CONSECUTIVE_SPEECHLIKE_FRAMES: int = 4
    MIN_RESUME_SPEECH_FRAMES: int = 3
    PRE_SPEECH_SECONDS: float = 0.4
    WEBRTC_VAD_ENABLED: bool = True
    WEBRTC_VAD_MODE: int = 2
    WEBRTC_VAD_MIN_VOICED_RATIO: float = 0.12
    WEBRTC_VAD_MIN_CONSECUTIVE_FRAMES: int = 4
    BARGEIN_MIN_WORDS: int = 4
    BARGEIN_MIN_CHARS: int = 12

    def __init__(
        self,
        voip_cfg: Dict[str, Any] | None = None,
        context: str = "",
    ) -> None:
        self._context = context
        self._noise_floor: float = 0.01
        self._last_speech_duration: float = 0.0
        self._webrtc_vad = None
        self._apply_config(voip_cfg or {})
        self._webrtc_vad = self._init_webrtc_vad()

    _FLOAT_CONFIGS = [
        ("silence_threshold", "SILENCE_THRESHOLD", 0.0, None),
        ("silence_emphasis_factor", "SILENCE_EMPHASIS_FACTOR", 1.0, 10.0),
        ("silence_seconds", "SILENCE_SECONDS", 0.1, None),
        ("min_speech_seconds", "MIN_SPEECH_SECONDS", 0.1, None),
        ("min_active_speech_ratio", "MIN_ACTIVE_SPEECH_RATIO", 0.0, 1.0),
        ("min_speech_band_ratio", "MIN_SPEECH_BAND_RATIO", 0.0, 1.0),
        ("max_spectral_flatness", "MAX_SPECTRAL_FLATNESS", 0.0, 1.0),
        ("min_speech_like_ratio", "MIN_SPEECH_LIKE_RATIO", 0.0, 1.0),
        ("pre_speech_seconds", "PRE_SPEECH_SECONDS", 0.0, None),
        ("webrtcvad_min_voiced_ratio", "WEBRTC_VAD_MIN_VOICED_RATIO", 0.0, 1.0),
    ]

    _INT_CONFIGS = [
        ("min_consecutive_speech_frames", "MIN_CONSECUTIVE_SPEECH_FRAMES", 1, None),
        ("min_consecutive_speechlike_frames", "MIN_CONSECUTIVE_SPEECHLIKE_FRAMES", 1, None),
        ("min_resume_speech_frames", "MIN_RESUME_SPEECH_FRAMES", 1, None),
        ("webrtcvad_mode", "WEBRTC_VAD_MODE", 0, 3),
        ("webrtcvad_min_consecutive_frames", "WEBRTC_VAD_MIN_CONSECUTIVE_FRAMES", 1, None),
    ]

    _BOOL_CONFIGS = [
        ("webrtcvad_enabled", "WEBRTC_VAD_ENABLED"),
    ]

    def _apply_config(self, cfg: Dict[str, Any]) -> None:
        for key, suffix, minimum, maximum in self._FLOAT_CONFIGS:
            setattr(self, suffix, get_float_config(
                cfg, key, getattr(self, suffix),
                context=self._context,
                minimum=minimum, maximum=maximum,
            ))

        for key, suffix, minimum, maximum in self._INT_CONFIGS:
            setattr(self, suffix, get_int_config(
                cfg, key, getattr(self, suffix),
                context=self._context,
                minimum=minimum, maximum=maximum,
            ))

        for key, suffix in self._BOOL_CONFIGS:
            setattr(self, suffix, get_bool_config(
                cfg, key, getattr(self, suffix),
                context=self._context,
            ))

    def _init_webrtc_vad(self):
        if not self.WEBRTC_VAD_ENABLED:
            return None
        vad = _build_webrtc_vad(self.WEBRTC_VAD_MODE)
        if vad is None:
            logger.info(
                "%s: webrtcvad unavailable, continuing without it",
                self._context,
            )
        return vad

    # ------------------------------------------------------------------
    # Properties for state shared with CallSession
    # ------------------------------------------------------------------

    @property
    def noise_floor(self) -> float:
        """Rolling EMA of background noise floor."""
        return self._noise_floor

    @property
    def last_speech_duration(self) -> float:
        """Duration of the most recently accepted speech chunk (seconds)."""
        return self._last_speech_duration

    @last_speech_duration.setter
    def last_speech_duration(self, value: float) -> None:
        self._last_speech_duration = value

    # ------------------------------------------------------------------
    # Noise-floor EMA update
    # ------------------------------------------------------------------

    def update_noise_floor(self, rms: float, during_speech: bool) -> None:
        """Update the rolling background noise floor EMA.

        While not accumulating speech, a *fast* EMA (~1 s time constant)
        adapts quickly to room changes.  During speech, a *very slow* EMA
        (~10 s) tracks slow drift (e.g. fan, HVAC cycling) without
        contamination from the speech signal.

        Call this once per audio frame before the normal VAD pipeline.
        """
        if not during_speech:
            self._noise_floor = 0.02 * rms + 0.98 * self._noise_floor
        elif rms < self._noise_floor * self.SILENCE_EMPHASIS_FACTOR:
            self._noise_floor = 0.002 * rms + 0.998 * self._noise_floor

    # ------------------------------------------------------------------
    # Frame-level speech classification
    # ------------------------------------------------------------------

    def is_speech_like_frame(
        self,
        pcm: "np.ndarray",
        sample_rate: int,
        adjusted_rms: float,
    ) -> bool:
        """Return True when a live frame looks like speech, not just loud noise.

        Uses an adaptive threshold based on the current noise floor, band-ratio
        (how much energy is in the 180–4000 Hz speech band), and spectral
        flatness (white noise vs. tonal signal).
        """
        effective_threshold = max(
            self.SILENCE_THRESHOLD,
            self._noise_floor * self.SILENCE_EMPHASIS_FACTOR,
        )
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

    # ------------------------------------------------------------------
    # Chunk-level speech analysis
    # ------------------------------------------------------------------

    def analyze_chunk(
        self,
        pcm: "np.ndarray",
        sample_rate: int,
        fps: int,
        agc_gain: float = 1.0,
        agc_active: bool = False,
    ) -> Dict[str, float]:
        """Summarize frame-level activity for a completed speech chunk.

        Returns a dict with keys: frame_count, active_frames, active_ratio,
        longest_run, voiced_frames, voiced_ratio, voiced_run,
        speech_like_frames, speech_like_ratio, speech_like_run,
        median_band_ratio, median_flatness, p90_rms.
        """
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
        if agc_active:
            frame_rms = frame_rms * agc_gain
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
                    logger.debug("%s: webrtcvad frame analysis failed: %s", self._context, e)
                    voiced_mask[idx] = False

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
        flatness = np.exp(np.mean(np.log(power + 1e-9), axis=1)) / np.maximum(
            np.mean(power + 1e-9, axis=1), 1e-9
        )
        speech_like_mask = (
            active_mask
            & (band_ratio >= self.MIN_SPEECH_BAND_RATIO)
            & (flatness <= self.MAX_SPECTRAL_FLATNESS)
        )

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

    def should_transcribe(
        self,
        pcm: "np.ndarray",
        sample_rate: int,
        fps: int,
        agc_gain: float = 1.0,
        agc_active: bool = False,
    ) -> bool:
        """Return True only when a chunk contains sustained speech-like activity."""
        stats = self.analyze_chunk(pcm, sample_rate, fps, agc_gain, agc_active)

        if self._noise_floor < 0.001:
            min_active_ratio = self.MIN_ACTIVE_SPEECH_RATIO * 0.5
            min_speech_like = self.MIN_SPEECH_LIKE_RATIO * 0.5
        else:
            min_active_ratio = self.MIN_ACTIVE_SPEECH_RATIO
            min_speech_like = self.MIN_SPEECH_LIKE_RATIO

        basic_match = (
            stats["active_ratio"] >= min_active_ratio
            and stats["longest_run"] >= self.MIN_CONSECUTIVE_SPEECH_FRAMES
            and stats["speech_like_ratio"] >= min_speech_like
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

    # ------------------------------------------------------------------
    # Pause / resume helpers
    # ------------------------------------------------------------------

    def resume_after_pause(
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

    @staticmethod
    def start_buffer(
        pre_speech_buffer: "deque[np.ndarray]",
        pcm: "np.ndarray",
    ) -> List["np.ndarray"]:
        """Seed a new speech chunk with a small pre-roll before the trigger frame."""
        chunk = list(pre_speech_buffer)
        chunk.append(pcm)
        return chunk

    # ------------------------------------------------------------------
    # Transcript filters (interrupt detection, hallucination filtering)
    # ------------------------------------------------------------------

    def is_meaningful_interrupt(self, text: str) -> bool:
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

    @staticmethod
    def looks_like_stt_hallucination(text: str) -> bool:
        """Filter common Whisper fallback phrases from non-speech chunks."""
        normalized = " ".join((text or "").strip().split())
        if not normalized:
            return False
        return bool(_STANDALONE_STT_HALLUCINATION_RE.fullmatch(normalized))
