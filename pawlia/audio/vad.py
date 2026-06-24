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
    # YouTube-style sign-off Whisper invents on noise ("Vielen Dank für's
    # Zuschauen", "Danke fürs Zuschauen") — nobody says this on a call.
    r"|(?:(?:vielen\s+dank|danke)\s+für'?s?\s+zuschauen)"
    r"|(?:tschüss|auf\s+wiedersehen)"
    r"|(?:untertitelung\s+des\s+zdf(?:,\s*\d{4})?)"
    r")\.?$",
    re.IGNORECASE,
)

# Subtitle/credits boilerplate Whisper invents on wind/road noise — caught even
# when embedded in surrounding garbage (e.g. "Untertitelung des ZDF für funk,
# 2017", "Untertitel im Auftrag des ZDF", "Untertitel der Amara.org-Community").
# Requires "untertitel" *plus* a corroborating marker so plain words like
# "Funkgerät" never trigger on their own.
_STT_HALLUCINATION_SUBSTR_RE = re.compile(
    r"(?=.*untertitel)(?=.*(?:zdf|amara|funk|auftrag))|amara\.org",
    re.IGNORECASE | re.DOTALL,
)

# Words that almost never end a finished spoken turn — when a transcript ends on
# one of these (or a comma), the caller is mid-thought and a trailing pause is a
# *thinking* pause, not an endpoint. Used by ``looks_like_incomplete_utterance``
# to hold the response back instead of replying to half a sentence. Kept high
# precision (dangling function words only) so a sentence ending in a content
# word ("...kauf bitte Milch") is never misread as unfinished. Bilingual:
# the caller may speak German or English.
_CONTINUATION_WORDS = frozenset({
    # --- German ---
    # coordinating / subordinating conjunctions
    "und", "oder", "aber", "sondern", "denn", "doch", "sowie", "bzw",
    "beziehungsweise", "weil", "dass", "daß", "ob", "obwohl", "damit", "sodass",
    "während", "bevor", "nachdem", "falls", "sobald", "indem", "als", "wenn",
    "wie", "wo", "wer", "was", "welche", "welcher", "welches",
    # prepositions
    "mit", "für", "von", "zu", "zur", "zum", "im", "in", "an", "am", "auf",
    "bei", "beim", "nach", "über", "unter", "vor", "durch", "gegen", "ohne",
    "um", "aus", "seit", "bis", "wegen", "trotz", "gegenüber",
    # dangling articles / determiners / possessives
    "der", "die", "das", "dem", "den", "des", "ein", "eine", "einen", "einem",
    "einer", "eines", "kein", "keine", "keinen", "mein", "dein", "sein", "ihr",
    "unser", "euer",
    # fillers / discourse markers
    "also", "äh", "ähm", "ehm", "öh", "naja", "halt", "quasi", "sozusagen",
    "zwar", "nämlich",
    # --- English ---
    "and", "or", "but", "so", "because", "that", "which", "who", "if", "when",
    "while", "since", "although", "though", "unless", "whereas",
    "with", "for", "from", "to", "of", "in", "on", "at", "by", "about",
    "into", "onto", "over", "under", "through", "between", "without",
    "the", "a", "an", "my", "your", "his", "her", "their", "our", "its",
    "like", "well", "um", "uh", "uhm", "basically", "actually",
})


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

        effective_threshold = max(SILENCE_THRESHOLD, noise_floor * emphasis)

    where ``emphasis`` is auto-scaled from ``SILENCE_EMPHASIS_FACTOR`` down to
    a minimum of 1.2 as the noise floor rises.  This keeps quiet speech
    detectable in silent environments while preventing AGC-amplified wind or
    road noise from being classified as speech in loud ones.
    """

    SILENCE_THRESHOLD: float = 0.018
    SILENCE_EMPHASIS_FACTOR: float = 2.0
    SILENCE_SECONDS: float = 1.8
    # Adaptive endpointing: the longer the caller has already been speaking in
    # the current utterance, the longer a mid-thought pause we tolerate before
    # closing the chunk — so a long, complex sentence with thinking pauses is
    # not chopped into pieces, while short replies ("ja", "nein") still close
    # promptly at SILENCE_SECONDS. Effective silence grows by
    # SILENCE_GROWTH_PER_SEC for each second already spoken, capped at
    # SILENCE_SECONDS_MAX. Set SILENCE_GROWTH_PER_SEC=0 for a fixed endpoint.
    #
    # The longer pause is only safe against gusty wind because we *also* require
    # a more sustained return to count as resumed speech as the tolerance grows
    # (resume_frames_for_silence): a short wind gust (1–5 frames) can no longer
    # cancel a long pause and hold the chunk open, while a real continuation
    # (the caller speaking again) still does. RESUME_FRAMES_AT_MAX is the resume
    # requirement when the pause tolerance is at SILENCE_SECONDS_MAX.
    SILENCE_GROWTH_PER_SEC: float = 0.12
    SILENCE_SECONDS_MAX: float = 3.0
    RESUME_FRAMES_AT_MAX: int = 8
    MIN_SPEECH_SECONDS: float = 0.4
    MIN_ACTIVE_SPEECH_RATIO: float = 0.12
    MIN_CONSECUTIVE_SPEECH_FRAMES: int = 8
    MIN_SPEECH_BAND_RATIO: float = 0.35
    MAX_SPECTRAL_FLATNESS: float = 0.72
    MIN_SPEECH_LIKE_RATIO: float = 0.08
    MIN_CONSECUTIVE_SPEECHLIKE_FRAMES: int = 4
    MIN_RESUME_SPEECH_FRAMES: int = 3
    PRE_SPEECH_SECONDS: float = 0.6
    WEBRTC_VAD_ENABLED: bool = True
    WEBRTC_VAD_MODE: int = 2
    WEBRTC_VAD_MIN_VOICED_RATIO: float = 0.12
    WEBRTC_VAD_MIN_CONSECUTIVE_FRAMES: int = 4
    BARGEIN_MIN_WORDS: int = 4
    BARGEIN_MIN_CHARS: int = 12
    # When the noise floor stays well above clean levels (~0.01), the
    # environment is genuinely loud (wind/road noise while cycling). Only then
    # does should_transcribe tighten its gate by HIGH_NOISE_STRICTNESS, so quiet
    # and moderately noisy (local) calls keep the unchanged behaviour.
    HIGH_NOISE_FLOOR: float = 0.03
    HIGH_NOISE_STRICTNESS: float = 1.3
    # In a persistently loud environment, also require the chunk's envelope to be
    # modulated like speech (syllabic loud/quiet alternation). Steady wind/road
    # noise has a near-constant envelope (CoV ~0.3-0.6) while real speech sits at
    # ~1.0+, so this rejects the most common steady-wind chunks at the VAD gate
    # instead of letting them reach STT and hallucinate. Only applied when
    # noise_floor > HIGH_NOISE_FLOOR, so quiet/local calls are unaffected.
    MIN_SPEECH_MODULATION: float = 0.7
    # Relative-energy pause: during an open utterance a frame whose level has
    # dropped below this fraction of the running speech level is treated as a
    # pause even if it still reads spectrally speech-like. In sustained wind a
    # real pause falls back to the wind floor (well under the speaker's own
    # loudness), so this lets the chunk close instead of staying open for tens
    # of seconds. 0 disables it.
    SPEECH_PAUSE_RATIO: float = 0.25
    # When the environment is persistently loud (noise_floor > HIGH_NOISE_FLOOR,
    # e.g. wind/train), the gentle, patient endpointing above stops working: the
    # noise fills the gaps between words so the silence counter never advances,
    # the relative pause at SPEECH_PAUSE_RATIO is too lax to register a real
    # pause against the speaker's own loudness, and the adaptive growth only
    # delays the close further. So in loud environments we switch to a more
    # aggressive endpoint — a higher relative-pause fraction, no adaptive growth
    # of the pause tolerance (clamped to HIGH_NOISE_SILENCE_SECONDS_MAX), and a
    # shorter max-chunk cap — trading the (already impossible) clean capture of a
    # long monologue for a bounded, responsive reply. Quiet/local calls keep the
    # patient values, so long sentences with thinking pauses are not chopped.
    # Set HIGH_NOISE_PAUSE_RATIO == SPEECH_PAUSE_RATIO to opt out of the coupling.
    HIGH_NOISE_PAUSE_RATIO: float = 0.40
    HIGH_NOISE_SILENCE_SECONDS_MAX: float = 1.8
    HIGH_NOISE_MAX_CHUNK_SECONDS: float = 8.0

    def __init__(
        self,
        voip_cfg: Dict[str, Any] | None = None,
        context: str = "",
    ) -> None:
        self._context = context
        self._noise_floor: float = 0.05
        self._last_speech_duration: float = 0.0
        self._webrtc_vad = None
        self._apply_config(voip_cfg or {})
        self._webrtc_vad = self._init_webrtc_vad()

    _FLOAT_CONFIGS = [
        ("silence_threshold", "SILENCE_THRESHOLD", 0.0, None),
        ("silence_emphasis_factor", "SILENCE_EMPHASIS_FACTOR", 1.0, 10.0),
        ("silence_seconds", "SILENCE_SECONDS", 0.1, None),
        ("silence_growth_per_sec", "SILENCE_GROWTH_PER_SEC", 0.0, None),
        ("silence_seconds_max", "SILENCE_SECONDS_MAX", 0.1, None),
        ("min_speech_seconds", "MIN_SPEECH_SECONDS", 0.1, None),
        ("min_active_speech_ratio", "MIN_ACTIVE_SPEECH_RATIO", 0.0, 1.0),
        ("min_speech_band_ratio", "MIN_SPEECH_BAND_RATIO", 0.0, 1.0),
        ("max_spectral_flatness", "MAX_SPECTRAL_FLATNESS", 0.0, 1.0),
        ("min_speech_like_ratio", "MIN_SPEECH_LIKE_RATIO", 0.0, 1.0),
        ("pre_speech_seconds", "PRE_SPEECH_SECONDS", 0.0, None),
        ("webrtcvad_min_voiced_ratio", "WEBRTC_VAD_MIN_VOICED_RATIO", 0.0, 1.0),
        ("high_noise_floor", "HIGH_NOISE_FLOOR", 0.0, None),
        ("high_noise_strictness", "HIGH_NOISE_STRICTNESS", 1.0, 3.0),
        ("min_speech_modulation", "MIN_SPEECH_MODULATION", 0.0, None),
        ("speech_pause_ratio", "SPEECH_PAUSE_RATIO", 0.0, 1.0),
        ("high_noise_pause_ratio", "HIGH_NOISE_PAUSE_RATIO", 0.0, 1.0),
        ("high_noise_silence_seconds_max", "HIGH_NOISE_SILENCE_SECONDS_MAX", 0.1, None),
        ("high_noise_max_chunk_seconds", "HIGH_NOISE_MAX_CHUNK_SECONDS", 0.0, None),
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

        The emphasis factor is auto-scaled: it starts at SILENCE_EMPHASIS_FACTOR
        (default 2.0) in quiet environments and decreases toward 1.2 as the
        noise floor rises.  This prevents AGC-amplified ambient noise from
        exceeding the effective threshold in loud settings.
        """
        emphasis = self.SILENCE_EMPHASIS_FACTOR
        if self._noise_floor > 0.002:
            emphasis = max(1.2, emphasis - (self._noise_floor - 0.002) * 30.0)
        effective_threshold = max(
            self.SILENCE_THRESHOLD,
            self._noise_floor * emphasis,
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
                "modulation": 1.0,
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

        # Envelope modulation: coefficient of variation of the active-frame RMS.
        # Speech is syllabic (loud/quiet alternation → high CoV ~1.0+); steady
        # wind/road noise is near-constant (low CoV ~0.3-0.6). Computed over
        # active frames only so leading/trailing silence doesn't inflate it.
        active_rms = frame_rms[active_mask]
        modulation = (
            float(np.std(active_rms) / (np.mean(active_rms) + 1e-9))
            if active_rms.size >= 5 else 1.0
        )

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
            "modulation": modulation,
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

        min_voiced_ratio = self.WEBRTC_VAD_MIN_VOICED_RATIO
        min_voiced_run = self.WEBRTC_VAD_MIN_CONSECUTIVE_FRAMES
        high_noise = False
        if self._noise_floor < 0.001:
            min_active_ratio = self.MIN_ACTIVE_SPEECH_RATIO * 0.5
            min_speech_like = self.MIN_SPEECH_LIKE_RATIO * 0.5
        elif self._noise_floor > self.HIGH_NOISE_FLOOR:
            high_noise = True
            # Persistently loud environment (wind/road noise): raise the bar so
            # AGC-amplified noise is less likely to pass as speech. Reached only
            # well above clean levels, so quiet/local calls are unaffected.
            s = self.HIGH_NOISE_STRICTNESS
            min_active_ratio = self.MIN_ACTIVE_SPEECH_RATIO * s
            min_speech_like = self.MIN_SPEECH_LIKE_RATIO * s
            min_voiced_ratio = self.WEBRTC_VAD_MIN_VOICED_RATIO * s
            min_voiced_run = self.WEBRTC_VAD_MIN_CONSECUTIVE_FRAMES + 1
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
        # In a loud environment, also demand speech-like envelope modulation so a
        # steady wind/road-noise chunk (near-constant loudness) is rejected here
        # rather than reaching STT. ``stats.get`` keeps callers that mock
        # analyze_chunk without this key working unchanged.
        if high_noise and stats.get("modulation", 1.0) < self.MIN_SPEECH_MODULATION:
            return False
        if self._webrtc_vad is None:
            return True
        return (
            stats["voiced_ratio"] >= min_voiced_ratio
            and stats["voiced_run"] >= min_voiced_run
        )

    # ------------------------------------------------------------------
    # Pause / resume helpers
    # ------------------------------------------------------------------

    def resume_after_pause(
        self,
        speech_like_frame: bool,
        silence_count: int,
        resume_speech_count: int,
        min_frames: Optional[int] = None,
    ) -> tuple[bool, int]:
        """Require a short sustained return before breaking an in-progress pause."""
        if not speech_like_frame or silence_count <= 0:
            return False, 0
        resume_speech_count += 1
        threshold = min_frames if min_frames is not None else self.MIN_RESUME_SPEECH_FRAMES
        return resume_speech_count >= threshold, resume_speech_count

    @property
    def noise_is_high(self) -> bool:
        """Persistently loud environment (wind/road/train noise).

        The same gate ``should_transcribe`` uses to tighten its acceptance: the
        noise floor sits well above clean levels. In this regime the patient
        endpointing is counter-productive (see :meth:`effective_pause_ratio`,
        :meth:`effective_max_chunk_seconds`, and the ``high_noise`` branch of
        :meth:`adaptive_silence_seconds`).
        """
        return self._noise_floor > self.HIGH_NOISE_FLOOR

    def effective_pause_ratio(self) -> float:
        """Relative-energy pause fraction for the *current* noise conditions.

        In a loud environment the gentle ``SPEECH_PAUSE_RATIO`` is too lax to
        register a real pause against the speaker's own (AGC-amplified) loudness,
        so the chunk only ever closes on the max-chunk cap. Raising the fraction
        lets a genuine mid-utterance drop register as a pause again. Quiet calls
        keep ``SPEECH_PAUSE_RATIO`` so deliberate, level mid-sentence pauses are
        not mistaken for an endpoint. ``SPEECH_PAUSE_RATIO == 0`` (disabled)
        stays disabled regardless of noise.
        """
        if self.SPEECH_PAUSE_RATIO <= 0.0:
            return 0.0
        if self.noise_is_high:
            return max(self.SPEECH_PAUSE_RATIO, self.HIGH_NOISE_PAUSE_RATIO)
        return self.SPEECH_PAUSE_RATIO

    def effective_max_chunk_seconds(self, base_cap: float) -> float:
        """Hard max-chunk cap for the current noise conditions.

        Shortened to ``HIGH_NOISE_MAX_CHUNK_SECONDS`` in a loud environment so
        the worst-case wait is bounded when no pause can be detected (the caller
        otherwise speaks into the void for the full quiet-call budget). A cap the
        operator has disabled (``base_cap <= 0``) stays disabled.
        """
        if base_cap <= 0.0:
            return 0.0
        if self.noise_is_high and self.HIGH_NOISE_MAX_CHUNK_SECONDS > 0.0:
            return min(base_cap, self.HIGH_NOISE_MAX_CHUNK_SECONDS)
        return base_cap

    def adaptive_silence_seconds(
        self, spoken_seconds: float, high_noise: bool = False
    ) -> float:
        """Pause tolerance (s) before an open chunk is finalised.

        Grows with how long the caller has already been speaking in the current
        utterance, so a long sentence with thinking pauses is not chopped into
        separate STT chunks, while short replies still close promptly at the
        ``SILENCE_SECONDS`` base. ``SILENCE_GROWTH_PER_SEC == 0`` disables growth.

        When ``high_noise`` is set (caller passes :attr:`noise_is_high`), the
        growth is suppressed and the tolerance is clamped to
        ``HIGH_NOISE_SILENCE_SECONDS_MAX``: in sustained noise a long tolerated
        pause can never be satisfied (the noise fills the gaps), so growing it
        only delays the close to the cap. A short, fixed tolerance lets the
        relative-energy pause actually fire.

        Wind safety for the *grown* (quiet-call) path is handled via
        :meth:`resume_frames_for_silence` — see the class-level note.
        """
        base = max(1.2, self.SILENCE_SECONDS)
        if high_noise:
            return min(base, max(1.2, self.HIGH_NOISE_SILENCE_SECONDS_MAX))
        if self.SILENCE_GROWTH_PER_SEC <= 0.0:
            return base
        grown = base + max(0.0, spoken_seconds) * self.SILENCE_GROWTH_PER_SEC
        return min(max(base, self.SILENCE_SECONDS_MAX), grown)

    def resume_frames_for_silence(self, silence_seconds: float) -> int:
        """Consecutive speech-like frames needed to cancel a pause of the given
        tolerance.

        Scales linearly from ``MIN_RESUME_SPEECH_FRAMES`` at the base
        ``SILENCE_SECONDS`` up to ``RESUME_FRAMES_AT_MAX`` at
        ``SILENCE_SECONDS_MAX``: a longer tolerated pause demands a more
        sustained return, so brief wind gusts cannot keep it open.
        """
        base = max(1.2, self.SILENCE_SECONDS)
        ceil = max(base, self.SILENCE_SECONDS_MAX)
        if ceil <= base:
            return self.MIN_RESUME_SPEECH_FRAMES
        frac = (max(base, min(ceil, silence_seconds)) - base) / (ceil - base)
        span = max(0, self.RESUME_FRAMES_AT_MAX - self.MIN_RESUME_SPEECH_FRAMES)
        return self.MIN_RESUME_SPEECH_FRAMES + int(round(frac * span))

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
        """Filter common Whisper fallback phrases from non-speech chunks.

        Two signatures, both absent from genuine speech, so this stays safe in
        quiet/local conditions:
          1. exact stock sign-off phrases ("Vielen Dank.", "Untertitelung …");
          2. subtitle/credits boilerplate embedded in surrounding garbage
             ("Untertitelung des ZDF für funk, 2017") — requires "untertitel"
             *plus* a corroborating marker, so plain words never trigger.
        """
        normalized = " ".join((text or "").strip().split())
        if not normalized:
            return False
        if _STANDALONE_STT_HALLUCINATION_RE.fullmatch(normalized):
            return True
        return bool(_STT_HALLUCINATION_SUBSTR_RE.search(normalized))

    @staticmethod
    def looks_like_incomplete_utterance(text: str) -> bool:
        """Return True when a transcript looks like a mid-thought fragment.

        Pure-text semantic endpointing: reply timers in the call pipeline are
        time-based, so a thinking pause after an unfinished clause ("...und das
        liegt daran, dass") would otherwise trigger a reply to half a sentence.
        This lets the caller hold the response back until the thought is closed.

        High precision by design — only the strongest dangling-fragment signals
        fire, so a complete sentence ending in a content word ("...kauf bitte
        Milch") is never misread as unfinished:

          1. a trailing comma (a comma never ends a spoken turn);
          2. the last word is a dangling function word (conjunction, preposition,
             article, filler) from :data:`_CONTINUATION_WORDS`.

        Signal 2 only applies once the utterance has at least three words, so a
        short standalone reply ("ja", "naja", "well") stays complete. Missing
        terminal punctuation alone is *not* a signal — Whisper drops it even on
        finished sentences, which would over-hold.
        """
        normalized = " ".join((text or "").strip().split())
        if not normalized:
            return False
        if normalized.endswith(","):
            return True
        words = re.findall(r"\b[\wäöüß']+\b", normalized, flags=re.UNICODE)
        if len(words) < 3:
            return False
        return words[-1].lower() in _CONTINUATION_WORDS
