"""Adaptive Gain Controller for VoIP audio pipelines.

The AGC normalizes incoming audio levels so that quiet speech is
amplified to a target RMS, while avoiding amplification when the bot
is actively speaking (which would amplify the bot's own TTS output
leaking through the mic).

Uses a dual-target approach:

- ``AGC_TARGET_RMS`` when the bot is speaking/generating — avoids
  amplifying over its own TTS.
- ``AGC_QUIET_TARGET_RMS`` when the bot is idle — boosts quiet speech
  in silent environments.

Gain is smoothed via an EMA (``AGC_SMOOTHING``) to avoid sudden
volume jumps.
"""

import logging
import time
from typing import Any, Dict

from pawlia.audio.config import get_float_config

logger = logging.getLogger("pawlia.audio.agc")


class AGCController:
    """Manages adaptive gain for incoming VoIP audio."""

    AGC_WINDOW_SECONDS: float = 15.0
    AGC_TARGET_RMS: float = 0.10
    AGC_QUIET_TARGET_RMS: float = 0.06
    AGC_MAX_GAIN: float = 12.0
    AGC_SMOOTHING: float = 0.15

    def __init__(
        self,
        voip_cfg: Dict[str, Any] | None = None,
        context: str = "",
    ) -> None:
        self._agc_until: float = 0.0
        self._agc_gain: float = 1.0
        self._context = context
        self._apply_config(voip_cfg or {})

    def _apply_config(self, cfg: Dict[str, Any]) -> None:
        self.AGC_WINDOW_SECONDS = get_float_config(
            cfg, "agc_window_seconds", self.AGC_WINDOW_SECONDS,
            context=self._context, minimum=0.1,
        )
        self.AGC_TARGET_RMS = get_float_config(
            cfg, "agc_target_rms", self.AGC_TARGET_RMS,
            context=self._context, minimum=0.001,
        )
        self.AGC_MAX_GAIN = get_float_config(
            cfg, "agc_max_gain", self.AGC_MAX_GAIN,
            context=self._context, minimum=1.0,
        )
        self.AGC_SMOOTHING = get_float_config(
            cfg, "agc_smoothing", self.AGC_SMOOTHING,
            context=self._context, minimum=0.001, maximum=1.0,
        )
        self.AGC_QUIET_TARGET_RMS = get_float_config(
            cfg, "agc_quiet_target_rms", self.AGC_QUIET_TARGET_RMS,
            context=self._context, minimum=0.01, maximum=1.0,
        )

    def activate(self) -> None:
        """Open an AGC window for the next ``AGC_WINDOW_SECONDS``."""
        self._agc_until = time.monotonic() + self.AGC_WINDOW_SECONDS

    @property
    def active(self) -> bool:
        """True while the AGC window is open."""
        return time.monotonic() < self._agc_until

    @property
    def gain(self) -> float:
        """Current smoothed gain factor."""
        return self._agc_gain

    def adjust_rms(self, raw_rms: float, bot_is_active: bool) -> float:
        """Return the AGC-adjusted RMS for VAD decisions.

        Always active — uses ``AGC_TARGET_RMS`` when the bot is
        speaking/generating and ``AGC_QUIET_TARGET_RMS`` otherwise.
        """
        if raw_rms > 1e-6:
            target = self.AGC_TARGET_RMS if bot_is_active else self.AGC_QUIET_TARGET_RMS
            ideal_gain = target / raw_rms
            ideal_gain = min(ideal_gain, self.AGC_MAX_GAIN)
            alpha = self.AGC_SMOOTHING
            self._agc_gain = alpha * ideal_gain + (1 - alpha) * self._agc_gain

        return raw_rms * self._agc_gain
