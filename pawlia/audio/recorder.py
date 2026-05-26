"""Call recorder — captures the full inbound audio stream for a VoIP call.

Records all received PCM frames into a single WAV file per call, optionally
compresses to FLAC, and provides a rotation helper that purges recordings
older than a configurable number of days.
"""

import logging
import os
import wave
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np

logger = logging.getLogger("pawlia.audio.recorder")

# ── Default paths ──
_PKG_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_RECORD_DIR = os.path.join(_PKG_DIR, "log", "call_recordings")
DEFAULT_RETENTION_DAYS = 7


class CallRecorder:
    """Accumulates inbound PCM frames for a single call and writes them on hangup.

    When both caller and Pawlia audio are recorded the output is a stereo
    WAV: left = caller, right = Pawlia.  If only the caller side has audio
    (e.g. recording was disabled for TTS) the result is mono.
    """

    def __init__(
        self,
        call_id: str,
        record_dir: str = DEFAULT_RECORD_DIR,
        sample_rate: int = 48000,
        compress_to_flac: bool = True,
    ) -> None:
        self.call_id = call_id
        self.record_dir = record_dir or DEFAULT_RECORD_DIR
        self.sample_rate = sample_rate
        self.compress_to_flac = compress_to_flac
        self._frames: List[np.ndarray] = []
        self._total_samples = 0
        self._frames_pawlia: List[np.ndarray] = []
        self._total_samples_pawlia = 0

    def push(self, pcm: np.ndarray) -> None:
        """Append a single PCM frame from the *caller* (float32 mono, [-1.0, 1.0])."""
        self._frames.append(pcm)
        self._total_samples += len(pcm)

    def push_pawlia(self, pcm: np.ndarray) -> None:
        """Append a single PCM frame from *Pawlia's TTS* (float32 mono, [-1.0, 1.0])."""
        self._frames_pawlia.append(pcm)
        self._total_samples_pawlia += len(pcm)

    def finish(self) -> Optional[str]:
        """Write accumulated audio to disk. Returns the output file path or None."""
        has_caller = self._frames and self._total_samples > 0
        has_pawlia = self._frames_pawlia and self._total_samples_pawlia > 0

        if not has_caller and not has_pawlia:
            logger.debug("call %s: no audio recorded, skipping", self.call_id[:8])
            return None

        os.makedirs(self.record_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{ts}_{self.call_id[:8]}"

        wav_path = os.path.join(self.record_dir, f"{base_name}.wav")
        try:
            if has_pawlia:
                caller = np.concatenate(self._frames) if has_caller else np.zeros(0, dtype=np.float32)
                pawlia = np.concatenate(self._frames_pawlia)
                max_len = max(len(caller), len(pawlia))
                stereo = np.empty((max_len, 2), dtype=np.float32)
                if len(caller):
                    stereo[: len(caller), 0] = caller
                if len(caller) < max_len:
                    stereo[len(caller):, 0] = 0.0
                stereo[: len(pawlia), 1] = pawlia
                if len(pawlia) < max_len:
                    stereo[len(pawlia):, 1] = 0.0
                stereo_int16 = (np.clip(stereo, -1.0, 1.0) * 32767).astype(np.int16)
                with wave.open(wav_path, "wb") as wf:
                    wf.setnchannels(2)
                    wf.setsampwidth(2)
                    wf.setframerate(self.sample_rate)
                    wf.writeframes(stereo_int16.tobytes())
                logger.info(
                    "call %s: recorded %.1fs caller + %.1fs pawlia (stereo) → %s",
                    self.call_id[:8],
                    self._total_samples / self.sample_rate,
                    self._total_samples_pawlia / self.sample_rate,
                    wav_path,
                )
            else:
                pcm_int16 = (np.clip(np.concatenate(self._frames), -1.0, 1.0) * 32767).astype(np.int16)
                with wave.open(wav_path, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(self.sample_rate)
                    wf.writeframes(pcm_int16.tobytes())
                logger.info(
                    "call %s: recorded %.1fs → %s",
                    self.call_id[:8], self._total_samples / self.sample_rate, wav_path,
                )
        except Exception as e:
            logger.error("call %s: failed to write WAV: %s", self.call_id[:8], e)
            return None

        # Optional FLAC compression
        if self.compress_to_flac:
            flac_path = self._compress_flac(wav_path)
            if flac_path:
                # Remove WAV after successful FLAC compression
                try:
                    os.remove(wav_path)
                except OSError:
                    pass
                return flac_path

        return wav_path

    @staticmethod
    def _compress_flac(wav_path: str) -> Optional[str]:
        """Compress a WAV file to FLAC using the ``flac`` CLI if available."""
        import shutil
        import subprocess

        if not shutil.which("flac"):
            logger.debug("flac CLI not found — keeping WAV recording")
            return None

        flac_path = wav_path.rsplit(".", 1)[0] + ".flac"
        try:
            result = subprocess.run(
                ["flac", "--best", "--silent", "-o", flac_path, wav_path],
                capture_output=True, timeout=60,
            )
            if result.returncode == 0 and os.path.exists(flac_path):
                wav_size = os.path.getsize(wav_path)
                flac_size = os.path.getsize(flac_path)
                ratio = (1 - flac_size / wav_size) * 100 if wav_size else 0
                logger.info(
                    "call recording compressed to FLAC: %s (%.0f%% savings)",
                    flac_path, ratio,
                )
                return flac_path
            else:
                logger.warning("flac compression failed: %s", result.stderr.decode(errors="replace"))
                return None
        except subprocess.TimeoutExpired:
            logger.warning("flac compression timed out")
            return None
        except Exception as e:
            logger.warning("flac compression error: %s", e)
            return None


def rotate_recordings(
    record_dir: str = DEFAULT_RECORD_DIR,
    retention_days: int = DEFAULT_RETENTION_DAYS,
) -> int:
    """Delete call recordings older than *retention_days*. Returns count of deleted files."""
    if not os.path.isdir(record_dir):
        return 0

    cutoff = datetime.now() - timedelta(days=retention_days)
    deleted = 0

    for fname in os.listdir(record_dir):
        if not (fname.endswith(".wav") or fname.endswith(".flac")):
            continue
        fpath = os.path.join(record_dir, fname)
        try:
            mtime = datetime.fromtimestamp(os.path.getmtime(fpath))
            if mtime < cutoff:
                os.remove(fpath)
                deleted += 1
                logger.debug("rotated old recording: %s", fpath)
        except OSError as e:
            logger.warning("could not rotate %s: %s", fpath, e)

    if deleted:
        logger.info("rotated %d recording(s) older than %d days from %s", deleted, retention_days, record_dir)
    return deleted
