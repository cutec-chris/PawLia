"""Audio transcription for PawLia via any OpenAI-compatible Whisper endpoint.

Config layout (YAML)::

    # API-based — Groq example (any compatible endpoint works):
    transcription:
      provider: groq
      groq:
        api_key: YOUR_GROQ_API_KEY
        model: whisper-large-v3-turbo
        # base_url: https://api.groq.com/openai/v1   # set automatically; override if needed
        # language: de

    # Other provider (OpenAI or self-hosted):
    # transcription:
    #   provider: openai
    #   openai:
    #     api_key: YOUR_API_KEY
    #     base_url: https://api.openai.com/v1
    #     model: whisper-1
    #     # language: de

    # Local (faster-whisper, requires FFmpeg) or self-hosted OpenAI-compatible:
    # transcription:
    #   provider: local
    #   local:
    #     model: base
    #     # base_url: http://127.0.0.1:8000/v1
    #     device: cpu
    #     compute_type: int8
    #     # language: de
"""

import asyncio
import logging
import os
import subprocess
import tempfile
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("pawlia.transcription")

# Default base URLs per known provider name
_PROVIDER_BASE_URLS: Dict[str, str] = {
    "groq":  "https://api.groq.com/openai/v1",
    "openai": "https://api.openai.com/v1",
}
_DEFAULT_MODEL = "whisper-large-v3-turbo"

_NATIVE_AUDIO_PROMPT = (
    "Transkribiere diese Audiodatei wörtlich. "
    "Antworte NUR mit dem gesprochenen Text, ohne Erklärungen oder Formatierung."
)

_DEFAULT_PREPROCESS: Dict[str, float] = {
    "highpass_hz": 140.0,
    "lowpass_hz": 7000.0,
    "denoise_strength": 1.25,
    "denoise_floor": 0.2,
    "adaptive_gate_percentile": 0.2,
    "adaptive_gate_multiplier": 2.2,
    "gate_threshold": 0.015,
    "gate_ratio": 0.2,
}


async def transcribe(audio_bytes: bytes, config: Dict[str, Any], mime: str = "audio/ogg") -> Optional[str]:
    """Transcribe *audio_bytes* to text.

    Uses the ``transcription`` section of *config*.  Returns the transcribed
    text, or ``None`` if transcription is not configured or fails.
    """
    cfg = config.get("transcription", {})
    if not cfg:
        logger.warning("transcription: no config — skipping")
        return None

    provider = cfg.get("provider", "groq")
    provider_cfg = cfg.get(provider, {})

    try:
        if provider == "local":
            if provider_cfg.get("base_url"):
                base_url = provider_cfg.get("base_url", "").rstrip("/")
                model = provider_cfg.get("model", _DEFAULT_MODEL)
                logger.info(
                    "transcription: sending to %s/audio/transcriptions (provider=%s model=%s)",
                    base_url,
                    provider,
                    model,
                )
                return await _transcribe_api(audio_bytes, provider, provider_cfg, mime)
            logger.debug("transcription: using local faster-whisper (model=%s)", provider_cfg.get("model", "base"))
            return await _transcribe_local(audio_bytes, provider_cfg, mime)
        base_url = provider_cfg.get("base_url", _PROVIDER_BASE_URLS.get(provider, "<no base_url>")).rstrip("/")
        model = provider_cfg.get("model", _DEFAULT_MODEL)
        logger.info("transcription: sending to %s/audio/transcriptions (provider=%s model=%s)", base_url, provider, model)
        return await _transcribe_api(audio_bytes, provider, provider_cfg, mime)
    except Exception as e:
        logger.error("transcription: error (provider=%s): %s", provider, e, exc_info=True)
        return None


def _bandpass_pcm(pcm: "np.ndarray", sample_rate: int, low_hz: float = 80.0, high_hz: float = 8000.0) -> "np.ndarray":
    """FFT-based bandpass filter — removes wind/rumble (<80 Hz) and high-freq hiss (>8 kHz).

    Pure numpy, no extra dependencies.
    """
    import numpy as np

    spectrum = np.fft.rfft(pcm)
    freqs = np.fft.rfftfreq(len(pcm), d=1.0 / sample_rate)
    spectrum[(freqs < low_hz) | (freqs > high_hz)] = 0.0
    filtered = np.fft.irfft(spectrum, n=len(pcm))
    return filtered.astype(np.float32)


def _spectral_denoise_pcm(
    pcm: "np.ndarray",
    sample_rate: int,
    strength: float = 1.25,
    floor_ratio: float = 0.2,
) -> "np.ndarray":
    """Reduce stationary background noise via simple spectral subtraction.

    This is intentionally lightweight and dependency-free. It works well for
    constant hiss, fan noise and parts of broadband wind noise, though it is
    not a full speech enhancement model.
    """
    import numpy as np

    if len(pcm) < max(512, sample_rate // 20):
        return pcm.astype(np.float32, copy=False)

    frame_size = 1024
    hop_size = frame_size // 2
    if len(pcm) < frame_size:
        frame_size = 1 << max(5, int(np.floor(np.log2(len(pcm)))))
        hop_size = max(frame_size // 2, 1)
    window = np.hanning(frame_size).astype(np.float32)
    padded_len = frame_size + hop_size * int(np.ceil(max(len(pcm) - frame_size, 0) / hop_size))
    padded = np.pad(pcm.astype(np.float32), (0, max(0, padded_len - len(pcm))))

    frames = []
    energies = []
    for start in range(0, len(padded) - frame_size + 1, hop_size):
        frame = padded[start : start + frame_size]
        windowed = frame * window
        frames.append(windowed)
        energies.append(float(np.mean(windowed ** 2)))

    if not frames:
        return pcm.astype(np.float32, copy=False)

    spectra = np.fft.rfft(np.stack(frames, axis=0), axis=1)
    magnitudes = np.abs(spectra)
    phases = np.angle(spectra)

    noise_frame_count = max(1, min(len(energies) // 5, 12))
    quiet_idx = np.argsort(np.asarray(energies))[:noise_frame_count]
    noise_profile = np.median(magnitudes[quiet_idx], axis=0)

    cleaned_mag = magnitudes - (strength * noise_profile[None, :])
    cleaned_mag = np.maximum(cleaned_mag, noise_profile[None, :] * floor_ratio)
    cleaned = cleaned_mag * np.exp(1j * phases)
    restored_frames = np.fft.irfft(cleaned, n=frame_size, axis=1).astype(np.float32)

    out = np.zeros(len(padded), dtype=np.float32)
    norm = np.zeros(len(padded), dtype=np.float32)
    for idx, start in enumerate(range(0, len(padded) - frame_size + 1, hop_size)):
        out[start : start + frame_size] += restored_frames[idx] * window
        norm[start : start + frame_size] += window ** 2

    norm = np.maximum(norm, 1e-6)
    out = out / norm
    return out[: len(pcm)].astype(np.float32)


def _noise_gate_pcm(
    pcm: "np.ndarray",
    threshold: float = 0.015,
    attenuation_ratio: float = 0.2,
) -> "np.ndarray":
    """Apply a soft gate to reduce residual low-level noise between syllables."""
    import numpy as np

    abs_pcm = np.abs(pcm)
    gate = np.where(abs_pcm >= threshold, 1.0, attenuation_ratio + (1.0 - attenuation_ratio) * (abs_pcm / max(threshold, 1e-6)))
    return (pcm * gate.astype(np.float32)).astype(np.float32)


def _adaptive_gate_pcm(
    pcm: "np.ndarray",
    sample_rate: int,
    base_threshold: float = 0.015,
    attenuation_ratio: float = 0.2,
    noise_percentile: float = 0.2,
    noise_multiplier: float = 2.2,
) -> "np.ndarray":
    """Apply a frame-wise soft gate based on the measured noise floor."""
    import numpy as np

    if len(pcm) < max(sample_rate // 20, 64):
        return _noise_gate_pcm(pcm, threshold=base_threshold, attenuation_ratio=attenuation_ratio)

    frame_size = max(sample_rate // 50, 64)  # ~20 ms
    hop_size = max(frame_size // 2, 1)
    window = np.hanning(frame_size).astype(np.float32)
    padded_len = frame_size + hop_size * int(np.ceil(max(len(pcm) - frame_size, 0) / hop_size))
    padded = np.pad(pcm.astype(np.float32), (0, max(0, padded_len - len(pcm))))

    frame_rms = []
    for start in range(0, len(padded) - frame_size + 1, hop_size):
        frame = padded[start : start + frame_size] * window
        frame_rms.append(float(np.sqrt(np.mean(frame ** 2))))

    if not frame_rms:
        return pcm.astype(np.float32, copy=False)

    noise_floor = float(np.quantile(np.asarray(frame_rms, dtype=np.float32), min(max(noise_percentile, 0.0), 1.0)))
    threshold = max(base_threshold, noise_floor * max(noise_multiplier, 1.0))

    gains = []
    for rms in frame_rms:
        if rms >= threshold:
            gains.append(1.0)
            continue
        rel = rms / max(threshold, 1e-6)
        gains.append(attenuation_ratio + (1.0 - attenuation_ratio) * np.sqrt(rel))

    gain_envelope = np.zeros(len(padded), dtype=np.float32)
    norm = np.zeros(len(padded), dtype=np.float32)
    for idx, start in enumerate(range(0, len(padded) - frame_size + 1, hop_size)):
        gain_envelope[start : start + frame_size] += gains[idx]
        norm[start : start + frame_size] += 1.0

    gain_envelope = gain_envelope / np.maximum(norm, 1e-6)
    return (padded[: len(pcm)] * gain_envelope[: len(pcm)]).astype(np.float32)


def _preprocess_pcm_for_stt(
    pcm: "np.ndarray",
    sample_rate: int,
    config: Optional[Dict[str, Any]] = None,
) -> "np.ndarray":
    """Preprocess PCM for more robust speech transcription in noisy environments."""
    import numpy as np

    preprocess_cfg = dict(_DEFAULT_PREPROCESS)
    if isinstance(config, dict):
        transcription_cfg = config.get("transcription", {})
        if isinstance(transcription_cfg, dict):
            user_cfg = transcription_cfg.get("preprocess", {})
            if isinstance(user_cfg, dict):
                for key, default in _DEFAULT_PREPROCESS.items():
                    value = user_cfg.get(key, default)
                    try:
                        preprocess_cfg[key] = float(value)
                    except (TypeError, ValueError):
                        logger.warning("transcription: ignoring invalid preprocess.%s=%r", key, value)

    pcm = _bandpass_pcm(
        pcm,
        sample_rate,
        low_hz=preprocess_cfg["highpass_hz"],
        high_hz=preprocess_cfg["lowpass_hz"],
    )
    pcm = _spectral_denoise_pcm(
        pcm,
        sample_rate,
        strength=preprocess_cfg["denoise_strength"],
        floor_ratio=preprocess_cfg["denoise_floor"],
    )
    pcm = _adaptive_gate_pcm(
        pcm,
        sample_rate,
        base_threshold=preprocess_cfg["gate_threshold"],
        attenuation_ratio=preprocess_cfg["gate_ratio"],
        noise_percentile=preprocess_cfg["adaptive_gate_percentile"],
        noise_multiplier=preprocess_cfg["adaptive_gate_multiplier"],
    )
    pcm = _noise_gate_pcm(
        pcm,
        threshold=preprocess_cfg["gate_threshold"],
        attenuation_ratio=preprocess_cfg["gate_ratio"],
    )

    peak = float(np.max(np.abs(pcm)))
    if peak > 1e-6:
        pcm = pcm * (0.9 / peak)
    return np.clip(pcm, -1.0, 1.0).astype(np.float32)


async def transcribe_pcm(
    pcm_float32: "np.ndarray",
    sample_rate: int,
    config: Dict[str, Any],
) -> Optional[str]:
    """Transcribe raw float32 mono PCM to text.

    Wraps the data in a WAV container and delegates to :func:`transcribe`.
    """
    import io
    import wave

    import numpy as np

    pcm_float32 = _preprocess_pcm_for_stt(pcm_float32, sample_rate, config)
    pcm_int16 = (np.clip(pcm_float32, -1.0, 1.0) * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_int16.tobytes())

    return await transcribe(buf.getvalue(), config, mime="audio/wav")


async def transcribe_via_model(
    audio_bytes: bytes,
    ollama_base: str,
    model: str,
    mime: str = "audio/wav",
    prompt: Optional[str] = None,
) -> Optional[str]:
    """Transcribe audio using a model with native audio support (e.g. Gemma4).

    Calls the Ollama ``/api/chat`` endpoint directly — the ``images`` field
    accepts any binary attachment including audio.

    *ollama_base* is the Ollama base URL **without** ``/v1`` (e.g.
    ``http://localhost:11434``).
    """
    import base64
    import httpx

    send_bytes, send_mime = _ensure_model_audio_format(audio_bytes, mime)

    audio_b64 = base64.b64encode(send_bytes).decode()
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt or _NATIVE_AUDIO_PROMPT,
                "images": [audio_b64],
            }
        ],
        "stream": False,
    }
    url = f"{ollama_base.rstrip('/')}/api/chat"
    logger.info(
        "transcribe_via_model: sending audio (%d bytes, mime=%s, input_mime=%s) to %s model=%s",
        len(send_bytes),
        send_mime,
        mime,
        url,
        model,
    )

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            text = data.get("message", {}).get("content", "").strip()
            logger.info("transcribe_via_model: result: %s", text[:120] if text else "(empty)")
            return text or None
    except Exception as e:
        logger.error("transcribe_via_model: error: %r", e, exc_info=True)
        return None


def _ensure_model_audio_format(audio_bytes: bytes, mime: str) -> Tuple[bytes, str]:
    """Convert audio to 16 kHz mono WAV for model-native transcription.

    Ollama native-audio models require 16 kHz mono WAV with a RIFF header.
    All input formats (including WAV at other sample rates) are converted
    via ffmpeg.  If conversion fails, return original bytes.
    """
    # Check if already 16 kHz mono WAV — skip conversion
    if mime in {"audio/wav", "audio/x-wav"} and _is_16khz_mono_wav(audio_bytes):
        return audio_bytes, "audio/wav"

    in_ext = _mime_to_ext(mime)
    if mime in {"audio/wav", "audio/x-wav"}:
        in_ext = "wav"

    tmp_in = tempfile.NamedTemporaryFile(suffix=f".{in_ext}", delete=False)
    tmp_out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_in_path = tmp_in.name
    tmp_out_path = tmp_out.name
    try:
        tmp_in.write(audio_bytes)
        tmp_in.close()
        tmp_out.close()

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            tmp_in_path,
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "wav",
            tmp_out_path,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            logger.warning(
                "transcribe_via_model: ffmpeg conversion failed for mime=%s (exit=%s): %s",
                mime,
                proc.returncode,
                (proc.stderr or "").strip()[:240],
            )
            return audio_bytes, mime

        with open(tmp_out_path, "rb") as f:
            wav_bytes = f.read()
        if not wav_bytes:
            logger.warning("transcribe_via_model: ffmpeg conversion produced empty WAV for mime=%s", mime)
            return audio_bytes, mime
        return wav_bytes, "audio/wav"
    except FileNotFoundError:
        logger.warning("transcribe_via_model: ffmpeg not found, sending original mime=%s", mime)
        return audio_bytes, mime
    except Exception as e:
        logger.warning("transcribe_via_model: audio conversion error for mime=%s: %r", mime, e)
        return audio_bytes, mime
    finally:
        for path in (tmp_in_path, tmp_out_path):
            try:
                os.unlink(path)
            except OSError:
                pass


_MODEL_AUDIO_RATE = 16000  # Ollama native-audio models expect 16 kHz mono WAV


async def transcribe_pcm_via_model(
    pcm_float32: "np.ndarray",
    sample_rate: int,
    ollama_base: str,
    model: str,
    prompt: Optional[str] = None,
) -> Optional[str]:
    """Transcribe raw float32 PCM via a native-audio model.

    Resamples to 16 kHz mono (required by Ollama native-audio models),
    wraps in WAV, and delegates to :func:`transcribe_via_model`.
    """
    import io
    import wave

    import numpy as np

    pcm_float32 = _preprocess_pcm_for_stt(pcm_float32, sample_rate)

    # Resample to 16 kHz if needed — Ollama audio models require it
    if sample_rate != _MODEL_AUDIO_RATE:
        num_samples = int(len(pcm_float32) * _MODEL_AUDIO_RATE / sample_rate)
        pcm_float32 = np.interp(
            np.linspace(0, len(pcm_float32), num_samples, endpoint=False),
            np.arange(len(pcm_float32)),
            pcm_float32,
        ).astype(np.float32)
        sample_rate = _MODEL_AUDIO_RATE

    pcm_int16 = (np.clip(pcm_float32, -1.0, 1.0) * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_int16.tobytes())

    return await transcribe_via_model(buf.getvalue(), ollama_base, model, mime="audio/wav", prompt=prompt)


# ---------------------------------------------------------------------------
# Implementations
# ---------------------------------------------------------------------------

async def _transcribe_api(audio_bytes: bytes, provider: str, cfg: Dict, mime: str) -> Optional[str]:
    """POST to any OpenAI-compatible /audio/transcriptions endpoint."""
    import httpx

    api_key  = cfg.get("api_key", "")
    base_url = cfg.get("base_url", _PROVIDER_BASE_URLS.get(provider, "")).rstrip("/")
    model    = cfg.get("model", _DEFAULT_MODEL)
    language = cfg.get("language")
    ext      = _mime_to_ext(mime)

    if not base_url:
        raise ValueError(f"transcription: no base_url for provider '{provider}'")

    data: Dict[str, Any] = {"model": model, "temperature": "0"}
    if language:
        data["language"] = language

    url = f"{base_url}/audio/transcriptions"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(
                url,
                headers=headers,
                files={"file": (f"audio.{ext}", audio_bytes, mime)},
                data=data,
                timeout=60,
            )
        except httpx.ConnectError as e:
            raise ConnectionError(f"STT: could not connect to {url} — {e}") from e
        except httpx.TimeoutException as e:
            raise TimeoutError(f"STT: request to {url} timed out — {e}") from e
        if resp.status_code >= 400:
            logger.error("transcription: HTTP %d from %s — %s", resp.status_code, url, resp.text[:300])
        resp.raise_for_status()
        return resp.json().get("text", "").strip() or None


async def _transcribe_local(audio_bytes: bytes, cfg: Dict, mime: str) -> Optional[str]:
    """Transcribe using faster-whisper locally (runs in thread pool)."""
    model_size   = cfg.get("model", "base")
    device       = cfg.get("device", "cpu")
    compute_type = cfg.get("compute_type", "int8")
    language     = cfg.get("language")
    ext          = _mime_to_ext(mime)

    def _run() -> Optional[str]:
        try:
            from faster_whisper import WhisperModel  # type: ignore
        except ImportError:
            raise RuntimeError("faster-whisper not installed — run: pip install faster-whisper")

        with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as f:
            f.write(audio_bytes)
            tmp_path = f.name

        try:
            model = WhisperModel(model_size, device=device, compute_type=compute_type)
            kw: Dict[str, Any] = {"temperature": 0, "no_speech_threshold": 0.6}
            if language:
                kw["language"] = language
            segments, _ = model.transcribe(tmp_path, **kw)
            return " ".join(s.text for s in segments).strip() or None
        finally:
            os.unlink(tmp_path)

    from pawlia.utils import run_sync_in_thread
    return await run_sync_in_thread(_run)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_16khz_mono_wav(data: bytes) -> bool:
    """Return True if *data* is a WAV file at 16 kHz, mono, 16-bit."""
    import struct

    if len(data) < 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return False
    try:
        channels, sample_rate, _, _, bits_per_sample = struct.unpack_from("<HHIIH", data, 22)
        return channels == 1 and sample_rate == 16000 and bits_per_sample == 16
    except struct.error:
        return False


def _mime_to_ext(mime: str) -> str:
    return {
        "audio/ogg":  "ogg",
        "audio/opus": "ogg",
        "audio/mpeg": "mp3",
        "audio/mp4":  "m4a",
        "audio/wav":  "wav",
        "audio/x-wav":"wav",
        "audio/webm": "webm",
    }.get(mime, "ogg")
