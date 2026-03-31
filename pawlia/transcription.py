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

    # Local (faster-whisper, requires FFmpeg):
    # transcription:
    #   provider: local
    #   local:
    #     model: base
    #     device: cpu
    #     compute_type: int8
    #     # language: de
"""

import asyncio
import logging
import os
import tempfile
from typing import Any, Dict, Optional

logger = logging.getLogger("pawlia.transcription")

# Default base URLs per known provider name
_PROVIDER_BASE_URLS: Dict[str, str] = {
    "groq":  "https://api.groq.com/openai/v1",
    "openai": "https://api.openai.com/v1",
}
_DEFAULT_MODEL = "whisper-large-v3-turbo"


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

    pcm_float32 = _bandpass_pcm(pcm_float32, sample_rate)
    pcm_int16 = (np.clip(pcm_float32, -1.0, 1.0) * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_int16.tobytes())

    return await transcribe(buf.getvalue(), config, mime="audio/wav")


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
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(
                url,
                headers={"Authorization": f"Bearer {api_key}"},
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

    return await asyncio.to_thread(_run)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
