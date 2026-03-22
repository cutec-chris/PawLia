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
            return await _transcribe_local(audio_bytes, provider_cfg, mime)
        return await _transcribe_api(audio_bytes, provider, provider_cfg, mime)
    except Exception as e:
        logger.error("transcription: error (%s): %s", provider, e)
        return None


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

    data: Dict[str, Any] = {"model": model}
    if language:
        data["language"] = language

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{base_url}/audio/transcriptions",
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": (f"audio.{ext}", audio_bytes, mime)},
            data=data,
            timeout=60,
        )
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
            kw: Dict[str, Any] = {}
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
