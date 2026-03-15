"""Text-to-speech for PawLia.

Config layout (YAML)::

    tts:
      provider: edge           # edge | piper
      edge:
        voice: de-DE-KatjaNeural   # any edge-tts voice name
      piper:
        executable: piper          # path to piper binary
        model: de_DE-thorsten-medium.onnx
        config: de_DE-thorsten-medium.onnx.json
"""

import asyncio
import io
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger("pawlia.tts")


async def synthesize(text: str, config: Dict[str, Any]) -> Optional[bytes]:
    """Synthesize *text* to audio bytes using the configured provider.

    Returns MP3 bytes (edge) or raw PCM bytes (piper), or ``None`` if TTS
    is not configured.
    """
    cfg = config.get("tts", {})
    if not cfg:
        return None

    provider = cfg.get("provider", "edge")
    try:
        if provider == "edge":
            return await _synthesize_edge(text, cfg.get("edge", {}))
        elif provider == "piper":
            return await _synthesize_piper(text, cfg.get("piper", {}))
        else:
            logger.error("tts: unknown provider '%s'", provider)
            return None
    except Exception as e:
        logger.error("tts: synthesis failed (%s): %s", provider, e)
        return None


async def synthesize_pcm(
    text: str,
    config: Dict[str, Any],
    sample_rate: int = 48000,
) -> Optional["np.ndarray"]:
    """Synthesize *text* and return float32 mono PCM at *sample_rate* Hz.

    Decodes the provider output (MP3/WAV/raw PCM) via PyAV and resamples.
    Returns a numpy float32 array, or ``None`` if TTS is not configured.
    """
    import numpy as np

    audio_bytes = await synthesize(text, config)
    if audio_bytes is None:
        return None

    return _decode_to_pcm(audio_bytes, config, sample_rate)


def _decode_to_pcm(audio_bytes: bytes, config: Dict[str, Any], target_rate: int) -> "np.ndarray":
    """Decode audio bytes to float32 mono PCM at *target_rate* Hz via PyAV."""
    import av  # type: ignore
    import numpy as np

    cfg = config.get("tts", {})
    provider = cfg.get("provider", "edge")

    if provider == "piper":
        # piper returns raw s16le PCM — wrap in WAV header for av
        piper_cfg = cfg.get("piper", {})
        src_rate = piper_cfg.get("sample_rate", 22050)
        audio_bytes = _raw_s16_to_wav(audio_bytes, src_rate, channels=1)

    container = av.open(io.BytesIO(audio_bytes))
    resampler = av.AudioResampler(
        format="fltp",
        layout="mono",
        rate=target_rate,
    )

    chunks = []
    for frame in container.decode(audio=0):
        for out_frame in resampler.resample(frame):
            arr = out_frame.to_ndarray()  # shape (1, samples) float32
            chunks.append(arr[0])

    # Flush resampler
    for out_frame in resampler.resample(None):
        arr = out_frame.to_ndarray()
        chunks.append(arr[0])

    if not chunks:
        return np.zeros(0, dtype=np.float32)

    return np.concatenate(chunks).astype(np.float32)


def _raw_s16_to_wav(pcm: bytes, rate: int, channels: int) -> bytes:
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------

async def _synthesize_edge(text: str, cfg: Dict) -> bytes:
    """Synthesize using edge-tts (Microsoft Edge TTS, requires internet)."""
    try:
        import edge_tts  # type: ignore
    except ImportError:
        raise RuntimeError("edge-tts not installed — run: pip install edge-tts")

    voice = cfg.get("voice", "de-DE-KatjaNeural")

    communicate = edge_tts.Communicate(text, voice)
    buf = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])

    data = buf.getvalue()
    if not data:
        raise RuntimeError("edge-tts returned empty audio")
    return data


async def _synthesize_piper(text: str, cfg: Dict) -> bytes:
    """Synthesize using piper-tts locally (must be installed separately)."""
    executable = cfg.get("executable", "piper")
    model = cfg.get("model", "")
    model_config = cfg.get("config", "")

    if not model:
        raise RuntimeError("tts.piper.model not configured")

    cmd = [executable, "--model", model, "--output_raw"]
    if model_config:
        cmd += ["--config", model_config]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    stdout, _ = await proc.communicate(input=text.encode("utf-8"))

    if proc.returncode != 0:
        raise RuntimeError(f"piper exited with code {proc.returncode}")

    return stdout
