"""Runtime vision-capability detection for chat models.

Not every configured model accepts image input, and the config does not
always say so. Rather than trust the config blindly, we *probe*: send a tiny
generated image with known content and check whether the model can actually
describe it. A model that silently ignores the image (common with local
text-only models) fails the check just like one that raises an error.

Results are cached in ``<session_dir>/model_capabilities.json`` (keyed by
``provider/model``) so each model is probed at most once.

Resolution order (see :func:`resolve_supports_images`):

1. explicit ``supports_images`` flag in the model config → authoritative
2. cached probe result
3. live probe (verified), result cached
4. name heuristic (only when the probe could not run; not cached)
"""

import asyncio
import base64
import json
import logging
import os
import struct
import time
import zlib
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage


logger = logging.getLogger(__name__)

CACHE_FILENAME = "model_capabilities.json"
_PROBE_TIMEOUT_S = 30.0

# Three horizontal bands, top→bottom. Primary RGB colors are named
# unambiguously across models, and requiring all three *in order* makes a
# lucky hallucinated guess from a blind model very unlikely.
_PROBE_BANDS = ("red", "green", "blue")
_PROBE_RGB = {"red": (220, 20, 20), "green": (20, 200, 20), "blue": (20, 20, 220)}

_PROBE_PROMPT = (
    "This image shows three horizontal color bands stacked vertically. "
    "Name the colors from top to bottom, separated by commas. "
    "Reply with only the color words, nothing else."
)

# Substrings that strongly imply image support when probing is impossible.
_VISION_NAME_HINTS = (
    "vl", "vision", "llava", "bakllava", "moondream", "minicpm-v",
    "gpt-4o", "gpt-4-turbo", "gpt-4.1", "gpt-5", "o3", "o4",
    "claude-3", "claude-4", "claude-opus", "claude-sonnet", "claude-haiku",
    "gemini", "pixtral", "qwen2-vl", "qwen2.5-vl", "qwen2.5vl",
    "internvl", "phi-3-vision", "phi-4-multimodal", "idefics", "smolvlm",
)


# ---------------------------------------------------------------------------
# Probe image generation (pure Python, no Pillow dependency)
# ---------------------------------------------------------------------------

def _png_bands(bands: List[tuple], width: int = 48, height: int = 48) -> bytes:
    """Encode a PNG of equal-height horizontal color bands (RGB, 8-bit)."""
    n = len(bands)
    rows = bytearray()
    for y in range(height):
        rgb = bands[min(y * n // height, n - 1)]
        rows.append(0)  # PNG filter type 0 (None) for this scanline
        rows.extend(bytes(rgb) * width)
    compressed = zlib.compress(bytes(rows), 9)

    def _chunk(typ: bytes, data: bytes) -> bytes:
        body = typ + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # color type 2 = RGB
    return sig + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", compressed) + _chunk(b"IEND", b"")


def _probe_data_uri() -> str:
    png = _png_bands([_PROBE_RGB[c] for c in _PROBE_BANDS])
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


# ---------------------------------------------------------------------------
# Capability cache
# ---------------------------------------------------------------------------

def cache_path(session_dir: str) -> str:
    return os.path.join(session_dir, CACHE_FILENAME)


def _cache_key(provider: str, model: str) -> str:
    return f"{provider or '?'}/{model or '?'}"


def _load_cache(session_dir: str) -> Dict[str, Any]:
    path = cache_path(session_dir)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def get_cached(session_dir: str, provider: str, model: str) -> Optional[bool]:
    entry = _load_cache(session_dir).get(_cache_key(provider, model))
    if isinstance(entry, dict) and isinstance(entry.get("supports_images"), bool):
        return entry["supports_images"]
    return None


def set_cached(session_dir: str, provider: str, model: str, supports: bool) -> None:
    path = cache_path(session_dir)
    cache = _load_cache(session_dir)
    cache[_cache_key(provider, model)] = {
        "supports_images": bool(supports),
        "probed_at": time.time(),
    }
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except OSError as exc:
        logger.debug("vision_probe: could not persist cache: %s", exc)


# ---------------------------------------------------------------------------
# Probing
# ---------------------------------------------------------------------------

def heuristic_supports_images(model_id: str) -> bool:
    """Best-effort guess from the model identifier when a probe is impossible."""
    name = (model_id or "").lower()
    return any(hint in name for hint in _VISION_NAME_HINTS)


def _verify_probe_answer(text: str) -> bool:
    """True iff *text* names the three probe colors in top→bottom order."""
    low = (text or "").lower()
    indices = [low.find(c) for c in _PROBE_BANDS]
    if any(i < 0 for i in indices):
        return False
    return indices[0] < indices[1] < indices[2]


async def probe_model(llm: Any) -> Optional[bool]:
    """Send the probe image to *llm* and verify the answer.

    Returns ``True``/``False`` on a conclusive probe, or ``None`` if the
    probe could not be carried out (timeout / unexpected error) so the caller
    can fall back to a heuristic without caching a wrong negative.
    """
    content = [
        {"type": "text", "text": _PROBE_PROMPT},
        {"type": "image_url", "image_url": {"url": _probe_data_uri()}},
    ]
    try:
        resp = await asyncio.wait_for(
            llm.ainvoke([HumanMessage(content=content)]), timeout=_PROBE_TIMEOUT_S
        )
    except asyncio.TimeoutError:
        logger.warning("vision_probe: probe timed out after %.0fs", _PROBE_TIMEOUT_S)
        return None
    except Exception as exc:
        # A model that rejects image content (HTTP 400, etc.) is conclusively
        # non-vision; we can't distinguish that from a transient error here, so
        # treat any failure as "no vision" — the common case for text models.
        logger.info("vision_probe: probe call failed (treating as no-vision): %s", exc)
        return False

    text = getattr(resp, "content", "") or ""
    if isinstance(text, list):  # some providers return content as a parts list
        text = " ".join(str(p.get("text", "") if isinstance(p, dict) else p) for p in text)
    ok = _verify_probe_answer(str(text))
    logger.info("vision_probe: probe %s (answer: %r)", "passed" if ok else "failed", str(text)[:120])
    return ok


async def resolve_supports_images(
    factory: Any,
    session_dir: str,
    model_name: str,
) -> bool:
    """Resolve whether the model behind *model_name* accepts image input.

    Order: explicit config flag → cache → live verified probe → name heuristic.
    """
    try:
        cfg = factory.get_model_config(model_name)
    except Exception:
        cfg = {}
    model_id = str(cfg.get("model") or model_name)

    flag = cfg.get("supports_images")
    if isinstance(flag, bool):
        return flag

    try:
        provider = factory.get_provider_name_for_model(model_name)
    except Exception:
        provider = "?"

    cached = get_cached(session_dir, provider, model_id)
    if cached is not None:
        return cached

    # Build a single-model LLM and probe it. Non-pawlia backends (e.g. Hermes)
    # can't be built here — fall back to the name heuristic without caching.
    try:
        llm = factory.get_with_model(model_name)
    except Exception as exc:
        logger.debug("vision_probe: cannot build '%s' for probing (%s); using heuristic", model_name, exc)
        return heuristic_supports_images(model_id)

    result = await probe_model(llm)
    if result is None:  # inconclusive — don't poison the cache
        return heuristic_supports_images(model_id)

    set_cached(session_dir, provider, model_id, result)
    return result


# ---------------------------------------------------------------------------
# Image description (vision-blind fallback)
# ---------------------------------------------------------------------------

_DESCRIBE_PROMPT = (
    "Describe this image in detail for someone who cannot see it. "
    "Transcribe any visible text verbatim, and note layout, people, objects, "
    "colors and overall context. Be thorough but concise."
)


async def describe_image(llm: Any, data_uri: str, prompt: Optional[str] = None) -> Optional[str]:
    """Ask a vision-capable *llm* to describe one image. Returns text or None."""
    content = [
        {"type": "text", "text": prompt or _DESCRIBE_PROMPT},
        {"type": "image_url", "image_url": {"url": data_uri}},
    ]
    try:
        resp = await asyncio.wait_for(
            llm.ainvoke([HumanMessage(content=content)]), timeout=_PROBE_TIMEOUT_S * 2
        )
    except Exception as exc:
        logger.warning("vision_probe: describe failed: %s", exc)
        return None
    text = getattr(resp, "content", "") or ""
    if isinstance(text, list):
        text = " ".join(str(p.get("text", "") if isinstance(p, dict) else p) for p in text)
    text = str(text).strip()
    return text or None
