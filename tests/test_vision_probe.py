"""Tests for runtime vision-capability detection (pawlia/vision_probe.py)."""

import struct
import zlib

import pytest

from pawlia import vision_probe as vp


# --------------------------------------------------------------------------
# Probe image generation
# --------------------------------------------------------------------------

def _decode_ihdr(png: bytes):
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    # First chunk after signature must be IHDR
    length = struct.unpack(">I", png[8:12])[0]
    assert png[12:16] == b"IHDR"
    w, h, depth, color_type = struct.unpack(">IIBB", png[16:16 + 10])
    return w, h, depth, color_type, length


def test_png_signature_and_header():
    png = vp._png_bands([(255, 0, 0), (0, 255, 0), (0, 0, 255)], width=48, height=48)
    w, h, depth, color_type, _ = _decode_ihdr(png)
    assert (w, h) == (48, 48)
    assert depth == 8
    assert color_type == 2  # RGB


def test_png_idat_decompresses_to_expected_size():
    w, h = 48, 48
    png = vp._png_bands([(255, 0, 0), (0, 255, 0), (0, 0, 255)], width=w, height=h)
    # Locate IDAT and inflate it; raw stream is h rows of (1 filter byte + w*3).
    idx = png.index(b"IDAT")
    length = struct.unpack(">I", png[idx - 4:idx])[0]
    idat = png[idx + 4: idx + 4 + length]
    raw = zlib.decompress(idat)
    assert len(raw) == h * (1 + w * 3)
    # Top row red, bottom row blue (filter byte at row start).
    assert raw[1:4] == bytes((255, 0, 0))
    assert raw[-(w * 3):][-3:] == bytes((0, 0, 255))


def test_probe_data_uri_is_png():
    uri = vp._probe_data_uri()
    assert uri.startswith("data:image/png;base64,")


# --------------------------------------------------------------------------
# Answer verification
# --------------------------------------------------------------------------

@pytest.mark.parametrize("answer,expected", [
    ("red, green, blue", True),
    ("Red Green Blue", True),
    ("The bands are red, then green, then blue.", True),
    ("blue, green, red", False),       # wrong order
    ("red and blue", False),           # missing green
    ("I cannot see the image", False),
    ("", False),
])
def test_verify_probe_answer(answer, expected):
    assert vp._verify_probe_answer(answer) is expected


def test_heuristic_name_hints():
    assert vp.heuristic_supports_images("qwen2.5vl:latest") is True
    assert vp.heuristic_supports_images("llava:13b") is True
    assert vp.heuristic_supports_images("gpt-4o") is True
    assert vp.heuristic_supports_images("qwen3:4b") is False
    assert vp.heuristic_supports_images("llama3.1:8b") is False


# --------------------------------------------------------------------------
# Cache roundtrip
# --------------------------------------------------------------------------

def test_cache_roundtrip(tmp_path):
    sd = str(tmp_path)
    assert vp.get_cached(sd, "ollama", "qwen3:4b") is None
    vp.set_cached(sd, "ollama", "qwen3:4b", False)
    vp.set_cached(sd, "ollama", "qwen2.5vl:latest", True)
    assert vp.get_cached(sd, "ollama", "qwen3:4b") is False
    assert vp.get_cached(sd, "ollama", "qwen2.5vl:latest") is True
    # Survives a reload (separate call reads the file fresh).
    assert vp.get_cached(sd, "ollama", "qwen2.5vl:latest") is True


def test_cache_handles_corrupt_file(tmp_path):
    path = vp.cache_path(str(tmp_path))
    with open(path, "w") as f:
        f.write("{ not json")
    assert vp.get_cached(str(tmp_path), "ollama", "x") is None


# --------------------------------------------------------------------------
# Probe + resolution (mocked LLM / factory)
# --------------------------------------------------------------------------

class _FakeResp:
    def __init__(self, content):
        self.content = content


class _FakeLLM:
    def __init__(self, reply=None, raises=None):
        self._reply = reply
        self._raises = raises
        self.calls = 0

    async def ainvoke(self, messages, **kwargs):
        self.calls += 1
        if self._raises:
            raise self._raises
        return _FakeResp(self._reply)


async def test_probe_model_pass():
    llm = _FakeLLM(reply="red, green, blue")
    assert await vp.probe_model(llm) is True


async def test_probe_model_fail_wrong_answer():
    llm = _FakeLLM(reply="sorry, I only process text")
    assert await vp.probe_model(llm) is False


async def test_probe_model_error_is_no_vision():
    llm = _FakeLLM(raises=RuntimeError("400: image input not supported"))
    assert await vp.probe_model(llm) is False


async def test_probe_handles_list_content():
    llm = _FakeLLM(reply=[{"text": "red"}, {"text": " green "}, {"text": "blue"}])
    assert await vp.probe_model(llm) is True


class _FakeFactory:
    def __init__(self, cfg=None, llm=None, provider="ollama", build_error=False):
        self._cfg = cfg or {}
        self._llm = llm
        self._provider = provider
        self._build_error = build_error

    def get_model_config(self, name):
        return {"model": name, **self._cfg}

    def get_provider_name_for_model(self, name):
        return self._provider

    def get_with_model(self, name):
        if self._build_error:
            raise RuntimeError("non-pawlia backend")
        return self._llm


async def test_resolve_explicit_flag_skips_probe(tmp_path):
    llm = _FakeLLM(reply="should not be called")
    factory = _FakeFactory(cfg={"supports_images": False}, llm=llm)
    assert await vp.resolve_supports_images(factory, str(tmp_path), "whatever") is False
    assert llm.calls == 0  # flag is authoritative, no probe


async def test_resolve_uses_cache(tmp_path):
    vp.set_cached(str(tmp_path), "ollama", "cached-model", True)
    llm = _FakeLLM(reply="should not be called")
    factory = _FakeFactory(llm=llm)
    assert await vp.resolve_supports_images(factory, str(tmp_path), "cached-model") is True
    assert llm.calls == 0


async def test_resolve_probes_and_caches(tmp_path):
    llm = _FakeLLM(reply="red, green, blue")
    factory = _FakeFactory(llm=llm)
    assert await vp.resolve_supports_images(factory, str(tmp_path), "probe-me") is True
    assert llm.calls == 1
    # Cached now → second call does not probe again.
    assert await vp.resolve_supports_images(factory, str(tmp_path), "probe-me") is True
    assert llm.calls == 1


async def test_resolve_heuristic_when_build_fails(tmp_path):
    factory = _FakeFactory(build_error=True)
    # Name hints vision → heuristic True, and nothing cached.
    assert await vp.resolve_supports_images(factory, str(tmp_path), "qwen2.5vl:latest") is True
    assert vp.get_cached(str(tmp_path), "ollama", "qwen2.5vl:latest") is None


async def test_describe_image(tmp_path):
    llm = _FakeLLM(reply="A red square on white background.")
    out = await vp.describe_image(llm, "data:image/png;base64,AAAA")
    assert out == "A red square on white background."


async def test_describe_image_error_returns_none():
    llm = _FakeLLM(raises=RuntimeError("boom"))
    assert await vp.describe_image(llm, "data:image/png;base64,AAAA") is None
