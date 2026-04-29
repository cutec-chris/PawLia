import numpy as np
import pytest
import httpx

from pawlia.transcription import _adaptive_gate_pcm, _noise_gate_pcm, _preprocess_pcm_for_stt, transcribe


def _tone(freq_hz: float, sample_rate: int, duration_s: float, amp: float) -> np.ndarray:
    t = np.arange(int(sample_rate * duration_s), dtype=np.float32) / sample_rate
    return (amp * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def test_preprocess_reduces_low_frequency_wind_component():
    sample_rate = 16000
    wind = _tone(30.0, sample_rate, 1.0, 0.9)
    voice = _tone(300.0, sample_rate, 1.0, 0.25)
    pcm = wind + voice

    processed = _preprocess_pcm_for_stt(pcm, sample_rate)

    spectrum_before = np.abs(np.fft.rfft(pcm))
    spectrum_after = np.abs(np.fft.rfft(processed))
    freqs = np.fft.rfftfreq(len(pcm), d=1.0 / sample_rate)

    low_band = (freqs >= 20.0) & (freqs <= 60.0)
    voice_band = (freqs >= 250.0) & (freqs <= 350.0)

    low_ratio = spectrum_after[low_band].mean() / max(spectrum_before[low_band].mean(), 1e-6)
    voice_ratio = spectrum_after[voice_band].mean() / max(spectrum_before[voice_band].mean(), 1e-6)

    assert low_ratio < 0.2
    assert voice_ratio > 0.4


def test_noise_gate_attenuates_low_level_residual_noise():
    pcm = np.array([0.001, -0.003, 0.006, -0.009], dtype=np.float32)

    gated = _noise_gate_pcm(pcm, threshold=0.01, attenuation_ratio=0.2)

    assert np.max(np.abs(gated)) < np.max(np.abs(pcm))
    assert np.all(np.signbit(gated) == np.signbit(pcm))


def test_adaptive_gate_reduces_stationary_noise_more_than_speech():
    sample_rate = 16000
    noise = _tone(900.0, sample_rate, 1.0, 0.01)
    speech = _tone(240.0, sample_rate, 1.0, 0.12)
    pcm = noise + speech

    gated = _adaptive_gate_pcm(
        pcm,
        sample_rate,
        base_threshold=0.015,
        attenuation_ratio=0.2,
        noise_percentile=0.2,
        noise_multiplier=2.2,
    )

    speech_only_gated = _adaptive_gate_pcm(
        speech,
        sample_rate,
        base_threshold=0.015,
        attenuation_ratio=0.2,
        noise_percentile=0.2,
        noise_multiplier=2.2,
    )

    residual_noise_before = np.sqrt(np.mean((pcm - speech) ** 2))
    residual_noise_after = np.sqrt(np.mean((gated - speech_only_gated) ** 2))
    speech_rms_before = np.sqrt(np.mean(speech ** 2))
    speech_rms_after = np.sqrt(np.mean(speech_only_gated ** 2))

    assert residual_noise_after < residual_noise_before * 0.8
    assert speech_rms_after > speech_rms_before * 0.7


@pytest.mark.asyncio
async def test_local_with_base_url_uses_openai_compatible_api(monkeypatch):
    called = {}

    async def fake_api(audio_bytes, provider, cfg, mime):
        called["args"] = (audio_bytes, provider, cfg, mime)
        return "Hallo"

    async def fake_local(audio_bytes, cfg, mime):
        raise AssertionError("local faster-whisper should not be used when base_url is set")

    monkeypatch.setattr("pawlia.transcription._transcribe_api", fake_api)
    monkeypatch.setattr("pawlia.transcription._transcribe_local", fake_local)

    cfg = {
        "transcription": {
            "provider": "local",
            "local": {
                "base_url": "http://192.168.177.120:8005",
                "model": "whisper-large-v3-turbo",
                "language": "de",
            },
        }
    }

    assert await transcribe(b"audio", cfg, mime="audio/wav") == "Hallo"
    assert called["args"][1:] == ("local", cfg["transcription"]["local"], "audio/wav")


@pytest.mark.asyncio
async def test_local_without_base_url_uses_faster_whisper(monkeypatch):
    called = {}

    async def fake_api(audio_bytes, provider, cfg, mime):
        raise AssertionError("API should not be used when local base_url is missing")

    async def fake_local(audio_bytes, cfg, mime):
        called["args"] = (audio_bytes, cfg, mime)
        return "Hallo lokal"

    monkeypatch.setattr("pawlia.transcription._transcribe_api", fake_api)
    monkeypatch.setattr("pawlia.transcription._transcribe_local", fake_local)

    cfg = {
        "transcription": {
            "provider": "local",
            "local": {"model": "base", "language": "de"},
        }
    }

    assert await transcribe(b"audio", cfg, mime="audio/wav") == "Hallo lokal"
    assert called["args"] == (b"audio", cfg["transcription"]["local"], "audio/wav")


@pytest.mark.asyncio
async def test_transcription_provider_fallback_tries_next_provider(monkeypatch):
    calls = []

    async def fake_api(audio_bytes, provider, cfg, mime):
        calls.append((provider, cfg, mime))
        if provider == "local":
            raise TimeoutError("local STT timed out")
        return "Hallo fallback"

    async def fake_local(audio_bytes, cfg, mime):
        raise AssertionError("local base_url should use the API path")

    monkeypatch.setattr("pawlia.transcription._transcribe_api", fake_api)
    monkeypatch.setattr("pawlia.transcription._transcribe_local", fake_local)

    cfg = {
        "transcription": {
            "providers": [
                {
                    "name": "lan-whisper",
                    "provider": "local",
                    "base_url": "http://192.168.177.120:8005/v1",
                    "model": "whisper-large-v3-turbo",
                    "timeout": 8,
                },
                "groq",
            ],
            "groq": {
                "api_key": "secret",
                "model": "whisper-large-v3-turbo",
                "language": "de",
            },
        }
    }

    assert await transcribe(b"audio", cfg, mime="audio/wav") == "Hallo fallback"
    assert calls[0][0] == "local"
    assert calls[0][1]["name"] == "lan-whisper"
    assert calls[0][1]["timeout"] == 8
    assert calls[1] == ("groq", cfg["transcription"]["groq"], "audio/wav")


@pytest.mark.asyncio
async def test_transcription_provider_fallback_accepts_comma_separated_string(monkeypatch):
    calls = []

    async def fake_api(audio_bytes, provider, cfg, mime):
        calls.append((provider, cfg, mime))
        if provider == "local":
            raise TimeoutError("local STT timed out")
        return "Hallo fallback"

    async def fake_local(audio_bytes, cfg, mime):
        raise AssertionError("local base_url should use the API path")

    monkeypatch.setattr("pawlia.transcription._transcribe_api", fake_api)
    monkeypatch.setattr("pawlia.transcription._transcribe_local", fake_local)

    cfg = {
        "transcription": {
            "providers": "local, groq",
            "local": {
                "base_url": "http://192.168.177.120:8005/v1",
                "model": "deepdml/faster-whisper-large-v3-turbo-ct2",
                "timeout": 10,
            },
            "groq": {
                "api_key": "secret",
                "model": "whisper-large-v3-turbo",
                "language": "de",
            },
        }
    }

    assert await transcribe(b"audio", cfg, mime="audio/wav") == "Hallo fallback"
    assert calls[0] == ("local", cfg["transcription"]["local"], "audio/wav")
    assert calls[1] == ("groq", cfg["transcription"]["groq"], "audio/wav")


@pytest.mark.asyncio
async def test_transcription_provider_fallback_continues_on_empty_text(monkeypatch):
    calls = []

    async def fake_api(audio_bytes, provider, cfg, mime):
        calls.append(provider)
        return None if provider == "local" else "Hallo danach"

    monkeypatch.setattr("pawlia.transcription._transcribe_api", fake_api)

    cfg = {
        "transcription": {
            "providers": [
                {"provider": "local", "base_url": "http://127.0.0.1:8005/v1"},
                {"provider": "openai", "api_key": "secret", "model": "whisper-1"},
            ],
        }
    }

    assert await transcribe(b"audio", cfg, mime="audio/wav") == "Hallo danach"
    assert calls == ["local", "openai"]


@pytest.mark.asyncio
async def test_transcription_blacklists_provider_after_three_failed_requests(monkeypatch):
    from pawlia import transcription

    transcription._stt_provider_state.clear()
    calls = []

    async def fake_api(audio_bytes, provider, cfg, mime):
        calls.append(provider)
        if provider == "local":
            raise TimeoutError("local STT timed out")
        return "Fallback ok"

    monkeypatch.setattr("pawlia.transcription._transcribe_api", fake_api)

    cfg = {
        "transcription": {
            "providers": [
                {"provider": "local", "base_url": "http://127.0.0.1:8005/v1", "model": "whisper"},
                {"provider": "groq", "api_key": "secret", "model": "whisper-large-v3-turbo"},
            ],
        }
    }

    for _ in range(3):
        assert await transcribe(b"audio", cfg, mime="audio/wav") == "Fallback ok"

    assert calls == ["local", "groq", "local", "groq", "local", "groq"]

    assert await transcribe(b"audio", cfg, mime="audio/wav") == "Fallback ok"
    assert calls == ["local", "groq", "local", "groq", "local", "groq", "groq"]


@pytest.mark.asyncio
async def test_transcription_blacklist_expires_after_cooldown(monkeypatch):
    from pawlia import transcription

    transcription._stt_provider_state.clear()
    now = [100.0]
    calls = []

    async def fake_api(audio_bytes, provider, cfg, mime):
        calls.append(provider)
        if provider == "local":
            raise TimeoutError("local STT timed out")
        return "Fallback ok"

    monkeypatch.setattr("pawlia.transcription._now", lambda: now[0])
    monkeypatch.setattr("pawlia.transcription._transcribe_api", fake_api)

    cfg = {
        "transcription": {
            "providers": [
                {"provider": "local", "base_url": "http://127.0.0.1:8005/v1", "model": "whisper"},
                {"provider": "groq", "api_key": "secret", "model": "whisper-large-v3-turbo"},
            ],
        }
    }

    for _ in range(3):
        assert await transcribe(b"audio", cfg, mime="audio/wav") == "Fallback ok"

    assert await transcribe(b"audio", cfg, mime="audio/wav") == "Fallback ok"
    assert calls[-1] == "groq"
    assert calls.count("local") == 3

    now[0] += 30 * 60 + 1

    assert await transcribe(b"audio", cfg, mime="audio/wav") == "Fallback ok"
    assert calls.count("local") == 4


@pytest.mark.asyncio
async def test_api_omits_authorization_header_without_api_key(monkeypatch):
    from pawlia import transcription
    captured = {}

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"text": "Hallo ohne key"}

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url, headers, files, data, timeout):
            captured["headers"] = headers
            return FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)

    text = await transcription._transcribe_api(
        b"audio",
        "local",
        {"base_url": "http://127.0.0.1:8005", "model": "whisper-large-v3-turbo"},
        "audio/wav",
    )

    assert text == "Hallo ohne key"
    assert captured["headers"] == {}
