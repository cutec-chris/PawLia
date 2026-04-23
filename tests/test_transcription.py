import numpy as np

from pawlia.transcription import _noise_gate_pcm, _preprocess_pcm_for_stt


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
