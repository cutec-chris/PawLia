import os
import wave

import numpy as np
import pytest

from pawlia.audio.recorder import CallRecorder, rotate_recordings


@pytest.fixture
def rec_dir(tmp_path):
    return str(tmp_path / "recordings")


def _make_pcm(samples=960):
    return np.zeros(samples, dtype=np.float32)


# ── CallRecorder ────────────────────────────────────────────────────────

def test_finish_returns_none_when_no_frames(rec_dir):
    r = CallRecorder("abc123", record_dir=rec_dir, compress_to_flac=False)
    assert r.finish() is None


def test_push_and_finish_writes_wav(rec_dir):
    r = CallRecorder("abc123", record_dir=rec_dir, sample_rate=48000, compress_to_flac=False)
    r.push(_make_pcm(960))
    r.push(_make_pcm(960))
    path = r.finish()
    assert path is not None
    assert path.endswith(".wav")
    assert os.path.exists(path)


def test_wav_has_correct_metadata(rec_dir):
    r = CallRecorder("abc123", record_dir=rec_dir, sample_rate=16000, compress_to_flac=False)
    r.push(_make_pcm(3200))
    path = r.finish()
    with wave.open(path, "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 16000
        assert wf.getnframes() == 3200


def test_total_samples_accumulates(rec_dir):
    r = CallRecorder("abc123", record_dir=rec_dir, compress_to_flac=False)
    r.push(_make_pcm(100))
    r.push(_make_pcm(200))
    assert r._total_samples == 300


def test_clipping_does_not_raise(rec_dir):
    r = CallRecorder("abc123", record_dir=rec_dir, compress_to_flac=False)
    pcm = np.array([2.0, -2.0, 0.5], dtype=np.float32)  # values outside [-1, 1]
    r.push(pcm)
    path = r.finish()
    assert path is not None


def test_compress_skipped_when_flac_not_found(rec_dir, monkeypatch):
    import shutil as _shutil
    monkeypatch.setattr(_shutil, "which", lambda _: None)
    r = CallRecorder("abc123", record_dir=rec_dir, compress_to_flac=True)
    r.push(_make_pcm(960))
    path = r.finish()
    # Falls back to WAV when flac CLI unavailable
    assert path is not None
    assert path.endswith(".wav")


# ── rotate_recordings ───────────────────────────────────────────────────

def test_rotate_deletes_old_files(tmp_path):
    rec_dir = str(tmp_path / "rec")
    os.makedirs(rec_dir)
    old_file = os.path.join(rec_dir, "old.wav")
    open(old_file, "w").close()
    # Back-date the file by 10 days
    old_time = os.path.getmtime(old_file) - 10 * 86400
    os.utime(old_file, (old_time, old_time))

    deleted = rotate_recordings(record_dir=rec_dir, retention_days=7)
    assert deleted == 1
    assert not os.path.exists(old_file)


def test_rotate_keeps_recent_files(tmp_path):
    rec_dir = str(tmp_path / "rec")
    os.makedirs(rec_dir)
    recent = os.path.join(rec_dir, "recent.wav")
    open(recent, "w").close()  # mtime = now

    deleted = rotate_recordings(record_dir=rec_dir, retention_days=7)
    assert deleted == 0
    assert os.path.exists(recent)


def test_rotate_returns_zero_for_missing_dir(tmp_path):
    assert rotate_recordings(record_dir=str(tmp_path / "nonexistent")) == 0
