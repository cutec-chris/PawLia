"""Regression tests for VAD endpointing / hallucination handling, built from
REAL call audio.

The fixtures under ``tests/fixtures/audio/`` are chunks cut from actual VoIP
call recordings, kept at *full* length so the surrounding silence/noise context
is preserved — that is what lets us test whether our own VAD finds the speech
boundaries, not just whether a pre-trimmed clip transcribes.

Ground-truth speech spans (``gt_start`` / ``gt_end``, seconds into the clip)
were obtained from Whisper word/segment timestamps (Groq whisper-large-v3-turbo,
temperature 0) and hand-checked against the transcripts. They are the reference
our frame-level detector is measured against.

What this pins down (see the diagnosis from the 2026-05-31 bike call):

  1. real speech (indoor *and* outdoor/mobile) must pass :meth:`should_transcribe`
     — we must never "fix" wind by tightening the gate until quiet speech drops;
  2. our boundary detection must find the speech: start not *late* (else we clip
     the first word) and cover most of the true span (recall);
  3. each wind chunk stays defended end-to-end — either the VAD rejects it, or
     the hallucination filter catches the transcript it produced.

Note on the bike call ``d6177c3e``: it is so wind-saturated that Whisper
hallucinates different text on every pass, so it yields *no* reliable speech
fixtures — only the wind/noise ones below. The genuine outdoor-speech samples
come from the lighter-noise mobile calls (``3bfe60ea`` / ``54e87832``).

Decoding: 48 kHz mono FLAC via ``soundfile`` / ``av`` if importable, else the
``flac`` CLI; the test skips only if none is available.
"""

import io
import os
import shutil
import subprocess
import wave

import numpy as np
import pytest

from pawlia.audio.vad import SpeechDetector

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "audio")

# category, filename, gt_start, gt_end, transcript
#   gt_start/gt_end are None for wind (no speech to localise).
SPEECH = [
    ("speech_outdoor", "speech_outdoor_unterbrochen_083728.flac", 0.80, 8.52,
     "Wir wurden gerade unterbrochen. Die Verbindung war, also oder beziehungsweise "
     "Elimit ist abgestürzt. Auf Home Assistant hast du noch keinen Zugang, oder?"),
    ("speech_outdoor", "speech_outdoor_webrtc_084107.flac", 1.12, 9.44,
     "Da war die Verbindung jetzt zum dritten Mal innerhalb von fünf Minuten weg. "
     "Ist das normal, dass WebRTC über Mobilfunk kurz abbricht?"),
    ("speech_outdoor", "speech_outdoor_pampa_084224.flac", 0.32, 7.90,
     "Nein, ich bin hier mitten in der Pampa. Also schlechter Empfang könnte "
     "natürlich sein. Wie ist denn die Netzwerkqualität?"),
    ("speech_indoor", "speech_indoor_zahnarzt_131512.flac", 1.96, 6.46,
     "Du kannst mir meinen Zahnarzttermin für morgen früh wie 7 Uhr in den "
     "Kalender eintragen."),
    ("speech_indoor", "speech_indoor_aufgaben_160045.flac", 4.06, 5.98,
     "Kannst du meine Aufgaben erzählen?"),
]

# wind-only chunks from the bike call; STT hallucinated these texts in production
WIND = [
    ("wind", "wind_zdf_070808.flac", None, None, "Untertitelung des ZDF, 2020"),
    ("wind", "wind_vielendank_070818.flac", None, None, "Vielen Dank."),
    ("wind", "wind_vielendank_070800.flac", None, None, "Vielen Dank."),
    ("wind", "wind_vielendank_0721_070652.flac", None, None, "Vielen Dank."),
]

ALL = SPEECH + WIND

# Current measured detector capability against ground truth (2026-05-31 audio).
# These are the *baseline* tolerances to tighten as endpointing improves:
#   - start must not be LATE by more than this (late start clips the first word);
#   - detected span must cover at least this fraction of the true speech.
MAX_LATE_START_S = 0.5
MIN_COVERAGE = 0.75


def _decode(path):
    """Return (float32 mono PCM in [-1, 1], sample_rate)."""
    try:
        import soundfile as sf  # type: ignore
        data, sr = sf.read(path, dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        return data.astype(np.float32), sr
    except ImportError:
        pass
    try:
        import av  # type: ignore
        container = av.open(path)
        chunks, sr = [], 48000
        for frame in container.decode(audio=0):
            sr = frame.sample_rate
            arr = frame.to_ndarray()
            chunks.append(arr.mean(axis=0) if arr.ndim > 1 else arr)
        pcm = np.concatenate(chunks).astype(np.float32)
        if pcm.max() > 1.5:
            pcm = pcm / 32768.0
        return pcm, sr
    except ImportError:
        pass
    if shutil.which("flac"):
        raw = subprocess.run(
            ["flac", "-d", "-c", "--silent", path], capture_output=True, check=True
        ).stdout
        w = wave.open(io.BytesIO(raw))
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        return pcm.astype(np.float32) / 32768.0, w.getframerate()
    pytest.skip("no FLAC decoder available (soundfile / av / flac CLI)")


def _load(fname):
    return _decode(os.path.join(FIXTURE_DIR, fname))


def _detector():
    """Detector on the energy+spectral path only (webrtcvad disabled), so results
    are deterministic regardless of whether the optional lib is installed."""
    d = SpeechDetector()
    d._webrtc_vad = None
    return d


def detect_speech_span(pcm, sr, fps=50, min_run=5):
    """Locate the speech region with the live frame classifier.

    Mirrors the streaming pipeline's per-frame decision: update the noise floor
    each frame, flag speech-like frames, then report the first/last frame that
    sits inside a run of at least ``min_run`` consecutive speech-like frames
    (so isolated blips don't define the boundaries). Returns (start_s, end_s) or
    (None, None) if no speech was found.
    """
    fs = sr // fps
    d = _detector()
    flags = []
    for i in range(0, len(pcm) - fs + 1, fs):
        frame = pcm[i:i + fs]
        rms = float(np.sqrt(np.mean(frame ** 2)))
        d.update_noise_floor(rms, during_speech=False)
        flags.append(d.is_speech_like_frame(frame, sr, rms))
    flags = np.array(flags, dtype=bool)
    qualifying = [
        i for i in range(min_run - 1, len(flags)) if flags[i - min_run + 1:i + 1].all()
    ]
    if not qualifying:
        return None, None
    return (qualifying[0] - min_run + 1) / fps, (qualifying[-1] + 1) / fps


def _ids(rows):
    return [r[1] for r in rows]


@pytest.mark.parametrize("row", ALL, ids=_ids(ALL))
def test_fixture_files_exist_and_decode(row):
    _, fname = row[0], row[1]
    pcm, sr = _load(fname)
    assert sr == 48000
    assert len(pcm) > 48000 * 0.5


@pytest.mark.parametrize("row", SPEECH, ids=_ids(SPEECH))
def test_real_speech_is_accepted_by_vad(row):
    """Indoor *and* outdoor speech must pass the VAD gate."""
    fname = row[1]
    pcm, sr = _load(fname)
    assert _detector().should_transcribe(pcm, sr, 50) is True, (
        f"VAD rejected real speech {fname} — likely an over-tightened noise gate"
    )


@pytest.mark.parametrize("row", SPEECH, ids=_ids(SPEECH))
def test_speech_boundaries_are_found(row):
    """Our boundary detection must localise the speech reasonably well.

    Two guards (baseline tolerances, to tighten as endpointing improves):
      * start not late by > MAX_LATE_START_S — a late start clips the first word;
      * detected span covers >= MIN_COVERAGE of the true speech (recall), which
        also bounds how early we may cut the trailing words.
    """
    _, fname, gt_start, gt_end, _ = row
    pcm, sr = _load(fname)
    det_start, det_end = detect_speech_span(pcm, sr)
    assert det_start is not None, f"no speech detected in {fname}"

    late_start = det_start - gt_start
    overlap = max(0.0, min(det_end, gt_end) - max(det_start, gt_start))
    coverage = overlap / (gt_end - gt_start)

    assert late_start <= MAX_LATE_START_S, (
        f"{fname}: speech start detected {late_start:.2f}s late "
        f"(det {det_start:.2f}s vs truth {gt_start:.2f}s) — first word would be clipped"
    )
    assert coverage >= MIN_COVERAGE, (
        f"{fname}: only {coverage:.0%} of the speech span "
        f"[{gt_start:.2f}, {gt_end:.2f}] was covered by detection "
        f"[{det_start:.2f}, {det_end:.2f}]"
    )


@pytest.mark.parametrize("row", WIND, ids=_ids(WIND))
def test_wind_chunk_is_defended_end_to_end(row):
    """A wind chunk must be stopped somewhere: VAD rejects it, or the
    hallucination filter catches the transcript it produced. Fails only if both
    layers let wind through."""
    _, fname, _, _, transcript = row
    detector = _detector()
    pcm, sr = _load(fname)
    vad_rejects = detector.should_transcribe(pcm, sr, 50) is False
    filter_catches = detector.looks_like_stt_hallucination(transcript)
    assert vad_rejects or filter_catches, (
        f"wind chunk {fname} passed the VAD and its transcript {transcript!r} "
        f"was not filtered — wind is no longer defended"
    )


@pytest.mark.parametrize("row", WIND, ids=_ids(WIND))
def test_steady_wind_is_rejected_by_vad(row):
    """Steady wind chunks (near-constant envelope) must be rejected at the VAD
    gate by the modulation check, so they never reach STT to hallucinate.

    This is stronger than the end-to-end guard above and pins the modulation
    gate specifically. Gusty/bursty wind can still slip through (higher
    modulation) — that remains the hallucination filter's job."""
    fname = row[1]
    pcm, sr = _load(fname)
    assert _detector().should_transcribe(pcm, sr, 50) is False, (
        f"steady wind {fname} passed the VAD — the envelope-modulation gate "
        f"is no longer catching it"
    )


@pytest.mark.parametrize("row", SPEECH, ids=_ids(SPEECH))
def test_real_transcript_is_not_filtered_as_hallucination(row):
    """The hallucination filter must never eat a genuine utterance."""
    transcript = row[4]
    assert SpeechDetector.looks_like_stt_hallucination(transcript) is False, (
        f"hallucination filter wrongly flagged real speech: {transcript!r}"
    )
