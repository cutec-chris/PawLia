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
    # 2026-06-01 bike ride (gusty wind). gt spans from Whisper word timestamps.
    ("speech_outdoor", "speech_outdoor_karte_135019.flac", 0.04, 2.45,
     "Und jetzt kommt es wieder auf die Karte."),
    ("speech_outdoor", "speech_outdoor_korrekt_141321.flac", 1.88, 9.70,
     "Nein, das war so nicht korrekt."),
    ("speech_outdoor", "speech_outdoor_tustdu_134950.flac", 2.24, 5.30,
     "Tust du das? Wiederhol mal einfach immer, was ich dir sage."),
    ("speech_outdoor", "speech_outdoor_geholfen_140438.flac", 2.14, 5.08,
     "Das ist nichts in unseren letzten Gesprächen geholfen."),
    ("speech_outdoor", "speech_outdoor_hallopauli_140239.flac", 3.10, 4.72,
     "Hallo Pauli, wie geht es?"),
    # Short/quiet utterance sitting right at the modulation threshold (CoV ~0.70):
    # guards that the wind gate does not start dropping brief real speech. Too
    # short to localise meaningfully, so no boundary span.
    ("speech_outdoor", "speech_outdoor_dasistgut_040055.flac", None, None,
     "Das ist gut."),
]

# wind-only chunks; STT hallucinated these texts in production.
#   WIND       — steady wind (low CoV), rejected at the VAD by the modulation gate.
#   WIND_GUSTY — gusty wind (high CoV ~1.3-1.7): modulates like speech, so it
#                passes the VAD gate and is only stopped downstream by the
#                hallucination filter. Documents the modulation gate's limit.
WIND = [
    ("wind", "wind_zdf_070808.flac", None, None, "Untertitelung des ZDF, 2020"),
    ("wind", "wind_vielendank_070818.flac", None, None, "Vielen Dank."),
    ("wind", "wind_vielendank_070800.flac", None, None, "Vielen Dank."),
    ("wind", "wind_vielendank_0721_070652.flac", None, None, "Vielen Dank."),
]
WIND_GUSTY = [
    ("wind_gusty", "wind_gusty_035825.flac", None, None, "Vielen Dank."),
    ("wind_gusty", "wind_gusty_140303.flac", None, None, "Vielen Dank."),
]

ALL = SPEECH + WIND + WIND_GUSTY

# Late-closing cases: real speech followed by prolonged wind. The user spoke
# briefly, then the endpointer kept the chunk open for tens of seconds (the
# field log shows chunks of 32-118s) because wind never lets the silence counter
# reach its threshold. (category, filename, gt_speech_end_seconds) — trimmed to
# speech + ~15s of trailing wind.
LATE_CLOSING = [
    ("late_closing", "lateclose_hallopauli_134750.flac", 2.0),   # "Hallo Pauli", then 95s wind in the wild
    ("late_closing", "lateclose_schiesse_135103.flac", 6.4),     # one sentence, then wind to 38s
]

# Speech fixtures the boundary detector localises poorly (xfail so the suite
# stays green but flags an xpass once fixed). Currently empty: the earlier
# apparent weakness was a flaw in the per-chunk noise-floor handling of the test
# helper (now fixed with a static ambient estimate), not real clipping.
WEAK_BOUNDARY = set()

# Current measured detector capability against ground truth.
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
    """Locate the speech region using the VAD's frame classifier.

    Returns (start_s, end_s) from the first/last frame inside a run of at least
    ``min_run`` consecutive speech-like frames, or (None, None) if none. Uses a
    static ambient noise-floor estimate (see below) because, unlike the live
    call, a single chunk offers no warm-up history to adapt the floor against.
    """
    fs = sr // fps
    frames = [pcm[i:i + fs] for i in range(0, len(pcm) - fs + 1, fs)]
    frame_rms = np.array([float(np.sqrt(np.mean(f ** 2))) for f in frames])

    # Seed a STATIC noise floor from the clip's quiet frames. In production the
    # floor is warmed over the whole call; a per-chunk dynamic EMA either
    # inflates on the speaker's own loud frames (masking quiet trailing
    # syllables → end too early) or stays stuck high on an all-speech clip. A
    # low percentile of the frame energy approximates the ambient level and
    # stays deterministic.
    d = _detector()
    d._noise_floor = max(d.SILENCE_THRESHOLD, float(np.percentile(frame_rms, 20)))

    flags = np.array([
        d.is_speech_like_frame(f, sr, rms) for f, rms in zip(frames, frame_rms)
    ], dtype=bool)
    qualifying = [
        i for i in range(min_run - 1, len(flags)) if flags[i - min_run + 1:i + 1].all()
    ]
    if not qualifying:
        return None, None
    return (qualifying[0] - min_run + 1) / fps, (qualifying[-1] + 1) / fps


def detect_close_time(pcm, sr, fps=50):
    """Time (s) at which the endpointer would finalise the chunk, or None if it
    never closes within the clip.

    Mirrors the live loop's silence logic: once speech has started, a frame that
    is not speech-like increments the silence counter (reset by any speech-like
    frame); the chunk closes after ``SILENCE_SECONDS`` of silence. In sustained
    wind, wind frames keep reading as speech-like, so the counter never reaches
    threshold and the chunk stays open — the late-closing bug.
    """
    fs = sr // fps
    frames = [pcm[i:i + fs] for i in range(0, len(pcm) - fs + 1, fs)]
    frame_rms = np.array([float(np.sqrt(np.mean(f ** 2))) for f in frames])
    d = _detector()
    d._noise_floor = max(d.SILENCE_THRESHOLD, float(np.percentile(frame_rms, 20)))
    silence_threshold = int(max(1.2, d.SILENCE_SECONDS) * fps)
    in_speech = False
    silence = 0
    for i, (frame, rms) in enumerate(zip(frames, frame_rms)):
        if d.is_speech_like_frame(frame, sr, rms):
            in_speech = True
            silence = 0
        elif in_speech:
            silence += 1
            if silence >= silence_threshold:
                return (i + 1) / fps
    return None


def _ids(rows):
    return [r[1] for r in rows]


def _boundary_params():
    """SPEECH entries that have a ground-truth span; WEAK_BOUNDARY ones are
    xfail (current detector localises them poorly — the speech-end fix target)."""
    params = []
    for row in SPEECH:
        if row[2] is None:
            continue
        marks = (
            [pytest.mark.xfail(reason="known speech-end localisation weakness",
                               strict=False)]
            if row[1] in WEAK_BOUNDARY else []
        )
        params.append(pytest.param(row, id=row[1], marks=marks))
    return params


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


@pytest.mark.parametrize("row", _boundary_params())
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


@pytest.mark.parametrize("row", WIND + WIND_GUSTY, ids=_ids(WIND + WIND_GUSTY))
def test_wind_chunk_is_defended_end_to_end(row):
    """A wind chunk must be stopped somewhere: VAD rejects it, or the
    hallucination filter catches the transcript it produced. Fails only if both
    layers let wind through. Covers both steady and gusty wind."""
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


MAX_CLOSE_LATENCY_S = 4.0


@pytest.mark.xfail(
    reason="late-closing in sustained wind: endpointer never reaches its silence "
           "threshold, so the chunk stays open for tens of seconds (not yet fixed)",
    strict=False,
)
@pytest.mark.parametrize("row", LATE_CLOSING, ids=_ids(LATE_CLOSING))
def test_endpointer_closes_soon_after_speech(row):
    """After the speaker stops, the endpointer must finalise the chunk promptly
    even if wind continues — otherwise the caller waits tens of seconds for a
    reply. Currently xfail: sustained wind keeps the chunk open indefinitely.
    """
    _, fname, gt_speech_end = row
    pcm, sr = _load(fname)
    close = detect_close_time(pcm, sr)
    assert close is not None, (
        f"{fname}: endpointer never closed — sustained wind kept the chunk open"
    )
    latency = close - gt_speech_end
    assert latency <= MAX_CLOSE_LATENCY_S, (
        f"{fname}: chunk closed {latency:.1f}s after speech ended "
        f"(speech ~{gt_speech_end:.1f}s, close {close:.1f}s)"
    )
