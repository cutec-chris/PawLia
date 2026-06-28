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
    # 2026-06-02 Zahlenspiel call (41fbf35f). Indoor, relatively quiet.
    # gt spans from Groq whisper-large-v3-turbo word timestamps.
    # Long sentence with natural pauses — the old SPEECH_PAUSE_RATIO=0.35 cut
    # these mid-sentence; 0.25 must keep them whole.
    ("speech_zahlenspiel", "zahlenspiel_spielregel_154645.flac", 1.14, 11.78,
     "Wenn du mich ausreden lässt, mache ich das. Wir machen jetzt ein Spiel und "
     "vornehme ich mal eine Ziffer und du addierst eins dazu. Und das muss eine "
     "kontinuierliche Kette werden."),
    ("speech_zahlenspiel", "zahlenspiel_addierst_154902.flac", 2.44, 9.96,
     "Nein, du sagst überhaupt nicht, wenn wir wieder anfangen. Du addierst "
     "immer eins dazu und sagst dann deine Zahl."),
    ("speech_zahlenspiel", "zahlenspiel_14sagen_155251.flac", 0.44, 10.78,
     "Ja, dann musst du 14 sagen, nicht 13. Außerdem machst du keine Vorgaben. "
     "Wenn ich 13 sage, musst du es sagen."),
    # Short but complete indoor utterances.
    ("speech_zahlenspiel", "zahlenspiel_keinfest_154516.flac", 0.36, 2.02,
     "Nein, kein Fest, ein Stil."),
    ("speech_zahlenspiel", "zahlenspiel_spiel_154549.flac", 0.38, 2.36,
     "Ja, wir machen jetzt ein Spiel."),
    # Single-word number responses in the counting game.
    # "Sieben" starts late (2.4s) in a noisy chunk — boundary detector picks up
    # leading noise, so no reliable boundary span.
    ("speech_zahlenspiel", "zahlenspiel_sieben_154916.flac", None, None,
     "Sieben."),
    ("speech_zahlenspiel", "zahlenspiel_acht_154951.flac", 2.24, 3.34,
     "Ich hab grad Acht gesagt."),
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
    # 2026-06-02 Zahlenspiel call — VAD cut these so aggressively that only
    # silence/noise remained; Whisper hallucinated "Vielen Dank" on all of them.
    # These are bursty noise (modulation 0.84-1.32), not steady wind, so the
    # modulation gate won't catch them — they rely on the hallucination filter
    # as the second defense layer (tested by test_wind_chunk_is_defended_end_to_end).
    ("wind_zahlenspiel", "zahlenspiel_wind_154439.flac", None, None, "Vielen Dank."),
    ("wind_zahlenspiel", "zahlenspiel_wind_154451.flac", None, None, "Vielen Dank."),
    ("wind_zahlenspiel", "zahlenspiel_wind_154453.flac", None, None, "Vielen Dank."),
    ("wind_zahlenspiel", "zahlenspiel_wind_154523.flac", None, None, "Vielen Dank."),
    ("wind_zahlenspiel", "zahlenspiel_wind_154607.flac", None, None, "Vielen Dank."),
    ("wind_zahlenspiel", "zahlenspiel_wind_154705.flac", None, None, "Vielen Dank."),
    ("wind_zahlenspiel", "zahlenspiel_wind_154707.flac", None, None, "Vielen Dank."),
    ("wind_zahlenspiel", "zahlenspiel_wind_154744.flac", None, None, "Vielen Dank."),
]

ALL = SPEECH + WIND + WIND_GUSTY

# Late-closing cases: real speech followed by prolonged wind. The user spoke
# briefly, then the endpointer kept the chunk open for tens of seconds (the
# field log shows chunks of 32-118s) because wind never lets the silence counter
# reach its threshold. (category, filename, gt_speech_end_seconds) — trimmed to
# speech + ~15s of trailing wind.
# (category, filename, gt_speech_end, max_close_latency)
#   schiesse: moderate wind → relative-energy pause closes it promptly (~2s).
#   hallopauli: extreme gusty wind → relative pause can't catch it, so the
#               max-utterance cap is the net (~13s — vs 95s in the field).
LATE_CLOSING = [
    ("late_closing", "lateclose_hallopauli_134750.flac", 2.0, 14.0),
    ("late_closing", "lateclose_schiesse_135103.flac", 6.4, 4.0),
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


# Max-utterance cap mirrored from CallSession.VAD_MAX_CHUNK_SECONDS (safety net).
MAX_CHUNK_SECONDS = 15.0


def detect_close_time(pcm, sr, fps=50, noise_floor=None):
    """Time (s) at which the endpointer finalises the chunk, or None if never.

    Mirrors the live loop's hybrid endpointing:
      * relative-energy pause — once the speaker's level is known, a frame that
        has dropped below SPEECH_PAUSE_RATIO of it counts as silence even if it
        still reads spectrally speech-like (so sustained wind, which falls back
        to the wind floor between words, lets the silence counter advance);
      * the chunk closes after SILENCE_SECONDS of such silence, or
      * a max-utterance cap force-flushes it as a safety net for extreme gusty
        wind that the relative pause cannot catch.
    """
    fs = sr // fps
    frames = [pcm[i:i + fs] for i in range(0, len(pcm) - fs + 1, fs)]
    frame_rms = np.array([float(np.sqrt(np.mean(f ** 2))) for f in frames])
    d = _detector()
    d._noise_floor = (
        noise_floor if noise_floor is not None
        else max(d.SILENCE_THRESHOLD, float(np.percentile(frame_rms, 20))))
    # Noise-coupled endpointing (mirror the live loop): in a loud environment the
    # relative pause is more aggressive and the max-chunk cap is shorter, while a
    # quiet call keeps the patient values.
    cap_frames = int(d.effective_max_chunk_seconds(MAX_CHUNK_SECONDS) * fps)
    ratio = d.effective_pause_ratio()
    in_speech = False
    silence = 0
    resume_speech_count = 0
    speech_ref = 0.0
    start = None
    for i, (frame, rms) in enumerate(zip(frames, frame_rms)):
        is_like = d.is_speech_like_frame(frame, sr, rms)
        if is_like and ratio > 0.0 and speech_ref > 0.0 and rms < speech_ref * ratio:
            is_like = False
        # adaptive endpoint: the longer this utterance has run, the longer a
        # mid-thought pause we tolerate — and the more sustained a return must
        # be to cancel that pause (so brief gusts can't hold the chunk open).
        # In a loud environment the growth is suppressed (high_noise).
        spoken_seconds = (i - start) / fps if (in_speech and start is not None) else 0.0
        adaptive_silence = d.adaptive_silence_seconds(spoken_seconds, high_noise=d.noise_is_high)
        silence_threshold = int(adaptive_silence * fps)
        resume_frames = d.resume_frames_for_silence(adaptive_silence)
        if is_like:
            if not in_speech:
                start = i
            in_speech = True
            speech_ref = 0.2 * rms + 0.8 * speech_ref if speech_ref > 0.0 else rms
            resume_confirmed, resume_speech_count = d.resume_after_pause(
                is_like, silence, resume_speech_count, min_frames=resume_frames)
            if resume_confirmed:
                silence = 0
                resume_speech_count = 0
            elif silence == 0:
                resume_speech_count = 0
        elif in_speech:
            silence += 1
            resume_speech_count = 0
            if silence >= silence_threshold:
                return (i + 1) / fps
        if in_speech and start is not None and (i - start) >= cap_frames:
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


@pytest.mark.parametrize("row", LATE_CLOSING, ids=_ids(LATE_CLOSING))
def test_endpointer_closes_soon_after_speech(row):
    """After the speaker stops, the endpointer must finalise the chunk within a
    bounded time even if wind continues — otherwise the caller waits tens of
    seconds for a reply (the field log had 32-118s chunks). The relative-energy
    pause handles moderate wind promptly; the max-utterance cap bounds the
    extreme gusty case.
    """
    _, fname, gt_speech_end, max_latency = row
    pcm, sr = _load(fname)
    close = detect_close_time(pcm, sr)
    assert close is not None, (
        f"{fname}: endpointer never closed — sustained wind kept the chunk open"
    )
    latency = close - gt_speech_end
    assert latency <= max_latency, (
        f"{fname}: chunk closed {latency:.1f}s after speech ended "
        f"(speech ~{gt_speech_end:.1f}s, close {close:.1f}s, budget {max_latency:.1f}s)"
    )


# --- adaptive endpointing invariants (unit-level, no fixtures) ---------------

def test_adaptive_silence_grows_with_speech_then_caps():
    """The longer the caller has been speaking, the longer a mid-thought pause
    we tolerate — monotonically, up to SILENCE_SECONDS_MAX. This is the whole
    point of adaptive endpointing: a long sentence is not chopped into pieces."""
    d = _detector()
    base = max(1.2, d.SILENCE_SECONDS)
    short = d.adaptive_silence_seconds(0.5)
    mid = d.adaptive_silence_seconds(6.0)
    long = d.adaptive_silence_seconds(30.0)
    assert short == pytest.approx(base, abs=0.2)        # short reply ≈ base
    assert mid > short                                   # grows while talking
    assert long > mid
    assert long <= d.SILENCE_SECONDS_MAX + 1e-6          # but capped
    # the cap must actually bind for a long monologue
    assert d.adaptive_silence_seconds(1e6) == pytest.approx(
        max(base, d.SILENCE_SECONDS_MAX), abs=1e-6)


def test_adaptive_silence_disabled_is_fixed():
    """SILENCE_GROWTH_PER_SEC == 0 restores the fixed endpoint (opt-out)."""
    d = _detector()
    d.SILENCE_GROWTH_PER_SEC = 0.0
    base = max(1.2, d.SILENCE_SECONDS)
    assert d.adaptive_silence_seconds(0.0) == base
    assert d.adaptive_silence_seconds(60.0) == base


def test_resume_requirement_grows_with_pause_tolerance():
    """Wind-safety invariant: as the tolerated pause grows, a more sustained
    return is required to cancel it — otherwise a brief gust could hold a long
    pause open until the max-chunk cap (the gusty-wind regression)."""
    d = _detector()
    base = max(1.2, d.SILENCE_SECONDS)
    at_base = d.resume_frames_for_silence(base)
    at_max = d.resume_frames_for_silence(d.SILENCE_SECONDS_MAX)
    assert at_base == d.MIN_RESUME_SPEECH_FRAMES       # base pause: prompt resume
    assert at_max >= d.RESUME_FRAMES_AT_MAX            # long pause: sustained return
    assert at_max > at_base                             # monotonic with tolerance


# --- noise-coupled endpointing invariants ------------------------------------
# In a persistently loud environment (noise_floor > HIGH_NOISE_FLOOR) the patient
# endpointing fails: wind/train noise fills the gaps so the silence counter never
# advances and the chunk only ever closes on the max-chunk cap (the field log of
# the 2026-06-15 ride showed ~21% of chunks running to the 15s cap). The fix
# couples the endpoint to the noise floor — aggressive when loud, patient when
# quiet — so pause detection works again without chopping quiet long sentences.

def test_noise_is_high_gate_matches_floor():
    d = _detector()
    d._noise_floor = d.HIGH_NOISE_FLOOR * 0.5
    assert d.noise_is_high is False
    d._noise_floor = d.HIGH_NOISE_FLOOR * 2.0
    assert d.noise_is_high is True


def test_effective_pause_ratio_rises_in_noise():
    """Quiet calls keep the gentle pause ratio (don't chop deliberate mid-
    sentence pauses); loud calls use the stronger HIGH_NOISE_PAUSE_RATIO so a
    real pause registers against the speaker's own loudness instead of running
    to the cap. The two regimes are fully independent knobs."""
    d = _detector()
    d._noise_floor = 0.01
    assert d.effective_pause_ratio() == pytest.approx(d.SPEECH_PAUSE_RATIO)
    d._noise_floor = 0.08
    assert d.effective_pause_ratio() == pytest.approx(d.HIGH_NOISE_PAUSE_RATIO)
    assert d.effective_pause_ratio() > d.SPEECH_PAUSE_RATIO


def test_speech_pause_ratio_disables_quiet_only_not_loud():
    """The config switch for the quiet-room "cuts me off" complaint:
    SPEECH_PAUSE_RATIO == 0 disables the relative-energy pause in quiet
    environments only — the loud regime keeps its own HIGH_NOISE_PAUSE_RATIO so
    wind/train calls still have a working pause detector. (Before decoupling,
    setting it to 0 disabled the pause everywhere.)"""
    d = _detector()
    d.SPEECH_PAUSE_RATIO = 0.0
    d._noise_floor = 0.01
    assert d.effective_pause_ratio() == 0.0                      # quiet: off
    d._noise_floor = 0.08
    assert d.effective_pause_ratio() == pytest.approx(d.HIGH_NOISE_PAUSE_RATIO)
    assert d.effective_pause_ratio() > 0.0                       # loud: still on


def test_high_noise_pause_ratio_disables_loud_only_not_quiet():
    """Symmetric switch: HIGH_NOISE_PAUSE_RATIO == 0 turns the relative pause off
    in loud environments without touching the quiet regime."""
    d = _detector()
    d.HIGH_NOISE_PAUSE_RATIO = 0.0
    d._noise_floor = 0.08
    assert d.effective_pause_ratio() == 0.0                      # loud: off
    d._noise_floor = 0.01
    assert d.effective_pause_ratio() == pytest.approx(d.SPEECH_PAUSE_RATIO)
    assert d.effective_pause_ratio() > 0.0                       # quiet: still on


def test_high_noise_suppresses_adaptive_growth():
    """In a loud environment the pause tolerance must NOT grow with speech (the
    growth only delays the close to the cap when noise fills the gaps); it is
    clamped to HIGH_NOISE_SILENCE_SECONDS_MAX. Quiet calls still grow."""
    d = _detector()
    base = max(1.2, d.SILENCE_SECONDS)
    # quiet: grows (existing behaviour, regression guard)
    assert d.adaptive_silence_seconds(30.0, high_noise=False) > base
    # loud: flat, short, independent of how long the caller has spoken
    short = d.adaptive_silence_seconds(0.5, high_noise=True)
    long = d.adaptive_silence_seconds(60.0, high_noise=True)
    assert short == long
    assert long == pytest.approx(max(1.2, d.HIGH_NOISE_SILENCE_SECONDS_MAX))


def test_quiet_and_loud_silence_trails_are_independent():
    """The STT-send timeout (silence trail) is fully separate for quiet vs loud:
    neither regime caps the other. A setup can be snappy in a quiet room yet
    patient in wind, or vice versa — the loud trail is NOT min()'d against the
    quiet base (the coupling that previously prevented loud > quiet)."""
    d = _detector()
    d.SILENCE_GROWTH_PER_SEC = 0.0  # isolate the base trails from adaptive growth
    # patient quiet, snappy loud
    d.SILENCE_SECONDS = 3.0
    d.HIGH_NOISE_SILENCE_SECONDS_MAX = 1.3
    assert d.adaptive_silence_seconds(0.0, high_noise=False) == pytest.approx(3.0)
    assert d.adaptive_silence_seconds(0.0, high_noise=True) == pytest.approx(1.3)
    # snappy quiet, patient loud — loud may exceed quiet (was impossible before)
    d.SILENCE_SECONDS = 1.3
    d.HIGH_NOISE_SILENCE_SECONDS_MAX = 2.5
    quiet = d.adaptive_silence_seconds(0.0, high_noise=False)
    loud = d.adaptive_silence_seconds(0.0, high_noise=True)
    assert quiet == pytest.approx(1.3)
    assert loud == pytest.approx(2.5)
    assert loud > quiet


def test_effective_max_chunk_shrinks_in_noise():
    d = _detector()
    base_cap = 15.0
    d._noise_floor = 0.01
    assert d.effective_max_chunk_seconds(base_cap) == base_cap   # quiet: unchanged
    d._noise_floor = 0.08
    assert d.effective_max_chunk_seconds(base_cap) == pytest.approx(
        min(base_cap, d.HIGH_NOISE_MAX_CHUNK_SECONDS))            # loud: shortened
    assert d.effective_max_chunk_seconds(base_cap) < base_cap


def test_effective_max_chunk_disabled_stays_disabled():
    """An operator who disabled the cap (<=0) keeps it disabled even in noise."""
    d = _detector()
    d._noise_floor = 0.08
    assert d.effective_max_chunk_seconds(0.0) == 0.0


def _speechlike_tone(seconds, sr=48000, level=0.3):
    """Deterministic, PII-free signal that passes is_speech_like_frame on every
    frame and never pauses (constant level, tonal content in the speech band) —
    so the ONLY thing that can close it is the max-chunk cap. Models the worst
    case: sustained noise the relative pause cannot catch."""
    n = int(seconds * sr)
    t = np.arange(n, dtype=np.float32) / sr
    sig = sum(np.sin(2 * np.pi * f * t) for f in (300.0, 700.0, 1300.0))
    return (sig / 3.0 * level).astype(np.float32), sr


def test_high_noise_cap_bounds_the_wait_end_to_end():
    """A sustained speech-like signal with no detectable pause must close at the
    SHORT high-noise cap when the floor is loud, and only at the long cap when
    quiet. This is the guarantee the field complaint needs: in wind/train the
    caller never waits the full quiet-call budget for a reply."""
    pcm, sr = _speechlike_tone(20.0)
    loud = detect_close_time(pcm, sr, noise_floor=0.08)
    quiet = detect_close_time(pcm, sr, noise_floor=0.01)
    d = _detector()
    assert loud is not None and quiet is not None
    assert loud == pytest.approx(d.HIGH_NOISE_MAX_CHUNK_SECONDS, abs=0.3)
    assert quiet == pytest.approx(MAX_CHUNK_SECONDS, abs=0.3)
    assert loud < quiet


# --- semantic endpointing: incomplete-utterance heuristic --------------------
# Reply timers in the call pipeline are time-based, so a thinking pause after an
# unfinished clause would reply to half a sentence. looks_like_incomplete_utterance
# lets the pipeline hold back. High precision by design: only dangling function
# words / trailing commas fire, so a sentence ending in a content word is never
# misread as unfinished (which would over-hold and feel laggy).

@pytest.mark.parametrize("text", [
    "und das liegt daran, dass",          # trailing subordinating conjunction (DE)
    "ich hätte gern einen Termin mit",    # trailing preposition (DE)
    "kannst du mir bitte die",            # trailing article (DE)
    "ich wollte nur kurz sagen, also",    # trailing filler (DE)
    "kauf bitte Milch, Eier,",            # trailing comma mid-list
    "so the thing is that",               # trailing conjunction (EN)
    "I would like to book a table for",   # trailing preposition (EN)
    "can you please send me the",         # trailing article (EN)
])
def test_incomplete_utterance_detected(text):
    assert SpeechDetector.looks_like_incomplete_utterance(text) is True


@pytest.mark.parametrize("text", [
    "kauf bitte Milch.",                  # complete, terminal punctuation
    "ich brauche einen neuen Termin",     # complete, ends on content word, no punct
    "ja",                                 # short standalone reply
    "nein danke",                         # short standalone reply
    "well",                               # short EN filler standalone (< 3 words)
    "please send the report tomorrow",    # complete EN, ends on content word
    "that's all for now",                 # complete EN
    "",                                   # empty
    "   ",                                # whitespace only
])
def test_complete_utterance_not_flagged(text):
    assert SpeechDetector.looks_like_incomplete_utterance(text) is False


def test_incomplete_heuristic_is_bilingual_and_case_insensitive():
    """The trailing-word check ignores case and covers both languages."""
    assert SpeechDetector.looks_like_incomplete_utterance("ICH GEHE NACH HAUSE UND") is True
    assert SpeechDetector.looks_like_incomplete_utterance("I AM GOING HOME AND") is True
