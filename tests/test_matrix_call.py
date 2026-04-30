from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio
import io
import logging
import sys

import numpy as np
import pytest

import pawlia.interfaces.matrix_call as matrix_call
from pawlia.interfaces.matrix_call import CallSession


def _make_pcm_from_frame_levels(levels, frame_size=960):
    return np.concatenate([
        np.full(frame_size, level, dtype=np.float32)
        for level in levels
    ])


def _make_tonal_pcm_from_frame_levels(levels, freq_hz=220.0, sample_rate=48000, frame_size=960):
    frames = []
    phase = 0.0
    phase_step = 2 * np.pi * freq_hz / sample_rate
    for level in levels:
        idx = np.arange(frame_size, dtype=np.float32)
        frame = (level * np.sin(phase + idx * phase_step)).astype(np.float32)
        phase = float((phase + frame_size * phase_step) % (2 * np.pi))
        frames.append(frame)
    return np.concatenate(frames)


class _FakeVad:
    def __init__(self, voiced_pattern):
        self._voiced_pattern = list(voiced_pattern)
        self._index = 0

    def is_speech(self, frame_bytes, sample_rate):
        if self._index < len(self._voiced_pattern):
            result = self._voiced_pattern[self._index]
        else:
            result = False
        self._index += 1
        return result


@pytest.mark.asyncio
async def test_process_speech_uses_call_system_prompt():
    pcm = MagicMock()
    pcm.__len__.return_value = 48000

    app = SimpleNamespace(config={}, llm=SimpleNamespace(audio_model_info=MagicMock(return_value=None)))
    client = SimpleNamespace(room_typing=AsyncMock())
    agent = MagicMock()
    agent.build_system_prompt.return_value = "CALL PROMPT"
    agent.run_streamed = AsyncMock(return_value="Kurze Antwort")

    send_cb = AsyncMock()
    session = CallSession(
        call_id="call-1",
        room_id="!room:test",
        caller_id="@user:test",
        thread_id="thread-1",
        client=client,
        app=app,
        cfg={},
        agent=agent,
        send_cb=send_cb,
    )

    session._tts_track = SimpleNamespace(
        start_hold=MagicMock(),
        stop_hold=MagicMock(),
        enqueue_pcm_float32=MagicMock(),
    )
    session._keep_typing = AsyncMock(return_value=None)

    with patch("pawlia.transcription.transcribe_pcm", new=AsyncMock(return_value="Hallo da")), patch(
        "pawlia.tts.synthesize_pcm", new=AsyncMock(return_value=[])
    ):
        await session._process_speech(pcm, 48000)

    agent.build_system_prompt.assert_called_once_with(mode="call", thread_id="thread-1")
    agent.run_streamed.assert_awaited_once_with(
        "Hallo da",
        system_prompt="CALL PROMPT",
        thread_id="thread-1",
        on_sentence=agent.run_streamed.await_args.kwargs["on_sentence"],
        on_skill_start=agent.run_streamed.await_args.kwargs["on_skill_start"],
        on_skill_done=agent.run_streamed.await_args.kwargs["on_skill_done"],
    )
    assert send_cb.await_args_list[0].args[0] == "🎙️ *Hallo da*"
    assert send_cb.await_args_list[1].args[0] == "Kurze Antwort"


def test_mark_activity_updates_last_activity_timestamp():
    session = CallSession(
        call_id="call-activity",
        room_id="!room:test",
        caller_id="@user:test",
        thread_id="thread-activity",
        client=SimpleNamespace(),
        app=SimpleNamespace(config={}, llm=SimpleNamespace(audio_model_info=MagicMock(return_value=None))),
        cfg={},
        agent=MagicMock(),
        send_cb=AsyncMock(),
    )

    session._last_activity_at = 10.0

    with patch("pawlia.interfaces.matrix_call.time.monotonic", return_value=42.0):
        session._mark_activity()

    assert session._last_activity_at == pytest.approx(42.0)


@pytest.mark.asyncio
async def test_watchdog_hangs_up_after_call_inactivity():
    session = CallSession(
        call_id="call-idle",
        room_id="!room:test",
        caller_id="@user:test",
        thread_id="thread-idle",
        client=SimpleNamespace(),
        app=SimpleNamespace(config={}, llm=SimpleNamespace(audio_model_info=MagicMock(return_value=None))),
        cfg={},
        agent=MagicMock(),
        send_cb=AsyncMock(),
    )

    session._last_activity_at = 0.0
    session.hangup = AsyncMock()
    session._send_hangup_event = AsyncMock()

    with patch("pawlia.interfaces.matrix_call.time.monotonic", return_value=181.0):
        await session._watchdog()

    session.hangup.assert_awaited_once()
    session._send_hangup_event.assert_awaited_once()


@pytest.mark.asyncio
async def test_process_speech_writes_debug_wav(tmp_path):
    app = SimpleNamespace(config={}, llm=SimpleNamespace(audio_model_info=MagicMock(return_value=None)))
    client = SimpleNamespace(room_typing=AsyncMock())
    agent = MagicMock()
    send_cb = AsyncMock()
    session = CallSession(
        call_id="call-debug",
        room_id="!room:test",
        caller_id="@user:test",
        thread_id="thread-debug",
        client=client,
        app=app,
        cfg={},
        agent=agent,
        send_cb=send_cb,
    )

    pcm = np.linspace(-0.25, 0.25, 4800, dtype=np.float32)
    fake_file = tmp_path / "pawlia" / "interfaces" / "matrix_call.py"
    fake_file.parent.mkdir(parents=True, exist_ok=True)
    fake_file.write_text("# test stub\n", encoding="utf-8")

    old_level = matrix_call.logger.level
    matrix_call.logger.setLevel(logging.DEBUG)
    try:
        with patch.object(matrix_call, "__file__", str(fake_file)), patch(
            "pawlia.transcription.transcribe_pcm", new=AsyncMock(return_value=None)
        ), patch("pawlia.tts.synthesize_pcm", new=AsyncMock(return_value=[])):
            await session._process_speech(pcm, 48000)
    finally:
        matrix_call.logger.setLevel(old_level)

    debug_dir = tmp_path / "log" / "debug_audio"
    files = list(debug_dir.glob("*.wav"))
    assert len(files) == 1
    assert files[0].stat().st_size > 44


def test_should_transcribe_chunk_rejects_background_noise():
    session = CallSession(
        call_id="call-noise",
        room_id="!room:test",
        caller_id="@user:test",
        thread_id="thread-noise",
        client=SimpleNamespace(),
        app=SimpleNamespace(config={}, llm=SimpleNamespace(audio_model_info=MagicMock(return_value=None))),
        cfg={},
        agent=MagicMock(),
        send_cb=AsyncMock(),
    )

    levels = [0.0] * 80
    for idx in (5, 22, 39, 57):
        levels[idx] = 0.05
    pcm = _make_tonal_pcm_from_frame_levels(levels)

    assert session._should_transcribe_chunk(pcm, 48000, fps=50) is False


def test_should_transcribe_chunk_accepts_sustained_speech():
    with patch.object(matrix_call, "_build_webrtc_vad", return_value=_FakeVad([False] * 10 + [True] * 30 + [False] * 40)):
        session = CallSession(
            call_id="call-speech",
            room_id="!room:test",
            caller_id="@user:test",
            thread_id="thread-speech",
            client=SimpleNamespace(),
            app=SimpleNamespace(config={}, llm=SimpleNamespace(audio_model_info=MagicMock(return_value=None))),
            cfg={},
            agent=MagicMock(),
            send_cb=AsyncMock(),
        )

        levels = [0.0] * 10 + [0.12] * 20 + [0.08] * 10 + [0.0] * 40
        pcm = _make_tonal_pcm_from_frame_levels(levels)

        assert session._should_transcribe_chunk(pcm, 48000, fps=50) is True


def test_should_transcribe_chunk_rejects_broadband_noise():
    with patch.object(matrix_call, "_build_webrtc_vad", return_value=_FakeVad([False] * 100)):
        session = CallSession(
            call_id="call-noise-broadband",
            room_id="!room:test",
            caller_id="@user:test",
            thread_id="thread-noise-broadband",
            client=SimpleNamespace(),
            app=SimpleNamespace(config={}, llm=SimpleNamespace(audio_model_info=MagicMock(return_value=None))),
            cfg={},
            agent=MagicMock(),
            send_cb=AsyncMock(),
        )

        rng = np.random.default_rng(7)
        pcm = rng.normal(0.0, 0.05, 48000 * 2).astype(np.float32)

        assert session._should_transcribe_chunk(pcm, 48000, fps=50) is False


def test_should_transcribe_chunk_rejects_when_webrtcvad_disagrees():
    with patch.object(matrix_call, "_build_webrtc_vad", return_value=_FakeVad([False] * 100)):
        session = CallSession(
            call_id="call-vad-reject",
            room_id="!room:test",
            caller_id="@user:test",
            thread_id="thread-vad-reject",
            client=SimpleNamespace(),
            app=SimpleNamespace(config={}, llm=SimpleNamespace(audio_model_info=MagicMock(return_value=None))),
            cfg={},
            agent=MagicMock(),
            send_cb=AsyncMock(),
        )

        levels = [0.0] * 10 + [0.12] * 20 + [0.08] * 10 + [0.0] * 40
        pcm = _make_tonal_pcm_from_frame_levels(levels)

        assert session._should_transcribe_chunk(pcm, 48000, fps=50) is False

def test_meaningful_interrupt_accepts_keywords_and_full_sentences():
    session = CallSession(
        call_id="call-interrupt-ok",
        room_id="!room:test",
        caller_id="@user:test",
        thread_id="thread-interrupt-ok",
        client=SimpleNamespace(),
        app=SimpleNamespace(config={}, llm=SimpleNamespace(audio_model_info=MagicMock(return_value=None))),
        cfg={},
        agent=MagicMock(),
        send_cb=AsyncMock(),
    )

    assert session._is_meaningful_interrupt("warte kurz") is True
    assert session._is_meaningful_interrupt("Kannst du kurz anhalten?") is True
    assert session._is_meaningful_interrupt("Ich bin gleich da.") is True


def test_meaningful_interrupt_rejects_short_noise_like_transcripts():
    session = CallSession(
        call_id="call-interrupt-no",
        room_id="!room:test",
        caller_id="@user:test",
        thread_id="thread-interrupt-no",
        client=SimpleNamespace(),
        app=SimpleNamespace(config={}, llm=SimpleNamespace(audio_model_info=MagicMock(return_value=None))),
        cfg={},
        agent=MagicMock(),
        send_cb=AsyncMock(),
    )

    assert session._is_meaningful_interrupt("hm") is False
    assert session._is_meaningful_interrupt("ja") is False
    assert session._is_meaningful_interrupt("fahrrad wind") is False


@pytest.mark.asyncio
async def test_process_speech_does_not_interrupt_for_non_meaningful_barge_in():
    app = SimpleNamespace(config={}, llm=SimpleNamespace(audio_model_info=MagicMock(return_value=None)))
    client = SimpleNamespace(room_typing=AsyncMock())
    agent = MagicMock()
    send_cb = AsyncMock()
    session = CallSession(
        call_id="call-barge-no",
        room_id="!room:test",
        caller_id="@user:test",
        thread_id="thread-barge-no",
        client=client,
        app=app,
        cfg={},
        agent=agent,
        send_cb=send_cb,
    )

    session._tts_track = SimpleNamespace(
        interrupt=MagicMock(),
        stop_after_current_sentence=MagicMock(),
        stop_hold=MagicMock(),
    )
    session._cancel_active_response = AsyncMock()
    session._respond_to_transcript = AsyncMock()

    with patch.object(session, "_transcribe_speech", new=AsyncMock(return_value="hm")):
        await session._process_speech(np.zeros(4800, dtype=np.float32), 48000, interrupt_playback=True)

    session._tts_track.interrupt.assert_not_called()
    session._tts_track.stop_after_current_sentence.assert_not_called()
    session._cancel_active_response.assert_not_awaited()
    session._respond_to_transcript.assert_not_awaited()
    send_cb.assert_not_awaited()


@pytest.mark.asyncio
async def test_process_speech_interrupts_for_meaningful_barge_in():
    app = SimpleNamespace(config={}, llm=SimpleNamespace(audio_model_info=MagicMock(return_value=None)))
    client = SimpleNamespace(room_typing=AsyncMock())
    agent = MagicMock()
    send_cb = AsyncMock()
    session = CallSession(
        call_id="call-barge-yes",
        room_id="!room:test",
        caller_id="@user:test",
        thread_id="thread-barge-yes",
        client=client,
        app=app,
        cfg={},
        agent=agent,
        send_cb=send_cb,
    )

    session._tts_track = SimpleNamespace(
        interrupt=MagicMock(),
        stop_after_current_sentence=MagicMock(),
        stop_hold=MagicMock(),
    )
    session._cancel_active_response = AsyncMock()
    session._respond_to_transcript = AsyncMock()

    with patch.object(session, "_transcribe_speech", new=AsyncMock(return_value="warte kurz")):
        await session._process_speech(np.zeros(4800, dtype=np.float32), 48000, interrupt_playback=True)

    session._tts_track.interrupt.assert_not_called()
    session._tts_track.stop_after_current_sentence.assert_called_once()
    session._cancel_active_response.assert_awaited_once()
    session._respond_to_transcript.assert_awaited_once_with("warte kurz")


@pytest.mark.asyncio
async def test_meaningful_barge_in_cancels_previous_response_task():
    app = SimpleNamespace(config={}, llm=SimpleNamespace(audio_model_info=MagicMock(return_value=None)))
    client = SimpleNamespace(room_typing=AsyncMock())
    session = CallSession(
        call_id="call-barge-cancel",
        room_id="!room:test",
        caller_id="@user:test",
        thread_id="thread-barge-cancel",
        client=client,
        app=app,
        cfg={},
        agent=MagicMock(),
        send_cb=AsyncMock(),
    )

    previous_response = asyncio.create_task(asyncio.sleep(30))
    session._track_response_task(previous_response)
    session._tts_track = SimpleNamespace(stop_after_current_sentence=MagicMock(), stop_hold=MagicMock())
    session._respond_to_transcript = AsyncMock()

    try:
        with patch.object(session, "_transcribe_speech", new=AsyncMock(return_value="warte kurz")):
            await session._process_speech(np.zeros(4800, dtype=np.float32), 48000, interrupt_playback=True)
    finally:
        if not previous_response.done():
            previous_response.cancel()

    assert previous_response.cancelled()
    session._tts_track.stop_after_current_sentence.assert_called_once()
    session._respond_to_transcript.assert_awaited_once_with("warte kurz")


@pytest.mark.skipif(not matrix_call._AIORTC_AVAILABLE, reason="aiortc not installed")
def test_tts_barge_in_finishes_current_sentence_and_discards_later_sentences():
    track = matrix_call._TTSAudioTrack()
    frame_count = track.SAMPLES_PER_FRAME * 2

    track.enqueue_pcm_float32(np.ones(frame_count, dtype=np.float32) * 0.1)
    track.enqueue_pcm_float32(np.ones(frame_count, dtype=np.float32) * 0.2)

    first_item = track._queue.get_nowait()
    assert first_item[1] == 1
    track._current_sentence_id = 1

    track.stop_after_current_sentence()

    kept = []
    while not track._queue.empty():
        kept.append(track._queue.get_nowait())

    assert kept
    assert {item[1] for item in kept} == {1}


def test_call_session_loads_voip_audio_thresholds_from_config():
    with patch.object(matrix_call, "_build_webrtc_vad", return_value=_FakeVad([])):
        session = CallSession(
            call_id="call-config",
            room_id="!room:test",
            caller_id="@user:test",
            thread_id="thread-config",
            client=SimpleNamespace(),
            app=SimpleNamespace(config={
                "voip": {
                    "silence_threshold": 0.03,
                    "silence_seconds": 2.2,
                    "min_speech_seconds": 0.7,
                    "min_active_speech_ratio": 0.25,
                    "min_consecutive_speech_frames": 11,
                    "min_speech_band_ratio": 0.42,
                    "max_spectral_flatness": 0.61,
                    "min_speech_like_ratio": 0.14,
                    "min_consecutive_speechlike_frames": 6,
                    "webrtcvad_enabled": True,
                    "webrtcvad_mode": 3,
                    "webrtcvad_min_voiced_ratio": 0.22,
                    "webrtcvad_min_consecutive_frames": 5,
                    "call_inactivity_seconds": 240,
                }
            }),
            cfg={},
            agent=MagicMock(),
            send_cb=AsyncMock(),
        )

        assert session.SILENCE_THRESHOLD == pytest.approx(0.03)
        assert session.SILENCE_SECONDS == pytest.approx(2.2)
        assert session.MIN_SPEECH_SECONDS == pytest.approx(0.7)
        assert session.MIN_ACTIVE_SPEECH_RATIO == pytest.approx(0.25)
        assert session.MIN_CONSECUTIVE_SPEECH_FRAMES == 11
        assert session.MIN_SPEECH_BAND_RATIO == pytest.approx(0.42)
        assert session.MAX_SPECTRAL_FLATNESS == pytest.approx(0.61)
        assert session.MIN_SPEECH_LIKE_RATIO == pytest.approx(0.14)
        assert session.MIN_CONSECUTIVE_SPEECHLIKE_FRAMES == 6
        assert session.WEBRTC_VAD_ENABLED is True
        assert session.WEBRTC_VAD_MODE == 3
        assert session.WEBRTC_VAD_MIN_VOICED_RATIO == pytest.approx(0.22)
        assert session.WEBRTC_VAD_MIN_CONSECUTIVE_FRAMES == 5
        assert session.CALL_INACTIVITY_SECONDS == 240


def test_call_session_invalid_voip_audio_thresholds_fall_back_to_defaults():
    session = CallSession(
        call_id="call-config-default",
        room_id="!room:test",
        caller_id="@user:test",
        thread_id="thread-config-default",
        client=SimpleNamespace(),
        app=SimpleNamespace(config={
            "voip": {
                "silence_threshold": -1,
                "silence_seconds": "bad",
                "min_speech_seconds": 0,
                "min_active_speech_ratio": 1.5,
                "min_consecutive_speech_frames": 0,
                "min_speech_band_ratio": 1.3,
                "max_spectral_flatness": -0.1,
                "min_speech_like_ratio": -1,
                "min_consecutive_speechlike_frames": 0,
                "webrtcvad_enabled": "maybe",
                "webrtcvad_mode": 99,
                "webrtcvad_min_voiced_ratio": 2,
                "webrtcvad_min_consecutive_frames": 0,
                "call_inactivity_seconds": 0,
            }
        }),
        cfg={},
        agent=MagicMock(),
        send_cb=AsyncMock(),
    )

    assert session.SILENCE_THRESHOLD == pytest.approx(CallSession.SILENCE_THRESHOLD)
    assert session.SILENCE_SECONDS == pytest.approx(CallSession.SILENCE_SECONDS)
    assert session.MIN_SPEECH_SECONDS == pytest.approx(CallSession.MIN_SPEECH_SECONDS)
    assert session.MIN_ACTIVE_SPEECH_RATIO == pytest.approx(CallSession.MIN_ACTIVE_SPEECH_RATIO)
    assert session.MIN_CONSECUTIVE_SPEECH_FRAMES == CallSession.MIN_CONSECUTIVE_SPEECH_FRAMES
    assert session.MIN_SPEECH_BAND_RATIO == pytest.approx(CallSession.MIN_SPEECH_BAND_RATIO)
    assert session.MAX_SPECTRAL_FLATNESS == pytest.approx(CallSession.MAX_SPECTRAL_FLATNESS)
    assert session.MIN_SPEECH_LIKE_RATIO == pytest.approx(CallSession.MIN_SPEECH_LIKE_RATIO)
    assert session.MIN_CONSECUTIVE_SPEECHLIKE_FRAMES == CallSession.MIN_CONSECUTIVE_SPEECHLIKE_FRAMES
    assert session.WEBRTC_VAD_ENABLED == CallSession.WEBRTC_VAD_ENABLED
    assert session.WEBRTC_VAD_MODE == CallSession.WEBRTC_VAD_MODE
    assert session.WEBRTC_VAD_MIN_VOICED_RATIO == pytest.approx(CallSession.WEBRTC_VAD_MIN_VOICED_RATIO)
    assert session.WEBRTC_VAD_MIN_CONSECUTIVE_FRAMES == CallSession.WEBRTC_VAD_MIN_CONSECUTIVE_FRAMES
    assert session.CALL_INACTIVITY_SECONDS == CallSession.CALL_INACTIVITY_SECONDS


def test_load_hold_audio_uses_ndarray_resampling():
    session = CallSession(
        call_id="call-2",
        room_id="!room:test",
        caller_id="@user:test",
        thread_id="thread-2",
        client=SimpleNamespace(),
        app=SimpleNamespace(config={"tts": {"hold_audio": "dummy.m4a", "hold_audio_volume": 1.0}}),
        cfg={},
        agent=MagicMock(),
        send_cb=AsyncMock(),
    )

    out_frame = SimpleNamespace(
        to_ndarray=lambda: np.array([[0.5, -0.5, 0.25]], dtype=np.float32)
    )
    resampler = MagicMock()
    resampler.resample.side_effect = [[out_frame], []]
    container = SimpleNamespace(decode=lambda audio=0: [object()])
    fake_av = SimpleNamespace(
        open=lambda stream: container,
        AudioResampler=lambda **kwargs: resampler,
    )

    with patch("os.path.exists", return_value=True), \
         patch("builtins.open", return_value=io.BytesIO(b"audio-bytes")), \
         patch.dict(sys.modules, {"av": fake_av}):
        pcm = session._load_hold_audio()

    assert pcm is not None
    assert pcm.dtype == np.int16
    assert pcm.tolist() == [16383, -16383, 8191]


def test_load_hold_audio_uses_mono_wav_default_without_m4a_fallback():
    session = CallSession(
        call_id="call-3",
        room_id="!room:test",
        caller_id="@user:test",
        thread_id="thread-3",
        client=SimpleNamespace(),
        app=SimpleNamespace(config={}, llm=SimpleNamespace(audio_model_info=MagicMock(return_value=None))),
        cfg={},
        agent=MagicMock(),
        send_cb=AsyncMock(),
    )

    chosen_paths = []

    def fake_exists(path):
        return path.endswith("keyboard_mono.wav")

    def fake_open(path, mode="rb"):
        chosen_paths.append(path)
        return io.BytesIO(b"audio-bytes")

    out_frame = SimpleNamespace(
        to_ndarray=lambda: np.array([[0.0, 0.0]], dtype=np.float32)
    )
    resampler = MagicMock()
    resampler.resample.side_effect = [[out_frame], []]
    container = SimpleNamespace(decode=lambda audio=0: [object()])
    fake_av = SimpleNamespace(
        open=lambda stream: container,
        AudioResampler=lambda **kwargs: resampler,
    )

    with patch("os.path.exists", side_effect=fake_exists), \
         patch("builtins.open", side_effect=fake_open), \
         patch.dict(sys.modules, {"av": fake_av}):
        session._load_hold_audio()

    assert chosen_paths
    assert all(path.endswith("keyboard_mono.wav") for path in chosen_paths)
