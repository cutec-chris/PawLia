from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio
import io
import logging
import sys

import numpy as np
import pytest

import pawlia.audio.vad as audio_vad
from pawlia.audio.vad import SpeechDetector
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
        is_playing=False,
    )
    session._keep_typing = AsyncMock(return_value=None)
    session.RESPONSE_DELAY_SECONDS = 0.0

    with patch("pawlia.transcription.transcribe_pcm", new=AsyncMock(return_value="Hallo da")), patch(
        "pawlia.tts.synthesize_pcm", new=AsyncMock(return_value=[])
    ):
        await session._process_speech(pcm, 48000)
        await session._pending_response_task

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


@pytest.mark.asyncio
async def test_process_speech_ignores_standalone_stt_hallucination():
    pcm = MagicMock()
    pcm.__len__.return_value = 48000

    app = SimpleNamespace(config={}, llm=SimpleNamespace(audio_model_info=MagicMock(return_value=None)))
    client = SimpleNamespace(room_typing=AsyncMock())
    agent = MagicMock()
    send_cb = AsyncMock()
    session = CallSession(
        call_id="call-hallucination",
        room_id="!room:test",
        caller_id="@user:test",
        thread_id="thread-hallucination",
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
        is_playing=False,
    )

    with patch("pawlia.transcription.transcribe_pcm", new=AsyncMock(return_value="Vielen Dank.")):
        await session._process_speech(pcm, 48000)

    send_cb.assert_not_awaited()
    agent.run_streamed.assert_not_called()
    session._tts_track.stop_hold.assert_called_once()


def test_standalone_stt_hallucination_filter_keeps_real_sentences():
    session = CallSession(
        call_id="call-hallucination-filter",
        room_id="!room:test",
        caller_id="@user:test",
        thread_id="thread-hallucination-filter",
        client=SimpleNamespace(),
        app=SimpleNamespace(config={}, llm=SimpleNamespace(audio_model_info=MagicMock(return_value=None))),
        cfg={},
        agent=MagicMock(),
        send_cb=AsyncMock(),
    )

    assert session._speech_detector.looks_like_stt_hallucination("Vielen Dank.") is True
    assert session._speech_detector.looks_like_stt_hallucination("Untertitelung des ZDF, 2020") is True
    assert session._speech_detector.looks_like_stt_hallucination("Ja, danke, das meinte ich.") is False
    assert session._speech_detector.looks_like_stt_hallucination("Ja, du, tschüss.") is False


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

    assert session._speech_detector.should_transcribe(pcm, 48000, fps=50) is False


def test_should_transcribe_chunk_accepts_sustained_speech():
    with patch.object(audio_vad, "_build_webrtc_vad", return_value=_FakeVad([False] * 10 + [True] * 30 + [False] * 40)):
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

        assert session._speech_detector.should_transcribe(pcm, 48000, fps=50) is True


def test_should_transcribe_chunk_rejects_broadband_noise():
    with patch.object(audio_vad, "_build_webrtc_vad", return_value=_FakeVad([False] * 100)):
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

        assert session._speech_detector.should_transcribe(pcm, 48000, fps=50) is False


def test_should_transcribe_chunk_rejects_when_webrtcvad_disagrees():
    with patch.object(audio_vad, "_build_webrtc_vad", return_value=_FakeVad([False] * 100)):
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

        assert session._speech_detector.should_transcribe(pcm, 48000, fps=50) is False


def test_live_frame_filter_rejects_broadband_wind_noise():
    session = CallSession(
        call_id="call-live-wind",
        room_id="!room:test",
        caller_id="@user:test",
        thread_id="thread-live-wind",
        client=SimpleNamespace(),
        app=SimpleNamespace(config={}, llm=SimpleNamespace(audio_model_info=MagicMock(return_value=None))),
        cfg={},
        agent=MagicMock(),
        send_cb=AsyncMock(),
    )

    rng = np.random.default_rng(9)
    pcm = rng.normal(0.0, 0.08, 960).astype(np.float32)

    assert session._speech_detector.is_speech_like_frame(pcm, 48000, adjusted_rms=0.08) is False


def test_live_frame_filter_accepts_speech_band_tone():
    session = CallSession(
        call_id="call-live-speech",
        room_id="!room:test",
        caller_id="@user:test",
        thread_id="thread-live-speech",
        client=SimpleNamespace(),
        app=SimpleNamespace(config={}, llm=SimpleNamespace(audio_model_info=MagicMock(return_value=None))),
        cfg={},
        agent=MagicMock(),
        send_cb=AsyncMock(),
    )

    pcm = _make_tonal_pcm_from_frame_levels([0.08], freq_hz=220.0)

    assert session._speech_detector.is_speech_like_frame(pcm, 48000, adjusted_rms=0.08) is True


def test_resume_speech_after_pause_requires_consecutive_frames():
    session = CallSession(
        call_id="call-resume-gate",
        room_id="!room:test",
        caller_id="@user:test",
        thread_id="thread-resume-gate",
        client=SimpleNamespace(),
        app=SimpleNamespace(config={}, llm=SimpleNamespace(audio_model_info=MagicMock(return_value=None))),
        cfg={},
        agent=MagicMock(),
        send_cb=AsyncMock(),
    )

    resumed, count = session._speech_detector.resume_after_pause(True, silence_count=10, resume_speech_count=0)
    assert resumed is False
    assert count == 1

    resumed, count = session._speech_detector.resume_after_pause(True, silence_count=10, resume_speech_count=count)
    assert resumed is False
    assert count == 2

    resumed, count = session._speech_detector.resume_after_pause(True, silence_count=10, resume_speech_count=count)
    assert resumed is True
    assert count == 3

    resumed, count = session._speech_detector.resume_after_pause(False, silence_count=10, resume_speech_count=2)
    assert resumed is False
    assert count == 0


def test_start_speech_buffer_includes_pre_speech_frames():
    session = CallSession(
        call_id="call-pre-roll",
        room_id="!room:test",
        caller_id="@user:test",
        thread_id="thread-pre-roll",
        client=SimpleNamespace(),
        app=SimpleNamespace(config={}, llm=SimpleNamespace(audio_model_info=MagicMock(return_value=None))),
        cfg={},
        agent=MagicMock(),
        send_cb=AsyncMock(),
    )

    pre = matrix_call.deque(
        [
            np.array([0.1, 0.2], dtype=np.float32),
            np.array([0.3, 0.4], dtype=np.float32),
        ]
    )
    trigger = np.array([0.5, 0.6], dtype=np.float32)

    chunk = SpeechDetector.start_buffer(pre, trigger)

    assert len(chunk) == 3
    assert np.array_equal(chunk[0], pre[0])
    assert np.array_equal(chunk[1], pre[1])
    assert np.array_equal(chunk[2], trigger)


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

    assert session._speech_detector.is_meaningful_interrupt("warte kurz") is True
    assert session._speech_detector.is_meaningful_interrupt("Kannst du kurz anhalten?") is True
    assert session._speech_detector.is_meaningful_interrupt("Ich bin gleich da.") is True


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

    assert session._speech_detector.is_meaningful_interrupt("hm") is False
    assert session._speech_detector.is_meaningful_interrupt("ja") is False
    assert session._speech_detector.is_meaningful_interrupt("fahrrad wind") is False


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
        is_playing=False,
    )
    session._cancel_active_response = AsyncMock()
    session._respond_to_transcript = AsyncMock()

    with patch.object(session, "_transcribe_speech", new=AsyncMock(return_value="hm")):
        await session._process_speech(np.zeros(4800, dtype=np.float32), 48000, interrupt_playback=True)

    session._tts_track.interrupt.assert_not_called()
    session._tts_track.stop_after_current_sentence.assert_not_called()
    session._cancel_active_response.assert_not_awaited()
    session._respond_to_transcript.assert_not_awaited()
    send_cb.assert_awaited_once_with("~~🎙️ *hm*~~ *(verworfen)*")


@pytest.mark.asyncio
async def test_process_speech_starts_hold_before_queueing_response():
    app = SimpleNamespace(config={}, llm=SimpleNamespace(audio_model_info=MagicMock(return_value=None)))
    client = SimpleNamespace(room_typing=AsyncMock())
    session = CallSession(
        call_id="call-hold-start",
        room_id="!room:test",
        caller_id="@user:test",
        thread_id="thread-hold-start",
        client=client,
        app=app,
        cfg={},
        agent=MagicMock(),
        send_cb=AsyncMock(),
    )

    session._tts_track = SimpleNamespace(
        start_hold=MagicMock(),
        stop_hold=MagicMock(),
        is_playing=False,
    )
    session._queue_transcript_response = AsyncMock()

    with patch.object(session, "_transcribe_speech", new=AsyncMock(return_value="Hallo da")):
        await session._process_speech(np.zeros(4800, dtype=np.float32), 48000)

    session._tts_track.start_hold.assert_called_once()
    session._queue_transcript_response.assert_awaited_once_with("Hallo da")
    session._tts_track.stop_hold.assert_not_called()


@pytest.mark.asyncio
async def test_process_speech_stops_hold_when_transcription_is_empty():
    app = SimpleNamespace(config={}, llm=SimpleNamespace(audio_model_info=MagicMock(return_value=None)))
    client = SimpleNamespace(room_typing=AsyncMock())
    session = CallSession(
        call_id="call-hold-empty",
        room_id="!room:test",
        caller_id="@user:test",
        thread_id="thread-hold-empty",
        client=client,
        app=app,
        cfg={},
        agent=MagicMock(),
        send_cb=AsyncMock(),
    )

    session._tts_track = SimpleNamespace(
        start_hold=MagicMock(),
        stop_hold=MagicMock(),
        is_playing=False,
    )
    session._queue_transcript_response = AsyncMock()

    with patch.object(session, "_transcribe_speech", new=AsyncMock(return_value=None)):
        await session._process_speech(np.zeros(4800, dtype=np.float32), 48000)

    session._tts_track.start_hold.assert_called_once()
    session._tts_track.stop_hold.assert_called_once()
    session._queue_transcript_response.assert_not_awaited()


@pytest.mark.asyncio
async def test_delayed_pending_response_ignores_hold_audio_only_playback():
    app = SimpleNamespace(config={}, llm=SimpleNamespace(audio_model_info=MagicMock(return_value=None)))
    session = CallSession(
        call_id="call-hold-gating",
        room_id="!room:test",
        caller_id="@user:test",
        thread_id="thread-hold-gating",
        client=SimpleNamespace(room_typing=AsyncMock()),
        app=app,
        cfg={},
        agent=MagicMock(),
        send_cb=AsyncMock(),
    )

    session._tts_track = SimpleNamespace(is_tts_playing=False, is_playing=True)
    session._respond_to_transcript = AsyncMock()
    session._pending_transcripts = ["Hallo da"]
    session._last_user_speech_at = 0.0
    session._speaking = False
    session.RESPONSE_DELAY_SECONDS = 0.0

    with patch("pawlia.interfaces.matrix_call.time.monotonic", return_value=1.0):
        await session._delayed_pending_response()

    session._respond_to_transcript.assert_awaited_once_with("Hallo da", announce_transcript=False)
    assert session._pending_transcripts == []


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
        is_playing=False,
    )
    session._cancel_active_response = AsyncMock()
    session._respond_to_transcript = AsyncMock()
    session.RESPONSE_DELAY_SECONDS = 0.0

    with patch.object(session, "_transcribe_speech", new=AsyncMock(return_value="warte kurz")):
        await session._process_speech(np.zeros(4800, dtype=np.float32), 48000, interrupt_playback=True)
        await session._pending_response_task

    session._tts_track.interrupt.assert_not_called()
    session._tts_track.stop_after_current_sentence.assert_called_once()
    session._cancel_active_response.assert_awaited_once()
    session._respond_to_transcript.assert_awaited_once_with("warte kurz", announce_transcript=False)


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
    session._tts_track = SimpleNamespace(
        stop_after_current_sentence=MagicMock(),
        stop_hold=MagicMock(),
        is_playing=False,
    )
    session._respond_to_transcript = AsyncMock()
    session.RESPONSE_DELAY_SECONDS = 0.0

    try:
        with patch.object(session, "_transcribe_speech", new=AsyncMock(return_value="warte kurz")):
            await session._process_speech(np.zeros(4800, dtype=np.float32), 48000, interrupt_playback=True)
            await session._pending_response_task
    finally:
        if not previous_response.done():
            previous_response.cancel()

    assert previous_response.cancelled()
    session._tts_track.stop_after_current_sentence.assert_called_once()
    session._respond_to_transcript.assert_awaited_once_with("warte kurz", announce_transcript=False)


@pytest.mark.asyncio
async def test_transcripts_are_debounced_while_user_keeps_speaking():
    session = CallSession(
        call_id="call-debounce",
        room_id="!room:test",
        caller_id="@user:test",
        thread_id="thread-debounce",
        client=SimpleNamespace(),
        app=SimpleNamespace(config={}, llm=SimpleNamespace(audio_model_info=MagicMock(return_value=None))),
        cfg={},
        agent=MagicMock(),
        send_cb=AsyncMock(),
    )

    session.RESPONSE_DELAY_SECONDS = 0.01
    session._respond_to_transcript = AsyncMock()
    session._speaking = True

    await session._queue_transcript_response("erster teil")
    await asyncio.sleep(0.03)

    session._respond_to_transcript.assert_not_awaited()

    session._mark_user_speech_ended()
    await session._pending_response_task

    session._respond_to_transcript.assert_awaited_once_with("erster teil", announce_transcript=False)


@pytest.mark.asyncio
async def test_new_transcript_restarts_response_debounce_and_combines_context():
    session = CallSession(
        call_id="call-debounce-combine",
        room_id="!room:test",
        caller_id="@user:test",
        thread_id="thread-debounce-combine",
        client=SimpleNamespace(),
        app=SimpleNamespace(config={}, llm=SimpleNamespace(audio_model_info=MagicMock(return_value=None))),
        cfg={},
        agent=MagicMock(),
        send_cb=AsyncMock(),
    )

    session.RESPONSE_DELAY_SECONDS = 0.05
    session._respond_to_transcript = AsyncMock()
    session._mark_user_speech_ended()

    await session._queue_transcript_response("erster teil")
    first_task = session._pending_response_task
    await session._queue_transcript_response("zweiter teil")
    await asyncio.sleep(0)
    assert first_task.cancelled() or first_task.done()
    await session._pending_response_task

    session._respond_to_transcript.assert_awaited_once_with(
        "erster teil\nzweiter teil",
        announce_transcript=False,
    )


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
    with patch.object(audio_vad, "_build_webrtc_vad", return_value=_FakeVad([])):
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
                    "min_resume_speech_frames": 5,
                    "pre_speech_seconds": 0.8,
                    "webrtcvad_enabled": True,
                    "webrtcvad_mode": 3,
                    "webrtcvad_min_voiced_ratio": 0.22,
                    "webrtcvad_min_consecutive_frames": 5,
                    "call_inactivity_seconds": 240,
                    "preanswer_warmup_enabled": False,
                    "preanswer_warmup_timeout_seconds": 7.5,
                    "preanswer_stt_silence_seconds": 0.2,
                    "response_delay_seconds": 3.5,
                    "connect_timeout_seconds": 12.0,
                    "hangup_on_media_end": False,
                }
            }),
            cfg={},
            agent=MagicMock(),
            send_cb=AsyncMock(),
        )
        sd = session._speech_detector
        assert sd.SILENCE_THRESHOLD == pytest.approx(0.03)
        assert sd.SILENCE_SECONDS == pytest.approx(2.2)
        assert sd.MIN_SPEECH_SECONDS == pytest.approx(0.7)
        assert sd.MIN_ACTIVE_SPEECH_RATIO == pytest.approx(0.25)
        assert sd.MIN_CONSECUTIVE_SPEECH_FRAMES == 11
        assert sd.MIN_SPEECH_BAND_RATIO == pytest.approx(0.42)
        assert sd.MAX_SPECTRAL_FLATNESS == pytest.approx(0.61)
        assert sd.MIN_SPEECH_LIKE_RATIO == pytest.approx(0.14)
        assert sd.MIN_CONSECUTIVE_SPEECHLIKE_FRAMES == 6
        assert sd.MIN_RESUME_SPEECH_FRAMES == 5
        assert sd.PRE_SPEECH_SECONDS == pytest.approx(0.8)
        assert sd.WEBRTC_VAD_ENABLED is True
        assert sd.WEBRTC_VAD_MODE == 3
        assert sd.WEBRTC_VAD_MIN_VOICED_RATIO == pytest.approx(0.22)
        assert sd.WEBRTC_VAD_MIN_CONSECUTIVE_FRAMES == 5
        assert session.CALL_INACTIVITY_SECONDS == 240
        assert session.PREANSWER_WARMUP_ENABLED is False
        assert session.PREANSWER_WARMUP_TIMEOUT_SECONDS == pytest.approx(7.5)
        assert session.PREANSWER_STT_SILENCE_SECONDS == pytest.approx(0.2)
        assert session.RESPONSE_DELAY_SECONDS == pytest.approx(3.5)
        assert session.CONNECT_TIMEOUT_SECONDS == pytest.approx(12.0)
        assert session.HANGUP_ON_MEDIA_END is False


def test_call_session_invalid_voip_audio_thresholds_fall_back_to_defaults():
    with patch.object(audio_vad, "_build_webrtc_vad", return_value=_FakeVad([])):
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
                    "min_resume_speech_frames": 0,
                    "pre_speech_seconds": -1,
                    "webrtcvad_enabled": "maybe",
                    "webrtcvad_mode": 99,
                    "webrtcvad_min_voiced_ratio": 2,
                    "webrtcvad_min_consecutive_frames": 0,
                    "call_inactivity_seconds": 0,
                    "preanswer_warmup_enabled": "perhaps",
                    "preanswer_warmup_timeout_seconds": 0,
                    "preanswer_stt_silence_seconds": 0,
                    "response_delay_seconds": -1,
                    "connect_timeout_seconds": 0,
                    "hangup_on_media_end": "perhaps",
                }
            }),
            cfg={},
            agent=MagicMock(),
            send_cb=AsyncMock(),
        )
        sd = session._speech_detector
        assert sd.SILENCE_THRESHOLD == pytest.approx(SpeechDetector.SILENCE_THRESHOLD)
        assert sd.SILENCE_SECONDS == pytest.approx(SpeechDetector.SILENCE_SECONDS)
        assert sd.MIN_SPEECH_SECONDS == pytest.approx(SpeechDetector.MIN_SPEECH_SECONDS)
        assert sd.MIN_ACTIVE_SPEECH_RATIO == pytest.approx(SpeechDetector.MIN_ACTIVE_SPEECH_RATIO)
        assert sd.MIN_CONSECUTIVE_SPEECH_FRAMES == SpeechDetector.MIN_CONSECUTIVE_SPEECH_FRAMES
        assert sd.MIN_SPEECH_BAND_RATIO == pytest.approx(SpeechDetector.MIN_SPEECH_BAND_RATIO)
        assert sd.MAX_SPECTRAL_FLATNESS == pytest.approx(SpeechDetector.MAX_SPECTRAL_FLATNESS)
        assert sd.MIN_SPEECH_LIKE_RATIO == pytest.approx(SpeechDetector.MIN_SPEECH_LIKE_RATIO)
        assert sd.MIN_CONSECUTIVE_SPEECHLIKE_FRAMES == SpeechDetector.MIN_CONSECUTIVE_SPEECHLIKE_FRAMES
        assert sd.MIN_RESUME_SPEECH_FRAMES == SpeechDetector.MIN_RESUME_SPEECH_FRAMES
        assert sd.PRE_SPEECH_SECONDS == pytest.approx(SpeechDetector.PRE_SPEECH_SECONDS)
        assert sd.WEBRTC_VAD_ENABLED == SpeechDetector.WEBRTC_VAD_ENABLED
        assert sd.WEBRTC_VAD_MODE == SpeechDetector.WEBRTC_VAD_MODE
        assert sd.WEBRTC_VAD_MIN_VOICED_RATIO == pytest.approx(SpeechDetector.WEBRTC_VAD_MIN_VOICED_RATIO)
        assert sd.WEBRTC_VAD_MIN_CONSECUTIVE_FRAMES == SpeechDetector.WEBRTC_VAD_MIN_CONSECUTIVE_FRAMES
        assert session.CALL_INACTIVITY_SECONDS == CallSession.CALL_INACTIVITY_SECONDS
        assert session.PREANSWER_WARMUP_ENABLED == CallSession.PREANSWER_WARMUP_ENABLED
        assert session.PREANSWER_WARMUP_TIMEOUT_SECONDS == pytest.approx(CallSession.PREANSWER_WARMUP_TIMEOUT_SECONDS)
        assert session.PREANSWER_STT_SILENCE_SECONDS == pytest.approx(CallSession.PREANSWER_STT_SILENCE_SECONDS)
        assert session.RESPONSE_DELAY_SECONDS == CallSession.RESPONSE_DELAY_SECONDS
        assert session.CONNECT_TIMEOUT_SECONDS == CallSession.CONNECT_TIMEOUT_SECONDS
        assert session.HANGUP_ON_MEDIA_END == CallSession.HANGUP_ON_MEDIA_END


@pytest.mark.asyncio
async def test_connect_timeout_watchdog_hangs_up_stale_call():
    client = SimpleNamespace(room_send=AsyncMock())
    send_cb = AsyncMock()
    session = CallSession(
        call_id="call-connect-timeout",
        room_id="!room:test",
        caller_id="@user:test",
        thread_id="thread-connect-timeout",
        client=client,
        app=SimpleNamespace(config={"voip": {"connect_timeout_seconds": 0.01}}, llm=SimpleNamespace(audio_model_info=MagicMock(return_value=None))),
        cfg={},
        agent=MagicMock(),
        send_cb=send_cb,
    )
    session._pc = SimpleNamespace(
        connectionState="connecting",
        iceConnectionState="checking",
        close=AsyncMock(),
    )
    session._answer_sent.set()

    await session._connect_timeout_watchdog()

    send_cb.assert_any_call("📞 Verbindung unterbrochen")
    client.room_send.assert_awaited_once()
    session._pc.close.assert_awaited_once()
    assert session.finished is True


@pytest.mark.asyncio
async def test_preanswer_warmup_prepares_greeting_without_playing_it():
    app = SimpleNamespace(config={}, llm=SimpleNamespace(audio_model_info=MagicMock(return_value=None)))
    client = SimpleNamespace(room_typing=AsyncMock())
    agent = MagicMock()
    agent.build_system_prompt.return_value = "CALL PROMPT"
    agent.run_streamed = AsyncMock(return_value="Hallo")
    send_cb = AsyncMock()
    session = CallSession(
        call_id="call-warmup",
        room_id="!room:test",
        caller_id="@user:test",
        thread_id="thread-warmup",
        client=client,
        app=app,
        cfg={},
        agent=agent,
        send_cb=send_cb,
    )
    session._tts_track = SimpleNamespace(enqueue_pcm_float32=MagicMock(), stop_hold=MagicMock())

    with patch("pawlia.transcription.transcribe_pcm", new=AsyncMock(return_value=None)) as transcribe_pcm, \
         patch("pawlia.tts.synthesize_pcm", new=AsyncMock(return_value=np.ones(480, dtype=np.float32))):
        await session._run_preanswer_warmup()

    transcribe_pcm.assert_awaited_once()
    agent.run_streamed.assert_awaited_once()
    send_cb.assert_not_awaited()
    session._tts_track.enqueue_pcm_float32.assert_not_called()
    assert session._greeting_sent is False
    assert session._prepared_greeting is not None


@pytest.mark.asyncio
async def test_preanswer_warmup_timeout_keeps_greeting_warmup_running():
    app = SimpleNamespace(config={}, llm=SimpleNamespace(audio_model_info=MagicMock(return_value=None)))
    agent = MagicMock()
    agent.build_system_prompt.return_value = "CALL PROMPT"

    greeting_ready = asyncio.Event()

    async def _slow_run_streamed(*args, on_sentence=None, **kwargs):
        await greeting_ready.wait()
        if on_sentence:
            await on_sentence("Hallo, ich bin da.")
        return "Hallo, ich bin da."

    agent.run_streamed = AsyncMock(side_effect=_slow_run_streamed)
    send_cb = AsyncMock()
    session = CallSession(
        call_id="call-warmup-timeout",
        room_id="!room:test",
        caller_id="@user:test",
        thread_id="thread-warmup-timeout",
        client=SimpleNamespace(),
        app=app,
        cfg={},
        agent=agent,
        send_cb=send_cb,
    )
    session.PREANSWER_WARMUP_TIMEOUT_SECONDS = 0.01
    session._tts_track = SimpleNamespace(enqueue_pcm_float32=MagicMock(), stop_hold=MagicMock())

    with patch("pawlia.transcription.transcribe_pcm", new=AsyncMock(return_value=None)), patch(
        "pawlia.tts.synthesize_pcm", new=AsyncMock(return_value=np.ones(480, dtype=np.float32))
    ):
        await session._run_preanswer_warmup()
        assert session._prepared_greeting is None
        assert session._prepare_greeting_task is not None
        assert session._prepare_greeting_task.done() is False

        greeting_ready.set()
        await session._prepare_greeting_task

    assert session._prepared_greeting is not None
    agent.run_streamed.assert_awaited_once()


@pytest.mark.asyncio
async def test_deferred_greeting_waits_for_answer_and_media_ready():
    app = SimpleNamespace(config={}, llm=SimpleNamespace(audio_model_info=MagicMock(return_value=None)))
    agent = MagicMock()
    agent.build_system_prompt.return_value = "CALL PROMPT"

    async def _fake_run_streamed(*args, on_sentence=None, **kwargs):
        if on_sentence:
            await on_sentence("Hallo, ich bin da.")
        return "Hallo, ich bin da."

    agent.run_streamed = AsyncMock(side_effect=_fake_run_streamed)
    send_cb = AsyncMock()
    session = CallSession(
        call_id="call-deferred-greeting",
        room_id="!room:test",
        caller_id="@user:test",
        thread_id="thread-deferred-greeting",
        client=SimpleNamespace(),
        app=app,
        cfg={},
        agent=agent,
        send_cb=send_cb,
    )
    session._tts_track = SimpleNamespace(enqueue_pcm_float32=MagicMock(), stop_hold=MagicMock())

    with patch("pawlia.tts.synthesize_pcm", new=AsyncMock(return_value=np.ones(480, dtype=np.float32))):
        task = asyncio.create_task(session._send_greeting_when_ready())
        await asyncio.sleep(0)
        agent.run_streamed.assert_not_awaited()

        await session.mark_answer_sent()
        await asyncio.sleep(0)
        agent.run_streamed.assert_not_awaited()

        session._media_connected.set()
        await task

    agent.run_streamed.assert_awaited_once()
    send_cb.assert_any_call("Hallo, ich bin da.")
    session._tts_track.enqueue_pcm_float32.assert_called_once()
    assert session._greeting_sent is True


@pytest.mark.asyncio
async def test_send_greeting_waits_for_existing_prepare_task():
    send_cb = AsyncMock()
    session = CallSession(
        call_id="call-await-prepared-greeting",
        room_id="!room:test",
        caller_id="@user:test",
        thread_id="thread-await-prepared-greeting",
        client=SimpleNamespace(),
        app=SimpleNamespace(config={}, llm=SimpleNamespace(audio_model_info=MagicMock(return_value=None))),
        cfg={},
        agent=MagicMock(),
        send_cb=send_cb,
    )
    pcm = np.ones(480, dtype=np.float32)
    session._tts_track = SimpleNamespace(enqueue_pcm_float32=MagicMock(), stop_hold=MagicMock())

    async def _prepare_later():
        await asyncio.sleep(0)
        session._prepared_greeting = ("Hallo, ich bin da.", [pcm])

    session._prepare_greeting_task = asyncio.create_task(_prepare_later())

    await session._send_greeting()

    session._tts_track.enqueue_pcm_float32.assert_called_once_with(pcm)
    session._tts_track.stop_hold.assert_called_once()
    send_cb.assert_any_call("Hallo, ich bin da.")
    assert session._greeting_sent is True


@pytest.mark.asyncio
async def test_deferred_greeting_plays_prepared_audio_when_ready():
    send_cb = AsyncMock()
    session = CallSession(
        call_id="call-prepared-greeting",
        room_id="!room:test",
        caller_id="@user:test",
        thread_id="thread-prepared-greeting",
        client=SimpleNamespace(),
        app=SimpleNamespace(config={}, llm=SimpleNamespace(audio_model_info=MagicMock(return_value=None))),
        cfg={},
        agent=MagicMock(),
        send_cb=send_cb,
    )
    pcm = np.ones(480, dtype=np.float32)
    session._prepared_greeting = ("Hallo, ich bin da.", [pcm])
    session._tts_track = SimpleNamespace(enqueue_pcm_float32=MagicMock(), stop_hold=MagicMock())

    task = asyncio.create_task(session._send_greeting_when_ready())
    await asyncio.sleep(0)
    session._tts_track.enqueue_pcm_float32.assert_not_called()

    await session.mark_answer_sent()
    session._media_connected.set()
    await task

    session._tts_track.enqueue_pcm_float32.assert_called_once_with(pcm)
    session._tts_track.stop_hold.assert_called_once()
    send_cb.assert_any_call("Hallo, ich bin da.")
    assert session._greeting_sent is True


@pytest.mark.asyncio
async def test_send_greeting_is_noop_after_preanswer_greeting():
    app = SimpleNamespace(config={}, llm=SimpleNamespace(audio_model_info=MagicMock(return_value=None)))
    agent = MagicMock()
    agent.run_streamed = AsyncMock(return_value="Hallo")
    session = CallSession(
        call_id="call-greeting-once",
        room_id="!room:test",
        caller_id="@user:test",
        thread_id="thread-greeting-once",
        client=SimpleNamespace(),
        app=app,
        cfg={},
        agent=agent,
        send_cb=AsyncMock(),
    )
    session._greeting_sent = True

    await session._send_greeting()

    agent.run_streamed.assert_not_awaited()


@pytest.mark.asyncio
async def test_send_greeting_drops_tts_after_hangup():
    """If the call hangs up before the LLM finishes, the greeting must not be
    enqueued onto a dead TTS track or posted to the room."""
    app = SimpleNamespace(config={}, llm=SimpleNamespace(audio_model_info=MagicMock(return_value=None)))
    agent = MagicMock()
    agent.build_system_prompt.return_value = "CALL PROMPT"

    send_cb = AsyncMock()
    enqueue = MagicMock()
    stop_hold = MagicMock()

    session = CallSession(
        call_id="call-hangup-mid-greeting",
        room_id="!room:test",
        caller_id="@user:test",
        thread_id="thread-hangup",
        client=SimpleNamespace(),
        app=app,
        cfg={},
        agent=agent,
        send_cb=send_cb,
    )
    session._tts_track = SimpleNamespace(enqueue_pcm_float32=enqueue, stop_hold=stop_hold)

    async def _fake_run_streamed(*args, on_sentence=None, **kwargs):
        # Simulate the call hanging up *during* greeting generation, before
        # any sentence callback fires.
        session._hungup = True
        session._done.set()
        if on_sentence:
            await on_sentence("Hallo, ich bin da.")
        return "Hallo, ich bin da."

    agent.run_streamed = AsyncMock(side_effect=_fake_run_streamed)

    with patch("pawlia.tts.synthesize_pcm", new=AsyncMock(return_value=np.ones(480, dtype=np.float32))):
        await session._send_greeting()

    enqueue.assert_not_called()
    stop_hold.assert_not_called()
    send_cb.assert_not_awaited()
    assert session._greeting_sent is False


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


def test_should_transcribe_chunk_relaxed_thresholds_in_quiet_env():
    """In very quiet environments (noise_floor < 0.001), relaxed thresholds
    should accept chunks that would be rejected with the default thresholds."""
    with patch.object(audio_vad, "_build_webrtc_vad", return_value=None):
        session = CallSession(
            call_id="call-quiet",
            room_id="!room:test",
            caller_id="@user:test",
            thread_id="thread-quiet",
            client=SimpleNamespace(),
            app=SimpleNamespace(config={}, llm=SimpleNamespace(audio_model_info=MagicMock(return_value=None))),
            cfg={},
            agent=MagicMock(),
            send_cb=AsyncMock(),
        )

        # Simulate a very quiet environment
        session._speech_detector._noise_floor = 0.0005

        # Create a chunk with low active_ratio (0.08) — would fail default 0.12
        # but should pass with relaxed threshold 0.06
        levels = [0.0] * 80
        # 8 active frames out of 80 = 10% active_ratio (between 0.06 and 0.12)
        for idx in range(8):
            levels[idx] = 0.15
        pcm = _make_tonal_pcm_from_frame_levels(levels)

        # With relaxed thresholds, this should be accepted
        assert session._speech_detector.should_transcribe(pcm, 48000, fps=50) is True


def test_should_transcribe_chunk_strict_thresholds_in_noisy_env():
    """In noisy environments (noise_floor >= 0.001), default strict thresholds
    should reject chunks with low active_ratio."""
    with patch.object(audio_vad, "_build_webrtc_vad", return_value=None):
        session = CallSession(
            call_id="call-noisy",
            room_id="!room:test",
            caller_id="@user:test",
            thread_id="thread-noisy",
            client=SimpleNamespace(),
            app=SimpleNamespace(config={}, llm=SimpleNamespace(audio_model_info=MagicMock(return_value=None))),
            cfg={},
            agent=MagicMock(),
            send_cb=AsyncMock(),
        )

        # Simulate a noisier environment
        session._speech_detector._noise_floor = 0.005

        # Same chunk as above — 10% active_ratio
        levels = [0.0] * 80
        for idx in range(8):
            levels[idx] = 0.15
        pcm = _make_tonal_pcm_from_frame_levels(levels)

        # With strict thresholds (0.12), this should be rejected
        assert session._speech_detector.should_transcribe(pcm, 48000, fps=50) is False

