from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
import io
import logging
import sys

import numpy as np
import pytest

import pawlia.interfaces.matrix_call as matrix_call
from pawlia.interfaces.matrix_call import CallSession


@pytest.mark.asyncio
async def test_process_speech_uses_call_system_prompt():
    pcm = MagicMock()
    pcm.__len__.return_value = 48000

    app = SimpleNamespace(config={})
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

    agent.build_system_prompt.assert_called_once_with(mode="call")
    agent.run_streamed.assert_awaited_once_with(
        "Hallo da",
        system_prompt="CALL PROMPT",
        thread_id="thread-1",
        on_sentence=agent.run_streamed.await_args.kwargs["on_sentence"],
    )
    assert send_cb.await_args_list[0].args[0] == "🎙️ *Hallo da*"
    assert send_cb.await_args_list[1].args[0] == "Kurze Antwort"


@pytest.mark.asyncio
async def test_process_speech_writes_debug_wav(tmp_path):
    app = SimpleNamespace(config={})
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
        app=SimpleNamespace(config={}),
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