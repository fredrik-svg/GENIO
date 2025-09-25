import sys
import types

import numpy as np
import pytest


class DummyRag:
    used_rag = False
    answer = "svar"

    def context_payload(self):
        return "[]"

    def contexts_for_client(self):
        return []


def _prepare_common_flow(monkeypatch, tmp_path, *, selected_device):
    if "multipart" not in sys.modules:
        multipart = types.ModuleType("multipart")
        multipart.__version__ = "0"
        sys.modules["multipart"] = multipart
        multipart_submodule = types.ModuleType("multipart.multipart")

        def _parse_options_header(value):  # pragma: no cover - tiny stub
            return value, {}

        multipart_submodule.parse_options_header = _parse_options_header
        sys.modules["multipart.multipart"] = multipart_submodule

    from backend import app

    notifications = []

    async def fake_notify(message):
        notifications.append(message)

    monkeypatch.setattr(app, "notify", fake_notify)
    monkeypatch.setattr(
        app,
        "record_until_silence",
        lambda: np.array([0.1, -0.2, 0.3], dtype=np.float32),
    )
    monkeypatch.setattr(app, "save_wav_mono16", lambda buf, audio: buf.write(b"pcm"))

    async def fake_stt(wav_bytes, *, language="sv"):
        return "hej"

    monkeypatch.setattr(app, "stt_transcribe_wav", fake_stt)

    async def fake_rag(text):
        return DummyRag()

    monkeypatch.setattr(app, "rag_answer", fake_rag)

    async def fake_tts(text):
        return b"raw-wav"

    monkeypatch.setattr(app, "tts_speak_sv", fake_tts)
    monkeypatch.setattr(app, "ensure_wav_pcm16", lambda data: b"final-wav")
    monkeypatch.setattr(app, "OUTPUT_WAV_PATH", tmp_path / "reply.wav")
    monkeypatch.setattr(app, "PLAY_CMD", "aplay -q")
    monkeypatch.setattr(app, "get_selected_output_device", lambda: selected_device)

    play_calls = []

    def fake_play_wav_bytes(data):
        play_calls.append(data)

    monkeypatch.setattr(app, "play_wav_bytes", fake_play_wav_bytes)

    run_calls = []

    def fake_run(*args, **kwargs):
        run_calls.append((args, kwargs))

    monkeypatch.setattr(app.subprocess, "run", fake_run)

    return app, notifications, play_calls, run_calls


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio("asyncio")
async def test_full_converse_flow_skips_play_cmd_when_device_selected(monkeypatch, tmp_path):
    app, notifications, play_calls, run_calls = _prepare_common_flow(
        monkeypatch, tmp_path, selected_device=7
    )

    result = await app.full_converse_flow(trigger="test")

    assert result["ok"] is True
    assert run_calls == []
    assert play_calls == [b"final-wav"]
    assert any("Lyssnar" in msg for msg in notifications)


@pytest.mark.anyio("asyncio")
async def test_full_converse_flow_uses_play_cmd_without_selected_device(
    monkeypatch, tmp_path
):
    app, notifications, play_calls, run_calls = _prepare_common_flow(
        monkeypatch, tmp_path, selected_device=None
    )

    result = await app.full_converse_flow(trigger="test")

    assert result["ok"] is True
    assert run_calls, "play command should be invoked when no device is selected"
    # ensure sounddevice fallback not used when command succeeds
    assert play_calls == []
    assert any("Lyssnar" in msg for msg in notifications)
