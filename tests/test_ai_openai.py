import base64
import io
import json
import wave

import pytest

pytest.importorskip("httpx")

from backend import ai


def _make_wav_bytes():
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)
        wav_file.writeframes(b"\x00\x01\x02\x03")
    return buf.getvalue()


class _DummyResponse:
    def __init__(self, payload, *, headers):
        self._payload = payload
        self.headers = headers

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload

    @property
    def content(self):
        return json.dumps(self._payload).encode()


class _ClientRecorder:
    def __init__(self, response):
        self._response = response
        self.post_calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, *, headers=None, json=None):
        self.post_calls.append({"url": url, "headers": headers, "json": json})
        return self._response


@pytest.mark.anyio
async def test_openai_synthesize_decodes_json_wav(monkeypatch):
    wav_bytes = _make_wav_bytes()
    payload = {"data": base64.b64encode(wav_bytes).decode("ascii")}
    response = _DummyResponse(payload, headers={"content-type": "application/json"})
    recorder = _ClientRecorder(response)

    monkeypatch.setattr(ai.httpx, "AsyncClient", lambda *args, **kwargs: recorder)

    provider = ai.OpenAIProvider(
        api_key="key",
        base_url="https://api.example.com",
        tts_model="model",
        tts_voice="alloy",
    )

    audio_bytes = await provider.synthesize("hej")

    assert audio_bytes == wav_bytes
    assert recorder.post_calls, "HTTP POST should have been invoked"
    assert recorder.post_calls[0]["url"].endswith("/audio/speech")
    assert recorder.post_calls[0]["json"]["input"] == "hej"


@pytest.mark.anyio
async def test_openai_synthesize_errors_on_missing_audio(monkeypatch):
    payload = {"data": "not-audio"}
    response = _DummyResponse(payload, headers={"content-type": "application/json"})
    recorder = _ClientRecorder(response)

    monkeypatch.setattr(ai.httpx, "AsyncClient", lambda *args, **kwargs: recorder)

    provider = ai.OpenAIProvider(
        api_key="key",
        base_url="https://api.example.com",
        tts_model="model",
        tts_voice="alloy",
    )

    with pytest.raises(RuntimeError):
        await provider.synthesize("hej")
