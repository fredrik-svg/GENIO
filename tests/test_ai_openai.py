import base64
import io
import json
import wave

import pytest

pytest.importorskip("httpx")

from backend import ai


@pytest.fixture
def anyio_backend():
    return "asyncio"


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


@pytest.mark.anyio("asyncio")
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
    request_json = recorder.post_calls[0]["json"]
    assert recorder.post_calls[0]["url"].endswith("/responses")
    assert request_json["model"] == "model"
    assert request_json["audio"] == {"voice": "alloy", "format": "wav"}
    assert request_json["modalities"] == ["audio"]
    assert request_json["input"] == [
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "hej"}],
        }
    ]


@pytest.mark.anyio("asyncio")
async def test_openai_synthesize_accepts_data_uri_audio(monkeypatch):
    wav_bytes = _make_wav_bytes()
    payload = {
        "audio": "data:audio/wav;base64," + base64.b64encode(wav_bytes).decode("ascii")
    }
    response = _DummyResponse(payload, headers={"content-type": "application/json"})
    recorder = _ClientRecorder(response)

    monkeypatch.setattr(ai.httpx, "AsyncClient", lambda *args, **kwargs: recorder)

    provider = ai.OpenAIProvider(
        api_key="key",
        base_url="https://api.example.com",
        tts_model="model",
        tts_voice="alloy",
    )

    audio_bytes = await provider.synthesize("hallå")

    assert audio_bytes == wav_bytes


@pytest.mark.anyio("asyncio")
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


@pytest.mark.anyio("asyncio")
async def test_openai_synthesize_supports_responses_audio_structure(monkeypatch):
    wav_bytes = _make_wav_bytes()
    payload = {
        "output": [
            {
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "Hej!"},
                    {
                        "type": "output_audio",
                        "audio": {
                            "id": "audio_0",
                            "format": "wav",
                            "data": base64.b64encode(wav_bytes).decode("ascii"),
                        },
                    },
                ],
            }
        ]
    }
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


@pytest.mark.anyio("asyncio")
async def test_openai_synthesize_joins_chunked_audio_payload(monkeypatch):
    wav_bytes = _make_wav_bytes()
    first_chunk = wav_bytes[: len(wav_bytes) // 2]
    second_chunk = wav_bytes[len(wav_bytes) // 2 :]
    payload = {
        "output": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "output_audio",
                        "audio": {
                            "format": "wav",
                            "data": [
                                {"index": 0, "data": base64.b64encode(first_chunk).decode("ascii")},
                                {"index": 1, "data": base64.b64encode(second_chunk).decode("ascii")},
                            ],
                        },
                    }
                ],
            }
        ]
    }
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
