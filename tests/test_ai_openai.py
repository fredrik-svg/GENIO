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
    def __init__(self, payload, *, headers, binary_content=None, text_content=None):
        self._payload = payload
        self.headers = headers
        self._binary_content = binary_content
        self._text_content = text_content

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload

    @property
    def content(self):
        if self._binary_content is not None:
            return self._binary_content
        return json.dumps(self._payload).encode()

    @property
    def text(self):
        if self._text_content is not None:
            return self._text_content
        return json.dumps(self._payload)


class _ClientRecorder:
    def __init__(self, response):
        self._response = response
        self.post_calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, *, headers=None, json=None, files=None):
        self.post_calls.append({"url": url, "headers": headers, "json": json, "files": files})
        return self._response


@pytest.mark.anyio("asyncio")
async def test_openai_synthesize_uses_correct_tts_api(monkeypatch):
    """Test that the synthesize method uses the correct OpenAI TTS API endpoint and format."""
    wav_bytes = _make_wav_bytes()
    response = _DummyResponse(payload={}, headers={"content-type": "audio/wav"}, binary_content=wav_bytes)
    recorder = _ClientRecorder(response)

    monkeypatch.setattr(ai.httpx, "AsyncClient", lambda *args, **kwargs: recorder)

    provider = ai.OpenAIProvider(
        api_key="key",
        base_url="https://api.example.com",
        tts_model="tts-1",
        tts_voice="alloy",
    )

    audio_bytes = await provider.synthesize("hej")

    assert audio_bytes == wav_bytes
    assert recorder.post_calls, "HTTP POST should have been invoked"
    
    # Verify correct endpoint
    assert recorder.post_calls[0]["url"].endswith("/audio/speech")
    
    # Verify correct payload format
    request_json = recorder.post_calls[0]["json"]
    assert request_json["model"] == "tts-1"
    assert request_json["input"] == "hej"
    assert request_json["voice"] == "alloy"
    assert request_json["response_format"] == "wav"


@pytest.mark.anyio("asyncio")
async def test_openai_synthesize_error_handling(monkeypatch):
    """Test that synthesize properly handles HTTP errors."""
    from httpx import HTTPStatusError, Request, Response
    
    class _ErrorResponse:
        def __init__(self):
            self.status_code = 400
            self.text = "Bad Request: Invalid model"
        
        def raise_for_status(self):
            request = Request("POST", "https://api.openai.com/v1/audio/speech")
            response = Response(400, content=b"Bad Request: Invalid model")
            raise HTTPStatusError("400 Bad Request", request=request, response=response)
    
    class _ErrorClient:
        async def __aenter__(self):
            return self
        
        async def __aexit__(self, exc_type, exc, tb):
            return False
        
        async def post(self, url, **kwargs):
            return _ErrorResponse()
    
    monkeypatch.setattr(ai.httpx, "AsyncClient", lambda *args, **kwargs: _ErrorClient())
    
    provider = ai.OpenAIProvider(
        api_key="key",
        base_url="https://api.example.com",
        tts_model="tts-1",
        tts_voice="alloy",
    )
    
    with pytest.raises(RuntimeError, match=r"OpenAI Voice API-TTS misslyckades \(400\)"):
        await provider.synthesize("test")


@pytest.mark.anyio("asyncio")
async def test_openai_synthesize_maps_legacy_model_names(monkeypatch):
    """Test that legacy model names like gpt-4o-mini-tts are mapped to tts-1."""
    wav_bytes = _make_wav_bytes()
    response = _DummyResponse(payload={}, headers={"content-type": "audio/wav"}, binary_content=wav_bytes)
    recorder = _ClientRecorder(response)

    monkeypatch.setattr(ai.httpx, "AsyncClient", lambda *args, **kwargs: recorder)

    provider = ai.OpenAIProvider(
        api_key="key",
        base_url="https://api.example.com",
        tts_model="gpt-4o-mini-tts",
        tts_voice="alloy",
    )

    audio_bytes = await provider.synthesize("hej")

    assert audio_bytes == wav_bytes
    
    # Verify that gpt-4o-mini-tts was mapped to tts-1
    request_json = recorder.post_calls[0]["json"]
    assert request_json["model"] == "tts-1"


@pytest.mark.skip(reason="Legacy test for deprecated OpenAI Responses API format")
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
    assert request_json["input"] == [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "hej"},
            ],
        }
    ]


@pytest.mark.skip(reason="Legacy test for deprecated OpenAI Responses API format")
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


@pytest.mark.skip(reason="Legacy test for deprecated OpenAI Responses API format")
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


@pytest.mark.skip(reason="Legacy test for deprecated OpenAI Responses API format")
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


@pytest.mark.skip(reason="Legacy test for deprecated OpenAI Responses API format")
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


@pytest.mark.anyio("asyncio")
async def test_openai_chat_reply_uses_responses_api(monkeypatch):
    payload = {
        "output": [
            {
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "Hej!"},
                    {"type": "output_text", "text": "Hur kan jag hjälpa dig?"},
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
        chat_model="model",
    )

    reply = await provider.chat_reply(
        "Hur är vädret?",
        context_sections=["Det regnar i Göteborg."],
    )

    assert reply == "Hej!\nHur kan jag hjälpa dig?"
    assert recorder.post_calls, "Responses endpoint should have been invoked"
    request_json = recorder.post_calls[0]["json"]
    assert recorder.post_calls[0]["url"].endswith("/responses")
    assert request_json["model"] == "model"
    assert request_json["input"][0]["role"] == "system"
    assert request_json["input"][-1]["role"] == "user"
    assert request_json["input"][-1]["content"][0]["type"] == "input_text"
    # Context should be present in a system message
    assert any("Göteborg" in chunk["content"][0]["text"] for chunk in request_json["input"])


@pytest.mark.anyio("asyncio")
async def test_openai_chat_reply_falls_back_to_legacy(monkeypatch):
    call_args: dict[str, dict] = {}

    async def fake_post_responses(self, payload):
        call_args["payload"] = payload
        return {}

    monkeypatch.setattr(ai.OpenAIProvider, "_post_responses", fake_post_responses)

    fallback_payload = {
        "choices": [{"message": {"content": "Fallback-svar"}}]
    }
    response = _DummyResponse(fallback_payload, headers={"content-type": "application/json"})
    recorder = _ClientRecorder(response)

    monkeypatch.setattr(ai.httpx, "AsyncClient", lambda *args, **kwargs: recorder)

    provider = ai.OpenAIProvider(
        api_key="key",
        base_url="https://api.example.com",
        chat_model="model",
    )

    reply = await provider.chat_reply("Hej")

    assert reply == "Fallback-svar"
    assert "payload" in call_args
    assert call_args["payload"]["input"][0]["role"] == "system"
    assert call_args["payload"]["input"][-1]["role"] == "user"
    assert recorder.post_calls, "Fallback chat completions should have been invoked"
    assert recorder.post_calls[0]["url"].endswith("/chat/completions")


def test_extract_text_handles_multiple_shapes():
    payload = {
        "output": [
            {
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "Hej"},
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "Hur mår du?"},
                        ],
                    },
                ],
            }
        ],
        "tool_response": {
            "output_text": ["Allt är bra!"],
        },
    }

    text = ai._extract_text_from_json_payload(payload)

    assert text == "Hej\nHur mår du?\nAllt är bra!"


@pytest.mark.anyio("asyncio")
async def test_openai_transcribe_uses_correct_api(monkeypatch):
    """Test that the transcribe method uses the correct OpenAI transcriptions API endpoint."""
    response = _DummyResponse(
        payload={}, 
        headers={"content-type": "text/plain"}, 
        text_content="Hej, det här är ett test."
    )
    recorder = _ClientRecorder(response)

    monkeypatch.setattr(ai.httpx, "AsyncClient", lambda *args, **kwargs: recorder)

    provider = ai.OpenAIProvider(
        api_key="key",
        base_url="https://api.example.com",
        stt_model="whisper-1",
    )

    wav_bytes = _make_wav_bytes()
    transcript = await provider.transcribe(wav_bytes, language="sv")

    assert transcript == "Hej, det här är ett test."
    assert recorder.post_calls, "HTTP POST should have been invoked"
    
    # Verify correct endpoint (should use audio/transcriptions, not responses)
    assert recorder.post_calls[0]["url"].endswith("/audio/transcriptions")
    
    # Should not use the responses endpoint
    assert not any(call["url"].endswith("/responses") for call in recorder.post_calls)


@pytest.mark.anyio("asyncio")
async def test_openai_transcribe_handles_errors(monkeypatch):
    """Test that transcribe properly handles HTTP errors."""
    from httpx import HTTPStatusError, Request, Response
    
    class _ErrorResponse:
        def __init__(self):
            self.status_code = 400
            self.text = "Bad Request: Invalid model"
        
        def raise_for_status(self):
            request = Request("POST", "https://api.openai.com/v1/audio/transcriptions")
            response = Response(400, content=b"Bad Request: Invalid model")
            raise HTTPStatusError("400 Bad Request", request=request, response=response)
    
    class _ErrorClient:
        async def __aenter__(self):
            return self
        
        async def __aexit__(self, exc_type, exc, tb):
            return False
        
        async def post(self, url, **kwargs):
            return _ErrorResponse()
    
    monkeypatch.setattr(ai.httpx, "AsyncClient", lambda *args, **kwargs: _ErrorClient())
    
    provider = ai.OpenAIProvider(
        api_key="key",
        base_url="https://api.example.com",
        stt_model="whisper-1",
    )
    
    wav_bytes = _make_wav_bytes()
    with pytest.raises(RuntimeError, match="Ljudtranskribering misslyckades"):
        await provider.transcribe(wav_bytes, language="sv")


@pytest.mark.anyio("asyncio")
async def test_openai_transcribe_handles_timeout(monkeypatch):
    """Test that transcribe properly handles timeout errors with custom timeout."""
    import asyncio
    from httpx import ReadTimeout
    
    class _TimeoutClient:
        async def __aenter__(self):
            return self
        
        async def __aexit__(self, exc_type, exc, tb):
            return False
        
        async def post(self, url, **kwargs):
            raise ReadTimeout("Request timed out")
    
    monkeypatch.setattr(ai.httpx, "AsyncClient", lambda *args, **kwargs: _TimeoutClient())
    
    provider = ai.OpenAIProvider(
        api_key="key",
        base_url="https://api.example.com",
        stt_model="whisper-1",
        stt_timeout=5.0,  # Short timeout for testing
    )
    
    wav_bytes = _make_wav_bytes()
    with pytest.raises(RuntimeError, match="Ljudtranskribering tog för lång tid"):
        await provider.transcribe(wav_bytes, language="sv")


def test_extract_text_skips_user_text():
    payload = {
        "input": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Hej, assistent"}],
            }
        ],
        "output": [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Hej!"},
                    {
                        "type": "output_text",
                        "text": "Hur kan jag hjälpa dig?",
                    },
                ],
            }
        ],
        "output_text": ["Hej!", "Hur kan jag hjälpa dig?"],
    }

    text = ai._extract_text_from_json_payload(payload)

    assert text == "Hej!\nHur kan jag hjälpa dig?"


@pytest.mark.anyio("asyncio")
async def test_openai_chat_reply_all_inputs_use_input_text_type(monkeypatch):
    """Test that all input messages (system and user) use 'input_text' type as required by OpenAI API."""
    payload = {
        "output": [
            {
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "Test response"},
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
        chat_model="model",
    )

    await provider.chat_reply(
        "Test question?",
        context_sections=["Test context information."],
    )

    # Verify the request was made
    assert recorder.post_calls, "Responses endpoint should have been invoked"
    request_json = recorder.post_calls[0]["json"]
    
    # Check that all input messages use "input_text" type
    for input_msg in request_json["input"]:
        for content_item in input_msg["content"]:
            assert content_item["type"] == "input_text", f"Expected 'input_text' but got '{content_item['type']}' for {input_msg['role']} message"
