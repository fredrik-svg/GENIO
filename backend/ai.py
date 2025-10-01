from __future__ import annotations

import asyncio
import base64
import binascii
import logging
from importlib import import_module
from typing import Any, Dict, Iterable, List, Optional, Sequence

import httpx

from .config import (
    AI_PROVIDER,
    AI_PROVIDER_CONFIG,
    CHAT_MODEL,
    EMBEDDING_MODEL,
    OPENAI_API_KEY,
    STT_MODEL,
    STT_TIMEOUT,
    TTS_MODEL,
    TTS_VOICE,
)

logger = logging.getLogger(__name__)


def _maybe_decode_wav_base64(candidate: str) -> Optional[bytes]:
    text = candidate.strip()
    if not text:
        return None

    if text.startswith("data:"):
        _prefix, _sep, rest = text.partition(",")
        if not _sep:
            return None
        text = rest.strip()

    if len(text) < 16:
        return None

    for strict in (True, False):
        try:
            decoded = base64.b64decode(text, validate=strict)
        except (binascii.Error, ValueError):
            continue
        if decoded.startswith((b"RIFF", b"RIFX")):
            return decoded
    return None


def _gather_wav_chunks(payload: Any, *, in_audio: bool = False) -> list[bytes]:
    """Return a list of decoded WAV byte chunks discovered in *payload*."""

    if isinstance(payload, str):
        audio = _maybe_decode_wav_base64(payload)
        if audio is not None:
            return [audio]
        if in_audio:
            try:
                decoded = base64.b64decode(payload)
            except (binascii.Error, ValueError):
                return []
            return [decoded] if decoded else []

    if isinstance(payload, dict):
        chunks: list[bytes] = []

        if "audio" in payload:
            chunks.extend(_gather_wav_chunks(payload["audio"], in_audio=True))

        if "b64_json" in payload:
            chunks.extend(_gather_wav_chunks(payload["b64_json"], in_audio=in_audio))

        if "data" in payload:
            data_value = payload["data"]
            if isinstance(data_value, list):
                data_chunks: list[bytes] = []
                for item in data_value:
                    data_chunks.extend(_gather_wav_chunks(item, in_audio=in_audio))
                if data_chunks:
                    chunks.append(b"".join(data_chunks))
            else:
                chunks.extend(_gather_wav_chunks(data_value, in_audio=in_audio))

        for key, value in payload.items():
            if key in {"b64_json", "data", "audio"}:
                continue
            chunks.extend(_gather_wav_chunks(value, in_audio=in_audio))

        return chunks

    if isinstance(payload, list):
        chunks: list[bytes] = []
        for item in payload:
            chunks.extend(_gather_wav_chunks(item, in_audio=in_audio))
        return chunks

    return []


def _extract_wav_from_json_payload(payload: Any) -> Optional[bytes]:
    chunks = _gather_wav_chunks(payload)
    if not chunks:
        return None
    if len(chunks) == 1:
        return chunks[0]

    for index, chunk in enumerate(chunks):
        if chunk.startswith((b"RIFF", b"RIFX")):
            tail = b"".join(chunks[:index] + chunks[index + 1 :])
            return chunk + tail

    return b"".join(chunks)


def _extract_wav_from_json_response(response: httpx.Response) -> Optional[bytes]:
    try:
        payload = response.json()
    except ValueError:
        return None
    return _extract_wav_from_json_payload(payload)


def _extract_text_from_json_payload(payload: Any) -> str:
    """Collect text fragments from a JSON payload returned by the Voice API."""

    texts: list[str] = []

    def _append(text: str | None) -> None:
        if not text:
            return
        stripped = text.strip()
        if stripped:
            texts.append(stripped)

    def _walk(node: Any, *, role: Optional[str] = None) -> None:
        if isinstance(node, dict):
            node_role = node.get("role") if isinstance(node.get("role"), str) else role
            node_type = node.get("type")

            text_value = node.get("text") if isinstance(node.get("text"), str) else None

            if node_type in {"output_text", "response.output_text"}:
                _append(text_value)
            elif node_type == "text" and node_role in {None, "assistant"}:
                _append(text_value)

            output_text_value = node.get("output_text")
            if isinstance(output_text_value, str):
                _append(output_text_value)
            elif isinstance(output_text_value, list):
                for item in output_text_value:
                    if isinstance(item, str):
                        _append(item)
                    else:
                        _walk(item, role=node_role)

            child_nodes: list[Any] = []

            for key in ("content", "data", "output", "tool_response"):
                value = node.get(key)
                if isinstance(value, (dict, list)):
                    child_nodes.append(value)

            for value in node.values():
                if isinstance(value, (dict, list)):
                    child_nodes.append(value)

            seen_ids: set[int] = set()
            for child in child_nodes:
                identifier = id(child)
                if identifier in seen_ids:
                    continue
                seen_ids.add(identifier)
                _walk(child, role=node_role)

        elif isinstance(node, list):
            for item in node:
                _walk(item, role=role)

    _walk(payload)

    unique: list[str] = []
    seen: set[str] = set()
    for chunk in texts:
        if chunk in seen:
            continue
        seen.add(chunk)
        unique.append(chunk)

    return "\n".join(unique)


class BaseAIProvider:
    """Abstract base class for AI providers."""

    async def transcribe(self, wav_bytes: bytes, *, language: str = "sv") -> str:
        raise NotImplementedError

    async def chat_reply(
        self,
        user_text: str,
        *,
        context_sections: Optional[Sequence[str]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        raise NotImplementedError

    async def synthesize(self, text: str) -> bytes:
        raise NotImplementedError

    async def create_embeddings(
        self, texts: Sequence[str], *, model: Optional[str] = None
    ) -> List[List[float]]:
        raise NotImplementedError


class OpenAIProvider(BaseAIProvider):
    """Provider implementation for OpenAI's REST API."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        chat_model: Optional[str] = None,
        stt_model: Optional[str] = None,
        tts_model: Optional[str] = None,
        tts_voice: Optional[str] = None,
        embedding_model: Optional[str] = None,
        request_timeout: float = 60.0,
        stt_timeout: Optional[float] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        self._api_key = (api_key or OPENAI_API_KEY or "").strip()
        self._base_url = base_url.rstrip("/") or "https://api.openai.com/v1"
        self._chat_model = (chat_model or CHAT_MODEL or "gpt-4o-mini").strip()
        self._stt_model = (stt_model or STT_MODEL or "gpt-4o-mini-transcribe").strip()
        self._tts_model = (tts_model or TTS_MODEL or "gpt-4o-mini-tts").strip()
        self._tts_voice = (tts_voice or TTS_VOICE or "alloy").strip()
        self._embedding_model = (
            embedding_model or EMBEDDING_MODEL or "text-embedding-3-small"
        ).strip()
        self._timeout = request_timeout
        self._stt_timeout = stt_timeout or STT_TIMEOUT
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        if extra_headers:
            headers.update(extra_headers)
        self._base_headers = headers
        self._responses_endpoint = f"{self._base_url}/responses"

    def _require_api_key(self) -> None:
        if not self._api_key:
            raise RuntimeError("OPENAI_API_KEY saknas.")

    def _headers(self, *, content_type: Optional[str] = None) -> Dict[str, str]:
        headers = dict(self._base_headers)
        if content_type:
            headers["Content-Type"] = content_type
        return headers

    async def _post_responses(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                self._responses_endpoint,
                headers=self._headers(content_type="application/json"),
                json=payload,
            )
            response.raise_for_status()
            return response.json()

    async def _legacy_transcribe(self, wav_bytes: bytes, language: str) -> str:
        model_name = self._stt_model or "whisper-1"
        if not model_name.lower().startswith("whisper"):
            model_name = "whisper-1"

        files = {
            "file": ("audio.wav", wav_bytes, "audio/wav"),
            "model": (None, model_name),
            "language": (None, language),
            "response_format": (None, "text"),
        }
        try:
            async with httpx.AsyncClient(timeout=self._stt_timeout) as client:
                response = await client.post(
                    f"{self._base_url}/audio/transcriptions",
                    headers=self._headers(),
                    files=files,
                )
                response.raise_for_status()
                return response.text.strip()
        except httpx.ReadTimeout as exc:
            logger.error(
                "OpenAI transcription timed out after %s seconds - audio may be too long or API is slow",
                self._stt_timeout
            )
            raise RuntimeError(
                f"Ljudtranskribering tog för lång tid (>{self._stt_timeout}s). "
                "Prova med kortare ljudklipp eller försök igen senare."
            ) from exc
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code if exc.response is not None else "okänt"
            body = exc.response.text if exc.response is not None else "<ingen text>"
            logger.error(
                "OpenAI transcription API failed (%s): %s – response: %s",
                status,
                exc,
                body,
            )
            raise RuntimeError(
                f"Ljudtranskribering misslyckades (HTTP {status}). Försök igen senare."
            ) from exc

    async def transcribe(self, wav_bytes: bytes, *, language: str = "sv") -> str:
        self._require_api_key()
        # Use the standard OpenAI audio transcriptions API directly
        # since the responses API doesn't support input_audio content type
        return await self._legacy_transcribe(wav_bytes, language)

    async def _legacy_chat_reply(
        self,
        user_text: str,
        *,
        context_sections: Optional[Sequence[str]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        messages: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": "Du är en hjälpsam röstassistent som pratar naturlig svenska. Svara kort och tydligt.",
            }
        ]
        if context_sections:
            context_text = "\n\n".join(context_sections)
            messages.append(
                {
                    "role": "system",
                    "content": "Använd följande bakgrundsinformation när du svarar, men hitta inte på fakta om inget passar:\n\n"
                    + context_text,
                }
            )
        messages.append({"role": "user", "content": user_text})
        payload = {
            "model": self._chat_model,
            "messages": messages,
        }
        
        # Add tools if provided
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                f"{self._base_url}/chat/completions",
                headers=self._headers(content_type="application/json"),
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()

    async def chat_reply(
        self,
        user_text: str,
        *,
        context_sections: Optional[Sequence[str]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        self._require_api_key()

        def _make_content(role: str, text: str, *, is_user: bool = False) -> Dict[str, Any]:
            entry_type = "input_text"
            return {"role": role, "content": [{"type": entry_type, "text": text}]}

        inputs: List[Dict[str, Any]] = [
            _make_content(
                "system",
                "Du är en hjälpsam röstassistent som pratar naturlig svenska. Svara kort och tydligt.",
            )
        ]

        if context_sections:
            context_text = "\n\n".join(context_sections)
            inputs.append(
                _make_content(
                    "system",
                    "Använd följande bakgrundsinformation när du svarar, men hitta inte på fakta om inget passar:\n\n"
                    + context_text,
                )
            )

        inputs.append(_make_content("user", user_text, is_user=True))

        payload = {
            "model": self._chat_model,
            "input": inputs,
        }
        
        # Add tools if provided (for OpenAI responses API)
        if tools:
            payload["tools"] = tools

        try:
            response_payload = await self._post_responses(payload)
        except httpx.HTTPStatusError as exc:  # pragma: no cover - nätverksfel
            status = exc.response.status_code if exc.response is not None else "okänt"
            body = exc.response.text if exc.response is not None else "<ingen text>"
            logger.error(
                "OpenAI Voice API-chatt misslyckades (%s): %s – svar: %s",
                status,
                exc,
                body,
            )
            return await self._legacy_chat_reply(user_text, context_sections=context_sections, tools=tools)
        except Exception as exc:  # pragma: no cover - nätverksfel
            logger.error("Kunde inte kontakta OpenAI Voice API för chatt: %s", exc)
            return await self._legacy_chat_reply(user_text, context_sections=context_sections, tools=tools)

        reply_text = _extract_text_from_json_payload(response_payload).strip()
        if reply_text:
            return reply_text

        logger.warning("Voice API-chatt saknade textsvar – återgår till klassisk chattmodell.")
        return await self._legacy_chat_reply(user_text, context_sections=context_sections, tools=tools)

    async def synthesize(self, text: str) -> bytes:
        self._require_api_key()
        cleaned_text = text.strip()
        if not cleaned_text:
            raise ValueError("Kan inte generera tal från tom text.")

        # Use the correct OpenAI TTS API format
        tts_model = self._tts_model
        # Map the model name to a valid OpenAI TTS model if needed
        if tts_model.startswith("gpt-"):
            tts_model = "tts-1"

        payload = {
            "model": tts_model,
            "input": cleaned_text,
            "voice": self._tts_voice,
            "response_format": "wav",
        }

        try:
            # Use the correct endpoint for OpenAI TTS
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    f"{self._base_url}/audio/speech",
                    headers=self._headers(content_type="application/json"),
                    json=payload,
                )
                response.raise_for_status()
                return response.content
        except httpx.HTTPStatusError as exc:  # pragma: no cover - nätverksfel
            status = exc.response.status_code if exc.response is not None else "okänt"
            body = exc.response.text if exc.response is not None else "<ingen text>"
            logger.error("OpenAI Voice API-TTS misslyckades (%s): %s – svar: %s", status, exc, body)
            raise RuntimeError(f"OpenAI Voice API-TTS misslyckades ({status}).") from exc

    async def create_embeddings(
        self, texts: Sequence[str], *, model: Optional[str] = None
    ) -> List[List[float]]:
        if not texts:
            return []
        self._require_api_key()
        payload = {
            "input": list(texts),
            "model": (model or self._embedding_model),
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                f"{self._base_url}/embeddings",
                headers=self._headers(content_type="application/json"),
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
        sorted_items = sorted(data["data"], key=lambda item: item.get("index", 0))
        return [item["embedding"] for item in sorted_items]


class EchoProvider(BaseAIProvider):
    """A minimal provider useful for development and offline demos."""

    def __init__(self, *, sample_rate: int = 16000) -> None:
        self._sample_rate = sample_rate

    async def transcribe(self, wav_bytes: bytes, *, language: str = "sv") -> str:
        return ""

    async def chat_reply(
        self,
        user_text: str,
        *,
        context_sections: Optional[Sequence[str]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        return user_text

    async def synthesize(self, text: str) -> bytes:
        duration_seconds = 0.2
        total_samples = int(self._sample_rate * duration_seconds)
        header = b"RIFF" + (36 + total_samples * 2).to_bytes(4, "little") + b"WAVEfmt "
        header += (16).to_bytes(4, "little")  # PCM header size
        header += (1).to_bytes(2, "little")  # PCM format
        header += (1).to_bytes(2, "little")  # mono
        header += self._sample_rate.to_bytes(4, "little")
        byte_rate = self._sample_rate * 2
        header += byte_rate.to_bytes(4, "little")
        block_align = (2).to_bytes(2, "little")
        header += block_align
        header += (16).to_bytes(2, "little")
        data_chunk = b"data" + (total_samples * 2).to_bytes(4, "little")
        silence = b"\x00\x00" * total_samples
        return header + data_chunk + silence

    async def create_embeddings(
        self, texts: Sequence[str], *, model: Optional[str] = None
    ) -> List[List[float]]:
        return [[float(len(text))] for text in texts]


_PROVIDER_ALIASES = {
    "openai": "backend.ai:OpenAIProvider",
    "echo": "backend.ai:EchoProvider",
    "mock": "backend.ai:EchoProvider",
}

_provider_instance: BaseAIProvider | None = None
_provider_lock: asyncio.Lock | None = None


def _resolve_provider_spec(spec: str) -> tuple[str, str]:
    target = _PROVIDER_ALIASES.get(spec.strip().lower(), spec)
    if ":" in target:
        module_name, class_name = target.split(":", 1)
    else:
        if "." not in target:
            raise ValueError(
                "AI provider specification must contain a module path or use a known alias"
            )
        module_name, class_name = target.rsplit(".", 1)
    return module_name, class_name


def _instantiate_provider() -> BaseAIProvider:
    spec = AI_PROVIDER or "openai"
    module_name, class_name = _resolve_provider_spec(spec)
    module = import_module(module_name)
    provider_cls = getattr(module, class_name)
    if not issubclass(provider_cls, BaseAIProvider):
        raise TypeError(f"{provider_cls!r} is not a BaseAIProvider subclass")
    kwargs = dict(AI_PROVIDER_CONFIG)
    try:
        provider = provider_cls(**kwargs)
    except TypeError:
        logger.warning("AI provider %s did not accept configuration %r", spec, kwargs)
        provider = provider_cls()
    return provider


async def get_ai_provider() -> BaseAIProvider:
    global _provider_instance, _provider_lock
    if _provider_instance is not None:
        return _provider_instance
    if _provider_lock is None:
        _provider_lock = asyncio.Lock()
    async with _provider_lock:
        if _provider_instance is None:
            _provider_instance = _instantiate_provider()
    return _provider_instance


async def stt_transcribe_wav(wav_bytes: bytes, *, language: str = "sv") -> str:
    provider = await get_ai_provider()
    return await provider.transcribe(wav_bytes, language=language)


async def chat_reply_sv(
    user_text: str, 
    *, 
    context_sections: Optional[Sequence[str]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> str:
    provider = await get_ai_provider()
    return await provider.chat_reply(user_text, context_sections=context_sections, tools=tools)


async def tts_speak_sv(text: str) -> bytes:
    provider = await get_ai_provider()
    return await provider.synthesize(text)


async def create_embeddings(
    texts: Sequence[str], *, model: Optional[str] = None
) -> List[List[float]]:
    provider = await get_ai_provider()
    return await provider.create_embeddings(texts, model=model)


__all__ = [
    "BaseAIProvider",
    "OpenAIProvider",
    "EchoProvider",
    "get_ai_provider",
    "stt_transcribe_wav",
    "chat_reply_sv",
    "tts_speak_sv",
    "create_embeddings",
]
