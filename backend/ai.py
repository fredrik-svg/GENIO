from __future__ import annotations

import asyncio
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
    TTS_MODEL,
    TTS_VOICE,
)

logger = logging.getLogger(__name__)


class BaseAIProvider:
    """Abstract base class for AI providers."""

    async def transcribe(self, wav_bytes: bytes, *, language: str = "sv") -> str:
        raise NotImplementedError

    async def chat_reply(
        self,
        user_text: str,
        *,
        context_sections: Optional[Sequence[str]] = None,
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
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        self._api_key = (api_key or OPENAI_API_KEY or "").strip()
        self._base_url = base_url.rstrip("/") or "https://api.openai.com/v1"
        self._chat_model = (chat_model or CHAT_MODEL or "gpt-4o-mini").strip()
        self._stt_model = (stt_model or STT_MODEL or "whisper-1").strip()
        self._tts_model = (tts_model or TTS_MODEL or "gpt-4o-mini-tts").strip()
        self._tts_voice = (tts_voice or TTS_VOICE or "alloy").strip()
        self._embedding_model = (
            embedding_model or EMBEDDING_MODEL or "text-embedding-3-small"
        ).strip()
        self._timeout = request_timeout
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        if extra_headers:
            headers.update(extra_headers)
        self._base_headers = headers

    def _require_api_key(self) -> None:
        if not self._api_key:
            raise RuntimeError("OPENAI_API_KEY saknas.")

    def _headers(self, *, content_type: Optional[str] = None) -> Dict[str, str]:
        headers = dict(self._base_headers)
        if content_type:
            headers["Content-Type"] = content_type
        return headers

    async def transcribe(self, wav_bytes: bytes, *, language: str = "sv") -> str:
        self._require_api_key()
        files = {
            "file": ("audio.wav", wav_bytes, "audio/wav"),
            "model": (None, self._stt_model),
            "language": (None, language),
            "response_format": (None, "text"),
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                f"{self._base_url}/audio/transcriptions",
                headers=self._headers(),
                files=files,
            )
            response.raise_for_status()
            return response.text.strip()

    async def chat_reply(
        self,
        user_text: str,
        *,
        context_sections: Optional[Sequence[str]] = None,
    ) -> str:
        self._require_api_key()
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
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                f"{self._base_url}/chat/completions",
                headers=self._headers(content_type="application/json"),
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()

    async def synthesize(self, text: str) -> bytes:
        self._require_api_key()
        payload = {
            "model": self._tts_model,
            "voice": self._tts_voice,
            "input": text,
            "format": "wav",
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                f"{self._base_url}/audio/speech",
                headers=self._headers(content_type="application/json"),
                json=payload,
            )
            response.raise_for_status()
            return response.content

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
    user_text: str, *, context_sections: Optional[Sequence[str]] = None
) -> str:
    provider = await get_ai_provider()
    return await provider.chat_reply(user_text, context_sections=context_sections)


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
