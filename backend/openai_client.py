
import httpx
from .config import (
    CHAT_MODEL,
    EMBEDDING_MODEL,
    OPENAI_API_KEY,
    STT_MODEL,
    TTS_MODEL,
    TTS_VOICE,
)

OPENAI_BASE = "https://api.openai.com/v1"

HEADERS = {
    "Authorization": f"Bearer {OPENAI_API_KEY}",
}

async def stt_transcribe_wav(wav_bytes: bytes, language: str = "sv") -> str:
    """Laddar upp WAV → text via OpenAI Transcriptions."""
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY saknas.")
    files = {
        "file": ("audio.wav", wav_bytes, "audio/wav"),
        "model": (None, STT_MODEL),
        "language": (None, language),
        "response_format": (None, "text"),
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(f"{OPENAI_BASE}/audio/transcriptions", headers=HEADERS, files=files)
        r.raise_for_status()
        return r.text.strip()

async def chat_reply_sv(user_text: str, *, context_sections: list[str] | None = None) -> str:
    """Skicka svensk prompt → få svensk text tillbaka."""
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY saknas.")
    messages = [
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
        "model": CHAT_MODEL,
        "messages": messages,
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(f"{OPENAI_BASE}/chat/completions", headers={**HEADERS,"Content-Type":"application/json"}, json=payload)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()

async def tts_speak_sv(text: str) -> bytes:
    """Text → WAV via OpenAI TTS. Returnerar WAV-bytes."""
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY saknas.")
    payload = {
        "model": TTS_MODEL,
        "voice": TTS_VOICE,
        "input": text,
        "format": "wav"
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(f"{OPENAI_BASE}/audio/speech", headers={**HEADERS,"Content-Type":"application/json"}, json=payload)
        r.raise_for_status()
        return r.content


async def create_embeddings(texts: list[str], model: str | None = None) -> list[list[float]]:
    if not texts:
        return []
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY saknas.")
    payload = {
        "input": texts,
        "model": model or EMBEDDING_MODEL,
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(
            f"{OPENAI_BASE}/embeddings",
            headers={**HEADERS, "Content-Type": "application/json"},
            json=payload,
        )
        r.raise_for_status()
        data = r.json()
    # Sortera efter index för att bevara ordningen
    sorted_items = sorted(data["data"], key=lambda item: item.get("index", 0))
    return [item["embedding"] for item in sorted_items]
