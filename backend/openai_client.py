
import os, io, base64, httpx
from .config import OPENAI_API_KEY, STT_MODEL, CHAT_MODEL, TTS_MODEL, TTS_VOICE

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

async def chat_reply_sv(user_text: str) -> str:
    """Skicka svensk prompt → få svensk text tillbaka."""
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY saknas.")
    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {"role":"system","content":"Du är en hjälpsam röstassistent som pratar naturlig svenska. Svara kort och tydligt."},
            {"role":"user","content": user_text}
        ]
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
