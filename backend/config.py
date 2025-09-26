from __future__ import annotations

import json
import os
from importlib import import_module, util
from typing import Any


def _load_dotenv() -> None:
    """Load environment variables from a ``.env`` file if python-dotenv exists."""

    spec = util.find_spec("dotenv")
    if spec is None:
        return
    module = import_module("dotenv")
    if hasattr(module, "load_dotenv"):
        module.load_dotenv()


_load_dotenv()


def env(key: str, default: str | None = None) -> str | None:
    return os.getenv(key, default)


def env_bool(key: str, default: str = "0") -> bool:
    value = os.getenv(key, default)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def env_json(key: str, default: str | None = None) -> dict[str, Any]:
    raw = os.getenv(key, default) or ""
    if not raw:
        return {}
    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON for {key}: {exc}") from exc
    if isinstance(loaded, dict):
        return loaded
    raise ValueError(f"Environment variable {key} must contain a JSON object")


def _parse_fallback_sample_rates(value: str | None) -> tuple[int, ...]:
    if not value:
        return ()
    rates: list[int] = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            rate = int(float(chunk))
        except ValueError:
            continue
        if rate > 0:
            rates.append(rate)
    return tuple(rates)


SAMPLE_RATE = int(env("SAMPLE_RATE", "16000"))
FALLBACK_SAMPLE_RATES = _parse_fallback_sample_rates(env("FALLBACK_SAMPLE_RATES"))
MAX_RECORD_SECONDS = float(env("MAX_RECORD_SECONDS", "12"))
# Tillåt längre tystnad efter aktiverad mikrofon innan inspelningen avslutas.
SILENCE_DURATION = float(env("SILENCE_DURATION", "2.5"))
ENERGY_THRESHOLD = float(env("ENERGY_THRESHOLD", "0.015"))  # justera vid behov

CHAT_MODEL = env("CHAT_MODEL", "gpt-4o-mini")
TTS_MODEL = env("TTS_MODEL", "gpt-4o-mini-tts")
TTS_VOICE = env("TTS_VOICE", "alloy")
STT_MODEL = env("STT_MODEL", "gpt-4o-mini-transcribe")
OPENAI_API_KEY = env("OPENAI_API_KEY", "")
HOST = env("HOST", "0.0.0.0")
PORT = int(env("PORT", "8080"))

EMBEDDING_MODEL = env("EMBEDDING_MODEL", "text-embedding-3-small")
RAG_ENABLED = env_bool("RAG_ENABLED", "1")
RAG_DB_PATH = env(
    "RAG_DB_PATH",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "rag_store")),
)
RAG_TOP_K = int(env("RAG_TOP_K", "4"))
RAG_MIN_SCORE = float(env("RAG_MIN_SCORE", "0.35"))
RAG_CHUNK_SIZE = int(env("RAG_CHUNK_SIZE", "400"))
RAG_CHUNK_OVERLAP = int(env("RAG_CHUNK_OVERLAP", "80"))

INPUT_DEVICE = env("INPUT_DEVICE", "")  # valfri: namn eller index för mic (sounddevice)
OUTPUT_DEVICE = env("OUTPUT_DEVICE", "")  # valfri: namn eller index för högtalare (sounddevice)
PLAY_CMD = env("PLAY_CMD", "aplay -q")  # t.ex. 'aplay -q' eller 'paplay'
OUTPUT_WAV_PATH = env("OUTPUT_WAV_PATH", "/tmp/reply.wav")

AI_PROVIDER = (env("AI_PROVIDER", "openai") or "openai").strip()
AI_PROVIDER_CONFIG = env_json("AI_PROVIDER_CONFIG", "")
