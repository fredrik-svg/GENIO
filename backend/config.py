import os
from importlib import import_module, util


def _load_dotenv() -> None:
    """Load environment variables from a ``.env`` file if python-dotenv exists."""

    spec = util.find_spec("dotenv")
    if spec is None:
        return
    module = import_module("dotenv")
    if hasattr(module, "load_dotenv"):
        module.load_dotenv()


_load_dotenv()

def env(key: str, default: str | None = None):
    return os.getenv(key, default)

SAMPLE_RATE = int(env("SAMPLE_RATE", "16000"))
MAX_RECORD_SECONDS = float(env("MAX_RECORD_SECONDS", "12"))
SILENCE_DURATION = float(env("SILENCE_DURATION", "1.0"))
ENERGY_THRESHOLD = float(env("ENERGY_THRESHOLD", "0.015"))  # justera vid behov

CHAT_MODEL = env("CHAT_MODEL","gpt-4o-mini")
TTS_MODEL = env("TTS_MODEL","gpt-4o-mini-tts")
TTS_VOICE = env("TTS_VOICE","alloy")
STT_MODEL = env("STT_MODEL","whisper-1")
OPENAI_API_KEY = env("OPENAI_API_KEY","")
HOST = env("HOST","0.0.0.0")
PORT = int(env("PORT","8080"))

INPUT_DEVICE = env("INPUT_DEVICE", "")  # valfri: namn eller index för mic (sounddevice)
PLAY_CMD = env("PLAY_CMD", "aplay -q")  # t.ex. 'aplay -q' eller 'paplay'
OUTPUT_WAV_PATH = env("OUTPUT_WAV_PATH", "/tmp/reply.wav")
