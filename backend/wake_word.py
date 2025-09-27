import asyncio
import io
import logging
import time
from typing import Optional, Callable, Awaitable

import numpy as np

from .config import (
    WAKE_WORD_ENABLED,
    WAKE_WORDS,
    WAKE_WORD_TIMEOUT,
    WAKE_WORD_COOLDOWN,
)
from .audio import record_until_silence, save_wav_mono16
from .ai import stt_transcribe_wav

logger = logging.getLogger(__name__)


class WakeWordDetector:
    """Wake word detector that continuously listens for configured wake words."""

    def __init__(self):
        self.is_listening = False
        self.last_detection_time = 0.0
        self._listen_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

    async def start_listening(self, on_wake_word: Callable[[], Awaitable[None]]) -> None:
        """Start continuous listening for wake words."""
        if not WAKE_WORD_ENABLED:
            logger.info("Wake word detection is disabled in configuration")
            return

        if self.is_listening:
            logger.warning("Wake word detector is already listening")
            return

        logger.info("Starting wake word detection with words: %s", WAKE_WORDS)
        self.is_listening = True
        self._stop_event.clear()

        self._listen_task = asyncio.create_task(self._listen_loop(on_wake_word))

    async def stop_listening(self) -> None:
        """Stop listening for wake words."""
        if not self.is_listening:
            return

        logger.info("Stopping wake word detection")
        self.is_listening = False
        self._stop_event.set()

        if self._listen_task:
            try:
                await asyncio.wait_for(self._listen_task, timeout=2.0)
            except asyncio.TimeoutError:
                logger.warning("Wake word listener didn't stop gracefully, cancelling")
                self._listen_task.cancel()
                try:
                    await self._listen_task
                except asyncio.CancelledError:
                    pass
            self._listen_task = None

    async def _listen_loop(self, on_wake_word: Callable[[], Awaitable[None]]) -> None:
        """Continuous listening loop for wake word detection."""
        logger.info("Wake word listening loop started")

        while not self._stop_event.is_set():
            try:
                # Check if we're in cooldown period
                current_time = time.time()
                if current_time - self.last_detection_time < WAKE_WORD_COOLDOWN:
                    await asyncio.sleep(0.1)
                    continue

                # Record audio for a short period
                logger.debug("Listening for wake word...")
                audio = await asyncio.get_event_loop().run_in_executor(
                    None, self._record_wake_word_audio
                )

                if audio.size == 0:
                    await asyncio.sleep(0.1)
                    continue

                # Transcribe and check for wake words
                detected = await self._check_for_wake_word(audio)
                if detected:
                    logger.info("Wake word detected!")
                    self.last_detection_time = time.time()

                    # Trigger the wake word callback
                    try:
                        await on_wake_word()
                    except Exception as e:
                        logger.error("Error in wake word callback: %s", e)

                # Small delay to prevent overwhelming the system
                await asyncio.sleep(0.1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in wake word detection loop: %s", e)
                await asyncio.sleep(1.0)  # Back off on errors

        logger.info("Wake word listening loop stopped")

    def _record_wake_word_audio(self) -> np.ndarray:
        """Record a short audio clip for wake word detection."""
        try:
            # Use a shorter timeout for wake word detection
            # We'll modify record_until_silence to support custom timeouts
            return record_until_silence(max_seconds=WAKE_WORD_TIMEOUT)
        except Exception as e:
            logger.debug("Failed to record wake word audio: %s", e)
            return np.array([], dtype=np.float32)

    async def _check_for_wake_word(self, audio: np.ndarray) -> bool:
        """Check if the audio contains a wake word."""
        if not WAKE_WORDS or audio.size == 0:
            return False

        try:
            # Convert audio to WAV format for STT
            buf = io.BytesIO()
            save_wav_mono16(buf, audio)
            wav_bytes = buf.getvalue()

            # Transcribe the audio
            text = await stt_transcribe_wav(wav_bytes, language="sv")
            if not text:
                return False

            text_lower = text.lower().strip()
            logger.debug("Transcribed wake word audio: '%s'", text_lower)

            # Check if any wake word is present
            for wake_word in WAKE_WORDS:
                if wake_word.lower() in text_lower:
                    logger.info("Wake word '%s' detected in text: '%s'", wake_word, text_lower)
                    return True

            return False

        except Exception as e:
            logger.debug("Error checking for wake word: %s", e)
            return False


# Global wake word detector instance
_wake_word_detector: Optional[WakeWordDetector] = None


def get_wake_word_detector() -> WakeWordDetector:
    """Get the global wake word detector instance."""
    global _wake_word_detector
    if _wake_word_detector is None:
        _wake_word_detector = WakeWordDetector()
    return _wake_word_detector