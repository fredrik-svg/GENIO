import asyncio
import io
import logging
import time
from typing import Optional, Callable, Awaitable

import numpy as np

from .config import (  # noqa: F401
    WAKE_WORD_ENABLED,
    WAKE_WORDS,
    WAKE_WORD_TIMEOUT,
    WAKE_WORD_COOLDOWN,
)
from .wake_word_settings import load_wake_word_settings
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
        self._current_settings = None
        self.last_error: Optional[str] = None
        self.last_error_time: Optional[float] = None
        self.total_detection_attempts = 0
        self.successful_detections = 0
        self.last_transcription: Optional[str] = None
        self.last_transcription_time: Optional[float] = None

    def _set_error(self, error_msg: str) -> None:
        """Set the last error message and timestamp."""
        self.last_error = error_msg
        self.last_error_time = time.time()
        logger.error("Wake word error: %s", error_msg)

    def _clear_error(self) -> None:
        """Clear any previous error state."""
        self.last_error = None
        self.last_error_time = None

    def get_status(self) -> dict:
        """Get comprehensive status including error information."""
        return {
            "is_listening": self.is_listening,
            "last_detection_time": self.last_detection_time,
            "last_error": self.last_error,
            "last_error_time": self.last_error_time,
            "total_detection_attempts": self.total_detection_attempts,
            "successful_detections": self.successful_detections,
            "last_transcription": self.last_transcription,
            "last_transcription_time": self.last_transcription_time,
        }

    def _get_current_settings(self):
        """Get current wake word settings, caching them for a short period."""
        if self._current_settings is None or time.time() - self._current_settings.get('_loaded_at', 0) > 30:
            # Reload settings every 30 seconds to pick up changes
            settings = load_wake_word_settings()
            self._current_settings = {
                'enabled': settings.enabled,
                'wake_words': settings.wake_words,
                'timeout': settings.timeout,
                'cooldown': settings.cooldown,
                '_loaded_at': time.time()
            }
        return self._current_settings

    async def start_listening(self, on_wake_word: Callable[[], Awaitable[None]]) -> None:
        """Start continuous listening for wake words."""
        settings = self._get_current_settings()

        if not settings['enabled']:
            error_msg = "Wake word detection is disabled in configuration"
            logger.info(error_msg)
            self._set_error(error_msg)
            raise RuntimeError(error_msg)

        if self.is_listening:
            error_msg = "Wake word detector is already listening"
            logger.warning(error_msg)
            raise RuntimeError(error_msg)

        # Clear any previous errors when starting
        self._clear_error()

        logger.info("Starting wake word detection with words: %s", settings['wake_words'])
        self.is_listening = True
        self._stop_event.clear()

        try:
            self._listen_task = asyncio.create_task(self._listen_loop(on_wake_word))
        except Exception as e:
            self.is_listening = False
            self._set_error(f"Failed to start wake word detection: {str(e)}")
            raise

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
                settings = self._get_current_settings()

                # Check if wake word detection was disabled
                if not settings['enabled']:
                    logger.info("Wake word detection disabled, stopping listener")
                    self._set_error("Wake word detection was disabled during operation")
                    break

                # Check if we're in cooldown period
                current_time = time.time()
                if current_time - self.last_detection_time < settings['cooldown']:
                    await asyncio.sleep(0.1)
                    continue

                # Record audio for a short period
                logger.debug("Listening for wake word...")
                try:
                    audio = await asyncio.get_event_loop().run_in_executor(
                        None, self._record_wake_word_audio, settings['timeout']
                    )
                except Exception as e:
                    self._set_error(f"Audio recording failed: {str(e)}")
                    await asyncio.sleep(1.0)
                    continue

                if audio.size == 0:
                    await asyncio.sleep(0.1)
                    continue

                # Transcribe and check for wake words
                try:
                    detected = await self._check_for_wake_word(audio, settings['wake_words'])
                    if detected:
                        logger.info("Wake word detected!")
                        self.last_detection_time = time.time()
                        # Clear any previous errors on successful detection
                        self._clear_error()

                        # Trigger the wake word callback
                        try:
                            await on_wake_word()
                        except Exception as e:
                            logger.error("Error in wake word callback: %s", e)
                except Exception as e:
                    self._set_error(f"Wake word processing failed: {str(e)}")
                    await asyncio.sleep(1.0)
                    continue

                # Small delay to prevent overwhelming the system
                await asyncio.sleep(0.1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in wake word detection loop: %s", e)
                self._set_error(f"Unexpected error in wake word detection: {str(e)}")
                await asyncio.sleep(1.0)  # Back off on errors

        logger.info("Wake word listening loop stopped")
        self.is_listening = False

    def _record_wake_word_audio(self, timeout: float = None) -> np.ndarray:
        """Record a short audio clip for wake word detection."""
        try:
            # Use custom timeout or fall back to config default
            max_seconds = timeout or WAKE_WORD_TIMEOUT
            return record_until_silence(max_seconds=max_seconds)
        except Exception as e:
            logger.debug("Failed to record wake word audio: %s", e)
            # Don't set error here as this will be caught by the caller
            return np.array([], dtype=np.float32)

    async def _check_for_wake_word(self, audio: np.ndarray, wake_words: list[str]) -> bool:
        """Check if the audio contains a wake word."""
        if not wake_words or audio.size == 0:
            return False

        start_time = time.time()
        try:
            # Convert audio to WAV format for STT
            buf = io.BytesIO()
            save_wav_mono16(buf, audio)
            wav_bytes = buf.getvalue()

            # Transcribe the audio
            text = await stt_transcribe_wav(wav_bytes, language="sv")
            transcription_time = time.time() - start_time

            # Store transcription for debugging
            self.last_transcription = text or ""
            self.last_transcription_time = time.time()
            self.total_detection_attempts += 1

            if not text:
                logger.debug("No transcription received (audio duration: %.2fs, transcription time: %.2fs)",
                             audio.size / 16000.0, transcription_time)
                return False

            text_lower = text.lower().strip()
            logger.debug("Transcribed wake word audio: '%s' (audio: %.2fs, transcription: %.2fs)",
                         text_lower, audio.size / 16000.0, transcription_time)

            # Check if any wake word is present
            for wake_word in wake_words:
                if wake_word.lower() in text_lower:
                    logger.info("✓ Wake word '%s' detected in text: '%s' (total time: %.2fs)",
                                wake_word, text_lower, transcription_time)
                    self.successful_detections += 1
                    return True

            logger.debug("✗ No wake word found. Expected one of %s, got: '%s'", wake_words, text_lower)
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
