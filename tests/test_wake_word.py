import asyncio
import io
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest

from backend.wake_word import WakeWordDetector, get_wake_word_detector


@pytest.fixture
def detector():
    """Create a fresh wake word detector for testing."""
    return WakeWordDetector()


@pytest.fixture
def mock_config():
    """Mock wake word configuration."""
    return {
        "WAKE_WORD_ENABLED": True,
        "WAKE_WORDS": ["hej genio", "genio", "hej assistant"],
        "WAKE_WORD_TIMEOUT": 2.0,
        "WAKE_WORD_COOLDOWN": 0.5,
    }


@pytest.mark.asyncio
async def test_wake_word_detector_init(detector):
    """Test wake word detector initialization."""
    assert not detector.is_listening
    assert detector.last_detection_time == 0.0
    assert detector._listen_task is None


@pytest.mark.asyncio
async def test_start_listening_when_disabled(detector):
    """Test that start_listening does nothing when wake word is disabled."""
    with patch("backend.wake_word.WAKE_WORD_ENABLED", False):
        callback = AsyncMock()
        await detector.start_listening(callback)
        assert not detector.is_listening
        callback.assert_not_called()


@pytest.mark.asyncio
async def test_start_listening_when_enabled(detector, mock_config):
    """Test starting wake word listening."""
    with patch.multiple("backend.wake_word", **mock_config):
        callback = AsyncMock()
        
        # Start listening
        await detector.start_listening(callback)
        assert detector.is_listening
        assert detector._listen_task is not None
        
        # Clean up
        await detector.stop_listening()


@pytest.mark.asyncio
async def test_stop_listening(detector, mock_config):
    """Test stopping wake word listening."""
    with patch.multiple("backend.wake_word", **mock_config):
        callback = AsyncMock()
        
        # Start then stop
        await detector.start_listening(callback)
        assert detector.is_listening
        
        await detector.stop_listening()
        assert not detector.is_listening
        assert detector._listen_task is None


@pytest.mark.asyncio 
async def test_check_for_wake_word_detects_wake_word(detector):
    """Test wake word detection in transcribed text."""
    audio = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    
    with patch("backend.wake_word.WAKE_WORDS", ["hej genio", "genio"]):
        with patch("backend.wake_word.save_wav_mono16") as mock_save:
            with patch("backend.wake_word.stt_transcribe_wav", return_value="hej genio vad händer"):
                mock_save.return_value = None
                result = await detector._check_for_wake_word(audio)
                assert result is True


@pytest.mark.asyncio
async def test_check_for_wake_word_no_wake_word(detector):
    """Test that non-wake-word text is not detected."""
    audio = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    
    with patch("backend.wake_word.WAKE_WORDS", ["hej genio", "genio"]):
        with patch("backend.wake_word.save_wav_mono16") as mock_save:
            with patch("backend.wake_word.stt_transcribe_wav", return_value="hej hur mår du"):
                mock_save.return_value = None
                result = await detector._check_for_wake_word(audio)
                assert result is False


@pytest.mark.asyncio
async def test_check_for_wake_word_empty_audio(detector):
    """Test that empty audio returns False."""
    audio = np.array([], dtype=np.float32)
    result = await detector._check_for_wake_word(audio)
    assert result is False


@pytest.mark.asyncio
async def test_check_for_wake_word_stt_error(detector):
    """Test that STT errors are handled gracefully."""
    audio = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    
    with patch("backend.wake_word.WAKE_WORDS", ["genio"]):
        with patch("backend.wake_word.save_wav_mono16") as mock_save:
            with patch("backend.wake_word.stt_transcribe_wav", side_effect=Exception("STT error")):
                mock_save.return_value = None
                result = await detector._check_for_wake_word(audio)
                assert result is False


def test_record_wake_word_audio(detector):
    """Test recording audio for wake word detection."""
    with patch("backend.wake_word.record_until_silence") as mock_record:
        mock_record.return_value = np.array([0.1, -0.2], dtype=np.float32)
        
        result = detector._record_wake_word_audio()
        
        mock_record.assert_called_once()
        # Check that custom timeout was passed
        assert "max_seconds" in str(mock_record.call_args)
        assert result.size == 2


def test_record_wake_word_audio_error(detector):
    """Test that recording errors return empty array."""
    with patch("backend.wake_word.record_until_silence", side_effect=Exception("Recording error")):
        result = detector._record_wake_word_audio()
        assert result.size == 0


def test_get_wake_word_detector():
    """Test that get_wake_word_detector returns singleton instance."""
    detector1 = get_wake_word_detector()
    detector2 = get_wake_word_detector()
    assert detector1 is detector2


@pytest.mark.asyncio
async def test_listen_loop_cooldown(detector, mock_config):
    """Test that cooldown period is respected."""
    with patch.multiple("backend.wake_word", **mock_config):
        with patch.object(detector, "_record_wake_word_audio", return_value=np.array([])):
            callback = AsyncMock()
            
            # Simulate recent detection
            detector.last_detection_time = asyncio.get_event_loop().time()
            
            # Start listening task
            task = asyncio.create_task(detector._listen_loop(callback))
            
            # Let it run briefly
            await asyncio.sleep(0.1)
            
            # Stop the task
            detector._stop_event.set()
            await task
            
            # Callback should not have been called due to cooldown
            callback.assert_not_called()


@pytest.mark.asyncio
async def test_listen_loop_wake_word_detection(detector, mock_config):
    """Test wake word detection in listen loop."""
    with patch.multiple("backend.wake_word", **mock_config):
        audio_data = np.array([0.1, -0.2, 0.3], dtype=np.float32)
        
        with patch.object(detector, "_record_wake_word_audio", return_value=audio_data):
            with patch.object(detector, "_check_for_wake_word", return_value=True) as mock_check:
                callback = AsyncMock()
                
                # Start listening task
                task = asyncio.create_task(detector._listen_loop(callback))
                
                # Let it run briefly to detect wake word
                await asyncio.sleep(0.2)
                
                # Stop the task
                detector._stop_event.set()
                await task
                
                # Check that wake word detection was called
                mock_check.assert_called()
                callback.assert_called_once()