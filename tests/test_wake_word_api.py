import pytest
from unittest.mock import patch, AsyncMock, Mock
from fastapi.testclient import TestClient

from backend.app import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


def test_wake_word_status_with_error_info(client):
    """Test wake word status includes error information."""
    mock_detector = Mock()
    mock_detector.get_status.return_value = {
        "is_listening": False,
        "last_detection_time": 1234567890.0,
        "last_error": "Audio device not found",
        "last_error_time": 1234567891.0,
    }
    
    with patch("backend.app.get_wake_word_detector", return_value=mock_detector):
        with patch("backend.app.load_wake_word_settings") as mock_settings:
            mock_settings.return_value.enabled = True
            mock_settings.return_value.wake_words = ["test"]
            
            response = client.get("/api/wake-word/status")
            assert response.status_code == 200
            data = response.json()
            assert data["enabled"] is True
            assert data["wake_words"] == ["test"]
            assert data["is_listening"] is False
            assert data["last_error"] == "Audio device not found"
            assert data["last_error_time"] == 1234567891.0
            assert data["last_detection_time"] == 1234567890.0


def test_wake_word_status_no_error(client):
    """Test wake word status when no error is present."""
    mock_detector = Mock()
    mock_detector.get_status.return_value = {
        "is_listening": True,
        "last_detection_time": 1234567890.0,
        "last_error": None,
        "last_error_time": None,
    }
    
    with patch("backend.app.get_wake_word_detector", return_value=mock_detector):
        with patch("backend.app.load_wake_word_settings") as mock_settings:
            mock_settings.return_value.enabled = True
            mock_settings.return_value.wake_words = ["test"]
            
            response = client.get("/api/wake-word/status")
            assert response.status_code == 200
            data = response.json()
            assert data["enabled"] is True
            assert data["is_listening"] is True
            assert data["last_error"] is None
            assert data["last_error_time"] is None


def test_wake_word_status_disabled(client):
    """Test wake word status when disabled."""
    mock_detector = Mock()
    mock_detector.get_status.return_value = {
        "is_listening": False,
        "last_detection_time": 0.0,
        "last_error": None,
        "last_error_time": None,
    }
    
    with patch("backend.app.get_wake_word_detector", return_value=mock_detector):
        with patch("backend.app.load_wake_word_settings") as mock_settings:
            mock_settings.return_value.enabled = False
            mock_settings.return_value.wake_words = ["test"]
            
            response = client.get("/api/wake-word/status")
            assert response.status_code == 200
            data = response.json()
            assert data["enabled"] is False
            assert data["wake_words"] == ["test"]
            assert data["is_listening"] is False


def test_wake_word_status_enabled(client):
    """Test wake word status when enabled."""
    mock_detector = Mock()
    mock_detector.get_status.return_value = {
        "is_listening": False,
        "last_detection_time": 0.0,
        "last_error": None,
        "last_error_time": None,
    }
    
    with patch("backend.app.get_wake_word_detector", return_value=mock_detector):
        with patch("backend.app.load_wake_word_settings") as mock_settings:
            mock_settings.return_value.enabled = True
            mock_settings.return_value.wake_words = ["hej genio", "genio"]
            
            response = client.get("/api/wake-word/status")
            assert response.status_code == 200
            data = response.json()
            assert data["enabled"] is True
            assert data["wake_words"] == ["hej genio", "genio"]
            assert data["is_listening"] is False


@pytest.mark.asyncio
async def test_start_wake_word_success(client):
    """Test successfully starting wake word detection."""
    mock_detector = AsyncMock()
    mock_detector.start_listening = AsyncMock()
    
    with patch("backend.app.get_wake_word_detector", return_value=mock_detector):
        with patch("backend.app.notify", AsyncMock()):
            response = client.post("/api/wake-word/start")
            assert response.status_code == 200
            data = response.json()
            assert data["ok"] is True
            assert "started" in data["message"]


@pytest.mark.asyncio
async def test_start_wake_word_error(client):
    """Test error handling when starting wake word detection fails."""
    mock_detector = AsyncMock()
    mock_detector.start_listening = AsyncMock(side_effect=Exception("Test error"))
    
    with patch("backend.app.get_wake_word_detector", return_value=mock_detector):
        response = client.post("/api/wake-word/start")
        assert response.status_code == 500
        data = response.json()
        assert data["ok"] is False
        assert "Test error" in data["error"]


@pytest.mark.asyncio
async def test_stop_wake_word_success(client):
    """Test successfully stopping wake word detection."""
    mock_detector = AsyncMock()
    mock_detector.stop_listening = AsyncMock()
    
    with patch("backend.app.get_wake_word_detector", return_value=mock_detector):
        with patch("backend.app.notify", AsyncMock()):
            response = client.post("/api/wake-word/stop")
            assert response.status_code == 200
            data = response.json()
            assert data["ok"] is True
            assert "stopped" in data["message"]


@pytest.mark.asyncio
async def test_stop_wake_word_error(client):
    """Test error handling when stopping wake word detection fails."""
    mock_detector = AsyncMock()
    mock_detector.stop_listening = AsyncMock(side_effect=Exception("Stop error"))
    
    with patch("backend.app.get_wake_word_detector", return_value=mock_detector):
        response = client.post("/api/wake-word/stop")
        assert response.status_code == 500
        data = response.json()
        assert data["ok"] is False
        assert "Stop error" in data["error"]