import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient

from backend.app import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


def test_wake_word_status_disabled(client):
    """Test wake word status when disabled."""
    with patch("backend.app.WAKE_WORD_ENABLED", False):
        with patch("backend.app.WAKE_WORDS", ["test"]):
            response = client.get("/api/wake-word/status")
            assert response.status_code == 200
            data = response.json()
            assert data["enabled"] is False
            assert data["wake_words"] == ["test"]
            assert data["is_listening"] is False


def test_wake_word_status_enabled(client):
    """Test wake word status when enabled."""
    with patch("backend.app.WAKE_WORD_ENABLED", True):
        with patch("backend.app.WAKE_WORDS", ["hej genio", "genio"]):
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