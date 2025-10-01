"""Tests for MCP API endpoints."""

import pytest
from fastapi.testclient import TestClient
from backend.app import app


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


def test_mcp_status_endpoint(client):
    """Test the /api/mcp/status endpoint."""
    response = client.get("/api/mcp/status")
    assert response.status_code == 200
    data = response.json()
    assert "ok" in data
    assert "enabled" in data
    assert "servers" in data
    assert "tools_count" in data
    assert isinstance(data["enabled"], bool)
    assert isinstance(data["servers"], list)
    assert isinstance(data["tools_count"], int)


def test_mcp_tools_endpoint(client):
    """Test the /api/mcp/tools endpoint."""
    response = client.get("/api/mcp/tools")
    assert response.status_code == 200
    data = response.json()
    assert "ok" in data
    assert "tools" in data
    assert isinstance(data["tools"], list)
    # If MCP is not enabled, ok should be False
    if not data["ok"]:
        assert "error" in data
