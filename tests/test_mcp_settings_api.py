"""Tests for MCP settings API endpoints."""

import os
import tempfile
import pytest
from fastapi.testclient import TestClient
from backend.app import app


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def temp_settings_dir(monkeypatch):
    """Create a temporary settings directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Patch the settings directory to use temp directory
        import backend.mcp_settings as mcp_settings
        original_dir = mcp_settings._SETTINGS_DIR
        original_file = mcp_settings._SETTINGS_FILE
        
        mcp_settings._SETTINGS_DIR = tmpdir
        mcp_settings._SETTINGS_FILE = os.path.join(tmpdir, "mcp_settings.json")
        
        yield tmpdir
        
        # Restore original paths
        mcp_settings._SETTINGS_DIR = original_dir
        mcp_settings._SETTINGS_FILE = original_file


def test_get_mcp_settings_default(client, temp_settings_dir):
    """Test getting MCP settings when none are configured."""
    response = client.get("/api/mcp/settings")
    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert data["enabled"] is False
    assert data["servers"] == {}


def test_update_mcp_settings(client, temp_settings_dir):
    """Test updating MCP settings."""
    payload = {
        "enabled": True,
        "servers": {
            "test-server": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-test"],
                "env": {"TEST_VAR": "value"}
            }
        }
    }
    
    response = client.post("/api/mcp/settings", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert "message" in data
    
    # Verify settings were saved
    response = client.get("/api/mcp/settings")
    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert data["enabled"] is True
    assert "test-server" in data["servers"]
    assert data["servers"]["test-server"]["command"] == "npx"


def test_update_mcp_settings_invalid_server(client, temp_settings_dir):
    """Test updating MCP settings with invalid server configuration."""
    payload = {
        "enabled": True,
        "servers": {
            "invalid-server": {
                "args": ["-y"],  # Missing required 'command' field
                "env": {}
            }
        }
    }
    
    response = client.post("/api/mcp/settings", json=payload)
    assert response.status_code == 400
    data = response.json()
    assert data["ok"] is False
    assert "error" in data


def test_update_mcp_settings_empty_server_name(client, temp_settings_dir):
    """Test updating MCP settings with empty server name."""
    payload = {
        "enabled": True,
        "servers": {
            "": {
                "command": "npx",
                "args": [],
                "env": {}
            }
        }
    }
    
    response = client.post("/api/mcp/settings", json=payload)
    assert response.status_code == 400
    data = response.json()
    assert data["ok"] is False
    assert "empty" in data["error"].lower()


def test_update_mcp_settings_disabled(client, temp_settings_dir):
    """Test disabling MCP."""
    # First enable MCP with a server
    payload = {
        "enabled": True,
        "servers": {
            "test-server": {
                "command": "npx",
                "args": [],
                "env": {}
            }
        }
    }
    client.post("/api/mcp/settings", json=payload)
    
    # Now disable MCP
    payload = {
        "enabled": False,
        "servers": {}
    }
    
    response = client.post("/api/mcp/settings", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    
    # Verify MCP is disabled
    response = client.get("/api/mcp/settings")
    data = response.json()
    assert data["enabled"] is False


def test_update_mcp_settings_multiple_servers(client, temp_settings_dir):
    """Test updating MCP settings with multiple servers."""
    payload = {
        "enabled": True,
        "servers": {
            "server1": {
                "command": "npx",
                "args": ["-y", "server1"],
                "env": {}
            },
            "server2": {
                "command": "node",
                "args": ["server2.js"],
                "env": {"VAR": "value"}
            }
        }
    }
    
    response = client.post("/api/mcp/settings", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    
    # Verify both servers were saved
    response = client.get("/api/mcp/settings")
    data = response.json()
    assert "server1" in data["servers"]
    assert "server2" in data["servers"]
    assert len(data["servers"]) == 2
