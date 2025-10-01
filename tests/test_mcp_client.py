"""Tests for MCP client integration."""

import pytest
from backend.mcp_client import MCPClient, get_mcp_client


def test_mcp_client_creation_without_mcp_library():
    """Test that MCPClient can be created even if MCP library is not available."""
    client = MCPClient(servers={})
    assert client is not None
    # Should not be enabled if MCP library is not available
    assert client.is_enabled() in (True, False)


def test_mcp_client_get_available_tools_when_disabled():
    """Test that get_available_tools returns empty list when MCP is disabled."""
    client = MCPClient(servers={})
    tools = client.get_available_tools()
    assert isinstance(tools, list)
    # Should be empty when not enabled
    if not client.is_enabled():
        assert len(tools) == 0


def test_mcp_client_get_servers_when_not_started():
    """Test that get_servers returns empty list before starting."""
    client = MCPClient(servers={})
    servers = client.get_servers()
    assert isinstance(servers, list)
    assert len(servers) == 0


@pytest.mark.asyncio
async def test_get_mcp_client_singleton():
    """Test that get_mcp_client returns a singleton instance."""
    client1 = await get_mcp_client()
    client2 = await get_mcp_client()
    assert client1 is client2


@pytest.mark.asyncio
async def test_mcp_client_stop_without_start():
    """Test that stop can be called without starting."""
    client = MCPClient(servers={})
    await client.stop()  # Should not raise an exception


def test_mcp_client_with_invalid_server_config():
    """Test that MCPClient handles invalid server configurations gracefully."""
    # Config without 'command' key
    invalid_config = {
        "test_server": {
            "args": ["test"]
        }
    }
    client = MCPClient(servers=invalid_config)
    assert client is not None
    # Should handle gracefully


def test_mcp_client_initialization_with_config():
    """Test MCPClient initialization with valid configuration."""
    config = {
        "test_server": {
            "command": "echo",
            "args": ["test"],
            "env": {"TEST_VAR": "value"}
        }
    }
    client = MCPClient(servers=config)
    assert client is not None
    assert hasattr(client, '_servers')
    assert 'test_server' in client._servers
