"""MCP (Model Context Protocol) client integration.

This module provides integration with MCP servers, allowing the assistant
to access external tools and data sources through the MCP protocol.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from .config import MCP_ENABLED, MCP_SERVERS

logger = logging.getLogger(__name__)

# MCP client will be conditionally imported
_mcp_available = False
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    _mcp_available = True
except ImportError:
    logger.info("MCP library not available - MCP server support disabled")


class MCPClient:
    """Client for managing MCP server connections and tool execution."""

    def __init__(self, servers: Optional[Dict[str, Any]] = None) -> None:
        """Initialize MCP client with server configurations.
        
        Args:
            servers: Dictionary mapping server names to their configurations.
                     Each config should have 'command' and optional 'args' and 'env'.
        """
        self._servers = servers or {}
        self._sessions: Dict[str, ClientSession] = {}
        self._tools: Dict[str, Dict[str, Any]] = {}
        self._enabled = MCP_ENABLED and _mcp_available
        
        if not self._enabled:
            if MCP_ENABLED and not _mcp_available:
                logger.warning("MCP_ENABLED is true but MCP library is not available")
            return

    async def start(self) -> None:
        """Start all configured MCP server connections."""
        if not self._enabled:
            return

        for server_name, config in self._servers.items():
            try:
                await self._connect_server(server_name, config)
            except Exception as exc:
                logger.error("Failed to connect to MCP server %s: %s", server_name, exc)

    async def _connect_server(self, name: str, config: Dict[str, Any]) -> None:
        """Connect to a single MCP server.
        
        Args:
            name: Server identifier
            config: Server configuration with 'command', optional 'args' and 'env'
        """
        command = config.get("command")
        if not command:
            logger.error("MCP server %s missing 'command' in configuration", name)
            return

        args = config.get("args", [])
        env = config.get("env")

        server_params = StdioServerParameters(
            command=command,
            args=args,
            env=env,
        )

        try:
            # Create stdio client context
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    # Initialize the session
                    await session.initialize()
                    
                    # Store session
                    self._sessions[name] = session
                    
                    # List available tools
                    tools_result = await session.list_tools()
                    server_tools = {}
                    
                    for tool in tools_result.tools:
                        tool_info = {
                            "name": tool.name,
                            "description": tool.description,
                            "input_schema": tool.inputSchema,
                        }
                        server_tools[tool.name] = tool_info
                        logger.info("Registered MCP tool: %s.%s - %s", name, tool.name, tool.description)
                    
                    self._tools[name] = server_tools
                    
        except Exception as exc:
            logger.error("Error connecting to MCP server %s: %s", name, exc)
            raise

    async def stop(self) -> None:
        """Stop all MCP server connections."""
        for name, session in self._sessions.items():
            try:
                # Sessions are closed automatically via context manager
                logger.info("Closed MCP server connection: %s", name)
            except Exception as exc:
                logger.error("Error closing MCP server %s: %s", name, exc)
        
        self._sessions.clear()
        self._tools.clear()

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get list of all available tools from all connected servers.
        
        Returns:
            List of tool definitions suitable for AI model function calling
        """
        tools = []
        for server_name, server_tools in self._tools.items():
            for tool_name, tool_info in server_tools.items():
                # Format tool for AI function calling
                tools.append({
                    "type": "function",
                    "function": {
                        "name": f"{server_name}__{tool_name}",
                        "description": tool_info.get("description", ""),
                        "parameters": tool_info.get("input_schema", {}),
                    }
                })
        return tools

    async def call_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool on a specific MCP server.
        
        Args:
            server_name: Name of the MCP server
            tool_name: Name of the tool to call
            arguments: Tool arguments
            
        Returns:
            Tool execution result
        """
        if not self._enabled:
            raise RuntimeError("MCP is not enabled")

        session = self._sessions.get(server_name)
        if not session:
            raise ValueError(f"MCP server '{server_name}' not connected")

        server_tools = self._tools.get(server_name, {})
        if tool_name not in server_tools:
            raise ValueError(f"Tool '{tool_name}' not found on server '{server_name}'")

        try:
            result = await session.call_tool(tool_name, arguments)
            return result
        except Exception as exc:
            logger.error("Error calling tool %s.%s: %s", server_name, tool_name, exc)
            raise

    def is_enabled(self) -> bool:
        """Check if MCP is enabled and available."""
        return self._enabled

    def get_servers(self) -> List[str]:
        """Get list of connected server names."""
        return list(self._sessions.keys())


# Global MCP client instance
_mcp_client: Optional[MCPClient] = None
_mcp_lock: Optional[asyncio.Lock] = None


async def get_mcp_client() -> MCPClient:
    """Get or create the global MCP client instance."""
    global _mcp_client, _mcp_lock
    
    if _mcp_client is not None:
        return _mcp_client
    
    if _mcp_lock is None:
        _mcp_lock = asyncio.Lock()
    
    async with _mcp_lock:
        if _mcp_client is None:
            _mcp_client = MCPClient(servers=MCP_SERVERS)
            if _mcp_client.is_enabled():
                await _mcp_client.start()
        return _mcp_client


async def shutdown_mcp_client() -> None:
    """Shutdown the global MCP client."""
    global _mcp_client
    if _mcp_client is not None:
        await _mcp_client.stop()
        _mcp_client = None
