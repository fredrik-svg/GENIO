"""MCP settings management."""

from __future__ import annotations

import json
import os
from typing import Any, Dict

from pydantic import BaseModel, Field, ConfigDict

# Path to MCP settings file
_SETTINGS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
_SETTINGS_FILE = os.path.join(_SETTINGS_DIR, "mcp_settings.json")


class MCPServerConfig(BaseModel):
    """Configuration for a single MCP server."""
    
    command: str = Field(..., description="Command to start the server")
    args: list[str] = Field(default_factory=list, description="Command arguments")
    env: Dict[str, str] = Field(default_factory=dict, description="Environment variables")
    
    model_config = ConfigDict(extra="forbid")


class MCPSettings(BaseModel):
    """MCP configuration settings."""
    
    enabled: bool = Field(default=False, description="Whether MCP is enabled")
    servers: Dict[str, MCPServerConfig] = Field(
        default_factory=dict,
        description="Configured MCP servers"
    )
    
    model_config = ConfigDict(extra="forbid")


def load_mcp_settings() -> MCPSettings:
    """Load MCP settings from file or return defaults."""
    if not os.path.exists(_SETTINGS_FILE):
        return MCPSettings()
    
    try:
        with open(_SETTINGS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return MCPSettings.model_validate(data)
    except Exception:
        # If file is corrupted or invalid, return defaults
        return MCPSettings()


def save_mcp_settings(settings: MCPSettings) -> None:
    """Save MCP settings to file."""
    os.makedirs(_SETTINGS_DIR, exist_ok=True)
    
    with open(_SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(settings.model_dump(), f, indent=2, ensure_ascii=False)


def get_mcp_servers_config() -> Dict[str, Any]:
    """Get MCP servers configuration in the format expected by MCPClient."""
    settings = load_mcp_settings()
    
    # Convert to dictionary format expected by MCPClient
    servers = {}
    for name, config in settings.servers.items():
        servers[name] = {
            "command": config.command,
            "args": config.args,
            "env": config.env,
        }
    
    return servers


def is_mcp_enabled() -> bool:
    """Check if MCP is enabled in settings."""
    settings = load_mcp_settings()
    return settings.enabled
