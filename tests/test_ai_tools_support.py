"""Tests for AI provider tools support."""

import pytest
from backend.ai import EchoProvider


@pytest.mark.asyncio
async def test_echo_provider_accepts_tools_parameter():
    """Test that EchoProvider accepts tools parameter without error."""
    provider = EchoProvider()
    
    # Test with tools parameter
    tools = [
        {
            "type": "function",
            "function": {
                "name": "test_function",
                "description": "Test function",
                "parameters": {}
            }
        }
    ]
    
    result = await provider.chat_reply("test message", tools=tools)
    assert result == "test message"


@pytest.mark.asyncio
async def test_echo_provider_accepts_context_and_tools():
    """Test that EchoProvider accepts both context_sections and tools."""
    provider = EchoProvider()
    
    context_sections = ["Context 1", "Context 2"]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "test_function",
                "description": "Test function",
                "parameters": {}
            }
        }
    ]
    
    result = await provider.chat_reply(
        "test message",
        context_sections=context_sections,
        tools=tools
    )
    assert result == "test message"


@pytest.mark.asyncio
async def test_echo_provider_backward_compatible():
    """Test that EchoProvider still works without tools parameter."""
    provider = EchoProvider()
    
    # Should work without tools parameter
    result = await provider.chat_reply("test message")
    assert result == "test message"
    
    # Should work with context but no tools
    result = await provider.chat_reply(
        "test message",
        context_sections=["Context"]
    )
    assert result == "test message"
