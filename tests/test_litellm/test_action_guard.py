from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
    MCPServerManager,
)
from litellm.proxy._types import MCPTransport
from litellm.responses.mcp.litellm_proxy_mcp_handler import (
    LiteLLM_Proxy_MCP_Handler,
)
from litellm.types.mcp import MCPAuth
from litellm.types.mcp_server.mcp_server_manager import MCPServer
from litellm.types.utils import ActionGuardDecision


@pytest.mark.asyncio
async def test_action_guard_blocks(monkeypatch):
    # Prepare a fake tool_call and mapping
    tool_calls = [
        {"id": "call_1", "name": "test_tool", "arguments": '{"x":1}'},
    ]
    tool_server_map = {"test_tool": "server1"}

    # Fake manager that would have been called if allowed
    class FakeManager:
        async def call_tool(self, **kwargs):
            return None

        def _get_mcp_server_from_tool_name(self, *args, **kwargs):
            return None

    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager",
        FakeManager(),
        raising=False,
    )

    # Guard that blocks all calls
    def blocking_guard(call: Dict[str, Any]):
        return ActionGuardDecision.BLOCK

    results = await LiteLLM_Proxy_MCP_Handler._execute_tool_calls(
        tool_server_map=tool_server_map,
        tool_calls=tool_calls,
        user_api_key_auth=None,
        action_guard=blocking_guard,
    )

    assert isinstance(results, list)
    assert len(results) == 1
    assert "blocked by action_guard" in results[0]["result"].lower()


@pytest.mark.asyncio
async def test_action_guard_allows(monkeypatch):
    tool_calls = [
        {"id": "call_2", "name": "test_tool2", "arguments": '{"y":2}'},
    ]
    tool_server_map = {"test_tool2": "server2"}

    # Fake result object without content -> _parse_mcp_result returns 'Tool executed successfully'
    class FakeResult:
        content = None

    class FakeManager:
        async def call_tool(self, **kwargs):
            return FakeResult()

        def _get_mcp_server_from_tool_name(self, *args, **kwargs):
            return None

    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager",
        FakeManager(),
        raising=False,
    )

    def allowing_guard(call: Dict[str, Any]):
        return ActionGuardDecision.ALLOW

    results = await LiteLLM_Proxy_MCP_Handler._execute_tool_calls(
        tool_server_map=tool_server_map,
        tool_calls=tool_calls,
        user_api_key_auth=None,
        action_guard=allowing_guard,
    )

    assert isinstance(results, list)
    assert len(results) == 1
    assert "tool executed successfully" in results[0]["result"].lower()


def _make_manager_with_server(server_name: str = "server1"):
    """Return an MCPServerManager with a fake registered server."""
    manager = MCPServerManager()
    server = MCPServer(
        server_id="srv-1",
        name=server_name,
        url="https://example.com",
        transport=MCPTransport.http,
        auth_type=MCPAuth.none,
    )
    manager.mcp_servers = [server]
    manager.tool_name_to_mcp_server_name = {f"{server_name}_tool": server_name}
    manager._server_name_to_server = {server_name: server}
    return manager, server


@pytest.mark.asyncio
async def test_action_guard_bool_true_allows():
    """bool True from action_guard should allow the tool call."""
    manager, server = _make_manager_with_server()
    fake_result = MagicMock(content=None)
    manager._call_regular_mcp_tool = AsyncMock(return_value=fake_result)

    def bool_allow_guard(call: Dict[str, Any]) -> bool:
        return True

    result = await manager.call_tool(
        server_name="server1",
        name="tool",
        arguments={},
        action_guard=bool_allow_guard,
    )
    manager._call_regular_mcp_tool.assert_called_once()
    assert result is fake_result


@pytest.mark.asyncio
async def test_action_guard_bool_false_blocks():
    """bool False from action_guard should block the tool call."""
    manager, server = _make_manager_with_server()
    manager._call_regular_mcp_tool = AsyncMock()

    def bool_block_guard(call: Dict[str, Any]) -> bool:
        return False

    result = await manager.call_tool(
        server_name="server1",
        name="tool",
        arguments={},
        action_guard=bool_block_guard,
    )
    manager._call_regular_mcp_tool.assert_not_called()
    assert result.isError is True
    assert "blocked by action_guard" in result.content[0].text.lower()


@pytest.mark.asyncio
async def test_action_guard_async_blocks():
    """Async action_guard returning BLOCK should block the tool call."""
    manager, server = _make_manager_with_server()
    manager._call_regular_mcp_tool = AsyncMock()

    async def async_blocking_guard(call: Dict[str, Any]):
        return ActionGuardDecision.BLOCK

    result = await manager.call_tool(
        server_name="server1",
        name="tool",
        arguments={},
        action_guard=async_blocking_guard,
    )
    manager._call_regular_mcp_tool.assert_not_called()
    assert result.isError is True
    assert "blocked by action_guard" in result.content[0].text.lower()


@pytest.mark.asyncio
async def test_action_guard_async_allows():
    """Async action_guard returning ALLOW should allow the tool call."""
    manager, server = _make_manager_with_server()
    fake_result = MagicMock(content=None)
    manager._call_regular_mcp_tool = AsyncMock(return_value=fake_result)

    async def async_allowing_guard(call: Dict[str, Any]):
        return ActionGuardDecision.ALLOW

    result = await manager.call_tool(
        server_name="server1",
        name="tool",
        arguments={},
        action_guard=async_allowing_guard,
    )
    manager._call_regular_mcp_tool.assert_called_once()
    assert result is fake_result


@pytest.mark.asyncio
async def test_action_guard_unexpected_return_type_blocks():
    """Unknown return type from action_guard should block (safe default)."""
    manager, server = _make_manager_with_server()
    manager._call_regular_mcp_tool = AsyncMock()

    def unexpected_guard(call: Dict[str, Any]):
        return 42  # neither bool, ActionGuardDecision, nor str

    result = await manager.call_tool(
        server_name="server1",
        name="tool",
        arguments={},
        action_guard=unexpected_guard,
    )
    manager._call_regular_mcp_tool.assert_not_called()
    assert result.isError is True
    assert "blocked by action_guard" in result.content[0].text.lower()
