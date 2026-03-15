from typing import Any, Dict

import pytest

from litellm.responses.mcp.litellm_proxy_mcp_handler import (
    LiteLLM_Proxy_MCP_Handler,
)
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
