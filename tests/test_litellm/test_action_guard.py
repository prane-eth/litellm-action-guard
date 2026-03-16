import asyncio
import inspect
from importlib import import_module
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock

import pytest

import litellm
from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
    MCPServerManager,
)
from litellm.proxy._experimental.mcp_server.utils import add_server_prefix_to_name
from litellm.proxy._types import MCPTransport
from litellm.responses.mcp.litellm_proxy_mcp_handler import (
    LiteLLM_Proxy_MCP_Handler,
)
from litellm.types.mcp import MCPAuth
from litellm.types.mcp_server.mcp_server_manager import MCPServer
from litellm.types.utils import ActionGuardDecision


# -----------------------------
# Non-MCP action_guard coverage
# -----------------------------


def test_completion_accepts_action_guard():
    def guard(call: Dict[str, Any]):
        return ActionGuardDecision.ALLOW

    response = litellm.completion(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hello"}],
        mock_response="ok",
        action_guard=guard,
    )

    assert response is not None


@pytest.mark.asyncio
async def test_acompletion_accepts_action_guard(monkeypatch):
    captured: Dict[str, Any] = {}

    def fake_completion(**kwargs):
        captured["action_guard"] = kwargs.get("action_guard")
        return {"id": "1", "choices": []}

    monkeypatch.setattr("litellm.main.completion", fake_completion)

    def guard(call: Dict[str, Any]):
        return ActionGuardDecision.ALLOW

    response = await litellm.acompletion(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hello"}],
        custom_llm_provider="openai",
        action_guard=guard,
    )

    assert captured["action_guard"] is guard
    assert response == {"id": "1", "choices": []}


# -------------------------
# MCP-mandatory test cases
# -------------------------


@pytest.mark.asyncio
async def test_action_guard_blocks_mcp(monkeypatch):
    from mcp.types import CallToolResult, TextContent

    # Prepare a fake tool_call and mapping
    tool_calls = [
        {"id": "call_1", "name": "test_tool", "arguments": '{"x":1}'},
    ]
    tool_server_map = {"test_tool": "server1"}

    # Fake manager that would have been called if allowed
    class FakeManager:
        async def call_tool(self, **kwargs):
            guard = kwargs.get("action_guard")
            assert guard is not None
            decision = guard(
                {
                    "name": kwargs.get("name"),
                    "arguments": kwargs.get("arguments"),
                    "server_name": kwargs.get("server_name"),
                }
            )
            if decision == ActionGuardDecision.BLOCK:
                return CallToolResult(
                    content=[
                        TextContent(type="text", text="Tool call blocked by action_guard")
                    ],
                    isError=True,
                )
            return CallToolResult(
                content=[TextContent(type="text", text="Tool executed successfully")],
                isError=False,
            )

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
async def test_action_guard_allows_mcp(monkeypatch):
    tool_calls = [
        {"id": "call_2", "name": "test_tool2", "arguments": '{"y":2}'},
    ]
    tool_server_map = {"test_tool2": "server2"}

    # Fake result object without content -> _parse_mcp_result returns 'Tool executed successfully'
    class FakeResult:
        def __init__(self):
            self.content = None
            self.isError = False

    class FakeManager:
        async def call_tool(self, **kwargs):
            guard = kwargs.get("action_guard")
            assert guard is not None
            decision = guard(
                {
                    "name": kwargs.get("name"),
                    "arguments": kwargs.get("arguments"),
                    "server_name": kwargs.get("server_name"),
                }
            )
            assert decision == ActionGuardDecision.ALLOW
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


def _make_manager_with_mcp_server(server_name: str = "server1"):
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
    prefixed_tool_name = add_server_prefix_to_name("tool", server_name)
    manager.tool_name_to_mcp_server_name = {prefixed_tool_name: server_name}
    manager._server_name_to_server = {server_name: server}
    manager._get_mcp_server_from_tool_name = MagicMock(return_value=server)
    return manager, server


@pytest.mark.asyncio
async def test_action_guard_bool_true_blocks_mcp():
    """bool True from action_guard should block (unsupported return type)."""
    manager, server = _make_manager_with_mcp_server()
    manager._call_regular_mcp_tool = AsyncMock()

    def bool_allow_guard(call: Dict[str, Any]) -> bool:
        return True

    result = await manager.call_tool(
        server_name="server1",
        name="tool",
        arguments={},
        action_guard=bool_allow_guard,
    )
    manager._call_regular_mcp_tool.assert_not_called()
    assert result.isError is True
    assert "blocked by action_guard" in result.content[0].text.lower()


@pytest.mark.asyncio
async def test_action_guard_bool_false_blocks_mcp():
    """bool False from action_guard should block the tool call."""
    manager, server = _make_manager_with_mcp_server()
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
async def test_action_guard_async_blocks_mcp():
    """Async action_guard returning BLOCK should block the tool call."""
    manager, server = _make_manager_with_mcp_server()
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
async def test_action_guard_exception_blocks_mcp():
    """Exceptions inside action_guard should block the tool call."""
    manager, server = _make_manager_with_mcp_server()
    manager._call_regular_mcp_tool = AsyncMock()

    def raising_guard(call: Dict[str, Any]):
        raise RuntimeError("guard failed")

    result = await manager.call_tool(
        server_name="server1",
        name="tool",
        arguments={},
        action_guard=raising_guard,
    )
    manager._call_regular_mcp_tool.assert_not_called()
    assert result.isError is True
    assert "action_guard exception" in result.content[0].text.lower()


@pytest.mark.asyncio
async def test_action_guard_async_allows_mcp():
    """Async action_guard returning ALLOW should allow the tool call."""
    manager, server = _make_manager_with_mcp_server()
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
async def test_action_guard_unexpected_return_type_blocks_mcp():
    """Unknown return type from action_guard should block (safe default)."""
    manager, server = _make_manager_with_mcp_server()
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


@pytest.mark.asyncio
async def test_action_guard_string_return_type_blocks_mcp():
    """String return values are unsupported; only ActionGuardDecision is accepted."""
    manager, server = _make_manager_with_mcp_server()
    manager._call_regular_mcp_tool = AsyncMock()

    def string_guard(call: Dict[str, Any]):
        return "ALLOW"

    result = await manager.call_tool(
        server_name="server1",
        name="tool",
        arguments={},
        action_guard=string_guard,
    )
    manager._call_regular_mcp_tool.assert_not_called()
    assert result.isError is True
    assert "blocked by action_guard" in result.content[0].text.lower()


@pytest.mark.asyncio
@pytest.mark.parametrize("entrypoint", ["completion", "acompletion"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "guard_mode, expected_blocked",
    [
        ("sync_allow", False),
        ("sync_block", True),
        ("async_allow", False),
        ("async_block", True),
    ],
)
async def test_completion_and_acompletion_action_guard_block_allow_streaming_mcp(
    monkeypatch,
    entrypoint: str,
    stream: bool,
    guard_mode: str,
    expected_blocked: bool,
):
    captured: Dict[str, Any] = {}
    tool_list = [
        {
            "type": "function",
            "function": {
                "name": "mock_tool",
                "description": "mock tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    async def fake_acompletion_with_mcp(**kwargs):
        guard = kwargs.get("action_guard")
        captured["action_guard"] = guard
        captured["stream"] = kwargs.get("stream")

        guard_decision = guard(
            {
                "name": "mock_tool",
                "arguments": {"x": 1},
                "server_name": "mock_server",
            }
        )
        if inspect.isawaitable(guard_decision):
            guard_decision = await guard_decision

        blocked = guard_decision == ActionGuardDecision.BLOCK
        return {"blocked": blocked, "stream": kwargs.get("stream")}

    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_should_use_litellm_mcp_gateway",
        staticmethod(lambda tools: True),
    )
    chat_mcp_module = import_module("litellm.responses.mcp.chat_completions_handler")
    monkeypatch.setattr(
        chat_mcp_module,
        "acompletion_with_mcp",
        fake_acompletion_with_mcp,
    )

    if guard_mode == "sync_allow":
        def guard(call: Dict[str, Any]):
            return ActionGuardDecision.ALLOW
    elif guard_mode == "sync_block":
        def guard(call: Dict[str, Any]):
            return ActionGuardDecision.BLOCK
    elif guard_mode == "async_allow":
        async def guard(call: Dict[str, Any]):
            return ActionGuardDecision.ALLOW
    else:
        async def guard(call: Dict[str, Any]):
            return ActionGuardDecision.BLOCK

    if entrypoint == "completion":
        response = litellm.completion(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hello"}],
            tools=tool_list,
            stream=stream,
            action_guard=guard,
        )
        if asyncio.iscoroutine(response):
            response = await response
    else:
        response = await litellm.acompletion(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hello"}],
            tools=tool_list,
            stream=stream,
            action_guard=guard,
        )

    assert captured["action_guard"] is guard
    assert captured["stream"] is stream
    assert response == {"blocked": expected_blocked, "stream": stream}


@pytest.mark.asyncio
async def test_aresponses_forwards_action_guard_to_mcp_handler(monkeypatch):
    captured: Dict[str, Any] = {}

    async def fake_aresponses_api_with_mcp(**kwargs):
        captured["action_guard"] = kwargs.get("action_guard")
        return {"ok": True}

    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_should_use_litellm_mcp_gateway",
        staticmethod(lambda tools: True),
    )
    responses_main_module = import_module("litellm.responses.main")
    monkeypatch.setattr(
        responses_main_module,
        "aresponses_api_with_mcp",
        fake_aresponses_api_with_mcp,
    )

    def guard(call: Dict[str, Any]):
        return ActionGuardDecision.ALLOW

    response = await litellm.aresponses(
        input="hello",
        model="gpt-4o",
        tools=[],
        action_guard=guard,
    )

    assert captured["action_guard"] is guard
    assert response == {"ok": True}
