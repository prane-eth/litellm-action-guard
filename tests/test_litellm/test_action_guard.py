import asyncio
import inspect
from importlib import import_module
from typing import Any, Callable, Dict
from unittest.mock import AsyncMock, MagicMock

import pytest

import litellm
from litellm.constants import ACTION_GUARD_BLOCKED_MESSAGE
from litellm._action_guard import call_action_guard_async, call_action_guard_sync
from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
    MCPServerManager,
)
from litellm.proxy._experimental.mcp_server.utils import add_server_prefix_to_name
from litellm.proxy._types import MCPTransport
from litellm.responses.mcp.litellm_proxy_mcp_handler import (
    LiteLLM_Proxy_MCP_Handler,
)
from litellm.types.llms.openai import ResponsesAPIResponse
from litellm.types.mcp import MCPAuth
from litellm.types.mcp_server.mcp_server_manager import MCPServer
from litellm.types.utils import ActionGuardDecision, ModelResponse


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
async def test_completion_with_tools_action_guard_blocking():
    tool_list = [
        {
            "type": "function",
            "function": {
                "name": "mock_tool",
                "description": "mock tool",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]
    response = litellm.completion(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hello"}],
        mock_tool_calls=[
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "mock_tool", "arguments": '{"x": 1}'},
            }
        ],
        mock_response="fallback",
        tools=tool_list,
        action_guard=lambda _: ActionGuardDecision.BLOCK,
    )

    if asyncio.iscoroutine(response):
        response = await response

    assert response.choices[0].message.tool_calls is None
    assert (
        "blocked by action_guard" in (response.choices[0].message.content or "").lower()
    )


@pytest.mark.asyncio
async def test_acompletion_accepts_action_guard(monkeypatch):
    captured: Dict[str, Any] = {}

    def fake_completion(**kwargs):
        captured["action_guard"] = kwargs.get("action_guard")
        return {"id": "1", "choices": []}

    monkeypatch.setattr("litellm.main.completion", fake_completion)

    def allowing_guard(call: Dict[str, Any]):
        return ActionGuardDecision.ALLOW

    response = await litellm.acompletion(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hello"}],
        custom_llm_provider="openai",
        action_guard=allowing_guard,
    )

    assert captured["action_guard"] is allowing_guard
    assert response == {"id": "1", "choices": []}


def test_completion_streaming_accepts_action_guard():
    """
    Non-MCP: streaming completion should accept action_guard without error.
    """

    def guard(call: Dict[str, Any]):
        return ActionGuardDecision.ALLOW

    stream = litellm.completion(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hello"}],
        mock_response="ok",
        stream=True,
        action_guard=guard,
    )

    # Ensure we can iterate the stream without errors
    for _ in stream:
        break


@pytest.mark.asyncio
async def test_acompletion_streaming_forwards_action_guard(monkeypatch):
    """
    Non-MCP: streaming acompletion should forward action_guard to core implementation.
    """

    captured: Dict[str, Any] = {}

    async def fake_acompletion(**kwargs):
        captured["action_guard"] = kwargs.get("action_guard")

        # Return a dummy async iterator; litellm.acompletion wrapper will not touch network
        class _DummyStream:
            def __aiter__(self):
                return self

            async def __anext__(self):
                raise StopAsyncIteration

        return _DummyStream()

    # Patch the *inner* completion used by litellm.acompletion so no real network is used.
    monkeypatch.setattr("litellm.main.completion", fake_acompletion)

    def guard(call: Dict[str, Any]):
        return ActionGuardDecision.ALLOW

    stream = await litellm.acompletion(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hello"}],
        custom_llm_provider="openai",
        stream=True,
        action_guard=guard,
    )

    # Ensure action_guard was forwarded
    assert captured["action_guard"] is guard

    # Ensure we can iterate async stream without errors
    async for _ in stream:
        break


def test_call_action_guard_sync_with_sync_guard():
    """Test async function passed to call_action_guard_sync"""

    def guard(call: Dict[str, Any]):
        return ActionGuardDecision.ALLOW

    result = call_action_guard_sync(action_guard=guard, guard_input={"x": 1})
    assert result == ActionGuardDecision.ALLOW


def test_call_action_guard_sync_with_async_guard_no_running_loop():
    """Test async function passed to call_action_guard_sync."""

    async def aguard(call: Dict[str, Any]):
        return ActionGuardDecision.BLOCK

    result = call_action_guard_sync(action_guard=aguard, guard_input={"x": 1})
    assert result == ActionGuardDecision.BLOCK


@pytest.mark.asyncio
async def test_call_action_guard_sync_with_async_guard_inside_running_loop():
    """
    Sync code can be invoked from async contexts (e.g. user calling a sync wrapper
    from within an async app). Ensure async guards still work.
    """

    async def aguard(call: Dict[str, Any]):
        await asyncio.sleep(0)
        return ActionGuardDecision.ALLOW

    result = call_action_guard_sync(action_guard=aguard, guard_input={"x": 1})
    assert result == ActionGuardDecision.ALLOW


@pytest.mark.asyncio
async def test_call_action_guard_async_with_sync_guard():
    def guard(call: Dict[str, Any]):
        return ActionGuardDecision.BLOCK

    result = await call_action_guard_async(action_guard=guard, guard_input={"x": 1})
    assert result == ActionGuardDecision.BLOCK


@pytest.mark.asyncio
async def test_call_action_guard_async_with_async_guard():
    async def aguard(call: Dict[str, Any]):
        return ActionGuardDecision.ALLOW

    result = await call_action_guard_async(action_guard=aguard, guard_input={"x": 1})
    assert result == ActionGuardDecision.ALLOW


@pytest.mark.asyncio
async def test_non_mcp_action_guard_blocks_internal_skills_tool_async_guard():
    """
    Non-MCP: Skills hook auto-executes internal tools. Ensure async guards can block.
    """

    from litellm.proxy.hooks.litellm_skills.main import SkillsInjectionHook

    hook = SkillsInjectionHook(max_iterations=1)

    class FakeExecutor:
        def __init__(self):
            self.called = False

        def execute(self, **kwargs):
            self.called = True
            return {"output": "ok"}

    class _Fn:
        name = "litellm_code_execution"
        arguments = '{"code": "print(1)"}'

    class _ToolCall:
        id = "tc_1"
        function = _Fn()

    async def aguard(call: Dict[str, Any]):
        return ActionGuardDecision.BLOCK

    executor = FakeExecutor()
    result = await hook._execute_code_tool(
        tool_call=_ToolCall(),
        skill_files={},
        executor=executor,
        generated_files=[],
        action_guard=aguard,
    )
    assert executor.called is False
    assert "blocked by action_guard" in result.lower()


@pytest.mark.asyncio
async def test_non_mcp_action_guard_allows_internal_skills_tool_sync_guard():
    """
    Non-MCP: Skills hook auto-executes internal tools. Ensure sync guards can allow.
    """

    from litellm.proxy.hooks.litellm_skills.main import SkillsInjectionHook

    hook = SkillsInjectionHook(max_iterations=1)

    class FakeExecutor:
        def __init__(self):
            self.called = False

        def execute(self, **kwargs):
            self.called = True
            return {"output": "ok"}

    class _Fn:
        name = "litellm_code_execution"
        arguments = '{"code": "print(1)"}'

    class _ToolCall:
        id = "tc_1"
        function = _Fn()

    def guard(call: Dict[str, Any]):
        return ActionGuardDecision.ALLOW

    executor = FakeExecutor()
    result = await hook._execute_code_tool(
        tool_call=_ToolCall(),
        skill_files={},
        executor=executor,
        generated_files=[],
        action_guard=guard,
    )
    assert executor.called is True
    assert "blocked by action_guard" not in result.lower()


# -------------------------
# MCP-mandatory test cases
# -------------------------


@pytest.mark.asyncio
async def test_action_guard_blocks_mcp(monkeypatch):
    # Prepare a fake tool_call and mapping
    tool_calls = [
        {"id": "call_1", "name": "test_tool", "arguments": '{"x":1}'},
    ]
    tool_server_map = {"test_tool": "server1"}

    class FakeManager:
        def __init__(self):
            self.call_tool = AsyncMock()

        def _get_mcp_server_from_tool_name(self, *args, **kwargs):
            return None

    fake_manager = FakeManager()

    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager",
        fake_manager,
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
    fake_manager.call_tool.assert_not_called()


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


@pytest.mark.asyncio
async def test_acompletion_with_mcp_action_guard_block_skips_follow_up(monkeypatch):
    initial_response = ModelResponse(
        id="chatcmpl-test",
        model="gpt-4o",
        created=123,
        object="chat.completion",
        choices=[
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "tool", "arguments": "{}"},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
    )

    acompletion_mock = AsyncMock(side_effect=[initial_response])
    monkeypatch.setattr(litellm, "acompletion", acompletion_mock)
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_parse_mcp_tools",
        staticmethod(
            lambda tools: (
                [{"type": "mcp", "server_url": "litellm_proxy/mcp/test"}],
                [],
            )
        ),
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_process_mcp_tools_without_openai_transform",
        AsyncMock(return_value=([], {"tool": "server1"})),
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_transform_mcp_tools_to_openai",
        staticmethod(lambda *args, **kwargs: []),
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_should_auto_execute_tools",
        staticmethod(lambda *args, **kwargs: True),
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_execute_tool_calls",
        AsyncMock(
            return_value=[
                {
                    "tool_call_id": "call-1",
                    "result": ACTION_GUARD_BLOCKED_MESSAGE,
                    "name": "tool",
                }
            ]
        ),
    )

    response = await import_module(
        "litellm.responses.mcp.chat_completions_handler"
    ).acompletion_with_mcp(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hello"}],
        tools=[{"type": "mcp", "server_url": "litellm_proxy/mcp/test"}],
        action_guard=lambda _: ActionGuardDecision.BLOCK,
    )

    assert isinstance(response, ModelResponse)
    assert response.choices[0].message.tool_calls is None
    assert response.choices[0].message.content == ACTION_GUARD_BLOCKED_MESSAGE
    assert acompletion_mock.await_count == 1


@pytest.mark.asyncio
async def test_aresponses_action_guard_block_skips_follow_up(monkeypatch):
    initial_response = ResponsesAPIResponse(
        id="resp_test",
        object="response",
        created_at=123,
        status="completed",
        error=None,
        incomplete_details=None,
        instructions=None,
        max_output_tokens=None,
        model="gpt-4o",
        output=[
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call-1",
                "name": "tool",
                "arguments": "{}",
                "status": "completed",
            }
        ],
        parallel_tool_calls=True,
        previous_response_id=None,
        reasoning={"effort": None, "summary": None},
        store=False,
        temperature=1.0,
        text={"format": {"type": "text"}},
        tool_choice="auto",
        tools=[],
        top_p=1.0,
        truncation="disabled",
        user=None,
        metadata={},
    )

    aresponses_mock = AsyncMock(side_effect=[initial_response])
    monkeypatch.setattr("litellm.responses.main.aresponses", aresponses_mock)
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_parse_mcp_tools",
        staticmethod(
            lambda tools: (
                [{"type": "mcp", "server_url": "litellm_proxy/mcp/test"}],
                [],
            )
        ),
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_process_mcp_tools_without_openai_transform",
        AsyncMock(return_value=([], {"tool": "server1"})),
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_transform_mcp_tools_to_openai",
        staticmethod(lambda *args, **kwargs: []),
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_should_auto_execute_tools",
        staticmethod(lambda *args, **kwargs: True),
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_execute_tool_calls",
        AsyncMock(
            return_value=[
                {
                    "tool_call_id": "call-1",
                    "result": ACTION_GUARD_BLOCKED_MESSAGE,
                    "name": "tool",
                }
            ]
        ),
    )

    response = await litellm.aresponses(
        input="hello",
        model="gpt-4o",
        tools=[{"type": "mcp", "server_url": "litellm_proxy/mcp/test"}],
        action_guard=lambda _: ActionGuardDecision.BLOCK,
    )

    assert isinstance(response, ResponsesAPIResponse)
    assert response.output_text == ACTION_GUARD_BLOCKED_MESSAGE
    assert aresponses_mock.await_count == 1


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
    manager.registry = {server.server_id: server}
    prefixed_tool_name = add_server_prefix_to_name("tool", server_name)
    manager.tool_name_to_mcp_server_name_mapping = {
        "tool": server_name,
        prefixed_tool_name: server_name,
    }
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

    def _sync_allow_guard(call: Dict[str, Any]) -> ActionGuardDecision:
        return ActionGuardDecision.ALLOW

    def _sync_block_guard(call: Dict[str, Any]) -> ActionGuardDecision:
        return ActionGuardDecision.BLOCK

    async def _async_allow_guard(call: Dict[str, Any]) -> ActionGuardDecision:
        return ActionGuardDecision.ALLOW

    async def _async_block_guard(call: Dict[str, Any]) -> ActionGuardDecision:
        return ActionGuardDecision.BLOCK

    guard: Callable[[Dict[str, Any]], Any]
    if guard_mode == "sync_allow":
        guard = _sync_allow_guard
    elif guard_mode == "sync_block":
        guard = _sync_block_guard
    elif guard_mode == "async_allow":
        guard = _async_allow_guard
    else:
        guard = _async_block_guard

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
