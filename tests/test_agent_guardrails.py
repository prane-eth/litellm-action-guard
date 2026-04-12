import pytest
from typing import Any
from litellm.proxy.guardrails.agent_guardrails import Agent

class MockTool:
    def __init__(self, name: str, is_safe: bool = True):
        self.name = name
        self.is_safe = is_safe

def pass_input_guardrail(tool_data: Any, agent_name: str) -> bool:
    return True

def fail_input_guardrail(tool_data: Any, agent_name: str) -> bool:
    return False

def pass_output_guardrail(output: Any, tool_data: Any, agent_name: str) -> bool:
    return True

def fail_output_guardrail(output: Any, tool_data: Any, agent_name: str) -> bool:
    return False

def test_agent_guardrails_pass():
    agent = Agent(
        "test_agent", 
        tool_input_guardrails=[pass_input_guardrail], 
        tool_output_guardrails=[pass_output_guardrail]
    )
    result = agent.execute_tool(MockTool("test"))
    assert result == "Executed test"

def test_agent_input_guardrail_fail():
    agent = Agent(
        "test_agent", 
        tool_input_guardrails=[fail_input_guardrail], 
        tool_output_guardrails=[pass_output_guardrail]
    )
    with pytest.raises(ValueError, match="Action blocked by tool_input_guardrails for agent 'test_agent'"):
        agent.execute_tool(MockTool("test_unsafe"))

def test_agent_output_guardrail_fail():
    agent = Agent(
        "test_agent", 
        tool_input_guardrails=[pass_input_guardrail], 
        tool_output_guardrails=[fail_output_guardrail]
    )
    with pytest.raises(ValueError, match="Action blocked by tool_output_guardrails for agent 'test_agent'"):
        agent.execute_tool(MockTool("test_unsafe_output"))

def test_agent_multiple_guardrails():
    agent = Agent(
        "test_agent", 
        tool_input_guardrails=[pass_input_guardrail, fail_input_guardrail],
        tool_output_guardrails=[]
    )
    with pytest.raises(ValueError, match="Action blocked by tool_input_guardrails for agent 'test_agent'"):
        agent.execute_tool(MockTool("test_multiple"))
