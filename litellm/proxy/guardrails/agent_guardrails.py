from typing import Callable, List, Optional, Any

class Agent:
    """
    An agent object that can process and execute tools, applying guardrails to tool inputs and outputs.
    """
    def __init__(
        self,
        agent_name: str,
        *args,
        tool_input_guardrails: Optional[List[Callable[[Any, str], bool]]] = None,
        tool_output_guardrails: Optional[List[Callable[[Any, Any, str], bool]]] = None,
        **kwargs
    ):
        """
        Initialize the agent.

        Args:
            agent_name: the name of the agent.
            tool_input_guardrails: optional list of guardrails to apply before a tool is executed.
                Each guardrail should accept the tool call data and the agent name as inputs, and return True/False.
            tool_output_guardrails: optional list of guardrails to apply after a tool is executed.
                Each guardrail should accept the tool output, tool call data, and agent name, and return True/False.
        """
        self.agent_name = agent_name
        self.tool_input_guardrails = tool_input_guardrails or []
        self.tool_output_guardrails = tool_output_guardrails or []

    def execute_tool(self, tool_data: Any) -> Any:
        """
        Execute a tool after passing input guardrails.
        Validates the output with output guardrails before returning.
        """
        # Check input guardrails to block harmful actions
        for guard in self.tool_input_guardrails:
            if not guard(tool_data, self.agent_name):
                raise ValueError(f"Action blocked by tool_input_guardrails for agent '{self.agent_name}'.")

        # Simulate executing the tool
        tool_result = f"Executed {getattr(tool_data, 'name', 'tool')}"

        # Check output guardrails to block harmful outputs
        for guard in self.tool_output_guardrails:
            if not guard(tool_result, tool_data, self.agent_name):
                raise ValueError(f"Action blocked by tool_output_guardrails for agent '{self.agent_name}'.")

        return tool_result
