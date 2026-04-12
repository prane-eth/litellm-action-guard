# Agent Guardrails Example

## Example Code

```python
from litellm.proxy.guardrails.agent_guardrails import Agent
from typing import Any

# Define a simple Tool class or structure
class Tool:
    def __init__(self, name: str, is_safe: bool = True):
        self.name = name
        self.is_safe = is_safe

# Define input guardrail
def check_harmful_input(tool_data: Any, agent_name: str) -> bool:
    print(f"[{agent_name}] Checking input safety for tool: {getattr(tool_data, 'name', 'unknown')}")
    # Return True to allow, False to block
    return getattr(tool_data, 'is_safe', False)

# Define output guardrail
def check_harmful_output(tool_result: Any, tool_data: Any, agent_name: str) -> bool:
    print(f"[{agent_name}] Checking output safety: {tool_result}")
    # Return True to allow, False to block
    return "harmful" not in tool_result.lower()

def main():
    agent = Agent(
        agent_name="SecurityAgent",
        tool_input_guardrails=[check_harmful_input],
        tool_output_guardrails=[check_harmful_output]
    )

    safe_tool = Tool("Calculate", is_safe=True)
    unsafe_tool = Tool("DeleteFiles", is_safe=False)

    try:
        result = agent.execute_tool(safe_tool)
        print("Success:", result)
    except ValueError as e:
        print("Error:", e)

    try:
        result = agent.execute_tool(unsafe_tool)
        print("Success:", result)
    except ValueError as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
```

## Run Test Cases

```bash
# Assuming you have pytest installed
# Run the agent guardrails test suite
uv run pytest tests/test_agent_guardrails.py
```
