import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# OpenAI Agents SDK

Use OpenAI Agents SDK with any LLM provider through LiteLLM Proxy.

The [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) is a lightweight framework for building multi-agent workflows. It includes an official LiteLLM extension that lets you use any of the 100+ supported providers.

## Quick Start

### 1. Install Dependencies

```bash
pip install "openai-agents[litellm]"
```

### 2. Add Model to Config

```yaml title="config.yaml"
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: "openai/gpt-4o"
      api_key: "os.environ/OPENAI_API_KEY"

  - model_name: claude-sonnet
    litellm_params:
      model: "anthropic/claude-3-5-sonnet-20241022"
      api_key: "os.environ/ANTHROPIC_API_KEY"

  - model_name: gemini-pro
    litellm_params:
      model: "gemini/gemini-2.0-flash-exp"
      api_key: "os.environ/GEMINI_API_KEY"
```

### 3. Start LiteLLM Proxy

```bash
litellm --config config.yaml
```

### 4. Use with Proxy

<Tabs>
<TabItem value="proxy" label="Via Proxy">

```python
from agents import Agent, Runner
from agents.extensions.models.litellm_model import LitellmModel


def block_harmful_tool_input(tool_data: dict, agent_name: str) -> bool:
    args = tool_data.get("arguments", {})
    command = str(args.get("command", ""))
    return "rm -rf /" not in command


def block_sensitive_tool_output(tool_data: dict, agent_name: str) -> bool:
    output_text = str(tool_data.get("result_text", ""))
    return "AWS_SECRET_ACCESS_KEY" not in output_text

# Point to LiteLLM proxy
agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant.",
    model=LitellmModel(
        model="claude-sonnet",  # Model from config.yaml
        api_key="sk-1234",      # LiteLLM API key
        base_url="http://localhost:4000"
    ),
    tool_input_guardrails=[block_harmful_tool_input],
    tool_output_guardrails=[block_sensitive_tool_output],
)

result = await Runner.run(agent, "What is LiteLLM?")
print(result.final_output)
```

</TabItem>
<TabItem value="direct" label="Direct (No Proxy)">

```python
from agents import Agent, Runner
from agents.extensions.models.litellm_model import LitellmModel

# Use any provider directly
agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant.",
    model=LitellmModel(
        model="anthropic/claude-3-5-sonnet-20241022",
        api_key="your-anthropic-key"
    )
)

result = await Runner.run(agent, "What is LiteLLM?")
print(result.final_output)
```

</TabItem>
</Tabs>

## Track Usage

Enable usage tracking to monitor token consumption:

```python
from agents import Agent, ModelSettings
from agents.extensions.models.litellm_model import LitellmModel

agent = Agent(
    name="Assistant",
    model=LitellmModel(model="claude-sonnet", api_key="sk-1234"),
    model_settings=ModelSettings(include_usage=True)
)

result = await Runner.run(agent, "Hello")
print(result.context_wrapper.usage)  # Token counts
```

## Tool Guardrails

Use `tool_input_guardrails` and `tool_output_guardrails` on `Agent(...)` to enforce
tool safety checks. Both arguments are optional and default to `None`.

- `tool_input_guardrails`: callable or list of callables with signature
  `(tool_call_data: dict, agent_name: str) -> bool`
- `tool_output_guardrails`: callable or list of callables with signature
  `(tool_output_data: dict, agent_name: str) -> bool`

If any guardrail returns `False` (or raises), the tool call is blocked.

## Environment Variables

| Variable | Value | Description |
|----------|-------|-------------|
| `LITELLM_BASE_URL` | `http://localhost:4000` | LiteLLM proxy URL |
| `LITELLM_API_KEY` | `sk-1234` | Your LiteLLM API key |

## Related Resources

- [OpenAI Agents SDK Documentation](https://openai.github.io/openai-agents-python/)
- [LiteLLM Extension Docs](https://openai.github.io/openai-agents-python/models/litellm/)
- [LiteLLM Proxy Quick Start](../proxy/quick_start)
