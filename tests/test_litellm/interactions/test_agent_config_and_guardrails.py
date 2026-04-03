from litellm.interactions.litellm_responses_transformation.transformation import (
    LiteLLMResponsesInteractionsConfig,
)
from litellm.interactions.utils import (
    InteractionsAPIRequestUtils,
    normalize_interactions_agent_config,
)
from litellm.llms.gemini.interactions.transformation import (
    GoogleAIStudioInteractionsConfig,
)
from litellm.types.router import GenericLiteLLMParams


def test_normalize_interactions_agent_config_should_merge_guardrails():
    result = normalize_interactions_agent_config(
        agent="deep-research-pro-preview-12-2025",
        agent_config={"type": "dynamic", "thinking_summaries": "enabled"},
        guardrails=["tool-input", "tool-output"],
    )

    assert result == {
        "type": "dynamic",
        "thinking_summaries": "enabled",
        "guardrails": ["tool-input", "tool-output"],
    }


def test_normalize_interactions_agent_config_should_create_dynamic_config_for_agent():
    result = normalize_interactions_agent_config(
        agent="deep-research-pro-preview-12-2025",
        agent_config=None,
        guardrails=["tool-input"],
    )

    assert result == {"type": "dynamic", "guardrails": ["tool-input"]}


def test_requested_optional_params_should_include_agent_config():
    params = {
        "model": None,
        "agent": "deep-research-pro-preview-12-2025",
        "input": "Research the latest model launches",
        "tools": None,
        "system_instruction": None,
        "generation_config": None,
        "stream": None,
        "store": None,
        "background": None,
        "response_modalities": None,
        "response_format": None,
        "response_mime_type": None,
        "previous_interaction_id": None,
        "agent_config": {"type": "dynamic", "guardrails": ["tool-input"]},
        "kwargs": {},
    }

    optional_params = (
        InteractionsAPIRequestUtils.get_requested_interactions_api_optional_params(
            params
        )
    )

    assert optional_params["agent_config"] == {
        "type": "dynamic",
        "guardrails": ["tool-input"],
    }


def test_gemini_transform_request_should_include_agent_config_for_agent_requests():
    config = GoogleAIStudioInteractionsConfig()

    request = config.transform_request(
        model=None,
        agent="deep-research-pro-preview-12-2025",
        input="Research the latest reasoning model releases",
        optional_params={
            "agent_config": {"type": "dynamic", "guardrails": ["tool-input"]}
        },
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert request["agent"] == "deep-research-pro-preview-12-2025"
    assert request["agent_config"] == {
        "type": "dynamic",
        "guardrails": ["tool-input"],
    }


def test_gemini_transform_request_should_not_include_agent_config_for_model_requests():
    config = GoogleAIStudioInteractionsConfig()

    request = config.transform_request(
        model="gemini/gemini-2.5-flash",
        agent=None,
        input="Hello",
        optional_params={
            "agent_config": {"type": "dynamic", "guardrails": ["tool-input"]}
        },
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert request["model"] == "gemini-2.5-flash"
    assert "agent_config" not in request


def test_responses_bridge_should_forward_explicit_guardrails():
    request = LiteLLMResponsesInteractionsConfig.transform_interactions_request_to_responses_request(
        model="gpt-4o",
        input="Hello",
        optional_params={},
        guardrails=["tool-input", "tool-output"],
    )

    assert request["guardrails"] == ["tool-input", "tool-output"]


def test_responses_bridge_should_forward_guardrails_from_agent_config():
    request = LiteLLMResponsesInteractionsConfig.transform_interactions_request_to_responses_request(
        model="gpt-4o",
        input="Hello",
        optional_params={
            "agent_config": {"type": "dynamic", "guardrails": ["tool-input"]}
        },
    )

    assert request["guardrails"] == ["tool-input"]
