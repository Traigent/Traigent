from __future__ import annotations

import pytest

from traigent.utils.cost_calculator import cost_from_tokens

# Static catalog only: do not call AWS list_inference_profiles in CI. This keeps
# the scheduled check credential-free and catches bundled litellm price-map drift.
CURATED_BEDROCK_INFERENCE_PROFILE_IDS = [
    # Seeded from the SDK's Bedrock catalog/skill material:
    # - traigent/config_generator/catalog/tvar_catalog.v1.json uses
    #   bedrock/us.anthropic.claude-haiku-4-5.
    # - traigent/config/models.yaml lists the Bedrock Anthropic and Meta families.
    # - examples/integrations/bedrock/bedrock_hello.py anchors Amazon Bedrock usage.
    "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    "us.anthropic.claude-3-haiku-20240307-v1:0",
    "us.amazon.nova-micro-v1:0",
    "us.amazon.nova-lite-v1:0",
    "us.meta.llama3-1-8b-instruct-v1:0",
    "us.meta.llama3-1-70b-instruct-v1:0",
]

COMMON_EXAMPLE_MODEL_IDS = [
    "gpt-4o-mini",
    "gpt-4o",
    "claude-3-haiku-20240307",
    "claude-3-5-sonnet-20241022",
    "claude-haiku-4-5-20251001",
]


@pytest.mark.parametrize(
    "model_id",
    CURATED_BEDROCK_INFERENCE_PROFILE_IDS + COMMON_EXAMPLE_MODEL_IDS,
)
def test_catalog_model_resolves_nonzero_price(model_id: str) -> None:
    input_cost, output_cost = cost_from_tokens(
        1,
        1,
        model_id,
        strict=True,
    )
    assert input_cost + output_cost > 0.0
