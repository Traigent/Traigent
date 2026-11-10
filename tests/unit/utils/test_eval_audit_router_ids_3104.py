"""model_family sees through router prefixes and vendor segments (Traigent#3104)."""

from __future__ import annotations

import pytest

from traigent.utils.eval_audit import UNKNOWN_FAMILY, model_family


@pytest.mark.parametrize(
    ("model", "family"),
    [
        ("gpt-4o", "openai"),
        ("anthropic/claude-3-5-sonnet", "anthropic"),
        ("azure/gpt-4o", "openai"),
        ("vertex_ai/gemini-1.5-pro", "google"),
        ("groq/llama3-8b-8192", "meta"),
        ("bedrock/anthropic.claude-3-sonnet-20240229-v1:0", "anthropic"),
        ("us.anthropic.claude-3-5-sonnet-20241022-v2:0", "anthropic"),
        ("eu.anthropic.claude-3-haiku-20240307-v1:0", "anthropic"),
        ("openrouter/anthropic/claude-3.5-sonnet", "anthropic"),
        ("openrouter/openai/gpt-4o", "openai"),
        ("bedrock/meta.llama3-70b-instruct-v1:0", "meta"),
        ("bedrock/cohere.command-r-plus-v1:0", "cohere"),
        ("bedrock/mistral.mistral-large-2402-v1:0", "mistral"),
    ],
)
def test_router_ids_resolve_to_vendor_family(model: str, family: str) -> None:
    assert model_family(model) == family


@pytest.mark.parametrize(
    "model",
    ["", "bedrock/amazon.titan-text-express-v1", "openrouter/acme/foo-1", "us.foo.bar"],
)
def test_unrecognised_router_ids_stay_unknown(model: str) -> None:
    assert model_family(model) == UNKNOWN_FAMILY


@pytest.mark.parametrize(
    ("model", "family"),
    [
        ("openrouter/x-ai/grok-2", "xai"),
        ("fireworks/accounts/fireworks/models/llama-v3p1-8b", "meta"),
        ("openrouter/meta-llama/llama-3-70b", "meta"),
    ],
)
def test_router_vendor_segments_still_classify(model: str, family: str) -> None:
    assert model_family(model) == family


@pytest.mark.parametrize(
    "model",
    [
        # Non-vendor orgs whose later segment happens to start with a family
        # prefix ("gpt", "claude"): these must not anchor a family.
        "together/EleutherAI/gpt-j-6b",
        "huggingface/openai-community/gpt2",
        "hf/EleutherAI/gpt-neox-20b",
        "sagemaker/my-endpoint.claude-proxy",
    ],
)
def test_non_vendor_orgs_with_family_like_model_stay_unknown(model: str) -> None:
    assert model_family(model) == UNKNOWN_FAMILY
