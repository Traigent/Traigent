"""Provider prompt-cache normalization (Traigent#2068).

The load-bearing behaviours, each of which produces a wrong *cost* if it regresses:

  1. every supported provider's shape is read;
  2. ``input_tokens`` comes out cache-EXCLUSIVE regardless of the wire convention,
     because OpenAI reports cached tokens inside ``prompt_tokens`` and Bedrock
     reports them alongside ``inputTokens``;
  3. a provider that did not report a cache field yields ``None``, never ``0``.
"""

import pytest

from traigent.core.cache_usage import CacheUsage, normalize_cache_usage


# ---------------------------------------------------------------------------
# Provider shapes
# ---------------------------------------------------------------------------


def test_openai_chat_completions_shape():
    usage = normalize_cache_usage(
        {"prompt_tokens": 1500, "prompt_tokens_details": {"cached_tokens": 1024}}
    )

    assert usage.provider_shape == "openai_chat"
    assert usage.cache_read_tokens == 1024
    # prompt_tokens INCLUDED the cached tokens, so fresh input is the remainder.
    assert usage.input_tokens == 476
    assert usage.billable_input_tokens == 1500


def test_openai_responses_shape():
    usage = normalize_cache_usage(
        {"input_tokens": 2048, "input_tokens_details": {"cached_tokens": 1024}}
    )

    assert usage.provider_shape == "openai_responses"
    assert usage.input_tokens == 1024
    assert usage.billable_input_tokens == 2048


def test_anthropic_shape():
    usage = normalize_cache_usage(
        {
            "input_tokens": 6,
            "cache_read_input_tokens": 4609,
            "cache_creation_input_tokens": 100,
        }
    )

    assert usage.provider_shape == "anthropic"
    # Anthropic reports cache reads ALONGSIDE input_tokens, so no subtraction.
    assert usage.input_tokens == 6
    assert usage.cache_read_tokens == 4609
    assert usage.cache_creation_tokens == 100
    assert usage.billable_input_tokens == 4715


def test_bedrock_converse_shape_is_the_probe_from_the_issue():
    """The exact payload measured on 2026-07-31, camelCase and disjoint."""
    usage = normalize_cache_usage({"inputTokens": 6, "cacheReadInputTokens": 4609})

    assert usage.provider_shape == "bedrock"
    assert usage.input_tokens == 6
    assert usage.cache_read_tokens == 4609
    # 4,609 tokens that were previously invisible to every cost path.
    assert usage.billable_input_tokens == 4615


def test_gemini_shape():
    usage = normalize_cache_usage({"input_tokens": 900, "cachedContentTokenCount": 300})

    assert usage.provider_shape == "gemini"
    assert usage.cache_read_tokens == 300
    assert usage.cache_creation_tokens is None


# ---------------------------------------------------------------------------
# The disjointness normalization -- the subtlety that makes cost provider-dependent
# ---------------------------------------------------------------------------


def test_the_same_numbers_mean_the_same_cost_across_conventions():
    """1500 total input of which 1024 cached must normalize identically.

    OpenAI states it inclusively, Anthropic exclusively. If the SDK does not pick one
    convention, downstream arithmetic is right for one provider and wrong for the
    other -- which is precisely the provider-dependent error direction #2068 reports.
    """
    inclusive = normalize_cache_usage(
        {"prompt_tokens": 1500, "prompt_tokens_details": {"cached_tokens": 1024}}
    )
    exclusive = normalize_cache_usage(
        {"input_tokens": 476, "cache_read_input_tokens": 1024}
    )

    assert inclusive.input_tokens == exclusive.input_tokens == 476
    assert inclusive.cache_read_tokens == exclusive.cache_read_tokens == 1024
    assert inclusive.billable_input_tokens == exclusive.billable_input_tokens == 1500


def test_subtraction_cannot_drive_fresh_input_negative():
    """A provider reporting more cached than total is malformed, not negative input."""
    usage = normalize_cache_usage(
        {"prompt_tokens": 100, "prompt_tokens_details": {"cached_tokens": 500}}
    )

    assert usage.input_tokens == 0


# ---------------------------------------------------------------------------
# Absent is not zero
# ---------------------------------------------------------------------------


def test_a_silent_provider_is_unknown_not_zero():
    """Amazon Nova omits the keys entirely when no cache is engaged."""
    usage = normalize_cache_usage({"inputTokens": 4615, "outputTokens": 120})

    assert usage.cache_read_tokens is None
    assert usage.cache_creation_tokens is None
    assert usage.is_complete is False
    assert set(usage.unreported_fields) == {
        "cache_read_tokens",
        "cache_creation_tokens",
    }


def test_a_reported_zero_is_a_different_fact_from_silence():
    usage = normalize_cache_usage(
        {"inputTokens": 100, "cacheReadInputTokens": 0, "cacheWriteInputTokens": 0}
    )

    assert usage.cache_read_tokens == 0
    assert usage.cache_creation_tokens == 0
    assert usage.is_complete is True


def test_partial_reporting_names_only_the_missing_field():
    usage = normalize_cache_usage({"input_tokens": 10, "cache_read_input_tokens": 50})

    assert usage.cache_read_tokens == 50
    assert usage.cache_creation_tokens is None
    assert usage.unreported_fields == ("cache_creation_tokens",)


@pytest.mark.parametrize("junk", ["nope", -5, None, {}, True])
def test_unusable_values_are_unknown_rather_than_coerced(junk):
    """A wrong number is worse than a gap -- only the gap is visible downstream."""
    usage = normalize_cache_usage({"inputTokens": 10, "cacheReadInputTokens": junk})

    assert usage.cache_read_tokens is None


def test_a_missing_payload_is_unknown_not_empty():
    for payload in (None, "not-a-dict", []):
        usage = normalize_cache_usage(payload)  # type: ignore[arg-type]
        assert usage.cache_read_tokens is None
        assert usage.is_complete is False


# ---------------------------------------------------------------------------
# Cache-write TTL tiers (TraigentSchema#383)
# ---------------------------------------------------------------------------


def test_anthropic_ttl_split_survives_rather_than_being_flattened():
    """The tiers are priced differently (1.25x vs 2x), so the split must not be lost."""
    usage = normalize_cache_usage(
        {
            "input_tokens": 10,
            "cache_creation": {
                "ephemeral_5m_input_tokens": 1000,
                "ephemeral_1h_input_tokens": 500,
            },
        }
    )

    assert usage.cache_creation_tokens_by_ttl == {
        "ephemeral_5m": 1000,
        "ephemeral_1h": 500,
    }
    # A flat total is still derivable for consumers that only need the aggregate.
    assert usage.cache_creation_tokens == 1500


def test_ttl_split_is_absent_when_the_provider_reports_only_a_total():
    usage = normalize_cache_usage({"inputTokens": 10, "cacheWriteInputTokens": 1500})

    assert usage.cache_creation_tokens == 1500
    assert usage.cache_creation_tokens_by_ttl == {}


# ---------------------------------------------------------------------------
# Metadata surface (#2069: make the contamination visible)
# ---------------------------------------------------------------------------


def test_metadata_marks_an_incomplete_reading_rather_than_looking_authoritative():
    metadata = normalize_cache_usage({"inputTokens": 100}).as_metadata()

    assert metadata["cache_usage_complete"] is False
    assert "cache_read_tokens" in metadata["unreported_usage_fields"]


def test_metadata_of_a_complete_reading_carries_no_caveat():
    metadata = normalize_cache_usage(
        {"inputTokens": 6, "cacheReadInputTokens": 4609, "cacheWriteInputTokens": 0}
    ).as_metadata()

    assert metadata["cache_usage_complete"] is True
    assert "unreported_usage_fields" not in metadata
    assert metadata["cache_read_tokens"] == 4609


def test_billable_input_is_none_when_fresh_input_itself_is_unknown():
    assert CacheUsage().billable_input_tokens is None


# ---------------------------------------------------------------------------
# with_usage() wiring -- the public surface users actually call
# ---------------------------------------------------------------------------


def test_with_usage_records_cache_dimensions_from_a_raw_provider_payload():
    from unittest.mock import patch

    from traigent.api import functions

    with patch.object(functions, "get_trial_context", return_value={"trial": 1}):
        result = functions.with_usage(
            "answer",
            total_cost=0.01,
            provider_usage={"inputTokens": 6, "cacheReadInputTokens": 4609},
        )

    meta = result["__traigent_meta__"]
    assert meta["cache_usage"]["cache_read_tokens"] == 4609
    # Fresh input is recorded from the payload when the caller passed no explicit
    # count, so the cache-exclusive figure is what downstream cost math sees.
    assert meta["usage"]["input_tokens"] == 6


def test_with_usage_is_unchanged_for_callers_that_pass_no_provider_payload():
    from unittest.mock import patch

    from traigent.api import functions

    with patch.object(functions, "get_trial_context", return_value={"trial": 1}):
        result = functions.with_usage(
            "answer", total_cost=0.01, input_tokens=100, output_tokens=50
        )

    meta = result["__traigent_meta__"]
    assert meta["usage"] == {"input_tokens": 100, "output_tokens": 50}
    assert "cache_usage" not in meta


def test_with_usage_keeps_an_explicit_input_count_over_the_payload():
    """An explicitly-passed count is the caller's intent and must win."""
    from unittest.mock import patch

    from traigent.api import functions

    with patch.object(functions, "get_trial_context", return_value={"trial": 1}):
        result = functions.with_usage(
            "answer",
            total_cost=0.01,
            input_tokens=999,
            provider_usage={"inputTokens": 6, "cacheReadInputTokens": 4609},
        )

    assert result["__traigent_meta__"]["usage"]["input_tokens"] == 999
