"""Pre-run estimate reads every model key and every output-cap spelling.

#2807: a model under any ``*_model`` knob (a cascade's ``expert_model``) was
never priced, yet the figure was labelled a conservative upper bound.
#2815: ``max_output_tokens`` and the other cap spellings were ignored (500 tokens
assumed).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from traigent.core.cost_estimator import OUTPUT_TOKEN_CAP_KEYS, CostEstimator
from traigent.core.model_config_keys import is_model_config_key, is_model_suffix_key
from traigent.core.optimized_function import (
    _extract_config_space_model_ids,
    _is_model_config_key,
)
from traigent.core.orchestrator import OptimizationOrchestrator
from traigent.evaluators.base import Dataset, EvaluationExample

_extract_models = OptimizationOrchestrator._extract_model_candidates_from_config_space
_extract_cap = OptimizationOrchestrator._extract_max_output_tokens_from_config_space
_extract_cap_key = OptimizationOrchestrator._max_output_tokens_key


def _dataset(n: int = 3) -> Dataset:
    return Dataset(
        examples=[
            EvaluationExample(input_data={"q": "hi"}, expected_output="x")
            for _ in range(n)
        ]
    )


def _estimate(space: dict) -> float:
    estimator = CostEstimator(
        MagicMock(is_mock_mode=False),
        max_trials=2,
        max_total_examples=None,
        candidate_models=_extract_models(space),
        configured_max_output_tokens=_extract_cap(space),
    )
    return estimator.estimate_optimization_cost(_dataset())


class TestModelKeys2807:
    def test_expert_model_is_collected(self) -> None:
        space = {"model": ["gpt-4o-mini"], "expert_model": ["gpt-4o"]}
        assert _extract_models(space) == ("gpt-4o-mini", "gpt-4o")

    def test_expert_arm_raises_the_estimate(self) -> None:
        cheap = _estimate({"model": ["gpt-4o-mini"]})
        cascade = _estimate({"model": ["gpt-4o-mini"], "expert_model": ["gpt-4o"]})
        expensive = _estimate({"model": ["gpt-4o"]})
        assert cascade > cheap * 5
        assert cascade == pytest.approx(expensive)

    def test_base_and_expert_keys_without_plain_model_key(self) -> None:
        space = {"base_model": ["gpt-4o-mini"], "expert_model": ["gpt-4o"]}
        assert set(_extract_models(space)) == {"gpt-4o-mini", "gpt-4o"}

    def test_union_not_first_key_wins(self) -> None:
        space = {"model": ["gpt-4o"], "model_name": ["gpt-4o-mini"]}
        assert set(_extract_models(space)) == {"gpt-4o", "gpt-4o-mini"}

    def test_non_model_keys_are_not_collected(self) -> None:
        assert _extract_models({"temperature": [0.1], "model_temperature": ["x"]}) == ()
        assert _extract_models({"use_model": [True, False]}) == ()

    def test_estimator_and_both_preflights_share_one_key_predicate(self) -> None:
        assert _is_model_config_key is is_model_config_key
        space = {"model": ["gpt-4o-mini"], "expert_model": ["claude-sonnet-5-5"]}
        assert set(_extract_config_space_model_ids(space)) == set(
            _extract_models(space)
        )
        for key in ("model", "model_name", "model_id", "engine"):
            assert is_model_config_key(key)
        assert is_model_suffix_key("judge_model")
        assert not is_model_config_key("judge_model")
        assert not is_model_config_key("modelling")
        assert not is_model_config_key(3)


class TestOutputCapKeys2815:
    @pytest.mark.parametrize("key", OUTPUT_TOKEN_CAP_KEYS)
    def test_every_cap_spelling_sets_the_assumption(self, key: str) -> None:
        space = {key: [256, 8000]}
        assert _extract_cap(space) == 8000
        assert _extract_cap_key(space) == key

    def test_alias_table_spellings_are_all_recognised(self) -> None:
        from traigent.integrations.config import FrameworkConstraints

        aliases = FrameworkConstraints.COMMON_CONSTRAINTS["max_tokens"].aliases
        assert set(aliases) <= set(OUTPUT_TOKEN_CAP_KEYS)
        assert "max_output_tokens" in OUTPUT_TOKEN_CAP_KEYS

    def test_maximum_over_spellings(self) -> None:
        space = {"max_tokens": [100], "max_output_tokens": [900]}
        assert _extract_cap(space) == 900
        assert _extract_cap_key(space) == "max_output_tokens"

    def test_estimate_scales_with_output_cap(self) -> None:
        base = _estimate({"model": ["gpt-4o"]})
        capped = _estimate({"model": ["gpt-4o"], "max_output_tokens": [8000]})
        assert capped > base * 4

    def test_assumptions_line_names_the_key(self) -> None:
        estimator = CostEstimator(
            MagicMock(is_mock_mode=False),
            max_trials=1,
            max_total_examples=None,
            model_name="gpt-4o",
            configured_max_output_tokens=8000,
            max_output_tokens_key="max_output_tokens",
        )
        line = estimator._assumptions_summary()
        assert "output tokens/example=8000" in line
        assert "max_output_tokens" in line

    def test_no_cap_key_keeps_default_wording(self) -> None:
        estimator = CostEstimator(
            MagicMock(is_mock_mode=False),
            max_trials=1,
            max_total_examples=None,
            model_name="gpt-4o",
        )
        assert "config-space key" not in estimator._assumptions_summary()


class TestSuffixKeysDoNotBreakPreflights:
    """A user ``*_model`` knob may hold a non-LLM model (embedding, reranker)."""

    def test_estimate_leaves_out_unpriced_suffix_values(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        space = {"model": ["gpt-4o-mini"], "embedding_model": ["all-MiniLM-L6-v2"]}
        with caplog.at_level("INFO"):
            assert _extract_models(space) == ("gpt-4o-mini",)
        assert "embedding_model" in caplog.text

    def test_unpriced_expert_model_stays_so_conservative_fallback_fires(self) -> None:
        space = {"model": ["gpt-4o-mini"], "expert_model": ["acme-paid-expert-1"]}
        assert "acme-paid-expert-1" in _extract_models(space)
        fallback = CostEstimator(
            MagicMock(is_mock_mode=False),
            max_trials=2,
            max_total_examples=None,
            candidate_models=("acme-paid-expert-1",),
        ).estimate_optimization_cost(_dataset())
        assert _estimate(space) == pytest.approx(fallback)
        assert _estimate(space) > _estimate({"model": ["gpt-4o-mini"]})

    @pytest.mark.parametrize(
        "key",
        ["embedding_model", "embedder_model", "rerank_model", "reranker_model"],
    )
    def test_named_non_llm_keys_are_left_out(self, key: str) -> None:
        space = {"model": ["gpt-4o-mini"], key: ["acme-unpriced-1"]}
        assert _extract_models(space) == ("gpt-4o-mini",)

    def test_free_local_embedder_does_not_inflate_the_estimate(self) -> None:
        base = {"model": ["gpt-4o-mini"]}
        with_embedder = {**base, "embedding_model": ["all-MiniLM-L6-v2"]}
        assert _estimate(with_embedder) == _estimate(base)

    def test_preflight_ids_skip_unpriced_suffix_values(self) -> None:
        space = {"model": ["gpt-4o-mini"], "embedding_model": ["all-MiniLM-L6-v2"]}
        assert _extract_config_space_model_ids(space) == ["gpt-4o-mini"]

    def test_priced_suffix_values_are_kept_in_both(self) -> None:
        space = {"model": ["gpt-4o-mini"], "expert_model": ["gpt-4o"]}
        assert set(_extract_config_space_model_ids(space)) == {
            "gpt-4o-mini",
            "gpt-4o",
        }
        assert set(_extract_models(space)) == {"gpt-4o-mini", "gpt-4o"}

    def test_engine_and_model_id_are_always_checked_by_preflights(self) -> None:
        space = {
            "model": ["gpt-4o-mini"],
            "engine": ["faiss", "annoy"],
            "model_id": ["acme-private-1"],
        }
        assert {"faiss", "annoy", "acme-private-1"} <= set(_extract_models(space))
        assert set(_extract_config_space_model_ids(space)) == {
            "gpt-4o-mini",
            "faiss",
            "annoy",
            "acme-private-1",
        }

    def test_fixed_keys_are_always_kept_even_when_unpriced(self) -> None:
        space = {"model": ["my-private-model"]}
        assert _extract_config_space_model_ids(space) == ["my-private-model"]
