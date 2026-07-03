"""Call-time kwarg rejection for decorator-only parameters (issue #1683 Bug A).

``.optimize(warm_start_from=...)`` (and sibling decorator-only options) used to
be silently swallowed into ``**algorithm_kwargs`` and stored inertly in
``BaseOptimizer.algorithm_config`` with zero effect. Per the no-silent-legacy
policy they must now hard-fail loudly at call time with an actionable message.
"""

import pytest

from traigent.core.optimized_function import OptimizedFunction


@pytest.fixture
def opt_func(simple_function, sample_config_space, sample_objectives, sample_dataset):
    return OptimizedFunction(
        func=simple_function,
        configuration_space=sample_config_space,
        objectives=sample_objectives,
        eval_dataset=sample_dataset,
        max_trials=2,
    )


class TestWarmStartFromCallTimeRejection:
    """warm_start_from at call time raises with a move-to-decorator message."""

    @pytest.mark.asyncio
    async def test_optimize_raises_typeerror(self, opt_func):
        with pytest.raises(TypeError, match=r"warm_start_from.*decorator"):
            await opt_func.optimize(warm_start_from="exp_prior_123")

    def test_optimize_sync_raises_typeerror(self, opt_func):
        with pytest.raises(TypeError, match=r"warm_start_from.*decorator"):
            opt_func.optimize_sync(warm_start_from="exp_prior_123")

    def test_message_is_actionable(self, opt_func):
        with pytest.raises(TypeError) as excinfo:
            opt_func._prepare_algorithm_kwargs({"warm_start_from": "exp_prior_123"})
        message = str(excinfo.value)
        assert "warm_start_from" in message
        assert "@traigent.optimize" in message
        assert "warm_start_from=..." in message  # tells the user exactly where


class TestDecoratorOnlyKwargDenylist:
    """Every denylisted decorator-only param is rejected at call time."""

    @pytest.mark.parametrize(
        "kwarg",
        [
            "warm_start_from",
            "eval_dataset",
            "experiment_name",
            "default_config",
            "constraints",
            "safety_constraints",
            "agents",
            "smart_pruning",
            "auto_load_best",
            "best_config_source",
        ],
    )
    def test_denylisted_kwarg_rejected(self, opt_func, kwarg):
        with pytest.raises(TypeError, match=rf"{kwarg}.*decorator"):
            opt_func._prepare_algorithm_kwargs({kwarg: "anything"})

    def test_multiple_rejected_kwargs_all_named(self, opt_func):
        with pytest.raises(TypeError) as excinfo:
            opt_func._prepare_algorithm_kwargs(
                {"warm_start_from": "exp_1", "eval_dataset": "data.jsonl"}
            )
        message = str(excinfo.value)
        assert "warm_start_from" in message
        assert "eval_dataset" in message

    def test_denylist_is_subset_of_decorator_defaults(self):
        """Every denylisted key must be a real decorator option, so the
        move-it-to-the-decorator message is always truthful."""
        from traigent.api.decorators import _OPTIMIZE_DEFAULTS

        deny = OptimizedFunction._DECORATOR_ONLY_OPTIMIZE_PARAMS
        assert deny <= set(_OPTIMIZE_DEFAULTS)


class TestConsumedAlgorithmKwargsStillAccepted:
    """Kwargs legitimately consumed downstream must not be rejected."""

    @pytest.mark.parametrize(
        "kwarg,value",
        [
            ("cost_limit", 1.0),
            ("cost_approved", True),
            ("plateau_window", 3),
            ("plateau_epsilon", 0.01),
            ("semantic_saturation", {"enabled": True}),
            ("cache_policy", "allow_repeats"),
            ("parallel_config", {"trial_concurrency": 2}),
            ("max_total_examples", 10),
            ("samples_include_pruned", True),
            ("parameter_order", {"model": 0}),
            ("seed", 42),
            ("random_seed", 42),
            ("invocations_per_example", 2),
            ("metric_limit", 5.0),
            ("tie_breakers", ["cost"]),
        ],
    )
    def test_consumed_kwarg_passes_validation(self, opt_func, kwarg, value):
        merged = opt_func._prepare_algorithm_kwargs({kwarg: value})
        assert merged[kwarg] == value

    def test_parallel_trials_still_rejected_with_original_message(self, opt_func):
        with pytest.raises(ValueError, match="parallel_trials is not a valid"):
            opt_func._prepare_algorithm_kwargs({"parallel_trials": 4})
