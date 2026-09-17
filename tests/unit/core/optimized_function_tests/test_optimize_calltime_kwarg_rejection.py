"""Call-time kwarg rejection for decorator-only parameters (issue #1683 Bug A)
and general allowlist validation of every unknown call-time kwarg (issue
#1705, the deferred follow-up).

``.optimize(warm_start_from=...)`` (and sibling decorator-only options) used to
be silently swallowed into ``**algorithm_kwargs`` and stored inertly in
``BaseOptimizer.algorithm_config`` with zero effect. Per the no-silent-legacy
policy they must now hard-fail loudly at call time with an actionable message.
Issue #1705 generalizes this: the set of decorator-only options is now
*derived* from an allowlist of keys genuinely consumed at call time, so a
plain typo of a real key (never a decorator option at all) is rejected too,
not just the pre-enumerated decorator-only names.
"""

import pytest

from traigent.core.optimized_function import (
    OptimizedFunction,
    _decorator_only_optimize_params,
)


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

        deny = _decorator_only_optimize_params()
        assert deny <= set(_OPTIMIZE_DEFAULTS)

    def test_denylist_is_derived_not_hand_maintained(self):
        """issue #1705: the decorator-only set is computed from
        _OPTIMIZE_DEFAULTS minus the call-time allowlist, so a newly added
        decorator-only default is rejected automatically without anyone
        having to remember to extend a separate hand-written list."""
        from traigent.api.decorators import _OPTIMIZE_DEFAULTS

        deny = _decorator_only_optimize_params()
        expected = (
            frozenset(_OPTIMIZE_DEFAULTS)
            - OptimizedFunction._EXPLICIT_OPTIMIZE_SIGNATURE_PARAMS
            - OptimizedFunction._CALL_TIME_ALGORITHM_KWARGS_ALLOWLIST
        )
        assert deny == expected

    def test_unknown_decorator_default_key_would_be_rejected(self, opt_func):
        """Simulates "a newly added decorator-only default" (the exact risk
        named in issue #1705): monkeypatch a fake decorator default that is
        not on the call-time allowlist and confirm it hard-fails, proving the
        auto-updating property without needing a real new decorator option."""
        import traigent.api.decorators as decorators_module

        fake_key = "__fake_decorator_only_option_1705__"
        assert fake_key not in decorators_module._OPTIMIZE_DEFAULTS
        assert fake_key not in OptimizedFunction._CALL_TIME_ALGORITHM_KWARGS_ALLOWLIST
        decorators_module._OPTIMIZE_DEFAULTS[fake_key] = None
        _decorator_only_optimize_params.cache_clear()
        try:
            with pytest.raises(TypeError):
                opt_func._prepare_algorithm_kwargs({fake_key: "anything"})
        finally:
            del decorators_module._OPTIMIZE_DEFAULTS[fake_key]
            _decorator_only_optimize_params.cache_clear()


class TestUnknownCallTimeKwargRejection:
    """issue #1705: a key that is neither a decorator option nor on the
    call-time allowlist is a typo, not a silent no-op."""

    @pytest.mark.parametrize(
        "kwarg",
        [
            "pralel_config",  # typo of parallel_config
            "cost_limitt",  # typo of cost_limit
            "totally_unknown_option",
            "warm_start",  # typo of warm_start_from (not even close enough to alias)
        ],
    )
    def test_unknown_kwarg_raises_typeerror(self, opt_func, kwarg):
        with pytest.raises(TypeError, match=r"Unknown keyword argument"):
            opt_func._prepare_algorithm_kwargs({kwarg: "anything"})

    def test_unknown_kwarg_message_names_the_key(self, opt_func):
        with pytest.raises(TypeError) as excinfo:
            opt_func._prepare_algorithm_kwargs({"totally_unknown_option": 1})
        assert "totally_unknown_option" in str(excinfo.value)


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
            # Additional allowlisted keys newly covered by issue #1705's
            # allowlist (previously accepted only by omission from the
            # denylist, with no positive test pinning them):
            ("metric_name", "cost"),
            ("metric_include_pruned", True),
            ("estimated_calls_per_example", 3),
            ("tvl_parameter_agents", ["agent_a"]),
            ("order", {"model": 0}),  # parameter_order alias
            ("max_grid_combinations", 1000),
            ("objective_weights", {"quality": 1.0}),
            ("optimizer_ready_timeout", 30.0),
            ("cloud_optimizer_ready_timeout", 30.0),
            # Legacy objective kwargs: _prepare_algorithm_kwargs() must let
            # these through so _validate_objectives_input() can reject them
            # downstream with its own "no longer supported" ValueError
            # (tests/unit/core/test_objectives_edge_cases.py::
            # test_runtime_objective_kwargs_rejected) instead of a generic
            # "unknown keyword argument" TypeError here.
            ("objective_weights", {"accuracy": 0.7}),
            ("objective_orientations", {"cost": "minimize"}),
        ],
    )
    def test_consumed_kwarg_passes_validation(self, opt_func, kwarg, value):
        merged = opt_func._prepare_algorithm_kwargs({kwarg: value})
        assert merged[kwarg] == value

    def test_parallel_trials_still_rejected_with_original_message(self, opt_func):
        with pytest.raises(ValueError, match="parallel_trials is not a valid"):
            opt_func._prepare_algorithm_kwargs({"parallel_trials": 4})
