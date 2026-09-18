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
    _explicit_optimize_signature_params,
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
            - _explicit_optimize_signature_params()
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


class TestShippedCallSitesSurviveTheAllowlist:
    """The allowlist makes ``.optimize()`` reject unknown kwargs, so every
    shipped call site that passes one stops working the day it lands.

    CI does not execute ``examples/``, ``walkthrough/`` or ``plugins/``, so a
    unit-test-green PR can still ship a crash into our most-copied code. Two
    such sites existed when this check was written, both passing a kwarg that
    had never had any effect:

      * ``plugins/traigent-ui/.../streamlit_core/optimization.py`` passed
        ``algorithm_params={"n_initial_points": 2}`` -- a name that appeared
        nowhere else in the repository, so the UI's bayesian hint was inert
        and the UI would have raised ``TypeError`` on every run.
      * ``examples/core/multi-objective-tradeoff/run_many_providers.py``
        passed ``model=args.model`` behind a ``--model`` CLI flag; ``model``
        is a *configuration_space* dimension, not an ``.optimize()``
        parameter, so the flag never pinned anything.

    This is the same defect class #2362 removed (``show_progress=``), which
    is why it is worth a standing check rather than a one-time sweep.
    """

    # Receivers whose ``.optimize()`` is a DIFFERENT method with its own
    # signature (orchestrator/optimizer/client APIs take func=, dataset=,
    # function_name=, invoker=, evaluator=, ...). Only OptimizedFunction's
    # ``.optimize()`` is governed by the allowlist.
    _FOREIGN_RECEIVERS = frozenset(
        {"orchestrator", "optimizer", "client", "mock_client", "self", "engine"}
    )

    _SHIPPED_DIRS = ("examples", "walkthrough", "plugins")

    def _offending_call_sites(self, repo_root=None, scan_dirs=None):
        """Walk the shipped trees and return every rejected .optimize() kwarg.

        ``repo_root``/``scan_dirs`` are injectable ONLY so the red control below
        can run this exact function against a planted offender. The control used
        to re-implement the walk, which meant a green control proved nothing
        about the code that actually guards the repo.
        """
        import ast
        from pathlib import Path

        from traigent.core.optimized_function import (
            OptimizedFunction,
            _decorator_only_optimize_params,
        )

        repo_root = (
            Path(__file__).resolve().parents[4]
            if repo_root is None
            else Path(repo_root)
        )
        allowed = (
            OptimizedFunction._CALL_TIME_ALGORITHM_KWARGS_ALLOWLIST
            | _explicit_optimize_signature_params()
        )
        # Every explicit parameter of .optimize() is routed by Python to the
        # named parameter and can never reach **algorithm_kwargs, so read the
        # real signature rather than trusting the curated subset above.
        import inspect

        allowed = allowed | set(
            inspect.signature(OptimizedFunction.optimize).parameters
        )
        decorator_only = _decorator_only_optimize_params()

        offenders = []
        scanned = 0
        for base in self._SHIPPED_DIRS if scan_dirs is None else scan_dirs:
            root = repo_root / base
            if not root.is_dir():
                continue
            for path in root.rglob("*.py"):
                try:
                    tree = ast.parse(path.read_text(encoding="utf-8"))
                except (SyntaxError, UnicodeDecodeError):
                    continue
                scanned += 1
                decorated = {
                    id(sub)
                    for node in ast.walk(tree)
                    for deco in getattr(node, "decorator_list", [])
                    for sub in ast.walk(deco)
                }
                for node in ast.walk(tree):
                    if id(node) in decorated:
                        continue  # @traigent.optimize(...) is the decorator
                    if not (
                        isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "optimize"
                    ):
                        continue
                    recv = node.func.value
                    if isinstance(recv, ast.Name) and (
                        recv.id in {"traigent", "tg"}
                        or recv.id in self._FOREIGN_RECEIVERS
                    ):
                        continue
                    for kw in node.keywords:
                        if kw.arg is None:
                            continue
                        if kw.arg in allowed:
                            continue
                        why = (
                            "decorator-only" if kw.arg in decorator_only else "unknown"
                        )
                        offenders.append(
                            f"{path.relative_to(repo_root)}:{node.lineno} "
                            f"passes {kw.arg}= ({why})"
                        )
        return offenders, scanned

    def test_no_shipped_call_site_passes_a_rejected_kwarg(self):
        offenders, scanned = self._offending_call_sites()

        # Without this, the guard passes vacuously if _SHIPPED_DIRS ever stops
        # matching the tree (a rename, a move, a wrong repo_root): zero files
        # scanned yields zero offenders and a green test that checks nothing.
        assert scanned > 100, (
            f"the census only parsed {scanned} files across {self._SHIPPED_DIRS}; "
            "the scan roots no longer match the repository layout, so a green "
            "result here would be meaningless"
        )

        assert not offenders, (
            "These shipped .optimize() call sites pass a kwarg the allowlist "
            "now rejects, so they raise TypeError at runtime even though CI "
            "never executes them:\n  " + "\n  ".join(sorted(offenders))
        )

    def test_the_census_can_actually_find_an_offender(self, tmp_path):
        """Red control: the REAL walk must flag a planted call site.

        This calls `_offending_call_sites` itself rather than a copy of it.
        An earlier version re-implemented the walk inline, so it could not
        detect the failure modes that matter -- an over-narrow receiver
        filter, a wrong repo root, or a missing scan directory would leave the
        guard vacuously green while the control stayed green too.
        """
        shipped = tmp_path / "examples"
        shipped.mkdir()
        (shipped / "planted.py").write_text(
            "import traigent\n"
            "@traigent.optimize(configuration_space={'t': [0.0]})\n"
            "def answer(t: float = 0.0) -> str:\n"
            "    return 'x'\n"
            "r = answer.optimize(max_trials=2, algorithm_params={'n': 2})\n",
            encoding="utf-8",
        )

        offenders, scanned = self._offending_call_sites(
            repo_root=tmp_path, scan_dirs=("examples",)
        )
        assert scanned == 1

        assert any("algorithm_params" in o for o in offenders), (
            "the guard's own walk missed a planted offender, so a green run of "
            f"it would prove nothing; got {offenders}"
        )

    def test_the_census_accepts_a_clean_shipped_tree(self, tmp_path):
        """The other half of the control: no false positives.

        A walk that flagged everything would also 'catch' the planted offender
        above, so pin that a legitimate call site stays clean -- including the
        decorator form and a foreign receiver, the two shapes the filter exists
        to distinguish.
        """
        shipped = tmp_path / "examples"
        shipped.mkdir()
        (shipped / "clean.py").write_text(
            "import traigent\n"
            "@traigent.optimize(configuration_space={'t': [0.0]}, offline=True)\n"
            "def answer(t: float = 0.0) -> str:\n"
            "    return 'x'\n"
            "r = answer.optimize(max_trials=2, cost_limit=1.0)\n"
            "s = orchestrator.optimize(func=answer, dataset=[], function_name='x')\n",
            encoding="utf-8",
        )

        offenders, scanned = self._offending_call_sites(
            repo_root=tmp_path, scan_dirs=("examples",)
        )
        assert scanned == 1
        assert offenders == []


class TestEveryRegisteredOptimizersOptionsAreAccepted:
    """The check that was missing, and that would have caught the break.

    ``.optimize()`` forwards ``**algorithm_kwargs`` into the chosen optimizer's
    constructor, so every explicit parameter of every registered optimizer is
    by definition consumed at call time. The first version of this PR checked
    call-time kwargs against a HAND-WRITTEN allowlist and rejected four of them
    -- ``batch_config``, ``pareto_frontier_size``, ``base_optimizer`` and
    ``remote_enabled`` -- with an error message asserting they were "not
    consumed by any optimizer at call time", which was false: on develop those
    calls ran and the values reached the optimizer.

    All 52 tests in this file passed while that was true, because none of them
    derived anything from the optimizer registry. This class does.
    """

    @staticmethod
    def _registry_params() -> dict[str, set[str]]:
        import inspect

        import traigent  # noqa: F401 - import populates the registry
        from traigent.optimizers.registry import _OPTIMIZER_REGISTRY

        out: dict[str, set[str]] = {}
        for name, cls in _OPTIMIZER_REGISTRY.items():
            try:
                parameters = inspect.signature(cls.__init__).parameters
            except (TypeError, ValueError):
                continue
            out[name] = {
                p
                for p, param in parameters.items()
                if p != "self"
                and param.kind
                not in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                )
            }
        return out

    def test_the_registry_is_populated(self) -> None:
        """Guard against a vacuous pass: an empty registry proves nothing."""
        params = self._registry_params()
        assert len(params) >= 4, (
            f"only {len(params)} optimizers registered; the assertions below "
            "would pass without checking anything"
        )

    def test_no_registered_optimizer_option_is_rejected(self, opt_func) -> None:
        """Every constructor parameter of every registered optimizer is accepted.

        Except the two Traigent supplies itself. ``config_space`` and
        ``objectives`` are real ``__init__`` parameters on every optimizer, but
        ``get_optimizer(algorithm, config_space, objectives, **kwargs)`` already
        passes them positionally -- accepting them from the caller as well is
        not "it works today", it is an internal ``got multiple values for
        argument`` TypeError three frames down. See
        ``TestOrchestratorSuppliedOptimizerParams``.
        """
        from traigent.core.optimized_function import (
            _ORCHESTRATOR_SUPPLIED_OPTIMIZER_PARAMS,
        )

        rejected: dict[str, str] = {}
        for optimizer_name, parameters in self._registry_params().items():
            for parameter in parameters - _ORCHESTRATOR_SUPPLIED_OPTIMIZER_PARAMS:
                try:
                    opt_func._prepare_algorithm_kwargs({parameter: None})
                except TypeError as exc:
                    rejected[f"{optimizer_name}.{parameter}"] = str(exc)[:80]

        assert not rejected, (
            "these are real constructor parameters of registered optimizers, so "
            "passing them to .optimize() works today and must keep working:\n  "
            + "\n  ".join(f"{k}: {v}" for k, v in sorted(rejected.items()))
        )

    @pytest.mark.parametrize(
        "kwarg,value",
        [
            ("batch_config", {"batch_size": 2}),
            ("pareto_frontier_size", 7),
            ("base_optimizer", "grid"),
            ("remote_enabled", True),
        ],
    )
    def test_the_four_that_regressed_specifically(self, opt_func, kwarg, value) -> None:
        """Named individually so a future reader sees exactly what broke."""
        opt_func._prepare_algorithm_kwargs({kwarg: value})

    def test_a_typo_of_a_registry_option_is_still_rejected(self, opt_func) -> None:
        """Widening to the registry must not turn the check off."""
        with pytest.raises(TypeError, match=r"Unknown keyword argument"):
            opt_func._prepare_algorithm_kwargs({"pareto_frontier_sizee": 7})

    def test_the_error_message_no_longer_overclaims(self, opt_func) -> None:
        """It used to say "not consumed by any optimizer", which was false.

        The message is the thing a user acts on, so it has to describe the
        actual test that was applied.
        """
        with pytest.raises(TypeError) as excinfo:
            opt_func._prepare_algorithm_kwargs({"totally_unknown_option": 1})
        text = str(excinfo.value)
        assert "registered optimizer" in text
        assert "runtime override" in text


class TestTheAllowlistRereadsTheRegistry:
    """The registry is mutable, so the derived allowlist must not be cached.

    ``_registered_optimizer_init_params`` carries a docstring saying in as many
    words that it is deliberately NOT cached, because ``register_optimizer``
    supports plugins and a cache would pin whatever happened to be registered
    at the first ``.optimize()`` call. An independent reviewer put
    ``@lru_cache(maxsize=1)`` back on it -- the exact thing the docstring
    forbids -- and all 60 tests in this module stayed green: every one of them
    reads the allowlist at most once per process, so a stale cache is
    indistinguishable from a fresh read.

    The ordering below is the whole test. Reading the allowlist BEFORE the
    plugin registers is what populates a hypothetical cache; asserting
    acceptance AFTER is what proves the second read saw the new optimizer.
    """

    def test_an_optimizer_registered_after_first_use_has_its_kwargs_accepted(self):
        from traigent.core.optimized_function import _registered_optimizer_init_params
        from traigent.optimizers.base import BaseOptimizer
        from traigent.optimizers.registry import _OPTIMIZER_REGISTRY, register_optimizer

        class _PluginOptimizer(BaseOptimizer):
            def __init__(self, *args, plugin_only_knob: int = 0, **kwargs):
                super().__init__(*args, **kwargs)
                self.plugin_only_knob = plugin_only_knob

            def suggest(self, *args, **kwargs):  # pragma: no cover - never run
                raise NotImplementedError

            def update(self, *args, **kwargs):  # pragma: no cover - never run
                raise NotImplementedError

        # Read it first: this is the call that would fill a cache.
        before = _registered_optimizer_init_params()
        assert "plugin_only_knob" not in before

        name = "_test_plugin_optimizer_1705"
        register_optimizer(name, _PluginOptimizer)
        try:
            after = _registered_optimizer_init_params()
        finally:
            _OPTIMIZER_REGISTRY.pop(name, None)

        assert "plugin_only_knob" in after, (
            "the allowlist did not pick up an optimizer registered after its "
            "first read -- it is being cached, which the function's own "
            "docstring forbids. A plugin's legitimate options would be "
            "rejected as typos for the life of the process."
        )


def test_the_signature_param_set_is_read_from_the_signature():
    """The set must equal ``optimize``'s real parameters, with nothing curated.

    The hand-written version this replaced was nine names behind the signature
    it claimed to mirror. That drift was inert -- a name is only misclassified
    when it appears in BOTH the signature and ``_OPTIMIZE_DEFAULTS``, and none
    of the nine did -- but it is the exact failure mode this PR exists to stop:
    a list someone has to remember to update.
    """
    import inspect

    expected = set()
    for method in (OptimizedFunction.optimize, OptimizedFunction.optimize_sync):
        for name, parameter in inspect.signature(method).parameters.items():
            if name == "self":
                continue
            if parameter.kind in (
                inspect.Parameter.VAR_KEYWORD,
                inspect.Parameter.VAR_POSITIONAL,
            ):
                continue
            expected.add(name)

    assert set(_explicit_optimize_signature_params()) == expected

    # The nine the curated list had lost. Named individually so the failure
    # says which one came back, not just that a set comparison differed.
    for lost in (
        "budget",
        "callbacks",
        "progress_bar",
        "save_to",
        "strategy",
        "strategy_params",
        "surrogate_evaluator",
        "surrogate_evaluator_name",
        "timeout",
    ):
        assert lost in _explicit_optimize_signature_params(), lost


def test_deriving_the_signature_params_rejects_exactly_what_it_did_before():
    """No behaviour change today -- the drift was inert, and this pins that.

    If deriving the set had quietly started accepting a name that used to be
    rejected, that would be a silent widening of the call-time surface hiding
    inside a refactor. It does not: measured, the decorator-only set is
    identical either way.
    """
    from traigent.api.decorators import _OPTIMIZE_DEFAULTS

    curated = frozenset(
        {
            "algorithm",
            "max_trials",
            "custom_evaluator",
            "configuration_space",
            "objectives",
            "tvl_spec",
            "tvl_environment",
            "tvl",
        }
    )
    as_curated = (
        frozenset(_OPTIMIZE_DEFAULTS)
        - curated
        - OptimizedFunction._CALL_TIME_ALGORITHM_KWARGS_ALLOWLIST
    )
    assert set(_decorator_only_optimize_params()) == set(as_curated)


class TestOrchestratorSuppliedOptimizerParams:
    """``config_space`` passed the allowlist and then crashed three frames down.

    It is a genuine ``BaseOptimizer.__init__`` parameter, so the registry-derived
    allowlist admitted it -- and then ``get_optimizer(algorithm, config_space,
    objectives, **kwargs)`` passes it positionally as well. Measured on the
    previous head:

        .optimize(config_space={"temperature": [9.9]})
        -> TypeError: InteractiveOptimizer.__init__() got multiple values for
           argument 'config_space'

    An internal error naming a class the caller never mentioned, from an API
    that was supposed to answer typos with a clear message. Deriving the
    allowlist from constructor signatures is right; it just has to subtract the
    arguments Traigent fills in itself.
    """

    def test_config_space_is_rejected_with_a_message_naming_the_real_parameter(
        self, opt_func
    ):
        with pytest.raises(TypeError) as excinfo:
            opt_func._prepare_algorithm_kwargs({"config_space": {"temperature": [9.9]}})

        message = str(excinfo.value)
        assert "config_space" in message
        assert "configuration_space" in message, (
            "the message must name the parameter the caller should have used"
        )
        assert "InteractiveOptimizer" not in message, (
            "the caller must not be shown an internal optimizer class they "
            f"never named: {message}"
        )

    def test_the_orchestrator_supplied_names_are_not_on_the_derived_allowlist(self):
        from traigent.core.optimized_function import (
            _ORCHESTRATOR_SUPPLIED_OPTIMIZER_PARAMS,
            _registered_optimizer_init_params,
        )

        derived = _registered_optimizer_init_params()
        assert not (derived & _ORCHESTRATOR_SUPPLIED_OPTIMIZER_PARAMS), (
            "an argument Traigent passes positionally must not also be accepted "
            f"from the caller: {sorted(derived & _ORCHESTRATOR_SUPPLIED_OPTIMIZER_PARAMS)}"
        )

    def test_a_real_optimizer_option_is_still_accepted(self, opt_func):
        """Control: the subtraction must not have narrowed the real allowlist.

        ``pareto_frontier_size`` is the constructor parameter whose wrongful
        rejection this PR exists to fix, so it is the right canary for an
        over-broad subtraction.
        """
        merged = opt_func._prepare_algorithm_kwargs({"pareto_frontier_size": 7})
        assert merged["pareto_frontier_size"] == 7
