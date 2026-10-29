"""Characterization of ``OptimizationOrchestrator._create_optimization_result``.

Pins every branch of the terminal result assembly with a bare orchestrator
instance and recorded module-level collaborators, so behaviour stays identical
across internal restructuring.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from traigent.api.types import OptimizationStatus, TrialStatus
from traigent.core import orchestrator as orch_mod
from traigent.core.orchestrator import OptimizationOrchestrator

NOW = datetime(2030, 1, 2, 3, 4, 5, tzinfo=UTC)


def _trial(
    trial_id: str = "t1",
    successful: bool = True,
    status: Any = TrialStatus.COMPLETED,
    metadata: Any = None,
    config: Any = None,
    metrics: Any = None,
) -> SimpleNamespace:
    metrics = metrics if metrics is not None else {"accuracy": 0.5}
    return SimpleNamespace(
        trial_id=trial_id,
        is_successful=successful,
        status=status,
        metadata=metadata,
        config=config if config is not None else {"a": 1},
        get_metric=lambda name: metrics.get(name),
    )


def _selection(**over: Any) -> SimpleNamespace:
    base = {
        "best_config": {"a": 1},
        "best_score": 0.9,
        "best_config_margin": None,
        "session_summary": None,
        "reason_code": "OK",
        "best_trial_id": None,
        "ranking_eligible_trial_ids": [],
    }
    base.update(over)
    return SimpleNamespace(**base)


class Optimizer:
    def __init__(self, objectives: Any = ("accuracy",), space: Any = None) -> None:
        self.objectives = list(objectives)
        if space is not None:
            self.config_space = space


def _make(**over: Any) -> OptimizationOrchestrator:
    o = object.__new__(OptimizationOrchestrator)
    o.optimizer = Optimizer()
    o.objective_schema = None
    o._trials = []
    o._status = OptimizationStatus.COMPLETED
    o._stop_reason = "max_trials"
    o._start_time = None
    o._optimization_id = "opt-1"
    o._function_descriptor = None
    o._best_trial_cached = None
    o._certified_promotions = 0
    o._tie_breakers = None
    o._band_target = None
    o._trials_prevented = 1
    o._examples_capped = 2
    o._cached_results_reused = 3
    o._ci_blocks = 4
    o.metric_registry = "registry"
    o._logger = None
    o._logger_facade = MagicMock()
    o._selection_receipt_binding = None
    o.traigent_config = SimpleNamespace(
        is_local_mode=lambda: True,
        get_comparability_mode=lambda: "cmp",
        fallback_reason=None,
        fallback_reason_code=None,
        persistence_rejection_reason=None,
        persistence_status=None,
    )
    o.cache_policy_handler = SimpleNamespace(
        configs_deduplicated=7, cache_policy_used="policy"
    )
    o.backend_session_manager = SimpleNamespace(
        result_source=MagicMock(return_value="local"),
        fallback_reason=None,
        fallback_reason_code=None,
        backend_rejection_reason=None,
    )
    o._stop_condition_manager = SimpleNamespace(
        semantic_saturation_diagnostics=MagicMock(return_value=None)
    )
    o._is_strict_evidence_mode = lambda: False
    o._build_result_metadata = MagicMock(side_effect=lambda s, t: {"base": s})
    for key, value in over.items():
        setattr(o, key, value)
    return o


class Env:
    """Patched collaborators, exposed for assertions and overrides."""

    def __init__(self, stack: Any) -> None:
        def p(name: str, **kw: Any) -> MagicMock:
            return stack.enter_context(patch.object(orch_mod, name, **kw))

        self.select = p("select_best_configuration", return_value=_selection())
        self.receipt = p("build_selection_receipt", return_value="receipt")
        self.never = p("_objective_metric_never_measured", return_value=True)
        self.names = p("_successful_trial_metric_names", return_value=["m1", "m2"])
        self.fmt_unmatched = p(
            "_format_unmatched_objective_warning", return_value=("UNMATCHED", "hint")
        )
        self.partial = p("_partial_coverage_trials", return_value=[])
        self.fmt_partial = p("_format_partial_coverage_warning", return_value="PARTIAL")
        self.weighted = p("resolve_weighted_selection_schema", return_value=None)
        self.inert = p("_detect_inert_constant_objectives", return_value=[])
        self.coerce = p("coerce_finite_objective_score", side_effect=lambda v: v)
        self.fmt_inert = p("_format_inert_objective_warning", return_value="INERT")
        self.agg = p(
            "aggregate_metrics",
            return_value=SimpleNamespace(
                processed_metrics={"pm": 1}, total_cost=0.5, total_tokens=10
            ),
        )
        self.safeguards = p("build_safeguards_telemetry", return_value={"sg": 1})
        self.label = p("generate_run_label", return_value="label-x")
        self.result_cls = p("OptimizationResult")
        self.logger = p("logger")
        dt = stack.enter_context(patch.object(orch_mod, "datetime"))
        dt.now.return_value = NOW
        self.datetime = dt
        self.time = stack.enter_context(patch.object(orch_mod, "time"))
        self.time.time.return_value = 1000.0
        self.apply_cost = stack.enter_context(
            patch.object(OptimizationOrchestrator, "_apply_abandoned_attempts_cost")
        )

    @property
    def kwargs(self) -> dict[str, Any]:
        return self.result_cls.call_args.kwargs


@pytest.fixture
def env():
    from contextlib import ExitStack

    with ExitStack() as stack:
        yield Env(stack)


def _run(o: OptimizationOrchestrator, env: Env) -> Any:
    return o._create_optimization_result()


class TestCertifiedSelection:
    def test_non_strict_passes_none_certified(self, env):
        o = _make()
        _run(o, env)
        kw = env.select.call_args.kwargs
        assert kw["require_certified"] is False
        assert kw["certified_config"] is None
        assert kw["certified_score"] is None
        assert kw["trials"] is o._trials
        assert kw["primary_objective"] == "accuracy"
        assert kw["config_space_keys"] == set()
        assert kw["aggregate_configs"] is False
        assert kw["tie_breakers"] is None
        assert kw["band_target"] is None
        assert kw["objective_order"] == ["accuracy"]
        assert kw["comparability_mode"] == "cmp"
        assert kw["objective_orientations"] is None
        assert kw["objective_schema"] is None

    def test_strict_without_promotions_has_no_certificate(self, env):
        o = _make(
            _is_strict_evidence_mode=lambda: True,
            _best_trial_cached=_trial(),
            _certified_promotions=0,
        )
        _run(o, env)
        kw = env.select.call_args.kwargs
        assert kw["require_certified"] is True
        assert kw["certified_config"] is None
        assert kw["certified_score"] is None

    def test_strict_without_cached_trial_has_no_certificate(self, env):
        o = _make(_is_strict_evidence_mode=lambda: True, _certified_promotions=2)
        _run(o, env)
        assert env.select.call_args.kwargs["certified_config"] is None

    def test_strict_with_promotion_filters_config_to_space(self, env):
        incumbent = _trial(
            config={"a": 1, "b": 2, "extra": 3}, metrics={"accuracy": 0.77}
        )
        o = _make(
            _is_strict_evidence_mode=lambda: True,
            _best_trial_cached=incumbent,
            _certified_promotions=1,
            optimizer=Optimizer(space={"a": [1], "b": [2]}),
        )
        _run(o, env)
        kw = env.select.call_args.kwargs
        assert kw["certified_config"] == {"a": 1, "b": 2}
        assert kw["certified_score"] == 0.77
        assert kw["config_space_keys"] == {"a", "b"}

    def test_strict_with_empty_space_keeps_all_keys(self, env):
        incumbent = _trial(config={"a": 1, "z": 9})
        o = _make(
            _is_strict_evidence_mode=lambda: True,
            _best_trial_cached=incumbent,
            _certified_promotions=1,
            optimizer=Optimizer(space={}),
        )
        _run(o, env)
        assert env.select.call_args.kwargs["certified_config"] == {"a": 1, "z": 9}

    def test_strict_none_config_gives_empty_dict(self, env):
        incumbent = _trial()
        incumbent.config = None
        o = _make(
            _is_strict_evidence_mode=lambda: True,
            _best_trial_cached=incumbent,
            _certified_promotions=1,
        )
        _run(o, env)
        assert env.select.call_args.kwargs["certified_config"] == {}

    def test_strict_without_objectives_score_none(self, env):
        o = _make(
            _is_strict_evidence_mode=lambda: True,
            _best_trial_cached=_trial(),
            _certified_promotions=1,
            optimizer=Optimizer(objectives=()),
        )
        _run(o, env)
        kw = env.select.call_args.kwargs
        assert kw["certified_score"] is None
        assert kw["primary_objective"] is None
        assert kw["certified_config"] == {"a": 1}

    def test_strict_incumbent_without_get_metric_score_none(self, env):
        incumbent = SimpleNamespace(config={"a": 1})
        o = _make(
            _is_strict_evidence_mode=lambda: True,
            _best_trial_cached=incumbent,
            _certified_promotions=1,
        )
        _run(o, env)
        assert env.select.call_args.kwargs["certified_score"] is None

    def test_non_local_mode_aggregates_and_tiebreakers_forwarded(self, env):
        o = _make(_tie_breakers=["cost"], _band_target="band")
        o.traigent_config.is_local_mode = lambda: False
        _run(o, env)
        kw = env.select.call_args.kwargs
        assert kw["aggregate_configs"] is True
        assert kw["tie_breakers"] == ["cost"]
        assert kw["band_target"] == "band"

    def test_empty_tiebreakers_become_none(self, env):
        o = _make(_tie_breakers=[])
        _run(o, env)
        assert env.select.call_args.kwargs["tie_breakers"] is None


class TestOrientations:
    def test_schema_orientations_mapped(self, env):
        schema = SimpleNamespace(
            objectives=[
                SimpleNamespace(name="accuracy", orientation="maximize"),
                SimpleNamespace(name="cost", orientation="minimize"),
            ],
            to_dict=lambda: {"schema": True},
        )
        o = _make(objective_schema=schema)
        _run(o, env)
        expected = {"accuracy": "maximize", "cost": "minimize"}
        assert env.select.call_args.kwargs["objective_orientations"] == expected
        assert env.select.call_args.kwargs["objective_schema"] is schema
        meta = env.kwargs["metadata"]
        assert meta["objective_schema"] == {"schema": True}
        assert meta["objective_orientations"] == expected

    def test_schema_without_objectives_gives_none_orientations_but_metadata(self, env):
        schema = SimpleNamespace(objectives=[], to_dict=lambda: {"s": 0})
        o = _make(objective_schema=schema)
        _run(o, env)
        assert env.select.call_args.kwargs["objective_orientations"] is None
        meta = env.kwargs["metadata"]
        assert meta["objective_schema"] == {"s": 0}
        assert meta["objective_orientations"] == {}


class TestStatisticalTie:
    def test_tie_logs_warning(self, env):
        margin = {
            "verdict": "statistical_tie",
            "primary_objective": "accuracy",
            "runner_up_trial_id": "t2",
            "delta": 0.01,
            "p_value": 0.4,
            "alpha": 0.05,
        }
        env.select.return_value = _selection(
            best_config_margin=margin, best_trial_id="t1"
        )
        _run(_make(), env)
        assert env.logger.warning.call_count == 1
        args = env.logger.warning.call_args.args
        assert args[0].startswith("best_config is a STATISTICAL TIE on '%s'")
        assert args[1:] == ("accuracy", "t1", "t2", 0.01, 0.4, 0.05)
        assert env.kwargs["best_config_margin"] is margin

    def test_non_tie_verdict_no_log(self, env):
        env.select.return_value = _selection(best_config_margin={"verdict": "clear"})
        _run(_make(), env)
        env.logger.warning.assert_not_called()

    def test_non_dict_margin_no_log(self, env):
        env.select.return_value = _selection(best_config_margin="statistical_tie")
        _run(_make(), env)
        env.logger.warning.assert_not_called()


class TestUnmatchedObjective:
    def _setup(self, env, summary=None, reason="NO_RANKING_ELIGIBLE_TRIALS"):
        env.select.return_value = _selection(
            session_summary=summary, reason_code=orch_mod.NO_RANKING_ELIGIBLE_TRIALS
        )

    def test_unmatched_sets_warning_summary_and_fails_status(self, env):
        self._setup(env)
        o = _make(_trials=[_trial()])
        _run(o, env)
        env.names.assert_called_once_with(o._trials)
        env.fmt_unmatched.assert_called_once_with("accuracy", ["m1", "m2"])
        kw = env.kwargs
        assert kw["warnings"] == ["UNMATCHED"]
        assert kw["warning_codes"] == [orch_mod._OBJECTIVE_UNMATCHED_WARNING_CODE]
        assert kw["status"] == OptimizationStatus.FAILED
        assert o._status == OptimizationStatus.FAILED
        summary = o._build_result_metadata.call_args.args[0]
        assert summary == {
            "reason": "UNMATCHED",
            "reason_code": orch_mod.NO_RANKING_ELIGIBLE_TRIALS,
            "unmatched_objective": "accuracy",
            "available_metrics": ["m1", "m2"],
            "did_you_mean": "hint",
        }

    def test_unmatched_copies_existing_summary_and_skips_empty_suggestion(self, env):
        original = {"keep": 1}
        self._setup(env, summary=original)
        env.fmt_unmatched.return_value = ("W", "")
        _run(_make(_trials=[_trial()]), env)
        assert original == {"keep": 1}
        summary = _make_summary(env)
        assert summary["keep"] == 1
        assert "did_you_mean" not in summary

    def test_unmatched_keeps_non_completed_status(self, env):
        self._setup(env)
        o = _make(_trials=[_trial()], _status=OptimizationStatus.CANCELLED)
        _run(o, env)
        assert env.kwargs["status"] == OptimizationStatus.CANCELLED
        assert o._status == OptimizationStatus.CANCELLED
        assert env.kwargs["warning_codes"] == [
            orch_mod._OBJECTIVE_UNMATCHED_WARNING_CODE
        ]

    def test_unmatched_requires_reason_code(self, env):
        env.select.return_value = _selection(reason_code="OTHER")
        _run(_make(_trials=[_trial()]), env)
        env.never.assert_not_called()
        assert env.kwargs["warnings"] == []

    def test_unmatched_requires_never_measured(self, env):
        self._setup(env)
        env.never.return_value = False
        _run(_make(_trials=[_trial()]), env)
        env.never.assert_called_once()
        env.fmt_unmatched.assert_not_called()
        assert env.kwargs["warnings"] == []

    def test_unmatched_requires_primary_objective(self, env):
        self._setup(env)
        _run(_make(optimizer=Optimizer(objectives=()), _trials=[_trial()]), env)
        env.never.assert_not_called()
        assert env.kwargs["warnings"] == []

    def test_unmatched_forces_zero_success_rate(self, env):
        self._setup(env)
        _run(_make(_trials=[_trial(), _trial("t2")]), env)
        assert env.kwargs["convergence_info"]["success_rate"] == 0.0
        assert env.kwargs["convergence_info"]["successful_trials"] == 2


def _make_summary(env) -> Any:
    return env.result_cls.call_args.kwargs["metadata"]["base"]


class TestAllTrialsFailed:
    def test_all_failed_warns_and_downgrades(self, env):
        trials = [
            _trial("a", False, TrialStatus.FAILED),
            _trial("b", False, TrialStatus.FAILED),
            _trial("c", False, TrialStatus.CANCELLED),
        ]
        o = _make(_trials=trials)
        _run(o, env)
        kw = env.kwargs
        assert kw["warnings"] == [
            "All trials failed: 2 of 3 trial(s) failed and none succeeded; "
            "no best_config was selected. See per-trial error messages."
        ]
        assert kw["warning_codes"] == [orch_mod._ALL_TRIALS_FAILED_WARNING_CODE]
        assert kw["status"] == OptimizationStatus.FAILED
        assert o._status == OptimizationStatus.FAILED

    def test_connection_failures_hint_mock_mode(self, env, monkeypatch):
        monkeypatch.delenv("TRAIGENT_MOCK_LLM", raising=False)
        monkeypatch.setattr(orch_mod, "is_mock_llm", lambda: False)
        failed = _trial("a", False, TrialStatus.FAILED)
        failed.error_message = "OpenAIException - Connection error."
        _run(_make(_trials=[failed]), env)
        (warning,) = env.kwargs["warnings"]
        assert warning.startswith("All trials failed: 1 of 1 trial(s)")
        assert "TRAIGENT_MOCK_LLM=true" in warning
        assert "python -m traigent.examples.quickstart" in warning

    def test_per_example_connection_failure_hints_mock_mode(self, env, monkeypatch):
        monkeypatch.setattr(orch_mod, "is_mock_llm", lambda: False)
        failed = _trial(
            "a",
            False,
            TrialStatus.FAILED,
            metadata={
                "example_results": [
                    {"error_message": "OpenAIException - Connection error."}
                ]
            },
        )
        failed.error_message = "No examples succeeded"
        _run(_make(_trials=[failed]), env)
        assert "TRAIGENT_MOCK_LLM=true" in env.kwargs["warnings"][0]

    def test_non_connection_failure_has_no_mock_hint(self, env, monkeypatch):
        monkeypatch.setattr(orch_mod, "is_mock_llm", lambda: False)
        failed = _trial("a", False, TrialStatus.FAILED)
        failed.error_message = "Objective metric 'accuracy' raised NameError"
        _run(_make(_trials=[failed]), env)
        assert "TRAIGENT_MOCK_LLM" not in env.kwargs["warnings"][0]

    def test_all_failed_keeps_cancelled_status(self, env):
        o = _make(
            _trials=[_trial("a", False, TrialStatus.FAILED)],
            _status=OptimizationStatus.CANCELLED,
        )
        _run(o, env)
        assert env.kwargs["status"] == OptimizationStatus.CANCELLED
        assert o._status == OptimizationStatus.CANCELLED
        assert len(env.kwargs["warnings"]) == 1

    def test_some_success_no_warning(self, env):
        trials = [_trial("a", False, TrialStatus.FAILED), _trial("b", True)]
        _run(_make(_trials=trials), env)
        assert env.kwargs["warnings"] == []
        assert env.kwargs["status"] == OptimizationStatus.COMPLETED

    def test_no_failed_status_no_warning(self, env):
        _run(_make(_trials=[_trial("a", False, TrialStatus.CANCELLED)]), env)
        assert env.kwargs["warnings"] == []

    def test_no_trials_no_warning_and_zero_success_rate(self, env):
        _run(_make(), env)
        assert env.kwargs["warnings"] == []
        info = env.kwargs["convergence_info"]
        assert info["success_rate"] == 0.0
        assert info["total_trials"] == 0


class TestPartialCoverage:
    def test_partial_coverage_warning(self, env):
        env.partial.return_value = ["p"]
        env.select.return_value = _selection(
            reason_code="RC", best_config={"x": 1}, ranking_eligible_trial_ids=["t1"]
        )
        o = _make(_trials=[_trial()])
        _run(o, env)
        env.partial.assert_called_once_with(o._trials, "accuracy")
        env.fmt_partial.assert_called_once_with(
            ["p"], "accuracy", "RC", {"x": 1}, ["t1"]
        )
        assert env.kwargs["warnings"] == ["PARTIAL"]
        assert env.kwargs["warning_codes"] == [
            orch_mod._METRIC_PARTIAL_COVERAGE_WARNING_CODE
        ]

    def test_empty_partial_no_warning(self, env):
        _run(_make(), env)
        env.fmt_partial.assert_not_called()

    def test_no_primary_objective_skips_check(self, env):
        _run(_make(optimizer=Optimizer(objectives=())), env)
        env.partial.assert_not_called()


class TestCustomEvaluatorFailedRows:
    def test_counts_only_positive_non_bool_ints(self, env):
        trials = [
            _trial("a", metadata={"custom_evaluator_failed_rows": 3}),
            _trial("b", metadata={"custom_evaluator_failed_rows": 0}),
            _trial("c", metadata={"custom_evaluator_failed_rows": True}),
            _trial("d", metadata={"custom_evaluator_failed_rows": 2.0}),
            _trial("e", metadata=None),
            _trial("f", metadata={"custom_evaluator_failed_rows": -1}),
            _trial("g", metadata={"custom_evaluator_failed_rows": 1}),
        ]
        o = _make(_trials=trials)
        _run(o, env)
        kw = env.kwargs
        assert kw["warnings"] == [
            "2 trial(s) had rows where the custom_evaluator raised; those rows "
            "scored 0.0 on quality metrics. See trial metadata "
            "'custom_evaluator_failed_rows'."
        ]
        assert kw["warning_codes"] == [
            orch_mod._CUSTOM_EVALUATOR_ROWS_FAILED_WARNING_CODE
        ]
        assert _make_summary(env) == {
            "custom_evaluator_failed_rows_by_trial": {"a": 3, "g": 1}
        }

    def test_trial_without_metadata_attribute_ignored(self, env):
        t = SimpleNamespace(
            trial_id="x",
            is_successful=True,
            status=TrialStatus.COMPLETED,
        )
        _run(_make(_trials=[t]), env)
        assert env.kwargs["warnings"] == []

    def test_existing_summary_copied_not_mutated(self, env):
        original = {"keep": 1}
        env.select.return_value = _selection(session_summary=original)
        trials = [_trial("a", metadata={"custom_evaluator_failed_rows": 5})]
        _run(_make(_trials=trials), env)
        assert original == {"keep": 1}
        assert _make_summary(env) == {
            "keep": 1,
            "custom_evaluator_failed_rows_by_trial": {"a": 5},
        }


class TestInertConstant:
    def _schema(self):
        return SimpleNamespace(
            objectives=[SimpleNamespace(name="accuracy"), SimpleNamespace(name="cost")]
        )

    def test_inert_warning_summary_and_log(self, env):
        schema = self._schema()
        env.weighted.return_value = schema
        env.inert.return_value = ["cost"]
        env.coerce.side_effect = lambda v: v
        env.select.return_value = _selection(
            ranking_eligible_trial_ids=["t1"], session_summary={"s": 1}
        )
        trials = [
            _trial("t1", metrics={"cost": 0.25}),
            _trial("t2", metrics={"cost": 0.25}),
        ]
        o = _make(_trials=trials)
        _run(o, env)
        env.weighted.assert_called_once_with(None)
        env.inert.assert_called_once_with([trials[0]], schema)
        env.fmt_inert.assert_called_once_with(["cost"], ["accuracy", "cost"], 0.25)
        env.logger.warning.assert_called_once_with("INERT")
        assert env.kwargs["warnings"] == ["INERT"]
        assert env.kwargs["warning_codes"] == [
            orch_mod._OBJECTIVE_INERT_CONSTANT_WARNING_CODE
        ]
        assert _make_summary(env) == {"s": 1, "inert_constant_objectives": ["cost"]}

    def test_constant_value_falls_back_to_zero_when_falsy(self, env):
        env.weighted.return_value = self._schema()
        env.inert.return_value = ["cost"]
        env.coerce.side_effect = lambda v: None
        env.select.return_value = _selection(ranking_eligible_trial_ids=["t1"])
        _run(_make(_trials=[_trial("t1", metrics={"cost": 1.0})]), env)
        assert env.fmt_inert.call_args.args[2] == 0.0
        assert _make_summary(env) == {"inert_constant_objectives": ["cost"]}

    def test_no_schema_skips(self, env):
        env.select.return_value = _selection(ranking_eligible_trial_ids=["t1"])
        _run(_make(_trials=[_trial("t1")]), env)
        env.inert.assert_not_called()

    def test_no_eligible_ids_skips(self, env):
        env.weighted.return_value = self._schema()
        _run(_make(_trials=[_trial("t1")]), env)
        env.inert.assert_not_called()

    def test_empty_inert_result_no_warning(self, env):
        env.weighted.return_value = self._schema()
        env.select.return_value = _selection(ranking_eligible_trial_ids=["t1"])
        _run(_make(_trials=[_trial("t1")]), env)
        env.inert.assert_called_once()
        env.fmt_inert.assert_not_called()
        assert env.kwargs["warnings"] == []
        env.logger.warning.assert_not_called()


class TestWarningOrdering:
    def test_all_warning_kinds_in_order(self, env):
        env.select.return_value = _selection(
            reason_code=orch_mod.NO_RANKING_ELIGIBLE_TRIALS,
            ranking_eligible_trial_ids=["t1"],
        )
        env.partial.return_value = ["p"]
        env.weighted.return_value = SimpleNamespace(
            objectives=[SimpleNamespace(name="q")]
        )
        env.inert.return_value = ["q"]
        trials = [
            _trial(
                "t1",
                False,
                TrialStatus.FAILED,
                metadata={"custom_evaluator_failed_rows": 1},
            )
        ]
        _run(_make(_trials=trials), env)
        assert env.kwargs["warning_codes"] == [
            orch_mod._OBJECTIVE_UNMATCHED_WARNING_CODE,
            orch_mod._ALL_TRIALS_FAILED_WARNING_CODE,
            orch_mod._METRIC_PARTIAL_COVERAGE_WARNING_CODE,
            orch_mod._CUSTOM_EVALUATOR_ROWS_FAILED_WARNING_CODE,
            orch_mod._OBJECTIVE_INERT_CONSTANT_WARNING_CODE,
        ]
        assert len(env.kwargs["warnings"]) == 5


class TestDurationAndConvergence:
    def test_duration_from_start_time(self, env):
        _run(_make(_start_time=400.0), env)
        assert env.kwargs["duration"] == 600.0

    def test_duration_zero_without_start(self, env):
        _run(_make(_start_time=None), env)
        assert env.kwargs["duration"] == 0.0

    def test_convergence_info(self, env):
        trials = [_trial("a"), _trial("b", False, TrialStatus.FAILED)]
        _run(_make(_trials=trials), env)
        assert env.kwargs["convergence_info"] == {
            "total_trials": 2,
            "successful_trials": 1,
            "success_rate": 0.5,
            "algorithm": "Optimizer",
        }
        assert env.kwargs["algorithm"] == "Optimizer"


class TestRunLabel:
    @pytest.mark.parametrize(
        ("descriptor", "expected"),
        [
            (None, "run"),
            (SimpleNamespace(display_name="Disp", identifier="ident"), "Disp"),
            (SimpleNamespace(display_name="", identifier="ident"), "ident"),
            (SimpleNamespace(display_name=None, identifier=None), "run"),
        ],
    )
    def test_func_name(self, env, descriptor, expected):
        _run(_make(_function_descriptor=descriptor), env)
        env.label.assert_called_once_with(expected, "opt-1", NOW)
        assert env.kwargs["run_label"] == "label-x"
        assert env.kwargs["timestamp"] is NOW


class TestMetricsAndTelemetry:
    def test_aggregation_and_safeguards_inputs(self, env):
        o = _make(_trials=[_trial()])
        _run(o, env)
        env.agg.assert_called_once_with(o._trials, "registry")
        env.safeguards.assert_called_once_with(
            trials_prevented=1,
            configs_deduplicated=7,
            examples_capped=2,
            cached_results_reused=3,
            ci_blocks=4,
            cache_policy_used="policy",
        )
        o._build_result_metadata.assert_called_once_with(None, {"sg": 1})
        assert env.kwargs["metrics"] == {"pm": 1}

    @pytest.mark.parametrize(
        ("cost", "tokens", "exp_cost", "exp_tokens"),
        [
            (0.5, 10, 0.5, 10),
            (0.0, 0, None, None),
            (None, 0, None, None),
            (-1.0, -5, None, None),
            (2.0, 1, 2.0, 1),
        ],
    )
    def test_cost_and_token_normalisation(
        self, env, cost, tokens, exp_cost, exp_tokens
    ):
        env.agg.return_value = SimpleNamespace(
            processed_metrics={}, total_cost=cost, total_tokens=tokens
        )
        _run(_make(), env)
        assert env.kwargs["total_cost"] == exp_cost
        assert env.kwargs["total_tokens"] == exp_tokens


class TestResultMetadata:
    def test_semantic_saturation_and_best_trial_id(self, env):
        env.select.return_value = _selection(best_trial_id="t9")
        o = _make()
        o._stop_condition_manager.semantic_saturation_diagnostics.return_value = {
            "sat": 1
        }
        _run(o, env)
        meta = env.kwargs["metadata"]
        assert meta["semantic_saturation"] == {"sat": 1}
        assert meta["best_trial_id"] == "t9"
        o._stop_condition_manager.semantic_saturation_diagnostics.assert_called_once_with(
            o._trials
        )

    def test_absent_saturation_and_trial_id_not_in_metadata(self, env):
        _run(_make(), env)
        meta = env.kwargs["metadata"]
        assert "semantic_saturation" not in meta
        assert "best_trial_id" not in meta
        assert meta["source"] == "local"

    def test_source_uses_trial_count(self, env):
        o = _make(_trials=[_trial(), _trial("b")])
        _run(o, env)
        o.backend_session_manager.result_source.assert_called_once_with(2)
        assert env.kwargs["source"] == "local"

    def test_fallback_reason_from_config_and_code(self, env):
        o = _make()
        o.backend_session_manager.result_source.return_value = (
            orch_mod.SOURCE_LOCAL_FALLBACK
        )
        o.traigent_config.fallback_reason = "cfg-reason"
        o.traigent_config.fallback_reason_code = "CFG_CODE"
        o.backend_session_manager.fallback_reason = "mgr-reason"
        o.backend_session_manager.fallback_reason_code = "MGR_CODE"
        _run(o, env)
        meta = env.kwargs["metadata"]
        assert meta["fallback_reason"] == "cfg-reason"
        assert meta["fallback_reason_code"] == "CFG_CODE"

    def test_fallback_reason_from_manager_when_config_empty(self, env):
        o = _make()
        o.backend_session_manager.result_source.return_value = (
            orch_mod.SOURCE_LOCAL_FALLBACK
        )
        o.backend_session_manager.fallback_reason = "mgr-reason"
        o.backend_session_manager.fallback_reason_code = "MGR_CODE"
        _run(o, env)
        meta = env.kwargs["metadata"]
        assert meta["fallback_reason"] == "mgr-reason"
        assert meta["fallback_reason_code"] == "MGR_CODE"

    def test_fallback_without_code_omits_code(self, env):
        o = _make()
        o.backend_session_manager.result_source.return_value = (
            orch_mod.SOURCE_LOCAL_FALLBACK
        )
        o.backend_session_manager.fallback_reason = "mgr-reason"
        _run(o, env)
        meta = env.kwargs["metadata"]
        assert meta["fallback_reason"] == "mgr-reason"
        assert "fallback_reason_code" not in meta

    def test_fallback_source_without_reason_omits_both(self, env):
        o = _make()
        o.backend_session_manager.result_source.return_value = (
            orch_mod.SOURCE_LOCAL_FALLBACK
        )
        o.backend_session_manager.fallback_reason_code = "MGR_CODE"
        _run(o, env)
        meta = env.kwargs["metadata"]
        assert "fallback_reason" not in meta
        assert "fallback_reason_code" not in meta

    def test_reason_ignored_when_source_not_fallback(self, env):
        o = _make()
        o.backend_session_manager.fallback_reason = "mgr-reason"
        _run(o, env)
        assert "fallback_reason" not in env.kwargs["metadata"]

    def test_rejection_from_manager(self, env):
        o = _make()
        o.backend_session_manager.backend_rejection_reason = "mgr-rej"
        o.traigent_config.persistence_rejection_reason = "cfg-rej"
        _run(o, env)
        meta = env.kwargs["metadata"]
        assert meta["persistence_reason"] == "rejected"
        assert meta["persistence_rejected"] is True
        assert meta["persistence_rejection_reason"] == "mgr-rej"

    def test_rejection_from_config(self, env):
        o = _make()
        o.traigent_config.persistence_rejection_reason = "cfg-rej"
        _run(o, env)
        assert env.kwargs["metadata"]["persistence_rejection_reason"] == "cfg-rej"

    def test_no_rejection_omits_keys(self, env):
        _run(_make(), env)
        meta = env.kwargs["metadata"]
        assert "persistence_reason" not in meta
        assert "persistence_rejected" not in meta

    def test_persistence_status(self, env):
        o = _make()
        o.traigent_config.persistence_status = "saved"
        _run(o, env)
        assert env.kwargs["metadata"]["persistence_status"] == "saved"

    def test_config_without_optional_attrs(self, env):
        o = _make()
        o.traigent_config = SimpleNamespace(
            is_local_mode=lambda: True, get_comparability_mode=lambda: "cmp"
        )
        _run(o, env)
        meta = env.kwargs["metadata"]
        assert "persistence_status" not in meta
        assert "persistence_rejected" not in meta


class TestResultConstructionAndLogging:
    def test_constructor_kwargs_and_return(self, env):
        env.select.return_value = _selection(
            best_config={"c": 1},
            best_score=0.42,
            reason_code="RC",
            ranking_eligible_trial_ids=["t1"],
            best_config_margin={"verdict": "clear"},
        )
        o = _make(_trials=[_trial()], _stop_reason="why", _start_time=990.0)
        result = _run(o, env)
        assert result is env.result_cls.return_value
        kw = env.kwargs
        assert kw["trials"] == o._trials
        assert kw["trials"] is not o._trials
        assert kw["best_config"] == {"c": 1}
        assert kw["best_score"] == 0.42
        assert kw["optimization_id"] == "opt-1"
        assert kw["objectives"] == ["accuracy"]
        assert kw["stop_reason"] == "why"
        assert kw["reason_code"] == "RC"
        assert kw["ranking_eligible_trial_ids"] == ["t1"]
        assert kw["best_config_margin"] == {"verdict": "clear"}
        assert kw["duration"] == 10.0
        assert kw["source"] == "local"
        assert kw["status"] == OptimizationStatus.COMPLETED

    def test_receipt_bound_and_cost_applied(self, env):
        o = _make()
        result = _run(o, env)
        env.receipt.assert_called_once_with(env.select.return_value)
        assert o._selection_receipt_binding == (result, "receipt")
        env.apply_cost.assert_called_once_with(o, result)

    def test_no_logger_no_session_end(self, env):
        o = _make(_logger=None)
        _run(o, env)
        o._logger_facade.log_session_end.assert_not_called()

    def test_logger_single_objective_no_weighted(self, env):
        o = _make(_logger=object())
        result = _run(o, env)
        result.calculate_weighted_scores.assert_not_called()
        o._logger_facade.log_session_end.assert_called_once_with(
            optimization_result=result, weighted_results=None
        )

    def test_logger_multi_objective_weighted(self, env):
        schema = SimpleNamespace(
            objectives=[
                SimpleNamespace(name="accuracy", orientation="maximize"),
                SimpleNamespace(name="cost", orientation="minimize"),
            ],
            to_dict=lambda: {},
        )
        o = _make(
            _logger=object(),
            optimizer=Optimizer(objectives=("accuracy", "cost")),
            objective_schema=schema,
        )
        result = _run(o, env)
        result.calculate_weighted_scores.assert_called_once_with(
            objective_schema=schema
        )
        o._logger_facade.log_session_end.assert_called_once_with(
            optimization_result=result,
            weighted_results=result.calculate_weighted_scores.return_value,
        )


@pytest.mark.parametrize(
    "text",
    [
        "All connection attempts failed",
        "Connection reset by peer",
        "[Errno -2] Name or service not known",
        "socket.gaierror: getaddrinfo failed",
        "ECONNREFUSED 127.0.0.1:443",
        "timed out connecting to host",
    ],
)
def test_connection_failure_detection_variants(text):
    assert orch_mod._is_connection_failure(text)


def test_connection_failure_detection_httpx_and_negatives():
    import httpx

    assert orch_mod._is_connection_failure(
        str(httpx.ConnectError("All connection attempts failed"))
    )
    assert not orch_mod._is_connection_failure("division by zero")
    assert not orch_mod._is_connection_failure(None)
