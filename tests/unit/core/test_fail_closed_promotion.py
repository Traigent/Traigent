"""Fail-closed strict-evidence promotion (FR-SDK-FAIL-CLOSED-PROMOTION-V1).

The four verified leak sites, closed strictly behind _is_strict_evidence_mode:
1. gate no_decision           -> never _simple_is_better (spy-verified)
2. insufficient samples       -> never _simple_is_better
3. gate exception             -> never _simple_is_better (Rule 1: deny)
4. terminal selector          -> NO_CERTIFIED_SELECTION, never re-derived
Non-strict behavior is pinned byte-identical (the legacy lane).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from traigent.api.types import TrialResult, TrialStatus
from traigent.core.orchestrator import OptimizationOrchestrator
from traigent.core.result_selection import (
    NO_CERTIFIED_SELECTION,
    select_best_configuration,
)
from traigent.tvl.models import ChanceConstraint, PromotionPolicy, RequireCalibration


def _trial(score: float = 0.9, config=None) -> TrialResult:
    return TrialResult(
        trial_id="t1",
        config=config or {"model": "a"},
        metrics={"accuracy": score},
        status=TrialStatus.COMPLETED,
        duration=0.1,
        timestamp=0.0,
        metadata={"successful_examples": 1},
    )


def _orchestrator(policy: PromotionPolicy | None) -> OptimizationOrchestrator:
    orchestrator = OptimizationOrchestrator.__new__(OptimizationOrchestrator)
    orchestrator.knob_resolver = None
    orchestrator._certified_promotions = 0
    orchestrator._strict_withheld_promotions = []
    orchestrator._promotion_gate = (
        None if policy is None else SimpleNamespace(policy=policy)
    )
    orchestrator._best_trial_cached = _trial(0.5)
    orchestrator.optimizer = SimpleNamespace(objectives=["accuracy"])
    orchestrator._config_metrics_history = {}
    orchestrator._incumbent_config_hash = "incumbent"
    return orchestrator


STRICT_POLICIES = {
    "require_calibration": PromotionPolicy(
        require_calibration=RequireCalibration(enabled=True)
    ),
    "chance_constraints": PromotionPolicy(
        chance_constraints=[
            ChanceConstraint(name="accuracy", threshold=0.8, confidence=0.95)
        ]
    ),
}

NON_STRICT_POLICY = PromotionPolicy()


class TestStrictModeDetection:
    @pytest.mark.parametrize("name,policy", sorted(STRICT_POLICIES.items()))
    def test_each_declared_trigger_is_strict(self, name, policy):
        assert _orchestrator(policy)._is_strict_evidence_mode(), name

    def test_plain_policy_is_not_strict(self):
        assert not _orchestrator(NON_STRICT_POLICY)._is_strict_evidence_mode()

    def test_no_gate_is_not_strict(self):
        assert not _orchestrator(None)._is_strict_evidence_mode()

    def test_disabled_require_calibration_is_not_strict(self):
        policy = PromotionPolicy(require_calibration=RequireCalibration(enabled=False))
        assert not _orchestrator(policy)._is_strict_evidence_mode()


@pytest.mark.parametrize("name,policy", sorted(STRICT_POLICIES.items()))
class TestStrictFailClosed:
    def test_no_decision_withholds_without_simple_fallback(self, name, policy):
        orchestrator = _orchestrator(policy)
        orchestrator._simple_is_better = MagicMock(name="simple")  # spy
        decision = SimpleNamespace(decision="no_decision", reason="too few samples")
        assert orchestrator._handle_promotion_decision(decision, _trial()) is False
        orchestrator._simple_is_better.assert_not_called()
        assert orchestrator._strict_withheld_promotions

    def test_insufficient_samples_withholds(self, name, policy):
        orchestrator = _orchestrator(policy)
        orchestrator._simple_is_better = MagicMock(name="simple")
        orchestrator._has_sufficient_samples = MagicMock(return_value=False)
        orchestrator._config_metrics_history = {
            "candidate": {"accuracy": [0.9]},
            "incumbent": {"accuracy": [0.5]},
        }
        assert orchestrator._evaluate_promotion("candidate", _trial()) is False
        orchestrator._simple_is_better.assert_not_called()

    def test_gate_exception_fails_closed(self, name, policy):
        orchestrator = _orchestrator(policy)
        orchestrator._simple_is_better = MagicMock(name="simple")
        orchestrator._has_sufficient_samples = MagicMock(return_value=True)
        orchestrator._config_metrics_history = {
            "candidate": {"accuracy": [0.9, 0.91]},
            "incumbent": {"accuracy": [0.5, 0.52]},
        }
        orchestrator._promotion_gate.evaluate = MagicMock(
            side_effect=RuntimeError("gate blew up")
        )
        orchestrator._promotion_gate.policy = policy
        assert orchestrator._evaluate_promotion("candidate", _trial()) is False
        orchestrator._simple_is_better.assert_not_called()
        assert any(
            "gate exception" in reason
            for reason in orchestrator._strict_withheld_promotions
        )

    def test_promote_and_reject_still_flow_through(self, name, policy):
        orchestrator = _orchestrator(policy)
        promote = SimpleNamespace(decision="promote", reason="dominates")
        reject = SimpleNamespace(decision="reject", reason="dominated")
        assert orchestrator._handle_promotion_decision(promote, _trial()) is True
        assert orchestrator._handle_promotion_decision(reject, _trial()) is False


class TestNonStrictLaneUnchanged:
    """The legacy lane must stay byte-identical (no behavior change for
    modules without strict declarations)."""

    def test_no_decision_falls_back_to_simple(self):
        orchestrator = _orchestrator(NON_STRICT_POLICY)
        orchestrator._simple_is_better = MagicMock(return_value=True)
        decision = SimpleNamespace(decision="no_decision", reason="few samples")
        assert orchestrator._handle_promotion_decision(decision, _trial()) is True
        orchestrator._simple_is_better.assert_called_once()

    def test_gate_exception_falls_back_to_simple(self):
        orchestrator = _orchestrator(NON_STRICT_POLICY)
        orchestrator._simple_is_better = MagicMock(return_value=True)
        orchestrator._has_sufficient_samples = MagicMock(return_value=True)
        orchestrator._config_metrics_history = {
            "candidate": {"accuracy": [0.9]},
            "incumbent": {"accuracy": [0.5]},
        }
        orchestrator._promotion_gate.evaluate = MagicMock(
            side_effect=RuntimeError("gate blew up")
        )
        assert orchestrator._evaluate_promotion("candidate", _trial()) is True
        orchestrator._simple_is_better.assert_called_once()


class TestTerminalSelector:
    """Leak 4: select_best_configuration is gate-independent by default —
    require_certified makes it honor the certified incumbent or return the
    explicit no-winner shape."""

    def test_no_certified_winner_returns_explicit_empty(self):
        result = select_best_configuration(
            trials=[_trial(0.99)],  # high score MUST NOT win
            primary_objective="accuracy",
            config_space_keys={"model"},
            aggregate_configs=False,
            require_certified=True,
            certified_config=None,
        )
        assert result.best_config == {}
        assert result.best_score is None
        assert result.reason_code == NO_CERTIFIED_SELECTION

    def test_certified_incumbent_returned_verbatim(self):
        result = select_best_configuration(
            trials=[_trial(0.99, config={"model": "hot"})],
            primary_objective="accuracy",
            config_space_keys={"model"},
            aggregate_configs=False,
            require_certified=True,
            certified_config={"model": "certified"},
            certified_score=0.7,
        )
        # the certified incumbent wins even though a higher-scoring trial exists
        assert result.best_config == {"model": "certified"}
        assert result.best_score == 0.7
        assert result.reason_code is None

    def test_default_mode_unchanged(self):
        result = select_best_configuration(
            trials=[_trial(0.99, config={"model": "hot"})],
            primary_objective="accuracy",
            config_space_keys={"model"},
            aggregate_configs=False,
            comparability_mode="legacy",
        )
        assert result.best_config.get("model") == "hot"


class TestCertifiedPromotionChain:
    """Review round-1 fixes: first-incumbent overclaim (B1), dict-policy
    fail-open (B2), per-run state reset (NB4), and the winner-claim guards
    on result surfaces (B3, scoped)."""

    def _orchestrator_with_real_chain(self, policy):
        orchestrator = _orchestrator(policy)
        orchestrator._best_trial_cached = None
        orchestrator._incumbent_config_hash = None
        orchestrator._certified_promotions = 0
        orchestrator._track_trial_metrics = lambda t: (
            "hash-" + str(t.config.get("model"))
        )
        return orchestrator

    def test_strict_zero_promotions_yields_no_certified_winner(self):
        """B1: the FIRST trial seeds the incumbent as initialization, not
        certification — a strict run with zero gate promotions must produce
        certified_promotions == 0, driving NO_CERTIFIED_SELECTION."""
        policy = STRICT_POLICIES["require_calibration"]
        orchestrator = self._orchestrator_with_real_chain(policy)
        orchestrator._update_best_trial_cache(_trial(0.9, config={"model": "a"}))
        assert orchestrator._best_trial_cached is not None  # seeded
        assert orchestrator._certified_promotions == 0  # NOT certified
        # second trial: gate says no_decision -> withheld, still zero
        orchestrator._has_sufficient_samples = MagicMock(return_value=True)
        orchestrator._config_metrics_history = {
            "hash-b": {"accuracy": [0.95]},
            "hash-a": {"accuracy": [0.9]},
        }
        orchestrator._incumbent_config_hash = "hash-a"
        orchestrator._promotion_gate.evaluate = MagicMock(
            return_value=SimpleNamespace(decision="no_decision", reason="thin")
        )
        orchestrator._update_best_trial_cache(_trial(0.95, config={"model": "b"}))
        assert orchestrator._certified_promotions == 0
        assert orchestrator._best_trial_cached.config == {"model": "a"}

    def test_strict_gate_promote_counts_as_certified(self):
        policy = STRICT_POLICIES["require_calibration"]
        orchestrator = self._orchestrator_with_real_chain(policy)
        orchestrator._update_best_trial_cache(_trial(0.5, config={"model": "a"}))
        orchestrator._has_sufficient_samples = MagicMock(return_value=True)
        orchestrator._config_metrics_history = {
            "hash-b": {"accuracy": [0.95]},
            "hash-a": {"accuracy": [0.5]},
        }
        orchestrator._incumbent_config_hash = "hash-a"
        orchestrator._promotion_gate.evaluate = MagicMock(
            return_value=SimpleNamespace(decision="promote", reason="dominates")
        )
        orchestrator._update_best_trial_cache(_trial(0.95, config={"model": "b"}))
        assert orchestrator._certified_promotions == 1
        assert orchestrator._best_trial_cached.config == {"model": "b"}

    def test_dict_policy_with_strict_keys_reads_strict(self):
        """B2: a raw dict policy (discovered specs) must not silently read
        as non-strict."""
        orchestrator = OptimizationOrchestrator.__new__(OptimizationOrchestrator)
        orchestrator.knob_resolver = None
        orchestrator._promotion_gate = SimpleNamespace(
            policy={
                "dominance": "epsilon_pareto",
                "require_calibration": {"enabled": True},
            }
        )
        assert orchestrator._is_strict_evidence_mode() is True

    def test_unparseable_dict_policy_with_strict_keys_fails_closed(self):
        orchestrator = OptimizationOrchestrator.__new__(OptimizationOrchestrator)
        orchestrator.knob_resolver = None
        orchestrator._promotion_gate = SimpleNamespace(
            policy={
                "require_calibration": {"enabled": "not-a-bool"},  # from_dict raises
            }
        )
        assert orchestrator._is_strict_evidence_mode() is True

    def test_plain_dict_policy_without_strict_keys_is_not_strict(self):
        orchestrator = OptimizationOrchestrator.__new__(OptimizationOrchestrator)
        orchestrator.knob_resolver = None
        orchestrator._promotion_gate = SimpleNamespace(
            policy={"dominance": "epsilon_pareto", "alpha": 0.05}
        )
        assert orchestrator._is_strict_evidence_mode() is False


class TestResultWinnerClaimGuards:
    """B3 (scoped): a no-winner result makes no winner claims on its other
    surfaces. Optimizer-internal best tracking and in-flight progress
    callbacks are search guidance / observability, NOT promoted-winner
    claims — recorded as a disposition in the feature trail."""

    def _result(self, best_config):
        from traigent.api.types import OptimizationResult

        return OptimizationResult(
            best_config=best_config,
            best_score=None if not best_config else 0.9,
            trials=[_trial(0.99, config={"model": "hot"})],
            total_cost=0.0,
            duration=1.0,
            objectives=["accuracy"],
            optimization_id="opt-test",
            convergence_info={},
            status="completed",
            algorithm="test",
            timestamp=0.0,
        )

    def test_best_metrics_empty_when_no_winner(self):
        assert self._result({}).best_metrics == {}

    def test_best_metrics_present_with_winner(self):
        assert self._result({"model": "hot"}).best_metrics

    def test_weighted_scores_omit_winner_claim_when_no_winner(self):
        result = self._result({})
        scores = result.calculate_weighted_scores(objective_weights={"accuracy": 1.0})
        # the winner-claim KEY is omitted entirely on a no-winner result
        assert "best_weighted_config" not in scores
        # descriptive per-trial statistics are still computed
        assert scores["weighted_scores"]

    def test_valid_empty_config_winner_not_regressed(self):
        """Round-2 blocker: a VALID winner whose config happens to be empty
        (score present) keeps its winner surfaces — only the
        NO_CERTIFIED_SELECTION shape (empty config AND None score) is a
        no-winner."""
        from traigent.api.types import OptimizationResult

        result = OptimizationResult(
            best_config={},
            best_score=0.99,  # a real winner, degenerate empty config
            trials=[_trial(0.99, config={})],
            total_cost=0.0,
            duration=1.0,
            objectives=["accuracy"],
            optimization_id="opt-test",
            convergence_info={},
            status="completed",
            algorithm="test",
            timestamp=0.0,
        )
        assert result.best_metrics  # winner metrics preserved
        scores = result.calculate_weighted_scores(objective_weights={"accuracy": 1.0})
        assert "best_weighted_config" in scores

    def test_weighted_scores_early_branch_also_omits_winner_key(self):
        """Round-3 residual: the no-successful-trials early return must obey
        the same omit-on-no-winner rule."""
        from traigent.api.types import OptimizationResult

        result = OptimizationResult(
            best_config={},
            best_score=None,
            trials=[],  # no trials -> the early-return branch
            total_cost=0.0,
            duration=1.0,
            objectives=["accuracy"],
            optimization_id="opt-test",
            convergence_info={},
            status="completed",
            algorithm="test",
            timestamp=0.0,
        )
        scores = result.calculate_weighted_scores(objective_weights={"accuracy": 1.0})
        assert "best_weighted_config" not in scores


class TestBestConfigRuntimeInterplay:
    """FR-SDK-BEST-CONFIG-RUNTIME-V1 interplay: a strict run with no certified
    winner produces NO snapshot/export/publish — the runtime keeps defaults.

    Exercises the REAL finalize path (_run_and_finalize_optimization's
    ``if result.best_config`` guard) and the REAL ConfigStateManager export
    guard, not re-implementations of them.
    """

    @staticmethod
    def _no_certified_result():
        from traigent.api.types import OptimizationResult

        return OptimizationResult(
            best_config={},  # NO_CERTIFIED_SELECTION shape
            best_score=None,
            trials=[],
            total_cost=0.0,
            duration=1.0,
            objectives=["accuracy"],
            optimization_id="opt-no-certified",
            convergence_info={},
            status="completed",
            algorithm="test",
            timestamp=0.0,
            metadata={"reason_code": NO_CERTIFIED_SELECTION},
        )

    @pytest.mark.asyncio
    async def test_no_certified_winner_yields_no_snapshot_or_export(self, tmp_path):
        from unittest.mock import AsyncMock

        from traigent.core.optimized_function import OptimizedFunction
        from traigent.evaluators.base import Dataset, EvaluationExample
        from traigent.utils.exceptions import ConfigurationError

        def answer(text: str) -> str:
            return text

        fn = OptimizedFunction(
            func=answer,
            configuration_space={"model": ["a", "b"]},
            default_config={"model": "a"},
        )
        fn.traigent_config = SimpleNamespace(is_local_mode=lambda: False)
        orchestrator = MagicMock()
        orchestrator.optimize = AsyncMock(return_value=self._no_certified_result())
        dataset = Dataset([EvaluationExample({"text": "x"}, "x")], name="d")

        result = await fn._run_and_finalize_optimization(
            orchestrator,
            dataset,
            {"model": ["a", "b"]},
            save_to=None,
        )

        # the no-winner result flowed through the real finalize path
        assert result.best_config == {}
        # no winner was applied: snapshot stays DEFAULT, best config is None
        assert fn.get_best_config() is None
        assert fn.current_config == {"model": "a"}
        assert fn.best_config_snapshot.source == "default"
        # and the export/publish surface fails closed (config_state_manager guard)
        with pytest.raises(ConfigurationError, match="No best configuration"):
            fn.export_best_config(tmp_path / "best-configs")

    @pytest.mark.asyncio
    async def test_certified_winner_still_exports(self, tmp_path):
        """Control: a result WITH a winner applies and exports — the guard
        only bites the no-winner shape."""
        from unittest.mock import AsyncMock

        from traigent.api.types import OptimizationResult
        from traigent.core.optimized_function import OptimizedFunction
        from traigent.evaluators.base import Dataset, EvaluationExample

        def answer(text: str) -> str:
            return text

        winner = OptimizationResult(
            best_config={"model": "b"},
            best_score=0.9,
            trials=[],
            total_cost=0.0,
            duration=1.0,
            objectives=["accuracy"],
            optimization_id="opt-certified",
            convergence_info={},
            status="completed",
            algorithm="test",
            timestamp=0.0,
        )
        fn = OptimizedFunction(
            func=answer,
            configuration_space={"model": ["a", "b"]},
            default_config={"model": "a"},
        )
        fn.traigent_config = SimpleNamespace(is_local_mode=lambda: False)
        orchestrator = MagicMock()
        orchestrator.optimize = AsyncMock(return_value=winner)
        dataset = Dataset([EvaluationExample({"text": "x"}, "x")], name="d")

        await fn._run_and_finalize_optimization(
            orchestrator, dataset, {"model": ["a", "b"]}, save_to=None
        )

        assert fn.get_best_config() == {"model": "b"}
        exported = fn.export_best_config(tmp_path / "best-configs")
        assert exported.exists()


class _GovernedResolverStub:
    """Duck-typed resolver declaring a governed CVAR (consolidation seam)."""

    def __init__(self, governed: bool = True):
        self._governed = governed

    def has_governed_cvars(self) -> bool:
        return self._governed

    def resolve(self, suggestion):  # pragma: no cover - not exercised here
        raise AssertionError("resolution is not under test")


class _OpaqueResolverStub:
    """A custom resolver WITHOUT governance introspection."""

    def resolve(self, suggestion):  # pragma: no cover - not exercised here
        raise AssertionError("resolution is not under test")


class TestPerCvarStrictDisjuncts:
    """RFC 0001 §3.6 consolidation: declared-governed CVARs make the run
    strict, gate-independently."""

    def test_governed_resolver_is_strict_without_any_gate(self):
        orchestrator = _orchestrator(None)
        orchestrator.knob_resolver = _GovernedResolverStub(governed=True)
        assert orchestrator._is_strict_evidence_mode()

    def test_governed_resolver_is_strict_with_non_strict_policy(self):
        orchestrator = _orchestrator(NON_STRICT_POLICY)
        orchestrator.knob_resolver = _GovernedResolverStub(governed=True)
        assert orchestrator._is_strict_evidence_mode()

    def test_ungoverned_resolver_alone_is_not_strict(self):
        orchestrator = _orchestrator(NON_STRICT_POLICY)
        orchestrator.knob_resolver = _GovernedResolverStub(governed=False)
        assert not orchestrator._is_strict_evidence_mode()

    def test_opaque_custom_resolver_contributes_no_disjunct(self):
        """Documented degradation: a duck-typed resolver without
        has_governed_cvars cannot opt the run into strictness — declare via
        the promotion policy instead."""
        orchestrator = _orchestrator(NON_STRICT_POLICY)
        orchestrator.knob_resolver = _OpaqueResolverStub()
        assert not orchestrator._is_strict_evidence_mode()
        strict = _orchestrator(STRICT_POLICIES["require_calibration"])
        strict.knob_resolver = _OpaqueResolverStub()
        assert strict._is_strict_evidence_mode()


class TestNoGateStrictClosure:
    """The no-gate lane must not promote under strict mode: a raw
    _simple_is_better win would be counted as a certified promotion by
    _update_best_trial_cache (fail open). Found during consolidation."""

    def test_no_gate_strict_withholds_and_never_consults_simple(self):
        orchestrator = _orchestrator(None)
        orchestrator.knob_resolver = _GovernedResolverStub(governed=True)
        orchestrator._simple_is_better = MagicMock(return_value=True)

        assert orchestrator._evaluate_promotion("candidate", _trial(0.99)) is False
        orchestrator._simple_is_better.assert_not_called()
        assert any(
            "no promotion gate" in reason
            for reason in orchestrator._strict_withheld_promotions
        )

    def test_no_gate_strict_full_cache_flow_certifies_nothing(self):
        orchestrator = _orchestrator(None)
        orchestrator.knob_resolver = _GovernedResolverStub(governed=True)
        orchestrator._best_trial_cached = None
        orchestrator._incumbent_config_hash = None
        orchestrator._simple_is_better = MagicMock(return_value=True)

        first = _trial(0.5, config={"model": "a"})
        second = _trial(0.99, config={"model": "b"})
        orchestrator._update_best_trial_cache(first)
        # first trial seeds the incumbent: initialization, not certification
        assert orchestrator._best_trial_cached is first
        assert orchestrator._certified_promotions == 0

        orchestrator._update_best_trial_cache(second)
        # the higher-scoring trial must NOT displace the incumbent rawly
        assert orchestrator._best_trial_cached is first
        assert orchestrator._certified_promotions == 0
        orchestrator._simple_is_better.assert_not_called()

    def test_no_gate_non_strict_keeps_legacy_simple_lane(self):
        orchestrator = _orchestrator(None)
        orchestrator._simple_is_better = MagicMock(return_value=True)
        assert orchestrator._evaluate_promotion("candidate", _trial(0.9)) is True
        orchestrator._simple_is_better.assert_called_once()
