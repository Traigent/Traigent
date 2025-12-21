"""Tests for Haystack advanced optimization module.

Coverage: Epic 5 (Advanced Optimization & Pareto Analysis)
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from traigent.integrations.haystack.advanced_optimization import (
    HaystackOptimizer,
    OptimizationDirection,
    OptimizationResult,
    OptimizationTarget,
    TrialResult,
    compute_pareto_frontier,
    export_optimization_history,
    get_hyperparameter_importance,
    rank_by_metric,
)


class TestOptimizationTarget:
    """Tests for OptimizationTarget dataclass."""

    def test_default_direction(self):
        """Test default direction is maximize."""
        target = OptimizationTarget(metric_name="accuracy")
        assert target.direction == OptimizationDirection.MAXIMIZE

    def test_string_direction_maximize(self):
        """Test string direction conversion for maximize."""
        target = OptimizationTarget("accuracy", "maximize")
        assert target.direction == OptimizationDirection.MAXIMIZE

    def test_string_direction_minimize(self):
        """Test string direction conversion for minimize."""
        target = OptimizationTarget("cost", "minimize")
        assert target.direction == OptimizationDirection.MINIMIZE

    def test_enum_direction(self):
        """Test enum direction is preserved."""
        target = OptimizationTarget("latency", OptimizationDirection.MINIMIZE)
        assert target.direction == OptimizationDirection.MINIMIZE

    def test_default_weight(self):
        """Test default weight is 1.0."""
        target = OptimizationTarget("accuracy")
        assert target.weight == 1.0

    def test_custom_weight(self):
        """Test custom weight is preserved."""
        target = OptimizationTarget("accuracy", weight=2.0)
        assert target.weight == 2.0


class TestTrialResult:
    """Tests for TrialResult dataclass."""

    def test_successful_trial(self):
        """Test successful trial detection."""
        trial = TrialResult(
            trial_id="trial_1",
            config={"temp": 0.7},
            metrics={"accuracy": 0.9},
        )
        assert trial.is_successful

    def test_failed_trial(self):
        """Test failed trial detection."""
        trial = TrialResult(
            trial_id="trial_1",
            config={"temp": 0.7},
            metrics={},
        )
        assert not trial.is_successful

    def test_constraints_default_true(self):
        """Test constraints_satisfied defaults to True."""
        trial = TrialResult(
            trial_id="trial_1",
            config={},
            metrics={"accuracy": 0.9},
        )
        assert trial.constraints_satisfied

    def test_timestamp_default(self):
        """Test timestamp is set by default."""
        trial = TrialResult(
            trial_id="trial_1",
            config={},
            metrics={},
        )
        assert trial.timestamp is not None


class TestOptimizationResult:
    """Tests for OptimizationResult dataclass."""

    def test_empty_result(self):
        """Test empty optimization result."""
        result = OptimizationResult()
        assert result.best_config is None
        assert result.history == []
        assert result.pareto_configs == []
        assert result.total_trials == 0

    def test_ranked_runs_empty(self):
        """Test ranked_runs with empty history."""
        result = OptimizationResult()
        assert result.ranked_runs == []

    def test_ranked_runs_sorted(self):
        """Test ranked_runs are sorted by primary metric."""
        trials = [
            TrialResult("t1", {"temp": 0.5}, {"accuracy": 0.7}),
            TrialResult("t2", {"temp": 0.7}, {"accuracy": 0.9}),
            TrialResult("t3", {"temp": 0.9}, {"accuracy": 0.8}),
        ]
        result = OptimizationResult(history=trials)
        ranked = result.ranked_runs

        assert len(ranked) == 3
        assert ranked[0].metrics["accuracy"] == 0.9
        assert ranked[1].metrics["accuracy"] == 0.8
        assert ranked[2].metrics["accuracy"] == 0.7


class TestComputeParetoFrontier:
    """Tests for compute_pareto_frontier function."""

    def test_empty_trials(self):
        """Test with empty trials list."""
        configs, metrics = compute_pareto_frontier([], [])
        assert configs == []
        assert metrics == []

    def test_single_trial(self):
        """Test with single trial."""
        trials = [
            TrialResult("t1", {"temp": 0.7}, {"accuracy": 0.9, "cost": 0.05}),
        ]
        targets = [
            OptimizationTarget("accuracy", "maximize"),
            OptimizationTarget("cost", "minimize"),
        ]

        configs, metrics = compute_pareto_frontier(trials, targets)
        assert len(configs) == 1
        assert configs[0] == {"temp": 0.7}

    def test_pareto_two_objectives(self):
        """Test Pareto frontier with two objectives."""
        trials = [
            TrialResult("t1", {"temp": 0.5}, {"accuracy": 0.7, "cost": 0.02}),
            TrialResult("t2", {"temp": 0.7}, {"accuracy": 0.9, "cost": 0.08}),
            TrialResult("t3", {"temp": 0.9}, {"accuracy": 0.85, "cost": 0.05}),
            # t4 is dominated by t3: lower accuracy (0.75 < 0.85) and higher cost (0.06 > 0.05)
            TrialResult("t4", {"temp": 0.6}, {"accuracy": 0.75, "cost": 0.06}),
        ]
        targets = [
            OptimizationTarget("accuracy", "maximize"),
            OptimizationTarget("cost", "minimize"),
        ]

        configs, metrics = compute_pareto_frontier(trials, targets)

        # t1 (0.7, 0.02) - Pareto (lowest cost)
        # t2 (0.9, 0.08) - Pareto (highest accuracy)
        # t3 (0.85, 0.05) - Pareto (good balance)
        # t4 (0.75, 0.06) - Dominated by t3 (worse in both objectives)

        assert len(configs) == 3
        # Check t4 is not in Pareto set
        assert {"temp": 0.6} not in configs

    def test_pareto_excludes_constraint_violations(self):
        """Test Pareto frontier excludes constraint violations."""
        trials = [
            TrialResult(
                "t1",
                {"temp": 0.5},
                {"accuracy": 0.95, "cost": 0.01},
                constraints_satisfied=False,
            ),
            TrialResult(
                "t2",
                {"temp": 0.7},
                {"accuracy": 0.9, "cost": 0.08},
                constraints_satisfied=True,
            ),
        ]
        targets = [
            OptimizationTarget("accuracy", "maximize"),
        ]

        configs, metrics = compute_pareto_frontier(trials, targets)

        # Only t2 should be in Pareto set
        assert len(configs) == 1
        assert configs[0] == {"temp": 0.7}

    def test_pareto_all_constraints_violated(self):
        """Test Pareto frontier when all constraints violated."""
        trials = [
            TrialResult(
                "t1", {"temp": 0.5}, {"accuracy": 0.9}, constraints_satisfied=False
            ),
            TrialResult(
                "t2", {"temp": 0.7}, {"accuracy": 0.8}, constraints_satisfied=False
            ),
        ]
        targets = [OptimizationTarget("accuracy", "maximize")]

        configs, metrics = compute_pareto_frontier(trials, targets)
        assert configs == []
        assert metrics == []


class TestRankByMetric:
    """Tests for rank_by_metric function."""

    def test_rank_maximize(self):
        """Test ranking with maximize direction."""
        trials = [
            TrialResult("t1", {"temp": 0.5}, {"accuracy": 0.7}),
            TrialResult("t2", {"temp": 0.7}, {"accuracy": 0.9}),
            TrialResult("t3", {"temp": 0.9}, {"accuracy": 0.8}),
        ]

        ranked = rank_by_metric(trials, "accuracy", OptimizationDirection.MAXIMIZE)

        assert len(ranked) == 3
        assert ranked[0].trial_id == "t2"  # 0.9
        assert ranked[1].trial_id == "t3"  # 0.8
        assert ranked[2].trial_id == "t1"  # 0.7

    def test_rank_minimize(self):
        """Test ranking with minimize direction."""
        trials = [
            TrialResult("t1", {"temp": 0.5}, {"cost": 0.05}),
            TrialResult("t2", {"temp": 0.7}, {"cost": 0.02}),
            TrialResult("t3", {"temp": 0.9}, {"cost": 0.08}),
        ]

        ranked = rank_by_metric(trials, "cost", OptimizationDirection.MINIMIZE)

        assert len(ranked) == 3
        assert ranked[0].trial_id == "t2"  # 0.02
        assert ranked[1].trial_id == "t1"  # 0.05
        assert ranked[2].trial_id == "t3"  # 0.08

    def test_rank_excludes_failed(self):
        """Test ranking excludes failed trials."""
        trials = [
            TrialResult("t1", {"temp": 0.5}, {"accuracy": 0.9}),
            TrialResult("t2", {"temp": 0.7}, {}),  # Failed
            TrialResult("t3", {"temp": 0.9}, {"accuracy": 0.8}),
        ]

        ranked = rank_by_metric(trials, "accuracy", OptimizationDirection.MAXIMIZE)
        assert len(ranked) == 2

    def test_rank_excludes_constraint_violations(self):
        """Test ranking excludes constraint violations."""
        trials = [
            TrialResult(
                "t1", {"temp": 0.5}, {"accuracy": 0.95}, constraints_satisfied=False
            ),
            TrialResult(
                "t2", {"temp": 0.7}, {"accuracy": 0.9}, constraints_satisfied=True
            ),
        ]

        ranked = rank_by_metric(
            trials, "accuracy", OptimizationDirection.MAXIMIZE, require_constraints=True
        )
        assert len(ranked) == 1
        assert ranked[0].trial_id == "t2"


class TestGetHyperparameterImportance:
    """Tests for get_hyperparameter_importance function."""

    def test_empty_trials(self):
        """Test with empty trials."""
        importance = get_hyperparameter_importance([], "accuracy")
        assert importance == {}

    def test_insufficient_trials(self):
        """Test with insufficient trials."""
        trials = [
            TrialResult("t1", {"temp": 0.7}, {"accuracy": 0.9}),
            TrialResult("t2", {"temp": 0.8}, {"accuracy": 0.8}),
        ]
        importance = get_hyperparameter_importance(trials, "accuracy")
        assert importance == {}

    def test_importance_computed(self):
        """Test importance is computed for parameters."""
        trials = [
            TrialResult("t1", {"temp": 0.5, "model": "gpt-4"}, {"accuracy": 0.7}),
            TrialResult("t2", {"temp": 0.5, "model": "gpt-3.5"}, {"accuracy": 0.6}),
            TrialResult("t3", {"temp": 0.9, "model": "gpt-4"}, {"accuracy": 0.9}),
            TrialResult("t4", {"temp": 0.9, "model": "gpt-3.5"}, {"accuracy": 0.75}),
            TrialResult("t5", {"temp": 0.7, "model": "gpt-4"}, {"accuracy": 0.8}),
        ]

        importance = get_hyperparameter_importance(trials, "accuracy")

        assert "temp" in importance
        assert "model" in importance
        assert sum(importance.values()) == pytest.approx(1.0)


class TestExportOptimizationHistory:
    """Tests for export_optimization_history function."""

    def test_export_json(self):
        """Test JSON export."""
        trials = [
            TrialResult(
                "t1",
                {"temp": 0.7},
                {"accuracy": 0.9},
                timestamp=datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC),
            ),
        ]
        result = OptimizationResult(
            best_config={"temp": 0.7},
            best_metrics={"accuracy": 0.9},
            history=trials,
            total_trials=1,
        )

        exported = export_optimization_history(result, "json")
        assert isinstance(exported, str)
        assert "temp" in exported
        assert "accuracy" in exported

    def test_export_dict(self):
        """Test dict export."""
        result = OptimizationResult(
            best_config={"temp": 0.7},
            best_metrics={"accuracy": 0.9},
            total_trials=1,
        )

        exported = export_optimization_history(result, "dict")
        assert isinstance(exported, dict)
        assert exported["best_config"] == {"temp": 0.7}
        assert exported["total_trials"] == 1

    def test_export_csv(self):
        """Test CSV export."""
        trials = [
            TrialResult(
                "t1",
                {"temp": 0.7},
                {"accuracy": 0.9},
                timestamp=datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC),
            ),
        ]
        result = OptimizationResult(history=trials)

        exported = export_optimization_history(result, "csv")
        assert isinstance(exported, str)
        assert "trial_id" in exported
        assert "t1" in exported

    def test_export_invalid_format(self):
        """Test invalid format raises error."""
        result = OptimizationResult()

        with pytest.raises(ValueError, match="Unknown format"):
            export_optimization_history(result, "xml")


class TestHaystackOptimizer:
    """Tests for HaystackOptimizer class."""

    def test_init_defaults(self):
        """Test optimizer initialization with defaults."""
        evaluator = MagicMock()
        evaluator.constraints = []

        optimizer = HaystackOptimizer(
            evaluator=evaluator,
            config_space={"temp": (0.0, 1.0)},
        )

        assert optimizer.strategy == "bayesian"
        assert optimizer.n_trials == 50
        assert optimizer.n_parallel == 1
        assert len(optimizer.targets) == 1

    def test_init_with_targets(self):
        """Test optimizer initialization with custom targets."""
        evaluator = MagicMock()

        targets = [
            OptimizationTarget("accuracy", "maximize"),
            OptimizationTarget("cost", "minimize"),
        ]

        optimizer = HaystackOptimizer(
            evaluator=evaluator,
            config_space={"temp": (0.0, 1.0)},
            targets=targets,
        )

        assert len(optimizer.targets) == 2

    def test_init_strategies(self):
        """Test different strategy options."""
        evaluator = MagicMock()

        for strategy in ["bayesian", "tpe", "evolutionary", "random", "grid"]:
            optimizer = HaystackOptimizer(
                evaluator=evaluator,
                config_space={"temp": [0.5, 0.7, 0.9]},
                strategy=strategy,
            )
            assert optimizer.strategy == strategy

    @pytest.mark.asyncio
    async def test_optimize_basic(self):
        """Test basic optimization flow."""
        # Create mock evaluator
        evaluator = MagicMock()
        evaluator.constraints = []

        # Mock evaluation result
        eval_result = MagicMock()
        eval_result.aggregated_metrics = {
            "accuracy": 0.85,
            "constraints_satisfied": True,
        }

        async def mock_evaluate(*args, **kwargs):
            return eval_result

        evaluator.evaluate = mock_evaluate
        evaluator.pipeline = MagicMock()
        evaluator.get_core_dataset = MagicMock(return_value=[])

        # Mock the optimizer creation
        with patch(
            "traigent.integrations.haystack.advanced_optimization.HaystackOptimizer._create_optimizer"
        ) as mock_create:
            mock_opt = MagicMock()
            mock_opt.suggest_next_trial.return_value = {"temp": 0.7}
            mock_create.return_value = mock_opt

            optimizer = HaystackOptimizer(
                evaluator=evaluator,
                config_space={"temp": (0.0, 1.0)},
                n_trials=3,
            )

            result = await optimizer.optimize()

            assert result.total_trials == 3
            assert len(result.history) == 3

    @pytest.mark.asyncio
    async def test_optimize_with_timeout(self):
        """Test optimization respects timeout."""
        evaluator = MagicMock()
        evaluator.constraints = []

        # Slow evaluation
        async def slow_evaluate(*args, **kwargs):
            import asyncio

            await asyncio.sleep(0.1)
            result = MagicMock()
            result.aggregated_metrics = {"accuracy": 0.8}
            return result

        evaluator.evaluate = slow_evaluate
        evaluator.pipeline = MagicMock()
        evaluator.get_core_dataset = MagicMock(return_value=[])

        with patch(
            "traigent.integrations.haystack.advanced_optimization.HaystackOptimizer._create_optimizer"
        ) as mock_create:
            mock_opt = MagicMock()
            mock_opt.suggest_next_trial.return_value = {"temp": 0.7}
            mock_create.return_value = mock_opt

            optimizer = HaystackOptimizer(
                evaluator=evaluator,
                config_space={"temp": (0.0, 1.0)},
                n_trials=100,
                timeout_seconds=0.15,  # Very short timeout
            )

            result = await optimizer.optimize()

            # Should stop before 100 trials due to timeout
            assert result.total_trials < 100

    @pytest.mark.asyncio
    async def test_warm_start(self):
        """Test warm start from previous results."""
        evaluator = MagicMock()

        previous = [
            TrialResult("t1", {"temp": 0.5}, {"accuracy": 0.7}),
            TrialResult("t2", {"temp": 0.7}, {"accuracy": 0.9}),
        ]

        optimizer = HaystackOptimizer(
            evaluator=evaluator,
            config_space={"temp": (0.0, 1.0)},
        )

        await optimizer.warm_start(previous)

        assert len(optimizer._history) == 2
        assert optimizer._trial_counter == 2


class TestOptimizerStrategies:
    """Tests for different optimization strategies."""

    def test_create_bayesian_optimizer(self):
        """Test Bayesian optimizer creation."""
        evaluator = MagicMock()

        optimizer = HaystackOptimizer(
            evaluator=evaluator,
            config_space={"temp": (0.0, 1.0)},
            strategy="bayesian",
        )

        with patch("traigent.optimizers.optuna_optimizer.OptunaTPEOptimizer") as mock:
            mock.return_value = MagicMock()
            opt = optimizer._create_optimizer()
            # Should try TPE first
            assert opt is not None

    def test_create_evolutionary_optimizer(self):
        """Test evolutionary optimizer creation."""
        evaluator = MagicMock()

        optimizer = HaystackOptimizer(
            evaluator=evaluator,
            config_space={"temp": (0.0, 1.0)},
            strategy="evolutionary",
        )

        with patch(
            "traigent.optimizers.optuna_optimizer.OptunaNSGAIIOptimizer"
        ) as mock:
            mock.return_value = MagicMock()
            opt = optimizer._create_optimizer()
            assert opt is not None

    def test_invalid_strategy(self):
        """Test invalid strategy raises error."""
        evaluator = MagicMock()

        optimizer = HaystackOptimizer(
            evaluator=evaluator,
            config_space={"temp": (0.0, 1.0)},
            strategy="invalid",
        )

        with pytest.raises(ValueError, match="Unknown strategy"):
            optimizer._create_optimizer()
