"""Test Optuna multi-objective optimization capabilities."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import optuna
import pytest

from traigent.api.types import TrialResult, TrialStatus
from traigent.optimizers.optuna_coordinator import BatchOptimizer, OptunaCoordinator
from traigent.optimizers.optuna_optimizer import (
    OptunaNSGAIIOptimizer,
    OptunaTPEOptimizer,
)


class TestOptunaMultiObjective:
    """Test multi-objective optimization with Optuna."""

    def test_pareto_front_discovery(self):
        """Test that Optuna finds Pareto-optimal solutions."""

        # Multi-objective: minimize f1 = x^2, minimize f2 = (x-2)^2 + y^2
        def objective(trial):
            x = trial.suggest_float("x", -10, 10)
            y = trial.suggest_float("y", -10, 10)

            f1 = x**2
            f2 = (x - 2) ** 2 + y**2

            return f1, f2

        study = optuna.create_study(
            directions=["minimize", "minimize"],
            sampler=optuna.samplers.NSGAIISampler(seed=42),
        )

        study.optimize(objective, n_trials=100)

        # Get Pareto front
        pareto_trials = study.best_trials

        # Verify we found multiple Pareto-optimal solutions
        assert len(pareto_trials) > 1

        # Verify Pareto dominance
        for trial in pareto_trials:
            # No other Pareto trial should dominate this one
            for other in pareto_trials:
                if trial != other:
                    dominates = all(
                        v1 <= v2 for v1, v2 in zip(other.values, trial.values)
                    ) and any(v1 < v2 for v1, v2 in zip(other.values, trial.values))
                    assert not dominates

    def test_multi_objective_with_constraints(self):
        """Test multi-objective optimization with constraints."""

        def objective_with_constraints(trial):
            x = trial.suggest_float("x", 0, 5)
            y = trial.suggest_float("y", 0, 5)

            # Objectives
            f1 = x + y  # Maximize sum
            f2 = x * y  # Maximize product

            # Constraint: x + 2*y <= 6
            constraint_violation = max(0, x + 2 * y - 6)

            # Penalize constraint violation
            if constraint_violation > 0:
                f1 -= constraint_violation * 10
                f2 -= constraint_violation * 10

            return -f1, -f2  # Minimize negative = maximize

        study = optuna.create_study(
            directions=["minimize", "minimize"],
            sampler=optuna.samplers.TPESampler(seed=42),
        )

        study.optimize(objective_with_constraints, n_trials=50)

        # Check that best solutions respect constraint
        for trial in study.best_trials:
            x = trial.params["x"]
            y = trial.params["y"]
            assert x + 2 * y <= 6.1  # Small tolerance for numerical errors

    def test_multi_objective_coordinator(self):
        """Test OptunaCoordinator with multi-objective optimization."""
        coordinator = OptunaCoordinator(
            directions=["maximize", "minimize", "minimize"],  # accuracy, cost, latency
            search_space={
                "model": {
                    "type": "categorical",
                    "choices": ["small", "medium", "large"],
                },
                "learning_rate": {
                    "type": "float",
                    "low": 1e-4,
                    "high": 1e-2,
                    "log": True,
                },
            },
        )

        def materialize_config(raw_config):
            # Translate AbstractConfig to pure dict for evaluation
            config = {
                param: value if not isinstance(value, tuple) else value[0]
                for param, value in raw_config.items()
            }
            config["_trial_id"] = raw_config.get("_trial_id")
            return config

        # Simulate optimization loop
        for _ in range(10):
            raw_configs, trials = coordinator.ask_batch(n_suggestions=2)
            configs = [materialize_config(cfg) for cfg in raw_configs]

            for config, _trial in zip(configs, trials):
                trial_id = config["_trial_id"]

                # Simulate different objectives based on model size
                if config["model"] == "small":
                    values = [0.80, 0.1, 50]  # Lower accuracy, lowest cost, fastest
                elif config["model"] == "medium":
                    values = [0.90, 0.3, 100]  # Medium everything
                else:  # large
                    values = [0.95, 0.5, 200]  # Highest accuracy, highest cost, slowest

                # Add some noise based on learning rate
                lr_factor = config["learning_rate"] * 100
                values[0] += (
                    lr_factor * 0.01
                )  # Accuracy improves with higher LR (simplified)

                coordinator.tell_result(trial_id, values)

        # Verify we have Pareto front
        pareto_trials = coordinator.study.best_trials
        assert len(pareto_trials) > 0

        # Verify different model sizes are represented
        model_sizes = {t.params["model"] for t in pareto_trials}
        assert len(model_sizes) >= 2  # Should have trade-offs

    def test_nsga2_multi_objective_optimizer(self):
        """Test NSGA-II optimizer for multi-objective problems."""
        config_space = {
            "x1": (-5.0, 5.0),
            "x2": (-5.0, 5.0),
            "x3": (-5.0, 5.0),
        }

        optimizer = OptunaNSGAIIOptimizer(
            config_space=config_space,
            objectives=["minimize_f1", "minimize_f2", "minimize_f3"],
            max_trials=50,
            population_size=20,
            sampler=optuna.samplers.NSGAIISampler(seed=42),
        )

        history = []

        for _ in range(30):
            config = optimizer.suggest_next_trial(history)
            trial_id = config.get("_optuna_trial_id")

            # Three-objective test problem
            x1, x2, x3 = config["x1"], config["x2"], config["x3"]
            f1 = x1**2 + x2**2
            f2 = (x1 - 1) ** 2 + x2**2 + x3**2
            f3 = x1**2 + (x2 - 1) ** 2 + x3**2

            optimizer.report_trial_result(trial_id, [f1, f2, f3])

            history.append(
                TrialResult(
                    trial_id=str(trial_id),
                    config=config,
                    metrics={
                        "minimize_f1": f1,
                        "minimize_f2": f2,
                        "minimize_f3": f3,
                    },
                    status=TrialStatus.COMPLETED,
                    duration=0.0,
                    timestamp=datetime.now(UTC),
                )
            )

        # Get Pareto front
        pareto_trials = optimizer.study.best_trials
        assert len(pareto_trials) > 1

        # Verify diversity in Pareto front
        pareto_values = [t.values for t in pareto_trials]
        # Check that we have variation in all objectives
        for obj_idx in range(3):
            obj_values = [v[obj_idx] for v in pareto_values]
            assert max(obj_values) - min(obj_values) > 0.1

    def test_weighted_multi_objective_fallback(self):
        """Test fallback to weighted sum for algorithms that don't support multi-objective."""
        config_space = {"param": [0.1, 0.5, 1.0]}

        optimizer = OptunaTPEOptimizer(
            config_space=config_space,
            objectives=["accuracy", "cost"],
            objective_weights=[0.7, 0.3],
            max_trials=10,
        )

        history = []
        for _ in range(5):
            config = optimizer.suggest_next_trial(history)
            trial_id = config.get("_optuna_trial_id")

            accuracy = config["param"]
            cost = 1.0 - config["param"]

            optimizer.report_trial_result(trial_id, [accuracy, cost])
            history.append(
                TrialResult(
                    trial_id=str(trial_id),
                    config=config,
                    metrics={"accuracy": accuracy, "cost": cost},
                    status=TrialStatus.COMPLETED,
                    duration=0.0,
                    timestamp=datetime.now(UTC),
                )
            )

        assert len(optimizer.study.trials) == 5

    def test_multi_objective_batch_optimization(self):
        """Test batch optimization with multiple objectives."""

        def worker_function(config):
            """Worker simulating multi-objective evaluation."""
            x = config["x"]
            y = config["y"]

            import time

            time.sleep(0.01)

            obj1 = x**2 + y**2
            obj2 = (x - 3) ** 2 + (y - 3) ** 2

            return {"values": [obj1, obj2]}

        config_space = {"x": (0.0, 5.0), "y": (0.0, 5.0)}

        batch_optimizer = BatchOptimizer(
            config_space=config_space,
            objectives=["minimize_distance_origin", "minimize_distance_target"],
            n_workers=3,
            worker_fn=worker_function,
        )

        batch_optimizer.optimize_batch(n_trials=15)

        study = batch_optimizer.coordinator.study
        pareto_trials = study.best_trials

        assert len(pareto_trials) > 1

        # Verify trade-off
        for trial in pareto_trials:
            x, y = trial.params["x"], trial.params["y"]
            dist_origin = x**2 + y**2
            dist_target = (x - 3) ** 2 + (y - 3) ** 2
            for other_trial in pareto_trials:
                if trial != other_trial:
                    other_x, other_y = other_trial.params["x"], other_trial.params["y"]
                    other_dist_origin = other_x**2 + other_y**2
                    other_dist_target = (other_x - 3) ** 2 + (other_y - 3) ** 2
                    if dist_origin < other_dist_origin:
                        assert dist_target >= other_dist_target - 0.01

    def test_multi_objective_with_failed_trials(self):
        """Test multi-objective optimization handling failed trials."""
        coordinator = OptunaCoordinator(
            directions=["maximize", "minimize"],
            search_space={
                "stable": {
                    "type": "categorical",
                    "choices": [True, False],
                },
                "param": {"type": "float", "low": 0.0, "high": 1.0},
            },
        )

        successful_count = 0
        failed_count = 0

        for _ in range(20):
            raw_configs, _ = coordinator.ask_batch(n_suggestions=1)
            config = {
                param: value if not isinstance(value, tuple) else value[0]
                for param, value in raw_configs[0].items()
            }
            trial_id = config["_trial_id"]

            if config["stable"]:
                values = [config["param"], 1 - config["param"]]
                coordinator.tell_result(trial_id, values)
                successful_count += 1
            else:
                coordinator.tell_failure(trial_id, "Unstable configuration")
                failed_count += 1

        all_trials = coordinator.study.trials
        assert len(all_trials) == 20

        completed = [
            t for t in all_trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
        failed = [t for t in all_trials if t.state == optuna.trial.TrialState.FAIL]

        assert len(completed) == successful_count
        assert len(failed) == failed_count

        pareto_trials = coordinator.study.best_trials
        assert all(t.state == optuna.trial.TrialState.COMPLETE for t in pareto_trials)

    def test_multi_objective_hypervolume_improvement(self):
        """Test that hypervolume improves over optimization."""
        try:
            from optuna._hypervolume import WFG
        except ImportError:  # pragma: no cover - optional dependency
            pytest.skip("Optuna WFG hypervolume support not available")

        study = optuna.create_study(
            directions=["minimize", "minimize"],
            sampler=optuna.samplers.RandomSampler(seed=42),
        )

        hypervolumes = []
        reference_point = [1.0, 1.0]

        def objective(trial):
            x = trial.suggest_float("x", 0, 1)
            y = trial.suggest_float("y", 0, 1)

            f1 = x
            f2 = (1 - x) ** 2 + y**2

            if len(study.trials) > 0:
                completed_trials = [
                    t
                    for t in study.trials
                    if t.state == optuna.trial.TrialState.COMPLETE
                ]
                if completed_trials:
                    pareto_trials = study.best_trials
                    if pareto_trials:
                        pareto_values = np.array([t.values for t in pareto_trials])
                        wfg = WFG()
                        hv = wfg.compute(pareto_values, reference_point)
                        hypervolumes.append(hv)

            return f1, f2

        study.optimize(objective, n_trials=30)

        if len(hypervolumes) > 10:
            early_avg = np.mean(hypervolumes[:5])
            late_avg = np.mean(hypervolumes[-5:])
            assert late_avg >= early_avg

    def test_multi_objective_visualization_data(self):
        """Test that multi-objective optimization generates visualization-ready data."""
        coordinator = OptunaCoordinator(
            directions=["maximize", "minimize"],
            search_space={
                "param1": {"type": "float", "low": 0.0, "high": 1.0},
                "param2": {"type": "int", "low": 1, "high": 10},
            },
        )

        pareto_trials = []
        for _ in range(20):
            raw_configs, _ = coordinator.ask_batch(n_suggestions=1)
            config = {
                param: value if not isinstance(value, tuple) else value[0]
                for param, value in raw_configs[0].items()
            }
            trial_id = config["_trial_id"]

            obj1 = config["param1"] * config["param2"]
            obj2 = config["param1"] + config["param2"] / 10

            coordinator.tell_result(trial_id, [obj1, obj2])

        study = coordinator.study

        all_trials = study.trials
        pareto_trials = study.best_trials

        all_values = [(t.values[0], t.values[1]) for t in all_trials if t.values]
        pareto_values = [(t.values[0], t.values[1]) for t in pareto_trials]

        assert len(all_values) == 20
        assert len(pareto_values) > 0
        assert len(pareto_values) <= len(all_values)

        for pv in pareto_values:
            assert pv in all_values
