"""Test Optuna ask/tell pattern implementation for distributed optimization."""

from __future__ import annotations

import optuna
import pytest

from traigent.optimizers.optuna_coordinator import OptunaCoordinator
from traigent.optimizers.optuna_optimizer import OptunaTPEOptimizer


class TestOptunaAskTellPattern:
    """Test the ask/tell pattern for distributed Optuna optimization."""

    def test_ask_tell_basic_flow(self):
        """Test basic ask/tell flow with single objective."""
        config_space = {
            "learning_rate": (1e-5, 1e-1),
            "batch_size": (16, 128),
        }

        coordinator = OptunaCoordinator(
            config_space=config_space,
            objectives=["accuracy"],
            directions=["maximize"],
            storage=None,  # In-memory
        )

        # Ask for configuration
        configs, _ = coordinator.ask_batch(n_suggestions=1)
        assert len(configs) == 1

        config = configs[0]
        trial_id = config["_trial_id"]

        # Verify configuration has expected parameters
        assert "learning_rate" in config
        assert "batch_size" in config
        assert 1e-5 <= config["learning_rate"] <= 1e-1
        assert 16 <= config["batch_size"] <= 128

        # Tell result
        coordinator.tell_result(trial_id, 0.95)

        # Verify trial was completed
        assert len(coordinator.study.trials) == 1
        assert coordinator.study.trials[0].value == 0.95

    def test_ask_tell_multi_objective(self):
        """Test ask/tell with multi-objective optimization."""
        config_space = {
            "model": ["gpt-4", "gpt-3.5", "claude"],
            "temperature": (0.0, 1.0),
        }

        coordinator = OptunaCoordinator(
            config_space=config_space,
            objectives=["accuracy", "cost"],
            directions=["maximize", "minimize"],
        )

        # Ask for multiple configurations
        configs, _ = coordinator.ask_batch(n_suggestions=3)
        assert len(configs) == 3

        # Simulate distributed execution
        results = []
        for _i, config in enumerate(configs):
            trial_id = config["_trial_id"]
            # Simulate different results for different models
            if config["model"] == "gpt-4":
                values = [0.95, 0.5]  # High accuracy, high cost
            elif config["model"] == "gpt-3.5":
                values = [0.85, 0.2]  # Medium accuracy, low cost
            else:
                values = [0.90, 0.3]  # Good balance

            results.append((trial_id, values))

        # Tell results
        for trial_id, values in results:
            coordinator.tell_result(trial_id, values)

        # Verify all trials completed
        assert len(coordinator.study.trials) == 3
        for trial in coordinator.study.trials:
            assert trial.values is not None
            assert len(trial.values) == 2

    def test_ask_tell_with_pruning(self):
        """Test trial pruning through intermediate reporting."""
        config_space = {
            "learning_rate": (1e-5, 1e-1),
        }

        coordinator = OptunaCoordinator(
            config_space=config_space, objectives=["accuracy"], directions=["maximize"]
        )

        # Ask for configuration
        configs, _ = coordinator.ask_batch(n_suggestions=1)
        trial_id = configs[0]["_trial_id"]

        # Report intermediate values that might trigger pruning
        for step in range(3):
            should_prune = coordinator.report_intermediate(
                trial_id, step, 0.1 + step * 0.01
            )
            if should_prune:
                coordinator.tell_pruned(trial_id, step=step)
                break
        else:
            # If not pruned, complete normally
            coordinator.tell_result(trial_id, 0.15)

        # Verify trial state
        trial = coordinator.study.trials[0]
        assert trial.state in (
            optuna.trial.TrialState.PRUNED,
            optuna.trial.TrialState.COMPLETE,
        )

    def test_ask_tell_failure_handling(self):
        """Test handling of failed trials."""
        config_space = {"param": (0.0, 1.0)}

        coordinator = OptunaCoordinator(
            config_space=config_space, objectives=["value"], directions=["minimize"]
        )

        # Ask for configurations
        configs, _ = coordinator.ask_batch(n_suggestions=2)

        # First trial succeeds
        coordinator.tell_result(configs[0]["_trial_id"], 0.5)

        # Second trial fails
        coordinator.tell_failure(configs[1]["_trial_id"], "Model initialization failed")

        # Verify trial states
        trials = coordinator.study.trials
        assert len(trials) == 2
        assert trials[0].state == optuna.trial.TrialState.COMPLETE
        assert trials[0].value == 0.5
        assert trials[1].state == optuna.trial.TrialState.FAIL
        assert trials[1].value is None

    def test_ask_tell_concurrent_execution(self):
        """Test concurrent trial execution with ask/tell."""
        config_space = {
            "worker_id": ["w1", "w2", "w3", "w4"],
            "param": (0.0, 1.0),
        }

        coordinator = OptunaCoordinator(
            config_space=config_space,
            objectives=["value"],
            directions=["maximize"],
            storage="sqlite:///test_concurrent.db",
            study_name="test_concurrent",
        )

        # Simulate multiple workers asking for work
        worker_configs = []
        for _ in range(4):
            configs, _ = coordinator.ask_batch(n_suggestions=1)
            worker_configs.append(configs[0])

        # Verify all pending trials are tracked
        assert len(coordinator._pending_trials) == 4

        # Workers complete in different order
        import random

        random.shuffle(worker_configs)
        for config in worker_configs:
            trial_id = config["_trial_id"]
            value = random.random()
            coordinator.tell_result(trial_id, value)

        # Verify all trials completed
        assert len(coordinator.study.trials) == 4
        assert all(
            t.state == optuna.trial.TrialState.COMPLETE
            for t in coordinator.study.trials
        )
        assert len(coordinator._pending_trials) == 0

    def test_tell_unknown_trial_id(self):
        """Test handling of unknown trial IDs in tell."""
        config_space = {"param": (0.0, 1.0)}
        coordinator = OptunaCoordinator(
            config_space=config_space, objectives=["value"], directions=["maximize"]
        )

        # Try to tell result for non-existent trial
        coordinator.tell_result(999, 0.5)  # Should log warning but not crash

        # Try to tell failure for non-existent trial
        coordinator.tell_failure(999, "Error")  # Should log warning but not crash

    def test_ask_without_config_space_raises_error(self):
        """Test that creating coordinator without config_space raises error."""
        with pytest.raises(TypeError):
            OptunaCoordinator(objectives=["value"], directions=["maximize"])


class TestOptunaTPEOptimizerAskTell:
    """Test OptunaTPEOptimizer with ask/tell pattern."""

    def test_suggest_and_report_cycle(self):
        """Test the suggest_next_trial and report_trial_result cycle."""
        config_space = {
            "model": ["gpt-4", "gpt-3.5"],
            "temperature": [0.0, 0.5, 1.0],
        }

        optimizer = OptunaTPEOptimizer(
            config_space=config_space, objectives=["accuracy", "cost"], max_trials=5
        )

        history = []

        for _i in range(3):
            # Ask for next configuration
            config = optimizer.suggest_next_trial(history)

            assert "model" in config
            assert "temperature" in config
            assert "_optuna_trial_id" in config

            trial_id = config["_optuna_trial_id"]

            # Simulate execution
            if config["model"] == "gpt-4":
                objectives = [0.95, 0.5]  # High accuracy, high cost
            else:
                objectives = [0.85, 0.2]  # Lower accuracy, lower cost

            # Tell the result
            optimizer.report_trial_result(trial_id, objectives)

            # Add to history for next iteration
            history.append(
                {"config": config, "objectives": objectives, "status": "completed"}
            )

        # Verify trials were recorded
        assert len(optimizer.study.trials) == 3

    def test_report_pruned_trial(self):
        """Test reporting a pruned trial."""
        config_space = {"param": [0.1, 0.5, 1.0]}
        optimizer = OptunaTPEOptimizer(config_space=config_space, objectives=["value"])

        # Get configuration
        config = optimizer.suggest_next_trial([])
        trial_id = config["_optuna_trial_id"]

        # Report as pruned
        optimizer.report_trial_result(trial_id, None, metadata={"state": "pruned"})

        # Verify trial state
        assert optimizer.study.trials[0].state == optuna.trial.TrialState.PRUNED

    def test_report_failed_trial(self):
        """Test reporting a failed trial."""
        config_space = {"param": [0.1, 0.5, 1.0]}
        optimizer = OptunaTPEOptimizer(config_space=config_space, objectives=["value"])

        # Get configuration
        config = optimizer.suggest_next_trial([])
        trial_id = config["_optuna_trial_id"]

        # Report as failed
        optimizer.report_trial_result(
            trial_id, None, metadata={"state": "failed", "error": "Connection timeout"}
        )

        # Verify trial state
        assert optimizer.study.trials[0].state == optuna.trial.TrialState.FAIL

    def test_parallel_suggestions(self):
        """Test getting multiple suggestions in parallel."""
        config_space = {
            "model": ["a", "b", "c"],
            "param": [0.1, 0.2, 0.3, 0.4, 0.5],
        }

        optimizer = OptunaTPEOptimizer(config_space=config_space, objectives=["score"])

        # Get multiple suggestions without reporting results
        configs = []
        for _ in range(3):
            config = optimizer.suggest_next_trial([])
            configs.append(config)

        # All should have different trial IDs
        trial_ids = [c["_optuna_trial_id"] for c in configs]
        assert len(set(trial_ids)) == 3

        # All should be tracked as active
        assert len(optimizer.active_trials) == 3

        # Report results out of order
        optimizer.report_trial_result(trial_ids[2], 0.9)
        optimizer.report_trial_result(trial_ids[0], 0.7)
        optimizer.report_trial_result(trial_ids[1], 0.8)

        # All trials should be completed
        assert len(optimizer.active_trials) == 0
        assert len(optimizer.study.trials) == 3
