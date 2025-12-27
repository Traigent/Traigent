"""Comprehensive tests for optimization result types."""

from dataclasses import asdict

from traigent.optimizers.results import OptimizationResult, Trial


class TestTrial:
    """Test suite for Trial dataclass."""

    def test_initialization_minimal(self):
        """Test initialization with minimal parameters."""
        trial = Trial(configuration={"x": 1, "y": 2}, score=0.85, duration=1.5)

        assert trial.configuration == {"x": 1, "y": 2}
        assert trial.score == 0.85
        assert trial.duration == 1.5
        assert trial.metadata == {}

    def test_initialization_with_metadata(self):
        """Test initialization with metadata."""
        metadata = {"optimizer": "random", "iteration": 5}
        trial = Trial(
            configuration={"learning_rate": 0.01},
            score=0.92,
            duration=10.3,
            metadata=metadata,
        )

        assert trial.metadata == metadata

    def test_is_successful_normal_score(self):
        """Test is_successful with normal score."""
        trial = Trial({"x": 1}, score=0.5, duration=1.0)
        assert trial.is_successful is True

    def test_is_successful_negative_infinity_score(self):
        """Test is_successful with negative infinity score."""
        trial = Trial({"x": 1}, score=float("-inf"), duration=1.0)
        assert trial.is_successful is False

    def test_is_successful_failed_metadata(self):
        """Test is_successful with failed flag in metadata."""
        trial = Trial({"x": 1}, score=0.9, duration=1.0, metadata={"failed": True})
        assert trial.is_successful is False

    def test_is_successful_failed_false(self):
        """Test is_successful with failed=False in metadata."""
        trial = Trial({"x": 1}, score=0.9, duration=1.0, metadata={"failed": False})
        assert trial.is_successful is True

    def test_dataclass_features(self):
        """Test dataclass features like repr and equality."""
        trial1 = Trial({"x": 1}, 0.5, 1.0)
        trial2 = Trial({"x": 1}, 0.5, 1.0)
        trial3 = Trial({"x": 2}, 0.5, 1.0)

        # Test repr
        repr_str = repr(trial1)
        assert "Trial" in repr_str
        assert "0.5" in repr_str

        # Test equality
        assert trial1 == trial2
        assert trial1 != trial3

    def test_asdict_conversion(self):
        """Test converting trial to dictionary."""
        trial = Trial(
            configuration={"param": "value"},
            score=0.75,
            duration=2.5,
            metadata={"key": "val"},
        )

        trial_dict = asdict(trial)

        assert trial_dict == {
            "configuration": {"param": "value"},
            "score": 0.75,
            "duration": 2.5,
            "metadata": {"key": "val"},
        }

    def test_metadata_default_factory(self):
        """Test that metadata uses default factory."""
        trial1 = Trial({"x": 1}, 0.5, 1.0)
        trial2 = Trial({"x": 2}, 0.6, 2.0)

        # Should have independent metadata dicts
        trial1.metadata["key"] = "value1"
        trial2.metadata["key"] = "value2"

        assert trial1.metadata["key"] == "value1"
        assert trial2.metadata["key"] == "value2"

    def test_various_score_types(self):
        """Test trials with various score types."""
        # Integer score
        trial = Trial({"x": 1}, score=10, duration=1.0)
        assert trial.score == 10
        assert trial.is_successful is True

        # Zero score
        trial = Trial({"x": 1}, score=0, duration=1.0)
        assert trial.score == 0
        assert trial.is_successful is True

        # Negative score
        trial = Trial({"x": 1}, score=-0.5, duration=1.0)
        assert trial.score == -0.5
        assert trial.is_successful is True

    def test_complex_configuration(self):
        """Test trial with complex configuration."""
        config = {
            "model": "gpt-4",
            "hyperparameters": {
                "learning_rate": 0.001,
                "batch_size": 32,
                "layers": [128, 64, 32],
            },
            "features": ["feature1", "feature2"],
            "use_dropout": True,
        }

        trial = Trial(config, score=0.95, duration=120.5)
        assert trial.configuration == config
        assert trial.configuration["hyperparameters"]["batch_size"] == 32


class TestOptimizationResult:
    """Test suite for OptimizationResult dataclass."""

    def test_initialization_minimal(self):
        """Test initialization with minimal parameters."""
        trials = [
            Trial({"x": 1}, 0.7, 1.0),
            Trial({"x": 2}, 0.8, 1.1),
            Trial({"x": 3}, 0.75, 1.2),
        ]

        result = OptimizationResult(
            best_config={"x": 2}, best_score=0.8, trials=trials, duration=3.3
        )

        assert result.best_config == {"x": 2}
        assert result.best_score == 0.8
        assert result.trials == trials
        assert result.duration == 3.3
        assert result.convergence_info == {}

    def test_initialization_with_convergence_info(self):
        """Test initialization with convergence info."""
        convergence_info = {"converged": True, "iterations": 50, "tolerance": 0.001}

        result = OptimizationResult(
            best_config={"x": 1},
            best_score=0.99,
            trials=[],
            duration=10.0,
            convergence_info=convergence_info,
        )

        assert result.convergence_info == convergence_info

    def test_total_trials_property(self):
        """Test total_trials property."""
        # Empty trials
        result = OptimizationResult({}, 0.0, [], 0.0)
        assert result.total_trials == 0

        # Multiple trials
        trials = [Trial({"x": i}, i * 0.1, 1.0) for i in range(5)]
        result = OptimizationResult({}, 0.0, trials, 5.0)
        assert result.total_trials == 5

    def test_successful_trials_property(self):
        """Test successful_trials property."""
        trials = [
            Trial({"x": 1}, 0.7, 1.0),  # Successful
            Trial({"x": 2}, float("-inf"), 1.0),  # Failed (negative inf)
            Trial({"x": 3}, 0.8, 1.0, metadata={"failed": True}),  # Failed
            Trial({"x": 4}, 0.9, 1.0),  # Successful
            Trial({"x": 5}, 0.6, 1.0, metadata={"failed": False}),  # Successful
        ]

        result = OptimizationResult({}, 0.0, trials, 5.0)
        assert result.successful_trials == 3

    def test_success_rate_property(self):
        """Test success_rate property."""
        # Empty trials
        result = OptimizationResult({}, 0.0, [], 0.0)
        assert result.success_rate == 0.0

        # All successful
        trials_success = [Trial({"x": i}, 0.5 + i * 0.1, 1.0) for i in range(4)]
        result = OptimizationResult({}, 0.0, trials_success, 4.0)
        assert result.success_rate == 1.0

        # Mixed success
        trials_mixed = [
            Trial({"x": 1}, 0.7, 1.0),  # Success
            Trial({"x": 2}, float("-inf"), 1.0),  # Fail
            Trial({"x": 3}, 0.8, 1.0),  # Success
            Trial({"x": 4}, 0.9, 1.0, metadata={"failed": True}),  # Fail
        ]
        result = OptimizationResult({}, 0.0, trials_mixed, 4.0)
        assert result.success_rate == 0.5

    def test_dataclass_features(self):
        """Test dataclass features like repr and equality."""
        trials = [Trial({"x": 1}, 0.5, 1.0)]

        result1 = OptimizationResult({"x": 1}, 0.5, trials, 1.0)
        result2 = OptimizationResult({"x": 1}, 0.5, trials, 1.0)
        result3 = OptimizationResult({"x": 2}, 0.5, trials, 1.0)

        # Test repr
        repr_str = repr(result1)
        assert "OptimizationResult" in repr_str

        # Test equality
        assert result1 == result2
        assert result1 != result3

    def test_asdict_conversion(self):
        """Test converting result to dictionary."""
        trials = [
            Trial({"x": 1}, 0.7, 1.0, metadata={"iter": 1}),
            Trial({"x": 2}, 0.8, 1.1, metadata={"iter": 2}),
        ]

        result = OptimizationResult(
            best_config={"x": 2},
            best_score=0.8,
            trials=trials,
            duration=2.1,
            convergence_info={"converged": True},
        )

        result_dict = asdict(result)

        assert result_dict["best_config"] == {"x": 2}
        assert result_dict["best_score"] == 0.8
        assert len(result_dict["trials"]) == 2
        assert result_dict["duration"] == 2.1
        assert result_dict["convergence_info"] == {"converged": True}

    def test_convergence_info_default_factory(self):
        """Test that convergence_info uses default factory."""
        result1 = OptimizationResult({}, 0.0, [], 0.0)
        result2 = OptimizationResult({}, 0.0, [], 0.0)

        # Should have independent convergence_info dicts
        result1.convergence_info["key"] = "value1"
        result2.convergence_info["key"] = "value2"

        assert result1.convergence_info["key"] == "value1"
        assert result2.convergence_info["key"] == "value2"

    def test_with_large_number_of_trials(self):
        """Test result with large number of trials."""
        trials = []
        for i in range(1000):
            score = 0.5 + (i % 10) * 0.05
            failed = i % 7 == 0  # Every 7th trial fails
            metadata = {"failed": failed} if failed else {}
            trials.append(Trial({"x": i}, score, 0.1, metadata))

        result = OptimizationResult(
            best_config={"x": 999}, best_score=0.95, trials=trials, duration=100.0
        )

        assert result.total_trials == 1000
        assert result.successful_trials == 857  # 1000 - 143 (failed)
        assert result.success_rate == 0.857

    def test_complex_best_config(self):
        """Test result with complex best configuration."""
        best_config = {
            "model": {
                "name": "neural_net",
                "architecture": {
                    "layers": [128, 64, 32, 16],
                    "activation": "relu",
                    "dropout_rates": [0.2, 0.3, 0.4],
                },
            },
            "training": {
                "optimizer": "adam",
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 100,
            },
            "preprocessing": {
                "normalize": True,
                "feature_selection": ["pca", "correlation"],
                "pca_components": 50,
            },
        }

        result = OptimizationResult(
            best_config=best_config, best_score=0.95, trials=[], duration=3600.0
        )

        assert result.best_config == best_config
        assert result.best_config["model"]["architecture"]["layers"][1] == 64

    def test_negative_and_zero_scores(self):
        """Test result with negative and zero scores."""
        trials = [
            Trial({"x": 1}, -10.5, 1.0),  # Negative score, still successful
            Trial({"x": 2}, 0.0, 1.0),  # Zero score, still successful
            Trial({"x": 3}, float("-inf"), 1.0),  # Failed
        ]

        result = OptimizationResult(
            best_config={"x": 2}, best_score=0.0, trials=trials, duration=3.0
        )

        assert result.successful_trials == 2
        assert result.success_rate == 2 / 3

    def test_edge_cases(self):
        """Test various edge cases."""
        # Result with single trial
        single_trial = [Trial({"x": 1}, 0.5, 1.0)]
        result = OptimizationResult({"x": 1}, 0.5, single_trial, 1.0)
        assert result.total_trials == 1
        assert result.successful_trials == 1
        assert result.success_rate == 1.0

        # Result with all failed trials
        failed_trials = [Trial({"x": i}, float("-inf"), 1.0) for i in range(5)]
        result = OptimizationResult({}, float("-inf"), failed_trials, 5.0)
        assert result.successful_trials == 0
        assert result.success_rate == 0.0

        # Result with zero duration
        result = OptimizationResult({}, 0.0, [], 0.0)
        assert result.duration == 0.0


class TestIntegration:
    """Test integration between Trial and OptimizationResult."""

    def test_building_optimization_result_from_trials(self):
        """Test building an optimization result from a series of trials."""
        trials = []
        best_score = float("-inf")
        best_config = None

        # Simulate optimization process
        configs = [
            {"learning_rate": 0.1, "batch_size": 16},
            {"learning_rate": 0.01, "batch_size": 32},
            {"learning_rate": 0.001, "batch_size": 64},
            {"learning_rate": 0.01, "batch_size": 128},
        ]

        scores = [0.75, 0.85, 0.82, 0.88]

        for i, (config, score) in enumerate(zip(configs, scores, strict=False)):
            trial = Trial(
                configuration=config,
                score=score,
                duration=10.5 + i,
                metadata={"iteration": i + 1},
            )
            trials.append(trial)

            if score > best_score:
                best_score = score
                best_config = config

        # Create result
        result = OptimizationResult(
            best_config=best_config,
            best_score=best_score,
            trials=trials,
            duration=sum(t.duration for t in trials),
            convergence_info={"final_iteration": 4, "improvement_threshold": 0.01},
        )

        assert result.best_score == 0.88
        assert result.best_config == {"learning_rate": 0.01, "batch_size": 128}
        assert result.total_trials == 4
        assert result.successful_trials == 4
        assert result.success_rate == 1.0
        assert result.duration == 48.0  # 10.5 + 11.5 + 12.5 + 13.5

    def test_handling_mixed_trial_outcomes(self):
        """Test handling optimization with mixed trial outcomes."""
        trials = []

        # Trial 1: Successful
        trials.append(Trial({"model": "A", "param": 1}, score=0.7, duration=5.0))

        # Trial 2: Failed due to error
        trials.append(
            Trial(
                {"model": "B", "param": 2},
                score=float("-inf"),
                duration=0.1,
                metadata={"failed": True, "error": "Model initialization failed"},
            )
        )

        # Trial 3: Successful with best score
        trials.append(Trial({"model": "A", "param": 3}, score=0.9, duration=6.0))

        # Trial 4: Timeout (marked as failed)
        trials.append(
            Trial(
                {"model": "C", "param": 4},
                score=0.0,
                duration=30.0,
                metadata={"failed": True, "error": "Timeout"},
            )
        )

        result = OptimizationResult(
            best_config={"model": "A", "param": 3},
            best_score=0.9,
            trials=trials,
            duration=41.1,
            convergence_info={"early_stopped": False, "total_failures": 2},
        )

        assert result.total_trials == 4
        assert result.successful_trials == 2
        assert result.success_rate == 0.5
        assert result.convergence_info["total_failures"] == 2
