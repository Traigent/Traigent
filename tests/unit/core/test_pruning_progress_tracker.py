"""Unit tests for PruningProgressTracker.

Tests for pruning progress tracking during Optuna trial optimization.
"""

# Traceability: CONC-Layer-Core CONC-Quality-Performance FUNC-OPT-ALGORITHMS FUNC-ORCH-LIFECYCLE REQ-OPT-ALG-004 REQ-ORCH-003 SYNC-OptimizationFlow

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from traigent.core.pruning_progress_tracker import PruningProgressTracker
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.utils.exceptions import TrialPrunedError


class TestPruningProgressTrackerInit:
    """Tests for PruningProgressTracker initialization."""

    def test_init_with_dataset_examples(self) -> None:
        """Test initialization extracts total_examples from dataset."""
        dataset = Dataset(
            examples=[
                EvaluationExample(input_data={"value": i}, expected_output=i)
                for i in range(5)
            ],
            name="test-dataset",
            description="Test dataset",
        )
        optimizer = MagicMock()
        optimizer.objectives = ["accuracy"]

        tracker = PruningProgressTracker(
            optimizer=optimizer,
            dataset=dataset,
            trial_id="trial-1",
            optuna_trial_id=123,
        )

        assert tracker.optimizer is optimizer
        assert tracker.dataset is dataset
        assert tracker.trial_id == "trial-1"
        assert tracker.optuna_trial_id == 123
        assert tracker.state["total_examples"] == 5
        assert tracker.state["evaluated"] == 0
        assert tracker.state["correct_sum"] == 0.0
        assert tracker.state["total_cost"] == 0.0

    def test_init_without_dataset_examples(self) -> None:
        """Test initialization with dataset without examples attribute."""
        dataset = MagicMock(spec=[])  # No examples attribute
        optimizer = MagicMock()
        optimizer.objectives = ["accuracy"]

        tracker = PruningProgressTracker(
            optimizer=optimizer,
            dataset=dataset,
            trial_id="trial-1",
            optuna_trial_id=123,
        )

        assert tracker.state["total_examples"] == 0


class TestPruningProgressTrackerCallback:
    """Tests for progress callback method."""

    @pytest.fixture
    def dataset(self) -> Dataset:
        """Create test dataset with 5 examples."""
        return Dataset(
            examples=[
                EvaluationExample(input_data={"value": i}, expected_output=i)
                for i in range(5)
            ],
            name="test-dataset",
            description="Test dataset",
        )

    @pytest.fixture
    def optimizer(self) -> MagicMock:
        """Create mock optimizer that doesn't prune."""
        optimizer = MagicMock()
        optimizer.objectives = ["accuracy"]
        optimizer.report_intermediate_value.return_value = False
        return optimizer

    def test_callback_updates_state_and_reports_value(
        self, optimizer: MagicMock, dataset: Dataset
    ) -> None:
        """Test callback updates state and reports to optimizer."""
        tracker = PruningProgressTracker(
            optimizer=optimizer,
            dataset=dataset,
            trial_id="trial-1",
            optuna_trial_id=123,
        )

        payload = {
            "success": True,
            "metrics": {"accuracy": 1.0},
            "output": 0,
        }

        tracker.callback(0, payload)

        assert tracker.state["evaluated"] == 1
        assert tracker.state["correct_sum"] == 1.0
        optimizer.report_intermediate_value.assert_called_once_with(
            123, step=0, value=1.0
        )

    def test_callback_reports_monotonic_steps(
        self, optimizer: MagicMock, dataset: Dataset
    ) -> None:
        """Test callback reports monotonic steps even with out-of-order evaluation."""
        steps: list[int] = []

        def capture_step(trial_id: int, step: int, value: float | list[float]) -> bool:
            steps.append(step)
            return False

        optimizer.report_intermediate_value = capture_step

        tracker = PruningProgressTracker(
            optimizer=optimizer,
            dataset=dataset,
            trial_id="trial-1",
            optuna_trial_id=123,
        )

        payload = {"success": True, "metrics": {"accuracy": 1.0}, "output": 0}

        # Report out of order: step 5, then step 0
        tracker.callback(5, payload)
        tracker.callback(0, payload)

        # Steps should be monotonic: 0, 1
        assert steps == [0, 1]

    def test_callback_raises_trial_pruned_error_when_should_prune(
        self, dataset: Dataset
    ) -> None:
        """Test callback raises TrialPrunedError when optimizer decides to prune."""
        optimizer = MagicMock()
        optimizer.objectives = ["accuracy"]
        optimizer.report_intermediate_value.return_value = True  # Should prune

        tracker = PruningProgressTracker(
            optimizer=optimizer,
            dataset=dataset,
            trial_id="trial-1",
            optuna_trial_id=123,
        )

        payload = {"success": True, "metrics": {"accuracy": 0.0}, "output": 99}

        with pytest.raises(TrialPrunedError) as excinfo:
            tracker.callback(0, payload)

        assert excinfo.value.step == 0

    def test_callback_skips_reporting_when_no_values(
        self, optimizer: MagicMock, dataset: Dataset
    ) -> None:
        """Test callback skips reporting when no objective values can be collected."""
        optimizer.objectives = []  # No objectives defined

        tracker = PruningProgressTracker(
            optimizer=optimizer,
            dataset=dataset,
            trial_id="trial-1",
            optuna_trial_id=123,
        )

        payload = {"success": True, "metrics": {}, "output": 0}

        # Should not raise, should not call report_intermediate_value
        tracker.callback(0, payload)

        optimizer.report_intermediate_value.assert_not_called()

    def test_callback_reports_list_value_for_multiple_objectives(
        self, dataset: Dataset
    ) -> None:
        """Test callback reports list value when multiple objectives present."""
        optimizer = MagicMock()
        optimizer.objectives = ["accuracy", "cost"]
        optimizer.report_intermediate_value.return_value = False

        tracker = PruningProgressTracker(
            optimizer=optimizer,
            dataset=dataset,
            trial_id="trial-1",
            optuna_trial_id=123,
        )

        payload = {
            "success": True,
            "metrics": {"accuracy": 0.8, "cost": 0.01},
            "output": 0,
        }

        tracker.callback(0, payload)

        # Should report list of optimistic values (accuracy is optimistic, cost is projected)
        call_args = optimizer.report_intermediate_value.call_args
        reported_value = call_args[1]["value"]
        assert isinstance(reported_value, list)
        assert len(reported_value) == 2
        # First value is optimistic accuracy (should be >= 0.8)
        assert reported_value[0] >= 0.8
        # Second value is projected cost (depends on state)
        assert isinstance(reported_value[1], float)


class TestPruningProgressTrackerUpdateState:
    """Tests for _update_state method."""

    @pytest.fixture
    def tracker(self) -> PruningProgressTracker:
        """Create tracker instance for testing."""
        dataset = Dataset(
            examples=[
                EvaluationExample(input_data={"value": i}, expected_output=i)
                for i in range(3)
            ],
            name="test-dataset",
            description="Test dataset",
        )
        optimizer = MagicMock()
        optimizer.objectives = ["accuracy"]
        return PruningProgressTracker(
            optimizer=optimizer,
            dataset=dataset,
            trial_id="trial-1",
            optuna_trial_id=123,
        )

    def test_update_state_increments_evaluated(
        self, tracker: PruningProgressTracker
    ) -> None:
        """Test _update_state increments evaluated count."""
        payload = {"success": True, "metrics": {}, "output": 0}

        tracker._update_state(0, payload)
        assert tracker.state["evaluated"] == 1

        tracker._update_state(1, payload)
        assert tracker.state["evaluated"] == 2

    def test_update_state_accumulates_accuracy_from_metrics(
        self, tracker: PruningProgressTracker
    ) -> None:
        """Test _update_state accumulates accuracy from metrics."""
        payload = {"success": True, "metrics": {"accuracy": 1.0}, "output": 0}

        tracker._update_state(0, payload)
        assert tracker.state["correct_sum"] == 1.0

        payload2 = {"success": True, "metrics": {"accuracy": 0.5}, "output": 1}
        tracker._update_state(1, payload2)
        assert tracker.state["correct_sum"] == 1.5

    def test_update_state_accumulates_cost_from_metrics(
        self, tracker: PruningProgressTracker
    ) -> None:
        """Test _update_state accumulates cost from metrics."""
        payload = {"success": True, "metrics": {"cost": 0.01}, "output": 0}

        tracker._update_state(0, payload)
        assert tracker.state["total_cost"] == 0.01

        payload2 = {"success": True, "metrics": {"cost": 0.02}, "output": 1}
        tracker._update_state(1, payload2)
        assert tracker.state["total_cost"] == 0.03

    def test_update_state_handles_none_metrics(
        self, tracker: PruningProgressTracker
    ) -> None:
        """Test _update_state handles None metrics gracefully."""
        payload = {"success": True, "metrics": None, "output": 0}

        tracker._update_state(0, payload)
        assert tracker.state["evaluated"] == 1
        # Even with None metrics, exact match can still determine accuracy
        # output=0 matches expected_output=0 for index 0
        assert tracker.state["correct_sum"] == 1.0
        assert tracker.state["total_cost"] == 0.0


class TestPruningProgressTrackerCollectObjectiveValues:
    """Tests for _collect_objective_values method."""

    @pytest.fixture
    def tracker(self) -> PruningProgressTracker:
        """Create tracker for testing objective collection."""
        dataset = Dataset(
            examples=[
                EvaluationExample(input_data={"value": i}, expected_output=i)
                for i in range(10)
            ],
            name="test-dataset",
            description="Test dataset",
        )
        optimizer = MagicMock()
        optimizer.objectives = ["accuracy", "cost"]
        return PruningProgressTracker(
            optimizer=optimizer,
            dataset=dataset,
            trial_id="trial-1",
            optuna_trial_id=123,
        )

    def test_collect_objective_values_from_metrics(
        self, tracker: PruningProgressTracker
    ) -> None:
        """Test collecting objective values from metrics payload."""
        payload = {"metrics": {"accuracy": 0.8, "cost": 0.05}}

        values, optimistic_values = tracker._collect_objective_values(payload)

        assert values == [0.8, 0.05]
        # Optimistic accuracy assumes remaining are correct
        assert len(optimistic_values) == 2
        assert optimistic_values[0] > 0.8  # Optimistic accuracy
        assert optimistic_values[1] == 0.05  # Cost stays same

    def test_collect_objective_values_projects_cost(
        self, tracker: PruningProgressTracker
    ) -> None:
        """Test cost projection for stronger pruning signal."""
        # Simulate 2 examples evaluated with total cost 0.02
        tracker.state["evaluated"] = 2
        tracker.state["total_cost"] = 0.02

        payload = {"metrics": {"accuracy": 1.0, "cost": 0.01}}

        values, _ = tracker._collect_objective_values(payload)

        # Cost should be projected: (0.02 / 2) * 10 = 0.1
        assert values[0] == 1.0  # Accuracy
        assert values[1] == 0.1  # Projected cost

    def test_collect_objective_values_with_total_cost_objective(
        self, tracker: PruningProgressTracker
    ) -> None:
        """Test cost projection works with total_cost objective name."""
        tracker.optimizer.objectives = ["accuracy", "total_cost"]
        tracker.state["evaluated"] = 1
        tracker.state["total_cost"] = 0.01

        payload = {"metrics": {"accuracy": 1.0, "total_cost": 0.01}}

        values, _ = tracker._collect_objective_values(payload)

        # Should project: (0.01 / 1) * 10 = 0.1
        assert values[1] == 0.1

    def test_collect_objective_values_uses_fallback(
        self, tracker: PruningProgressTracker
    ) -> None:
        """Test fallback to state-based calculation when metric missing."""
        tracker.state["evaluated"] = 2
        tracker.state["correct_sum"] = 1.5
        tracker.state["total_cost"] = 0.02

        payload = {"metrics": {"cost": 0.01}}  # No accuracy in metrics

        values, _ = tracker._collect_objective_values(payload)

        # Should use fallback: 1.5 / 2 = 0.75
        assert values[0] == 0.75
        # Cost is projected: (0.02 / 2) * 10 = 0.1
        assert values[1] == 0.1

    def test_collect_objective_values_skips_non_numeric(
        self, tracker: PruningProgressTracker
    ) -> None:
        """Test non-numeric values are skipped."""
        payload = {"metrics": {"accuracy": "invalid", "cost": 0.01}}

        values, _ = tracker._collect_objective_values(payload)

        # Should only include cost
        assert len(values) == 1
        assert values[0] == 0.01

    def test_collect_objective_values_empty_when_no_objectives(
        self, tracker: PruningProgressTracker
    ) -> None:
        """Test empty values when no objectives defined."""
        tracker.optimizer.objectives = []

        payload = {"metrics": {"accuracy": 1.0}}

        values, optimistic_values = tracker._collect_objective_values(payload)

        assert values == []
        assert optimistic_values == []


class TestPruningProgressTrackerExtractAccuracy:
    """Tests for _extract_accuracy method."""

    @pytest.fixture
    def tracker(self) -> PruningProgressTracker:
        """Create tracker for accuracy extraction tests."""
        dataset = Dataset(
            examples=[
                EvaluationExample(input_data={"value": i}, expected_output=i)
                for i in range(3)
            ],
            name="test-dataset",
            description="Test dataset",
        )
        optimizer = MagicMock()
        optimizer.objectives = ["accuracy"]
        return PruningProgressTracker(
            optimizer=optimizer,
            dataset=dataset,
            trial_id="trial-1",
            optuna_trial_id=123,
        )

    def test_extract_accuracy_from_metrics(
        self, tracker: PruningProgressTracker
    ) -> None:
        """Test extracting accuracy from metrics payload."""
        payload = {"output": 0}
        metrics_payload = {"accuracy": 1.0}

        accuracy = tracker._extract_accuracy(0, payload, metrics_payload)

        assert accuracy == 1.0

    def test_extract_accuracy_via_exact_match(
        self, tracker: PruningProgressTracker
    ) -> None:
        """Test accuracy via exact match when metric not provided."""
        payload = {"output": 1}
        metrics_payload = {}

        accuracy = tracker._extract_accuracy(1, payload, metrics_payload)

        assert accuracy == 1.0  # output == expected_output

    def test_extract_accuracy_via_exact_match_incorrect(
        self, tracker: PruningProgressTracker
    ) -> None:
        """Test accuracy via exact match returns 0.0 for mismatch."""
        payload = {"output": 99}
        metrics_payload = {}

        accuracy = tracker._extract_accuracy(0, payload, metrics_payload)

        assert accuracy == 0.0  # output != expected_output

    def test_extract_accuracy_returns_none_when_no_expected_output(
        self, tracker: PruningProgressTracker
    ) -> None:
        """Test returns None when cannot determine accuracy."""
        # Use invalid index so expected_output is None
        payload = {"output": 0}
        metrics_payload = {}

        accuracy = tracker._extract_accuracy(999, payload, metrics_payload)

        assert accuracy is None


class TestPruningProgressTrackerExtractCost:
    """Tests for _extract_cost method."""

    @pytest.fixture
    def tracker(self) -> PruningProgressTracker:
        """Create tracker for cost extraction tests."""
        dataset = Dataset(
            examples=[
                EvaluationExample(input_data={"value": i}, expected_output=i)
                for i in range(3)
            ],
            name="test-dataset",
            description="Test dataset",
        )
        optimizer = MagicMock()
        optimizer.objectives = ["cost"]
        return PruningProgressTracker(
            optimizer=optimizer,
            dataset=dataset,
            trial_id="trial-1",
            optuna_trial_id=123,
        )

    def test_extract_cost_from_total_cost_metric(
        self, tracker: PruningProgressTracker
    ) -> None:
        """Test extracting cost from total_cost metric."""
        payload = {}
        metrics_payload = {"total_cost": 0.05}

        cost = tracker._extract_cost(payload, metrics_payload)

        assert cost == 0.05

    def test_extract_cost_from_cost_metric(
        self, tracker: PruningProgressTracker
    ) -> None:
        """Test extracting cost from cost metric."""
        payload = {}
        metrics_payload = {"cost": 0.03}

        cost = tracker._extract_cost(payload, metrics_payload)

        assert cost == 0.03

    def test_extract_cost_from_example_cost_metric(
        self, tracker: PruningProgressTracker
    ) -> None:
        """Test extracting cost from example_cost metric."""
        payload = {}
        metrics_payload = {"example_cost": 0.02}

        cost = tracker._extract_cost(payload, metrics_payload)

        assert cost == 0.02

    def test_extract_cost_from_payload_total_cost(
        self, tracker: PruningProgressTracker
    ) -> None:
        """Test extracting cost from payload total_cost."""
        payload = {"total_cost": 0.04}
        metrics_payload = {}

        cost = tracker._extract_cost(payload, metrics_payload)

        assert cost == 0.04

    def test_extract_cost_from_payload_cost(
        self, tracker: PruningProgressTracker
    ) -> None:
        """Test extracting cost from payload cost."""
        payload = {"cost": 0.01}
        metrics_payload = {}

        cost = tracker._extract_cost(payload, metrics_payload)

        assert cost == 0.01

    def test_extract_cost_returns_none_when_missing(
        self, tracker: PruningProgressTracker
    ) -> None:
        """Test returns None when cost not in payload or metrics."""
        payload = {}
        metrics_payload = {}

        cost = tracker._extract_cost(payload, metrics_payload)

        assert cost is None

    def test_extract_cost_returns_none_for_invalid_value(
        self, tracker: PruningProgressTracker
    ) -> None:
        """Test returns None for non-numeric cost value."""
        payload = {}
        metrics_payload = {"cost": "invalid"}

        cost = tracker._extract_cost(payload, metrics_payload)

        assert cost is None


class TestPruningProgressTrackerFallbackObjectiveValue:
    """Tests for _fallback_objective_value method."""

    @pytest.fixture
    def tracker(self) -> PruningProgressTracker:
        """Create tracker for fallback tests."""
        dataset = Dataset(
            examples=[
                EvaluationExample(input_data={"value": i}, expected_output=i)
                for i in range(5)
            ],
            name="test-dataset",
            description="Test dataset",
        )
        optimizer = MagicMock()
        optimizer.objectives = ["accuracy"]
        return PruningProgressTracker(
            optimizer=optimizer,
            dataset=dataset,
            trial_id="trial-1",
            optuna_trial_id=123,
        )

    def test_fallback_accuracy_calculation(
        self, tracker: PruningProgressTracker
    ) -> None:
        """Test fallback accuracy calculation from state."""
        tracker.state["evaluated"] = 4
        tracker.state["correct_sum"] = 3.0

        accuracy = tracker._fallback_objective_value("accuracy")

        assert accuracy == 0.75  # 3.0 / 4

    def test_fallback_success_rate_calculation(
        self, tracker: PruningProgressTracker
    ) -> None:
        """Test fallback success_rate calculation from state."""
        tracker.state["evaluated"] = 5
        tracker.state["correct_sum"] = 4.0

        success_rate = tracker._fallback_objective_value("success_rate")

        assert success_rate == 0.8  # 4.0 / 5

    def test_fallback_accuracy_returns_none_when_no_evaluated(
        self, tracker: PruningProgressTracker
    ) -> None:
        """Test fallback returns None when no examples evaluated."""
        tracker.state["evaluated"] = 0

        accuracy = tracker._fallback_objective_value("accuracy")

        assert accuracy is None

    def test_fallback_cost_from_state(self, tracker: PruningProgressTracker) -> None:
        """Test fallback cost returns total_cost from state."""
        tracker.state["total_cost"] = 0.15

        cost = tracker._fallback_objective_value("cost")

        assert cost == 0.15

    def test_fallback_latency_from_state(self, tracker: PruningProgressTracker) -> None:
        """Test fallback latency returns total_cost from state."""
        tracker.state["total_cost"] = 1.5

        latency = tracker._fallback_objective_value("latency")

        assert latency == 1.5

    def test_fallback_error_from_state(self, tracker: PruningProgressTracker) -> None:
        """Test fallback error returns total_cost from state."""
        tracker.state["total_cost"] = 0.05

        error = tracker._fallback_objective_value("error")

        assert error == 0.05

    def test_fallback_unknown_objective_returns_none(
        self, tracker: PruningProgressTracker
    ) -> None:
        """Test fallback returns None for unknown objective."""
        result = tracker._fallback_objective_value("unknown_objective")

        assert result is None


class TestPruningProgressTrackerOptimisticAccuracy:
    """Tests for _optimistic_accuracy method."""

    @pytest.fixture
    def tracker(self) -> PruningProgressTracker:
        """Create tracker for optimistic accuracy tests."""
        dataset = Dataset(
            examples=[
                EvaluationExample(input_data={"value": i}, expected_output=i)
                for i in range(10)
            ],
            name="test-dataset",
            description="Test dataset",
        )
        optimizer = MagicMock()
        optimizer.objectives = ["accuracy"]
        return PruningProgressTracker(
            optimizer=optimizer,
            dataset=dataset,
            trial_id="trial-1",
            optuna_trial_id=123,
        )

    def test_optimistic_accuracy_with_partial_evaluation(
        self, tracker: PruningProgressTracker
    ) -> None:
        """Test optimistic accuracy assumes remaining examples are correct."""
        tracker.state["evaluated"] = 3
        tracker.state["correct_sum"] = 2.0  # 2 out of 3 correct

        optimistic = tracker._optimistic_accuracy()

        # (2 + 7) / 10 = 0.9 (assumes remaining 7 are correct)
        assert optimistic == 0.9

    def test_optimistic_accuracy_with_all_correct(
        self, tracker: PruningProgressTracker
    ) -> None:
        """Test optimistic accuracy when all evaluated are correct."""
        tracker.state["evaluated"] = 5
        tracker.state["correct_sum"] = 5.0

        optimistic = tracker._optimistic_accuracy()

        # (5 + 5) / 10 = 1.0
        assert optimistic == 1.0

    def test_optimistic_accuracy_with_no_evaluation(
        self, tracker: PruningProgressTracker
    ) -> None:
        """Test optimistic accuracy when nothing evaluated yet."""
        tracker.state["evaluated"] = 0
        tracker.state["correct_sum"] = 0.0

        optimistic = tracker._optimistic_accuracy()

        # (0 + 10) / 10 = 1.0 (all remaining assumed correct)
        assert optimistic == 1.0

    def test_optimistic_accuracy_with_all_evaluated(
        self, tracker: PruningProgressTracker
    ) -> None:
        """Test optimistic accuracy when all examples evaluated."""
        tracker.state["evaluated"] = 10
        tracker.state["correct_sum"] = 7.0

        optimistic = tracker._optimistic_accuracy()

        # (7 + 0) / 10 = 0.7 (no remaining)
        assert optimistic == 0.7

    def test_optimistic_accuracy_with_zero_total_examples(
        self, tracker: PruningProgressTracker
    ) -> None:
        """Test optimistic accuracy handles zero total examples."""
        tracker.state["total_examples"] = 0
        tracker.state["evaluated"] = 0
        tracker.state["correct_sum"] = 0.0

        optimistic = tracker._optimistic_accuracy()

        # Uses max(total_examples, 1) = 1 to avoid division by zero
        # (0 + max(0 - 0, 0)) / max(0, 1) = 0 / 1 = 0.0
        # But with remaining = max(0 - 0, 0) = 0
        # So (0.0 + 0) / 1 = 0.0
        # Actually: remaining = max(1 - 0, 0) = 1, so (0 + 1) / 1 = 1.0
        assert optimistic == 1.0


class TestPruningProgressTrackerGetExpectedOutput:
    """Tests for _get_expected_output method."""

    @pytest.fixture
    def tracker(self) -> PruningProgressTracker:
        """Create tracker with dataset for expected output tests."""
        dataset = Dataset(
            examples=[
                EvaluationExample(input_data={"value": 0}, expected_output="zero"),
                EvaluationExample(input_data={"value": 1}, expected_output="one"),
                EvaluationExample(input_data={"value": 2}, expected_output="two"),
            ],
            name="test-dataset",
            description="Test dataset",
        )
        optimizer = MagicMock()
        optimizer.objectives = ["accuracy"]
        return PruningProgressTracker(
            optimizer=optimizer,
            dataset=dataset,
            trial_id="trial-1",
            optuna_trial_id=123,
        )

    def test_get_expected_output_valid_index(
        self, tracker: PruningProgressTracker
    ) -> None:
        """Test getting expected output for valid index."""
        expected = tracker._get_expected_output(0)
        assert expected == "zero"

        expected = tracker._get_expected_output(1)
        assert expected == "one"

        expected = tracker._get_expected_output(2)
        assert expected == "two"

    def test_get_expected_output_invalid_index_negative(
        self, tracker: PruningProgressTracker
    ) -> None:
        """Test getting expected output for negative index."""
        expected = tracker._get_expected_output(-1)
        assert expected is None

    def test_get_expected_output_invalid_index_too_large(
        self, tracker: PruningProgressTracker
    ) -> None:
        """Test getting expected output for index beyond dataset size."""
        expected = tracker._get_expected_output(999)
        assert expected is None

    def test_get_expected_output_dataset_without_examples(self) -> None:
        """Test getting expected output when dataset has no examples attribute."""
        dataset = MagicMock(spec=[])  # No examples attribute
        optimizer = MagicMock()
        optimizer.objectives = ["accuracy"]

        tracker = PruningProgressTracker(
            optimizer=optimizer,
            dataset=dataset,
            trial_id="trial-1",
            optuna_trial_id=123,
        )

        expected = tracker._get_expected_output(0)
        assert expected is None

    def test_get_expected_output_handles_exception(
        self, tracker: PruningProgressTracker
    ) -> None:
        """Test getting expected output handles exceptions gracefully."""
        # Mock dataset to raise exception when accessing examples
        tracker.dataset.examples = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("Test error"))
        )

        expected = tracker._get_expected_output(0)
        assert expected is None


class TestPruningProgressTrackerLogProgress:
    """Tests for _log_progress method."""

    @pytest.fixture
    def tracker(self) -> PruningProgressTracker:
        """Create tracker for logging tests."""
        dataset = Dataset(
            examples=[
                EvaluationExample(input_data={"value": i}, expected_output=i)
                for i in range(5)
            ],
            name="test-dataset",
            description="Test dataset",
        )
        optimizer = MagicMock()
        optimizer.objectives = ["accuracy"]
        return PruningProgressTracker(
            optimizer=optimizer,
            dataset=dataset,
            trial_id="trial-1",
            optuna_trial_id=123,
        )

    @patch("traigent.core.pruning_progress_tracker.logger")
    def test_log_progress_logs_single_value(
        self, mock_logger: MagicMock, tracker: PruningProgressTracker
    ) -> None:
        """Test logging progress with single value."""
        tracker.state["evaluated"] = 3

        tracker._log_progress(2, 0.75)

        mock_logger.info.assert_called_once()
        # Check format string and arguments separately
        args = mock_logger.info.call_args[0]
        # args[0] is the format string with %s placeholders
        assert "%s" in args[0]  # Contains format placeholders
        assert "Trial" in args[0]
        assert "Step" in args[0]
        # Check the actual values passed
        assert args[1] == 123  # optuna_trial_id
        assert args[2] == 2  # step_index
        assert args[3] == 3  # evaluated
        assert args[4] == 5  # total_examples
        assert args[5] == 0.75  # value

    @patch("traigent.core.pruning_progress_tracker.logger")
    def test_log_progress_logs_list_value(
        self, mock_logger: MagicMock, tracker: PruningProgressTracker
    ) -> None:
        """Test logging progress with list of values."""
        tracker.state["evaluated"] = 2

        tracker._log_progress(1, [0.8, 0.05])

        mock_logger.info.assert_called_once()
        args = mock_logger.info.call_args[0]
        assert args[5] == [0.8, 0.05]
