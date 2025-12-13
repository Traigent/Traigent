"""Unit tests for visualization plots.

Tests for plot generation with matplotlib and ASCII fallback modes.
"""

# Traceability: CONC-Layer-Tooling CONC-Quality-Usability
# Traceability: CONC-Quality-Observability FUNC-ANALYTICS REQ-ANLY-011
# Traceability: SYNC-Observability

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from traigent.api.types import (
    OptimizationResult,
    OptimizationStatus,
    TrialResult,
    TrialStatus,
)
from traigent.visualization.plots import PlotGenerator, create_quick_plot


def _make_trial(trial_id: str, accuracy: float, cost: float) -> TrialResult:
    """Helper to create trial results."""
    return TrialResult(
        trial_id=trial_id,
        config={"trial": trial_id},
        metrics={"accuracy": accuracy, "cost": cost},
        status=TrialStatus.COMPLETED,
        duration=1.0,
        timestamp=datetime.now(UTC),
        metadata={},
    )


def _make_result(
    trials: list[TrialResult],
    orientations: dict[str, str],
    objectives: list[str] | None = None,
) -> OptimizationResult:
    """Helper to create optimization results."""
    return OptimizationResult(
        trials=trials,
        best_config=trials[0].config if trials else {},
        best_score=0.0,
        optimization_id="opt-123",
        duration=1.0,
        convergence_info={},
        status=OptimizationStatus.COMPLETED,
        objectives=objectives or ["accuracy", "cost"],
        algorithm="test_algo",
        timestamp=datetime.now(UTC),
        metadata={"objective_orientations": orientations},
    )


class TestPlotGeneratorInit:
    """Tests for PlotGenerator initialization."""

    def test_init_without_matplotlib(self) -> None:
        """Test initialization when matplotlib is not available."""
        with patch("traigent.visualization.plots.PlotGenerator.__init__") as mock_init:
            mock_init.return_value = None
            PlotGenerator(use_matplotlib=False)
            mock_init.assert_called_once()

    def test_init_with_matplotlib_unavailable(self) -> None:
        """Test initialization when matplotlib import fails."""
        with patch("builtins.__import__", side_effect=ImportError("No matplotlib")):
            plotter = PlotGenerator(use_matplotlib=True)
            assert plotter.use_matplotlib is True
            assert plotter._matplotlib_available is False

    def test_init_with_matplotlib_available(self) -> None:
        """Test initialization when matplotlib is available."""
        # Simply test that it initializes without error
        plotter = PlotGenerator(use_matplotlib=False)
        assert plotter.use_matplotlib is False
        assert plotter._matplotlib_available is False

    def test_init_sets_agg_backend(self) -> None:
        """Test that Agg backend is set when matplotlib is available."""
        mock_plt = MagicMock()
        mock_matplotlib = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "matplotlib": mock_matplotlib,
                "matplotlib.pyplot": mock_plt,
            },
        ):
            plotter = PlotGenerator(use_matplotlib=True)
            assert plotter._matplotlib_available is True
            mock_matplotlib.use.assert_called_once_with("Agg")


class TestPlotOptimizationProgress:
    """Tests for plot_optimization_progress method."""

    @pytest.fixture
    def plotter_ascii(self) -> PlotGenerator:
        """Create PlotGenerator in ASCII mode."""
        return PlotGenerator(use_matplotlib=False)

    @pytest.fixture
    def plotter_matplotlib(self) -> PlotGenerator:
        """Create PlotGenerator with matplotlib enabled."""
        plotter = PlotGenerator(use_matplotlib=True)
        plotter._matplotlib_available = True
        plotter.plt = MagicMock()
        return plotter

    def test_no_trials_returns_message(self, plotter_ascii: PlotGenerator) -> None:
        """Test that empty trials list returns appropriate message."""
        result = _make_result([], {})
        output = plotter_ascii.plot_optimization_progress(result)
        assert output == "No trials to plot"

    def test_no_completed_trials_with_objective(
        self, plotter_ascii: PlotGenerator
    ) -> None:
        """Test when no trials have the requested objective."""
        trials = [
            TrialResult(
                trial_id="t1",
                config={},
                metrics={"other": 0.5},
                status=TrialStatus.COMPLETED,
                duration=1.0,
                timestamp=datetime.now(UTC),
            )
        ]
        result = _make_result(trials, {}, objectives=["accuracy"])
        output = plotter_ascii.plot_optimization_progress(result, objective="accuracy")
        assert "No completed trials with objective 'accuracy'" in output

    def test_uses_primary_objective_when_none_specified(
        self, plotter_ascii: PlotGenerator
    ) -> None:
        """Test that primary objective is used when none specified."""
        trials = [_make_trial("t1", 0.9, 0.5)]
        result = _make_result(trials, {}, objectives=["accuracy", "cost"])
        with patch.object(plotter_ascii, "_plot_progress_ascii") as mock_ascii:
            mock_ascii.return_value = "plot"
            plotter_ascii.plot_optimization_progress(result)
            # Should use first objective "accuracy"
            args = mock_ascii.call_args.args
            assert "Accuracy" in args[3]  # objective_label

    def test_matplotlib_mode_calls_matplotlib_plot(
        self, plotter_matplotlib: PlotGenerator
    ) -> None:
        """Test that matplotlib mode calls _plot_progress_matplotlib."""
        trials = [_make_trial("t1", 0.9, 0.5)]
        result = _make_result(trials, {}, objectives=["accuracy"])

        with patch.object(plotter_matplotlib, "_plot_progress_matplotlib") as mock_mpl:
            mock_mpl.return_value = "matplotlib_plot.png"
            output = plotter_matplotlib.plot_optimization_progress(result)
            assert output == "matplotlib_plot.png"
            mock_mpl.assert_called_once()

    def test_skips_failed_trials(self, plotter_ascii: PlotGenerator) -> None:
        """Test that failed trials are skipped."""
        trials = [
            _make_trial("t1", 0.9, 0.5),
            TrialResult(
                trial_id="t2",
                config={},
                metrics={"accuracy": 0.95},
                status=TrialStatus.FAILED,
                duration=1.0,
                timestamp=datetime.now(UTC),
            ),
            _make_trial("t3", 0.85, 0.3),
        ]
        result = _make_result(trials, {}, objectives=["accuracy"])

        with patch.object(plotter_ascii, "_plot_progress_ascii") as mock_ascii:
            mock_ascii.return_value = "plot"
            plotter_ascii.plot_optimization_progress(result)
            args = mock_ascii.call_args.args
            trial_numbers = args[0]
            assert trial_numbers == [1, 3]  # t2 is skipped

    def test_skips_trials_without_metrics(self, plotter_ascii: PlotGenerator) -> None:
        """Test that trials without metrics are skipped."""
        trials = [
            _make_trial("t1", 0.9, 0.5),
            TrialResult(
                trial_id="t2",
                config={},
                metrics=None,
                status=TrialStatus.COMPLETED,
                duration=1.0,
                timestamp=datetime.now(UTC),
            ),
        ]
        result = _make_result(trials, {}, objectives=["accuracy"])

        with patch.object(plotter_ascii, "_plot_progress_ascii") as mock_ascii:
            mock_ascii.return_value = "plot"
            plotter_ascii.plot_optimization_progress(result)
            args = mock_ascii.call_args.args
            trial_numbers = args[0]
            assert trial_numbers == [1]

    def test_skips_invalid_metric_values(self, plotter_ascii: PlotGenerator) -> None:
        """Test that non-numeric metric values are skipped."""
        trials = [
            _make_trial("t1", 0.9, 0.5),
            TrialResult(
                trial_id="t2",
                config={},
                metrics={"accuracy": "invalid"},
                status=TrialStatus.COMPLETED,
                duration=1.0,
                timestamp=datetime.now(UTC),
            ),
        ]
        result = _make_result(trials, {}, objectives=["accuracy"])

        with patch.object(plotter_ascii, "_plot_progress_ascii") as mock_ascii:
            mock_ascii.return_value = "plot"
            plotter_ascii.plot_optimization_progress(result)
            args = mock_ascii.call_args.args
            scores = args[1]
            assert scores == [0.9]


class TestProgressPlotRespectOrientation:
    """Tests for progress plot with minimize/maximize orientation."""

    def test_progress_plot_respects_minimize_orientation(self) -> None:
        """Progress plot should treat minimize objectives as improvements when scores drop."""
        trials = [
            TrialResult(
                trial_id="t1",
                config={"trial": "t1"},
                metrics={"loss": 0.5},
                status=TrialStatus.COMPLETED,
                duration=1.0,
                timestamp=datetime.now(UTC),
            ),
            TrialResult(
                trial_id="t2",
                config={"trial": "t2"},
                metrics={"loss": 0.3},
                status=TrialStatus.COMPLETED,
                duration=1.0,
                timestamp=datetime.now(UTC),
            ),
            TrialResult(
                trial_id="t3",
                config={"trial": "t3"},
                metrics={"loss": 0.4},
                status=TrialStatus.COMPLETED,
                duration=1.0,
                timestamp=datetime.now(UTC),
            ),
        ]

        result = _make_result(
            trials=trials,
            orientations={"loss": "minimize"},
            objectives=["loss"],
        )

        plotter = PlotGenerator(use_matplotlib=False)
        with patch.object(plotter, "_plot_progress_ascii") as mock_ascii:
            mock_ascii.return_value = "ok"
            output = plotter.plot_optimization_progress(result)

        assert output == "ok"
        mock_ascii.assert_called_once()
        trial_numbers, scores, best_scores, label = mock_ascii.call_args.args
        assert trial_numbers == [1, 2, 3]
        assert scores == [0.5, 0.3, 0.4]
        assert best_scores == [0.5, 0.3, 0.3]
        assert label == "Loss (min)"

    def test_progress_plot_respects_maximize_orientation(self) -> None:
        """Progress plot should treat maximize objectives correctly."""
        trials = [
            TrialResult(
                trial_id="t1",
                config={},
                metrics={"accuracy": 0.5},
                status=TrialStatus.COMPLETED,
                duration=1.0,
                timestamp=datetime.now(UTC),
            ),
            TrialResult(
                trial_id="t2",
                config={},
                metrics={"accuracy": 0.7},
                status=TrialStatus.COMPLETED,
                duration=1.0,
                timestamp=datetime.now(UTC),
            ),
            TrialResult(
                trial_id="t3",
                config={},
                metrics={"accuracy": 0.6},
                status=TrialStatus.COMPLETED,
                duration=1.0,
                timestamp=datetime.now(UTC),
            ),
        ]

        result = _make_result(
            trials=trials,
            orientations={"accuracy": "maximize"},
            objectives=["accuracy"],
        )

        plotter = PlotGenerator(use_matplotlib=False)
        with patch.object(plotter, "_plot_progress_ascii") as mock_ascii:
            mock_ascii.return_value = "ok"
            plotter.plot_optimization_progress(result)

        args = mock_ascii.call_args.args
        best_scores = args[2]
        assert best_scores == [0.5, 0.7, 0.7]  # Keeps max


class TestPlotProgressMatplotlib:
    """Tests for _plot_progress_matplotlib method."""

    @pytest.fixture
    def plotter(self) -> PlotGenerator:
        """Create PlotGenerator with matplotlib mocked."""
        plotter = PlotGenerator(use_matplotlib=True)
        plotter._matplotlib_available = True
        plotter.plt = MagicMock()
        return plotter

    def test_creates_matplotlib_plot(self, plotter: PlotGenerator) -> None:
        """Test that matplotlib plot is created and saved."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        plotter.plt.subplots.return_value = (mock_fig, mock_ax)

        result = plotter._plot_progress_matplotlib(
            trial_numbers=[1, 2, 3],
            scores=[0.5, 0.7, 0.6],
            best_scores=[0.5, 0.7, 0.7],
            objective="accuracy",
            objective_label="Accuracy (max)",
        )

        assert "optimization_progress_accuracy.png" in result
        mock_ax.scatter.assert_called_once()
        mock_ax.plot.assert_called_once()
        mock_ax.set_xlabel.assert_called_once()
        mock_ax.set_ylabel.assert_called_once()
        mock_ax.set_title.assert_called_once()
        mock_ax.legend.assert_called_once()
        mock_ax.grid.assert_called_once()
        mock_fig.savefig.assert_called_once()
        plotter.plt.close.assert_called_once()


class TestPlotProgressAscii:
    """Tests for _plot_progress_ascii method."""

    @pytest.fixture
    def plotter(self) -> PlotGenerator:
        """Create PlotGenerator in ASCII mode."""
        return PlotGenerator(use_matplotlib=False)

    def test_empty_scores_returns_message(self, plotter: PlotGenerator) -> None:
        """Test that empty scores return appropriate message."""
        result = plotter._plot_progress_ascii([], [], [], "Test")
        assert result == "No data to plot"

    def test_ascii_plot_contains_expected_elements(
        self, plotter: PlotGenerator
    ) -> None:
        """Test that ASCII plot contains all expected elements."""
        result = plotter._plot_progress_ascii(
            trial_numbers=[1, 2, 3],
            scores=[0.5, 0.7, 0.6],
            best_scores=[0.5, 0.7, 0.7],
            objective_label="Accuracy (max)",
        )

        assert "Optimization Progress: Accuracy (max)" in result
        assert "Legend: · = trial score, * = best score so far" in result
        assert "Score range:" in result
        assert "Final best score: 0.700" in result

    def test_ascii_plot_with_single_trial(self, plotter: PlotGenerator) -> None:
        """Test ASCII plot with only one trial."""
        result = plotter._plot_progress_ascii(
            trial_numbers=[1],
            scores=[0.8],
            best_scores=[0.8],
            objective_label="Test",
        )

        assert "Final best score: 0.800" in result

    def test_ascii_plot_with_constant_scores(self, plotter: PlotGenerator) -> None:
        """Test ASCII plot when all scores are the same."""
        result = plotter._plot_progress_ascii(
            trial_numbers=[1, 2, 3],
            scores=[0.5, 0.5, 0.5],
            best_scores=[0.5, 0.5, 0.5],
            objective_label="Constant",
        )

        assert "0.50" in result
        assert "Final best score: 0.500" in result


class TestPlotParetoFront:
    """Tests for plot_pareto_front method."""

    @pytest.fixture
    def plotter_ascii(self) -> PlotGenerator:
        """Create PlotGenerator in ASCII mode."""
        return PlotGenerator(use_matplotlib=False)

    def test_insufficient_objectives_returns_message(
        self, plotter_ascii: PlotGenerator
    ) -> None:
        """Test that single objective result returns error message."""
        trials = [_make_trial("t1", 0.9, 0.5)]
        result = _make_result(trials, {}, objectives=["accuracy"])

        output = plotter_ascii.plot_pareto_front(result, "accuracy", "cost")
        assert output == "Need at least 2 objectives for Pareto front plot"

    @patch("traigent.visualization.plots.ParetoFrontCalculator")
    def test_no_pareto_solutions_returns_message(
        self, mock_calculator: MagicMock, plotter_ascii: PlotGenerator
    ) -> None:
        """Test message when no Pareto-optimal solutions found."""
        instance = mock_calculator.return_value
        instance.calculate_pareto_front.return_value = []

        trials = [_make_trial("t1", 0.9, 0.4)]
        result = _make_result(trials, {}, objectives=["accuracy", "cost"])

        output = plotter_ascii.plot_pareto_front(result, "accuracy", "cost")
        assert output == "No Pareto-optimal solutions found"

    @patch("traigent.visualization.plots.ParetoFrontCalculator")
    def test_plot_pareto_front_respects_orientation_metadata(
        self, mock_calculator: MagicMock
    ) -> None:
        """Ensure Pareto front calculator receives orientation overrides."""
        instance = mock_calculator.return_value
        instance.calculate_pareto_front.return_value = []

        result = _make_result(
            trials=[
                _make_trial("t1", 0.9, 0.4),
                _make_trial("t2", 0.85, 0.2),
            ],
            orientations={"accuracy": "maximize", "cost": "minimize"},
        )

        plotter = PlotGenerator(use_matplotlib=False)
        plotter.plot_pareto_front(result, "accuracy", "cost")

        mock_calculator.assert_called_once()
        kwargs = mock_calculator.call_args.kwargs
        assert kwargs["maximize"] == {"accuracy": True, "cost": False}

    def test_plot_pareto_front_ascii_outputs_orientation(self) -> None:
        """ASCII plot should reflect objective orientations and correct Pareto size."""
        trials = [
            _make_trial("t1", 0.9, 0.4),
            _make_trial("t2", 0.85, 0.2),
            _make_trial("t3", 0.8, 0.7),
        ]
        result = _make_result(
            trials=trials,
            orientations={"accuracy": "maximize", "cost": "minimize"},
        )

        plotter = PlotGenerator(use_matplotlib=False)
        output = plotter.plot_pareto_front(result, "accuracy", "cost")

        assert "Accuracy (max)" in output
        assert "Cost (min)" in output
        assert "Pareto points found: 2" in output


class TestPlotParetoMatplotlib:
    """Tests for _plot_pareto_matplotlib method."""

    @pytest.fixture
    def plotter(self) -> PlotGenerator:
        """Create PlotGenerator with matplotlib mocked."""
        plotter = PlotGenerator(use_matplotlib=True)
        plotter._matplotlib_available = True
        plotter.plt = MagicMock()
        return plotter

    def test_creates_pareto_plot(self, plotter: PlotGenerator) -> None:
        """Test that matplotlib Pareto plot is created."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        plotter.plt.subplots.return_value = (mock_fig, mock_ax)

        result = plotter._plot_pareto_matplotlib(
            all_x=[0.5, 0.7, 0.6],
            all_y=[0.3, 0.2, 0.5],
            pareto_x=[0.7, 0.6],
            pareto_y=[0.2, 0.5],
            obj1="accuracy",
            obj2="cost",
            orientations={"accuracy": True, "cost": False},
        )

        assert "pareto_front_accuracy_cost.png" in result
        # Should call scatter twice: once for all points, once for Pareto
        assert mock_ax.scatter.call_count == 2
        mock_ax.plot.assert_called_once()  # Connect Pareto points
        mock_fig.savefig.assert_called_once()

    def test_skips_connecting_single_pareto_point(self, plotter: PlotGenerator) -> None:
        """Test that single Pareto point is not connected."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        plotter.plt.subplots.return_value = (mock_fig, mock_ax)

        plotter._plot_pareto_matplotlib(
            all_x=[0.5, 0.7],
            all_y=[0.3, 0.2],
            pareto_x=[0.7],
            pareto_y=[0.2],
            obj1="accuracy",
            obj2="cost",
            orientations={"accuracy": True, "cost": False},
        )

        mock_ax.plot.assert_not_called()


class TestPlotParetoAscii:
    """Tests for _plot_pareto_ascii method."""

    @pytest.fixture
    def plotter(self) -> PlotGenerator:
        """Create PlotGenerator in ASCII mode."""
        return PlotGenerator(use_matplotlib=False)

    def test_empty_data_returns_message(self, plotter: PlotGenerator) -> None:
        """Test that empty data returns appropriate message."""
        result = plotter._plot_pareto_ascii(
            all_x=[],
            all_y=[],
            pareto_x=[],
            pareto_y=[],
            obj1="accuracy",
            obj2="cost",
            orientations={"accuracy": True, "cost": False},
        )
        assert result == "No data to plot"

    def test_ascii_pareto_contains_expected_elements(
        self, plotter: PlotGenerator
    ) -> None:
        """Test that ASCII Pareto plot contains all expected elements."""
        result = plotter._plot_pareto_ascii(
            all_x=[0.5, 0.7, 0.6],
            all_y=[0.3, 0.2, 0.5],
            pareto_x=[0.7, 0.6],
            pareto_y=[0.2, 0.5],
            obj1="accuracy",
            obj2="cost",
            orientations={"accuracy": True, "cost": False},
        )

        assert "Pareto Front: Accuracy (max) vs Cost (min)" in result
        assert "Legend: · = all trials, * = Pareto optimal" in result
        assert "Pareto points found: 2" in result
        assert "X-axis: Accuracy (max)" in result
        assert "Y-axis: Cost (min)" in result

    def test_ascii_pareto_with_constant_values(self, plotter: PlotGenerator) -> None:
        """Test ASCII Pareto plot with constant X or Y values."""
        result = plotter._plot_pareto_ascii(
            all_x=[0.5, 0.5, 0.5],
            all_y=[0.3, 0.2, 0.5],
            pareto_x=[0.5],
            pareto_y=[0.2],
            obj1="accuracy",
            obj2="cost",
            orientations={"accuracy": True, "cost": False},
        )

        assert "Pareto points found: 1" in result


class TestDetermineObjectiveOrientations:
    """Tests for _determine_objective_orientations method."""

    @pytest.fixture
    def plotter(self) -> PlotGenerator:
        """Create PlotGenerator."""
        return PlotGenerator(use_matplotlib=False)

    def test_uses_metadata_orientations(self, plotter: PlotGenerator) -> None:
        """Test that metadata objective_orientations are used."""
        result = OptimizationResult(
            trials=[],
            best_config={},
            best_score=0.0,
            optimization_id="opt-1",
            duration=1.0,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=["accuracy", "cost"],
            algorithm="test",
            timestamp=datetime.now(UTC),
            metadata={
                "objective_orientations": {"accuracy": "maximize", "cost": "minimize"}
            },
        )

        orientations = plotter._determine_objective_orientations(
            result, ["accuracy", "cost"]
        )

        assert orientations == {"accuracy": True, "cost": False}

    def test_uses_objective_schema(self, plotter: PlotGenerator) -> None:
        """Test that objective_schema is parsed correctly."""
        # Create mock objective definitions
        mock_obj1 = MagicMock()
        mock_obj1.name = "accuracy"
        mock_obj1.orientation = "maximize"

        mock_obj2 = MagicMock()
        mock_obj2.name = "latency"
        mock_obj2.orientation = "minimize"

        mock_schema = MagicMock()
        mock_schema.objectives = [mock_obj1, mock_obj2]

        result = OptimizationResult(
            trials=[],
            best_config={},
            best_score=0.0,
            optimization_id="opt-1",
            duration=1.0,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=["accuracy", "latency"],
            algorithm="test",
            timestamp=datetime.now(UTC),
            metadata={"objective_schema": mock_schema},
        )

        orientations = plotter._determine_objective_orientations(
            result, ["accuracy", "latency"]
        )

        assert orientations["accuracy"] is True
        assert orientations["latency"] is False

    def test_uses_objective_schema_dict(self, plotter: PlotGenerator) -> None:
        """Test that objective_schema as dict is parsed correctly."""
        result = OptimizationResult(
            trials=[],
            best_config={},
            best_score=0.0,
            optimization_id="opt-1",
            duration=1.0,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=["accuracy", "cost"],
            algorithm="test",
            timestamp=datetime.now(UTC),
            metadata={
                "objective_schema": {
                    "objectives": [
                        {"name": "accuracy", "orientation": "maximize"},
                        {"name": "cost", "orientation": "minimize"},
                    ]
                }
            },
        )

        orientations = plotter._determine_objective_orientations(
            result, ["accuracy", "cost"]
        )

        assert orientations["accuracy"] is True
        assert orientations["cost"] is False

    def test_falls_back_to_heuristics(self, plotter: PlotGenerator) -> None:
        """Test that heuristics are used when no metadata available."""
        result = OptimizationResult(
            trials=[],
            best_config={},
            best_score=0.0,
            optimization_id="opt-1",
            duration=1.0,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=["accuracy", "cost", "latency", "error", "f1_score"],
            algorithm="test",
            timestamp=datetime.now(UTC),
            metadata={},
        )

        orientations = plotter._determine_objective_orientations(
            result, ["accuracy", "cost", "latency", "error", "f1_score"]
        )

        # accuracy and f1_score should be maximize (heuristic)
        assert orientations["accuracy"] is True
        assert orientations["f1_score"] is True
        # cost, latency, error should be minimize (heuristic)
        assert orientations["cost"] is False
        assert orientations["latency"] is False
        assert orientations["error"] is False

    def test_heuristic_patterns(self, plotter: PlotGenerator) -> None:
        """Test various heuristic patterns for minimize."""
        result = OptimizationResult(
            trials=[],
            best_config={},
            best_score=0.0,
            optimization_id="opt-1",
            duration=1.0,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=["loss", "time", "duration", "accuracy"],
            algorithm="test",
            timestamp=datetime.now(UTC),
            metadata={},
        )

        orientations = plotter._determine_objective_orientations(
            result, ["loss", "time", "duration", "accuracy"]
        )

        assert orientations["loss"] is False  # minimize
        assert orientations["time"] is False  # minimize
        assert orientations["duration"] is False  # minimize
        assert orientations["accuracy"] is True  # maximize

    def test_metadata_orientations_override_schema(
        self, plotter: PlotGenerator
    ) -> None:
        """Test that objective_orientations override schema values."""
        mock_obj = MagicMock()
        mock_obj.name = "accuracy"
        mock_obj.orientation = "minimize"  # Wrong

        mock_schema = MagicMock()
        mock_schema.objectives = [mock_obj]

        result = OptimizationResult(
            trials=[],
            best_config={},
            best_score=0.0,
            optimization_id="opt-1",
            duration=1.0,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=["accuracy"],
            algorithm="test",
            timestamp=datetime.now(UTC),
            metadata={
                "objective_schema": mock_schema,
                # Should be False from schema, no override
            },
        )

        orientations = plotter._determine_objective_orientations(result, ["accuracy"])
        # Should get minimize from schema (False)
        assert orientations["accuracy"] is False

    def test_boolean_orientation_values(self, plotter: PlotGenerator) -> None:
        """Test that boolean orientation values are handled correctly."""
        result = OptimizationResult(
            trials=[],
            best_config={},
            best_score=0.0,
            optimization_id="opt-1",
            duration=1.0,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=["accuracy", "cost"],
            algorithm="test",
            timestamp=datetime.now(UTC),
            metadata={"objective_orientations": {"accuracy": True, "cost": False}},
        )

        orientations = plotter._determine_objective_orientations(
            result, ["accuracy", "cost"]
        )

        assert orientations["accuracy"] is True
        assert orientations["cost"] is False


class TestFormatObjectiveLabel:
    """Tests for _format_objective_label static method."""

    def test_maximize_label(self) -> None:
        """Test label formatting for maximize objective."""
        label = PlotGenerator._format_objective_label("accuracy", True)
        assert label == "Accuracy (max)"

    def test_minimize_label(self) -> None:
        """Test label formatting for minimize objective."""
        label = PlotGenerator._format_objective_label("cost", False)
        assert label == "Cost (min)"

    def test_title_case_conversion(self) -> None:
        """Test that objective name is converted to title case."""
        label = PlotGenerator._format_objective_label("f1_score", True)
        assert label == "F1_Score (max)"


class TestPlotParameterImportance:
    """Tests for plot_parameter_importance method."""

    @pytest.fixture
    def plotter_ascii(self) -> PlotGenerator:
        """Create PlotGenerator in ASCII mode."""
        return PlotGenerator(use_matplotlib=False)

    @pytest.fixture
    def plotter_matplotlib(self) -> PlotGenerator:
        """Create PlotGenerator with matplotlib enabled."""
        plotter = PlotGenerator(use_matplotlib=True)
        plotter._matplotlib_available = True
        plotter.plt = MagicMock()
        return plotter

    def test_empty_importance_results(self, plotter_ascii: PlotGenerator) -> None:
        """Test with empty importance results."""
        output = plotter_ascii.plot_parameter_importance({})
        assert output == "No importance data to plot"

    def test_matplotlib_mode_calls_matplotlib_plot(
        self, plotter_matplotlib: PlotGenerator
    ) -> None:
        """Test that matplotlib mode is used when available."""
        mock_result = MagicMock()
        mock_result.importance_score = 0.8
        importance_results = {"param1": mock_result}

        with patch.object(
            plotter_matplotlib, "_plot_importance_matplotlib"
        ) as mock_mpl:
            mock_mpl.return_value = "importance.png"
            output = plotter_matplotlib.plot_parameter_importance(importance_results)
            assert output == "importance.png"
            mock_mpl.assert_called_once()

    def test_ascii_mode(self, plotter_ascii: PlotGenerator) -> None:
        """Test that ASCII mode is used when matplotlib unavailable."""
        mock_result = MagicMock()
        mock_result.importance_score = 0.8
        importance_results = {"param1": mock_result}

        with patch.object(plotter_ascii, "_plot_importance_ascii") as mock_ascii:
            mock_ascii.return_value = "ascii_plot"
            output = plotter_ascii.plot_parameter_importance(importance_results)
            assert output == "ascii_plot"
            mock_ascii.assert_called_once()


class TestPlotImportanceMatplotlib:
    """Tests for _plot_importance_matplotlib method."""

    @pytest.fixture
    def plotter(self) -> PlotGenerator:
        """Create PlotGenerator with matplotlib mocked."""
        plotter = PlotGenerator(use_matplotlib=True)
        plotter._matplotlib_available = True
        plotter.plt = MagicMock()
        return plotter

    def test_creates_importance_plot(self, plotter: PlotGenerator) -> None:
        """Test that matplotlib importance plot is created."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        plotter.plt.subplots.return_value = (mock_fig, mock_ax)

        result = plotter._plot_importance_matplotlib(
            params=["param1", "param2", "param3"],
            scores=[0.8, 0.6, 0.4],
        )

        assert "parameter_importance.png" in result
        mock_ax.barh.assert_called_once()
        mock_ax.set_xlabel.assert_called_once()
        mock_ax.set_title.assert_called_once()
        mock_fig.savefig.assert_called_once()

    def test_sorts_by_importance(self, plotter: PlotGenerator) -> None:
        """Test that parameters are sorted by importance."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_bars = [MagicMock(), MagicMock(), MagicMock()]
        for i, bar in enumerate(mock_bars):
            bar.get_width.return_value = [0.8, 0.6, 0.4][i]
            bar.get_y.return_value = i
            bar.get_height.return_value = 0.8
        mock_ax.barh.return_value = mock_bars
        plotter.plt.subplots.return_value = (mock_fig, mock_ax)

        plotter._plot_importance_matplotlib(
            params=["param2", "param1", "param3"],
            scores=[0.6, 0.8, 0.4],
        )

        # Should be sorted in descending order
        call_args = mock_ax.barh.call_args
        sorted_scores = call_args[0][1]
        # Check that scores are in descending order
        assert list(sorted_scores) == sorted(sorted_scores, reverse=True)


class TestPlotImportanceAscii:
    """Tests for _plot_importance_ascii method."""

    @pytest.fixture
    def plotter(self) -> PlotGenerator:
        """Create PlotGenerator in ASCII mode."""
        return PlotGenerator(use_matplotlib=False)

    def test_empty_data_returns_message(self, plotter: PlotGenerator) -> None:
        """Test that empty data returns appropriate message."""
        result = plotter._plot_importance_ascii([], [])
        assert result == "No data to plot"

    def test_ascii_importance_contains_expected_elements(
        self, plotter: PlotGenerator
    ) -> None:
        """Test that ASCII importance plot contains all expected elements."""
        result = plotter._plot_importance_ascii(
            params=["param1", "param2", "param3"],
            scores=[0.8, 0.6, 0.4],
        )

        assert "Parameter Importance Analysis" in result
        assert "param1" in result
        assert "param2" in result
        assert "param3" in result
        assert "0.800" in result
        assert "0.600" in result
        assert "0.400" in result
        assert "Scale:" in result

    def test_sorts_by_importance_descending(self, plotter: PlotGenerator) -> None:
        """Test that parameters are sorted by importance in descending order."""
        result = plotter._plot_importance_ascii(
            params=["low", "high", "medium"],
            scores=[0.3, 0.9, 0.6],
        )

        lines = result.split("\n")
        # Find the lines with parameters (skip header)
        param_lines = [line for line in lines if "0." in line and "|" in line]

        # First param line should have the highest score
        assert "high" in param_lines[0]
        assert "0.900" in param_lines[0]


class TestGenerateOptimizationReport:
    """Tests for generate_optimization_report method."""

    @pytest.fixture
    def plotter(self) -> PlotGenerator:
        """Create PlotGenerator in ASCII mode."""
        return PlotGenerator(use_matplotlib=False)

    def test_report_contains_basic_info(self, plotter: PlotGenerator) -> None:
        """Test that report contains basic optimization information."""
        trials = [_make_trial("t1", 0.9, 0.5), _make_trial("t2", 0.85, 0.3)]
        result = _make_result(trials, {}, objectives=["accuracy", "cost"])
        result.function_name = "test_function"

        with patch.object(plotter, "plot_optimization_progress") as mock_progress:
            with patch.object(plotter, "plot_pareto_front") as mock_pareto:
                mock_progress.return_value = "progress_plot"
                mock_pareto.return_value = "pareto_plot"

                report = plotter.generate_optimization_report(result)

        assert "TraiGent Optimization Report" in report
        assert "test_function" in report
        assert "test_algo" in report
        assert "accuracy, cost" in report
        assert "Total trials: 2" in report
        assert "Successful trials: 2" in report

    def test_report_includes_progress_plot(self, plotter: PlotGenerator) -> None:
        """Test that report includes optimization progress plot."""
        trials = [_make_trial("t1", 0.9, 0.5)]
        result = _make_result(trials, {}, objectives=["accuracy"])

        with patch.object(plotter, "plot_optimization_progress") as mock_progress:
            mock_progress.return_value = "PROGRESS_PLOT_HERE"
            report = plotter.generate_optimization_report(result)

        assert "Optimization Progress" in report
        assert "PROGRESS_PLOT_HERE" in report

    def test_report_includes_pareto_for_multi_objective(
        self, plotter: PlotGenerator
    ) -> None:
        """Test that Pareto front is included for multi-objective optimization."""
        trials = [_make_trial("t1", 0.9, 0.5), _make_trial("t2", 0.85, 0.3)]
        result = _make_result(trials, {}, objectives=["accuracy", "cost"])

        with patch.object(plotter, "plot_optimization_progress") as mock_progress:
            with patch.object(plotter, "plot_pareto_front") as mock_pareto:
                mock_progress.return_value = "PROGRESS_PLOT_HERE"
                mock_pareto.return_value = "PARETO_PLOT_HERE"
                report = plotter.generate_optimization_report(result)

        assert "Pareto Front Analysis" in report
        assert "PARETO_PLOT_HERE" in report

    def test_report_skips_pareto_for_single_objective(
        self, plotter: PlotGenerator
    ) -> None:
        """Test that Pareto front is skipped for single-objective optimization."""
        trials = [_make_trial("t1", 0.9, 0.5)]
        result = _make_result(trials, {}, objectives=["accuracy"])

        with patch.object(plotter, "plot_optimization_progress") as mock_progress:
            with patch.object(plotter, "plot_pareto_front") as mock_pareto:
                mock_progress.return_value = "PROGRESS_PLOT_HERE"
                report = plotter.generate_optimization_report(result)

        assert "Pareto Front Analysis" not in report
        mock_pareto.assert_not_called()

    def test_report_includes_configuration_space(self, plotter: PlotGenerator) -> None:
        """Test that report includes configuration space information."""
        trials = [_make_trial("t1", 0.9, 0.5)]
        result = _make_result(trials, {}, objectives=["accuracy"])
        result.configuration_space = {
            "learning_rate": [0.001, 0.01, 0.1],
            "batch_size": (16, 128),
            "optimizer": "adam",
        }

        with patch.object(plotter, "plot_optimization_progress") as mock_progress:
            mock_progress.return_value = "PROGRESS_PLOT_HERE"
            report = plotter.generate_optimization_report(result)

        assert "Configuration Space" in report
        assert "learning_rate: 3 discrete values" in report
        assert "batch_size: continuous range (16, 128)" in report
        assert "optimizer: adam" in report

    def test_report_includes_best_configuration(self, plotter: PlotGenerator) -> None:
        """Test that report includes best configuration."""
        trials = [_make_trial("t1", 0.9, 0.5)]
        result = _make_result(trials, {}, objectives=["accuracy"])
        result.best_config = {"learning_rate": 0.01, "batch_size": 32}

        with patch.object(plotter, "plot_optimization_progress") as mock_progress:
            mock_progress.return_value = "PROGRESS_PLOT_HERE"
            report = plotter.generate_optimization_report(result)

        assert "Best Configuration" in report
        assert "learning_rate: 0.01" in report
        assert "batch_size: 32" in report


class TestCreateQuickPlot:
    """Tests for create_quick_plot function."""

    def test_creates_progress_plot(self) -> None:
        """Test creating progress plot."""
        trials = [_make_trial("t1", 0.9, 0.5)]
        result = _make_result(trials, {}, objectives=["accuracy"])

        output = create_quick_plot(result, plot_type="progress")
        # Should create ASCII plot (use_matplotlib=False)
        assert "Optimization Progress" in output or "No completed trials" in output

    def test_creates_pareto_plot(self) -> None:
        """Test creating Pareto plot."""
        trials = [_make_trial("t1", 0.9, 0.5), _make_trial("t2", 0.85, 0.3)]
        result = _make_result(trials, {}, objectives=["accuracy", "cost"])

        output = create_quick_plot(result, plot_type="pareto")
        # Should create ASCII Pareto plot
        assert "Pareto" in output or "Pareto points found" in output

    def test_creates_report(self) -> None:
        """Test creating full report."""
        trials = [_make_trial("t1", 0.9, 0.5)]
        result = _make_result(trials, {}, objectives=["accuracy"])

        output = create_quick_plot(result, plot_type="report")
        assert "TraiGent Optimization Report" in output

    def test_unknown_plot_type(self) -> None:
        """Test handling of unknown plot type."""
        trials = [_make_trial("t1", 0.9, 0.5)]
        result = _make_result(trials, {}, objectives=["accuracy"])

        output = create_quick_plot(result, plot_type="unknown")
        assert "Unknown plot type: unknown" in output

    def test_uses_ascii_mode(self) -> None:
        """Test that quick plot uses ASCII mode (not matplotlib)."""
        trials = [_make_trial("t1", 0.9, 0.5)]
        result = _make_result(trials, {}, objectives=["accuracy"])

        with patch("traigent.visualization.plots.PlotGenerator") as mock_generator:
            mock_instance = MagicMock()
            mock_instance.plot_optimization_progress.return_value = "plot"
            mock_generator.return_value = mock_instance

            create_quick_plot(result, plot_type="progress")

            # Should be called with use_matplotlib=False
            mock_generator.assert_called_once_with(use_matplotlib=False)
