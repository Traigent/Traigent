"""Visualization utilities for TraiGent optimization results."""

# Traceability: CONC-Layer-Tooling CONC-Quality-Usability CONC-Quality-Observability FUNC-ANALYTICS REQ-ANLY-011 SYNC-Observability

from __future__ import annotations

from typing import Any

from ..api.types import OptimizationResult
from ..utils.multi_objective import ParetoFrontCalculator


class PlotGenerator:
    """Generates plots for optimization results using ASCII and optional matplotlib."""

    def __init__(self, use_matplotlib: bool = True) -> None:
        """Initialize plot generator.

        Args:
            use_matplotlib: Whether to use matplotlib for rich plots
        """
        self.use_matplotlib = use_matplotlib
        self._matplotlib_available = False

        if use_matplotlib:
            try:
                import matplotlib
                import matplotlib.pyplot as plt

                self._matplotlib_available = True
                self.plt = plt
                # Use non-interactive backend for server environments
                matplotlib.use("Agg")
            except ImportError:
                self._matplotlib_available = False

    def plot_optimization_progress(
        self, result: OptimizationResult, objective: str | None = None
    ) -> str:
        """Plot optimization progress over trials.

        Args:
            result: Optimization result
            objective: Objective to plot (uses primary if None)

        Returns:
            Plot as string (ASCII or path to saved image)
        """
        if not result.trials:
            return "No trials to plot"

        # Determine objective
        if objective is None:
            objective = result.objectives[0] if result.objectives else "score"

        # Determine orientation for the selected objective
        orientations = self._determine_objective_orientations(result, [objective])
        maximize = orientations.get(objective, True)
        objective_label = self._format_objective_label(objective, maximize)

        # Extract trial data
        trial_numbers = []
        scores = []
        best_scores = []

        current_best: float | None = None

        for i, trial in enumerate(result.trials):
            if trial.status == "completed" and trial.metrics:
                score_value = trial.metrics.get(objective)
                if score_value is None:
                    continue
                try:
                    score = float(score_value)
                except (TypeError, ValueError):
                    continue
                trial_numbers.append(i + 1)
                scores.append(score)

                if current_best is None:
                    current_best = score
                elif maximize:
                    current_best = max(current_best, score)
                else:
                    current_best = min(current_best, score)

                best_scores.append(current_best)

        if not scores:
            return f"No completed trials with objective '{objective}'"

        if self._matplotlib_available:
            return self._plot_progress_matplotlib(
                trial_numbers,
                scores,
                best_scores,
                objective,
                objective_label,
            )
        else:
            return self._plot_progress_ascii(
                trial_numbers,
                scores,
                best_scores,
                objective_label,
            )

    def _plot_progress_matplotlib(
        self,
        trial_numbers: list[int],
        scores: list[float],
        best_scores: list[float],
        objective: str,
        objective_label: str,
    ) -> str:
        """Create matplotlib progress plot."""
        fig, ax = self.plt.subplots(figsize=(10, 6))

        # Plot individual trial scores
        ax.scatter(
            trial_numbers, scores, alpha=0.6, color="lightblue", label="Trial scores"
        )

        # Plot best score progression
        ax.plot(
            trial_numbers, best_scores, color="red", linewidth=2, label="Best score"
        )

        ax.set_xlabel("Trial Number")
        ax.set_ylabel(objective_label)
        ax.set_title(f"Optimization Progress: {objective_label}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Save plot
        filename = f"optimization_progress_{objective}.png"
        fig.savefig(filename, dpi=150, bbox_inches="tight")
        self.plt.close(fig)

        return f"Plot saved: {filename}"

    def _plot_progress_ascii(
        self,
        trial_numbers: list[int],
        scores: list[float],
        best_scores: list[float],
        objective_label: str,
    ) -> str:
        """Create ASCII progress plot."""
        if not scores:
            return "No data to plot"

        # Normalize scores for plotting
        min_score = min(scores)
        max_score = max(scores)
        score_range = max_score - min_score if max_score > min_score else 1

        width = 60
        height = 20

        lines = [f"Optimization Progress: {objective_label}"]
        lines.append("=" * width)
        lines.append("")

        # Create ASCII plot
        plot_lines = [""] * height

        for i, (_trial_num, score, best_score) in enumerate(
            zip(trial_numbers, scores, best_scores, strict=False)
        ):
            # Calculate positions
            x_pos = (
                int((i / (len(trial_numbers) - 1)) * (width - 1))
                if len(trial_numbers) > 1
                else 0
            )

            # Score position
            score_y = int(((score - min_score) / score_range) * (height - 1))
            score_y = height - 1 - score_y  # Flip Y axis

            # Best score position
            best_y = int(((best_score - min_score) / score_range) * (height - 1))
            best_y = height - 1 - best_y  # Flip Y axis

            # Place characters
            if len(plot_lines[score_y]) <= x_pos:
                plot_lines[score_y] += " " * (x_pos - len(plot_lines[score_y]) + 1)

            if x_pos < len(plot_lines[score_y]):
                plot_lines[score_y] = (
                    plot_lines[score_y][:x_pos] + "·" + plot_lines[score_y][x_pos + 1 :]
                )

            if len(plot_lines[best_y]) <= x_pos:
                plot_lines[best_y] += " " * (x_pos - len(plot_lines[best_y]) + 1)

            if x_pos < len(plot_lines[best_y]):
                plot_lines[best_y] = (
                    plot_lines[best_y][:x_pos] + "*" + plot_lines[best_y][x_pos + 1 :]
                )

        # Add Y-axis labels
        for i, line in enumerate(plot_lines):
            y_value = max_score - (i / (height - 1)) * score_range
            label = f"{y_value:.2f}|"
            lines.append(f"{label:>7s}{line}")

        # Add X-axis
        x_axis = " " * 8 + "".join([str(i % 10) for i in range(width)])
        lines.append(x_axis)
        lines.append("")
        lines.append("Legend: · = trial score, * = best score so far")
        lines.append(f"Score range: {min_score:.2f} -> {max_score:.2f}")
        lines.append(f"Final best score: {best_scores[-1]:.3f}")

        return "\n".join(lines)

    def plot_pareto_front(
        self, result: OptimizationResult, obj1: str, obj2: str
    ) -> str:
        """Plot Pareto front for two objectives.

        Args:
            result: Optimization result
            obj1: First objective name
            obj2: Second objective name

        Returns:
            Plot as string (ASCII or path to saved image)
        """
        if len(result.objectives) < 2:
            return "Need at least 2 objectives for Pareto front plot"

        # Calculate Pareto front
        orientations = self._determine_objective_orientations(result, [obj1, obj2])
        pareto_calc = ParetoFrontCalculator(maximize=orientations)
        pareto_front = pareto_calc.calculate_pareto_front(
            result.successful_trials, [obj1, obj2]
        )

        if not pareto_front:
            return "No Pareto-optimal solutions found"

        # Extract all trial points
        all_points_x = []
        all_points_y = []
        pareto_points_x = []
        pareto_points_y = []

        for trial in result.successful_trials:
            if obj1 in trial.metrics and obj2 in trial.metrics:
                all_points_x.append(trial.metrics[obj1])
                all_points_y.append(trial.metrics[obj2])

        for point in pareto_front:
            if obj1 in point.objectives and obj2 in point.objectives:
                pareto_points_x.append(point.objectives[obj1])
                pareto_points_y.append(point.objectives[obj2])

        if self._matplotlib_available:
            return self._plot_pareto_matplotlib(
                all_points_x,
                all_points_y,
                pareto_points_x,
                pareto_points_y,
                obj1,
                obj2,
                orientations,
            )
        else:
            return self._plot_pareto_ascii(
                all_points_x,
                all_points_y,
                pareto_points_x,
                pareto_points_y,
                obj1,
                obj2,
                orientations,
            )

    def _plot_pareto_matplotlib(
        self,
        all_x: list[float],
        all_y: list[float],
        pareto_x: list[float],
        pareto_y: list[float],
        obj1: str,
        obj2: str,
        orientations: dict[str, bool],
    ) -> str:
        """Create matplotlib Pareto front plot."""
        fig, ax = self.plt.subplots(figsize=(10, 8))

        # Plot all points
        ax.scatter(all_x, all_y, alpha=0.6, color="lightblue", label="All trials")

        # Plot Pareto front
        ax.scatter(
            pareto_x,
            pareto_y,
            color="red",
            s=100,
            label="Pareto front",
            marker="*",
            edgecolor="darkred",
        )

        # Connect Pareto points
        if len(pareto_x) > 1:
            reverse_sort = orientations.get(obj1, True)
            sorted_pareto = sorted(
                zip(pareto_x, pareto_y, strict=False),
                key=lambda p: p[0],
                reverse=reverse_sort,
            )
            sorted_x, sorted_y = zip(*sorted_pareto, strict=False)
            ax.plot(sorted_x, sorted_y, "r--", alpha=0.7)

        label1 = self._format_objective_label(obj1, orientations.get(obj1, True))
        label2 = self._format_objective_label(obj2, orientations.get(obj2, True))
        ax.set_xlabel(label1)
        ax.set_ylabel(label2)
        ax.set_title(f"Pareto Front: {label1} vs {label2}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Save plot
        filename = f"pareto_front_{obj1}_{obj2}.png"
        fig.savefig(filename, dpi=150, bbox_inches="tight")
        self.plt.close(fig)

        return f"Plot saved: {filename}"

    def _plot_pareto_ascii(
        self,
        all_x: list[float],
        all_y: list[float],
        pareto_x: list[float],
        pareto_y: list[float],
        obj1: str,
        obj2: str,
        orientations: dict[str, bool],
    ) -> str:
        """Create ASCII Pareto front plot."""
        if not all_x or not all_y:
            return "No data to plot"

        width = 60
        height = 20

        # Calculate ranges
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        range_x = max_x - min_x if max_x > min_x else 1
        range_y = max_y - min_y if max_y > min_y else 1

        label1 = self._format_objective_label(obj1, orientations.get(obj1, True))
        label2 = self._format_objective_label(obj2, orientations.get(obj2, True))
        lines = [f"Pareto Front: {label1} vs {label2}"]
        lines.append("=" * width)
        lines.append("")

        # Create plot grid
        plot_grid = [[" " for _ in range(width)] for _ in range(height)]

        # Plot all points
        for x, y in zip(all_x, all_y, strict=False):
            plot_x = int(((x - min_x) / range_x) * (width - 1))
            plot_y = int(((y - min_y) / range_y) * (height - 1))
            plot_y = height - 1 - plot_y  # Flip Y axis

            if 0 <= plot_x < width and 0 <= plot_y < height:
                plot_grid[plot_y][plot_x] = "·"

        # Plot Pareto points
        for x, y in zip(pareto_x, pareto_y, strict=False):
            plot_x = int(((x - min_x) / range_x) * (width - 1))
            plot_y = int(((y - min_y) / range_y) * (height - 1))
            plot_y = height - 1 - plot_y  # Flip Y axis

            if 0 <= plot_x < width and 0 <= plot_y < height:
                plot_grid[plot_y][plot_x] = "*"

        # Convert grid to strings with Y-axis labels
        for i, row in enumerate(plot_grid):
            y_value = max_y - (i / (height - 1)) * range_y
            label = f"{y_value:.2f}|"
            lines.append(f"{label:>7s}{''.join(row)}")

        # Add X-axis
        x_axis = " " * 8 + "".join([str(i % 10) for i in range(width)])
        lines.append(x_axis)
        lines.append("")
        lines.append(f"X-axis: {label1} ({min_x:.2f} to {max_x:.2f})")
        lines.append(f"Y-axis: {label2} ({min_y:.2f} to {max_y:.2f})")
        lines.append("Legend: · = all trials, * = Pareto optimal")
        lines.append(f"Pareto points found: {len(pareto_x)}")

        return "\n".join(lines)

    def _determine_objective_orientations(
        self, result: OptimizationResult, objectives: list[str]
    ) -> dict[str, bool]:
        """Determine objective orientations from metadata or heuristics."""
        orientations: dict[str, bool] = {}
        metadata = result.metadata or {}

        schema = metadata.get("objective_schema")
        if schema is not None:
            definitions = []
            if hasattr(schema, "objectives"):
                definitions = schema.objectives
            elif isinstance(schema, dict):
                definitions = schema.get("objectives", [])

            for obj_def in definitions:
                name = getattr(obj_def, "name", None)
                if name is None and isinstance(obj_def, dict):
                    name = obj_def.get("name")
                orientation = getattr(obj_def, "orientation", None)
                if orientation is None and isinstance(obj_def, dict):
                    orientation = obj_def.get("orientation")

                if name and orientation:
                    orientations[name] = str(orientation).lower() != "minimize"

        metadata_orientations = metadata.get("objective_orientations")
        if isinstance(metadata_orientations, dict):
            for name, orientation in metadata_orientations.items():
                if name not in orientations:
                    if isinstance(orientation, str):
                        orientations[name] = orientation.lower() != "minimize"
                    else:
                        orientations[name] = bool(orientation)

        minimize_patterns = ["cost", "latency", "error", "loss", "time", "duration"]
        for name in objectives:
            if name not in orientations:
                lower = name.lower()
                orientations[name] = not any(
                    pattern in lower for pattern in minimize_patterns
                )

        return {name: orientations.get(name, True) for name in objectives}

    @staticmethod
    def _format_objective_label(name: str, maximize: bool) -> str:
        """Format axis label with orientation indicator."""
        direction = "max" if maximize else "min"
        return f"{name.title()} ({direction})"

    def plot_parameter_importance(self, importance_results: dict[str, Any]) -> str:
        """Plot parameter importance analysis results.

        Args:
            importance_results: Results from ParameterImportanceAnalyzer

        Returns:
            Plot as string (ASCII or path to saved image)
        """
        if not importance_results:
            return "No importance data to plot"

        # Extract parameter names and importance scores
        params = list(importance_results.keys())
        scores = [result.importance_score for result in importance_results.values()]

        if self._matplotlib_available:
            return self._plot_importance_matplotlib(params, scores)
        else:
            return self._plot_importance_ascii(params, scores)

    def _plot_importance_matplotlib(
        self, params: list[str], scores: list[float]
    ) -> str:
        """Create matplotlib parameter importance plot."""
        fig, ax = self.plt.subplots(figsize=(10, 6))

        # Sort by importance
        sorted_data = sorted(
            zip(params, scores, strict=False), key=lambda x: x[1], reverse=True
        )
        sorted_params, sorted_scores = zip(*sorted_data, strict=False)

        # Create horizontal bar plot
        y_pos = range(len(sorted_params))
        bars = ax.barh(y_pos, sorted_scores, color="skyblue", edgecolor="navy")

        # Add value labels on bars
        for _i, (bar, score) in enumerate(zip(bars, sorted_scores, strict=False)):
            ax.text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{score:.3f}",
                va="center",
                ha="left",
            )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_params)
        ax.set_xlabel("Importance Score")
        ax.set_title("Parameter Importance Analysis")
        ax.grid(True, axis="x", alpha=0.3)

        # Save plot
        filename = "parameter_importance.png"
        fig.savefig(filename, dpi=150, bbox_inches="tight")
        self.plt.close(fig)

        return f"Plot saved: {filename}"

    def _plot_importance_ascii(self, params: list[str], scores: list[float]) -> str:
        """Create ASCII parameter importance plot."""
        if not params or not scores:
            return "No data to plot"

        # Sort by importance
        sorted_data = sorted(
            zip(params, scores, strict=False), key=lambda x: x[1], reverse=True
        )

        lines = ["Parameter Importance Analysis"]
        lines.append("=" * 50)
        lines.append("")

        max_score = max(scores) if scores else 1
        max_param_len = max(len(param) for param in params) if params else 10
        bar_width = 40

        for param, score in sorted_data:
            # Create bar
            normalized_score = score / max_score if max_score > 0 else 0
            bar_length = int(normalized_score * bar_width)
            bar = "█" * bar_length + "░" * (bar_width - bar_length)

            # Format line
            param_str = f"{param:>{max_param_len}s}"
            score_str = f"{score:.3f}"
            lines.append(f"{param_str} |{bar}| {score_str}")

        lines.append("")
        lines.append(f"Scale: 0 to {max_score:.3f}")

        return "\n".join(lines)

    def generate_optimization_report(
        self, result: OptimizationResult, save_plots: bool = True
    ) -> str:
        """Generate comprehensive optimization report with plots.

        Args:
            result: Optimization result
            save_plots: Whether to save plot files

        Returns:
            Comprehensive report with embedded plots
        """
        lines = []
        lines.append("🚀 TraiGent Optimization Report")
        lines.append("=" * 50)
        lines.append("")

        # Basic information
        lines.append(f"Function: {result.function_name}")
        lines.append(f"Algorithm: {result.algorithm}")
        lines.append(f"Objectives: {', '.join(result.objectives)}")
        lines.append(f"Duration: {result.duration:.1f}s")
        lines.append(f"Total trials: {len(result.trials)}")
        lines.append(f"Successful trials: {len(result.successful_trials)}")
        lines.append(f"Success rate: {result.success_rate:.1%}")
        lines.append(f"Best score: {result.best_score:.3f}")
        lines.append("")

        # Progress plot
        lines.append("📈 Optimization Progress")
        lines.append("-" * 30)
        progress_plot = self.plot_optimization_progress(result)
        lines.append(progress_plot)
        lines.append("")

        # Pareto front (if multi-objective)
        if len(result.objectives) >= 2:
            lines.append("🎯 Pareto Front Analysis")
            lines.append("-" * 30)
            pareto_plot = self.plot_pareto_front(
                result, result.objectives[0], result.objectives[1]
            )
            lines.append(pareto_plot)
            lines.append("")

        # Configuration space summary
        lines.append("⚙️ Configuration Space")
        lines.append("-" * 30)
        for param, values in result.configuration_space.items():
            if isinstance(values, list):
                lines.append(f"{param}: {len(values)} discrete values")
            elif isinstance(values, tuple):
                lines.append(f"{param}: continuous range {values}")
            else:
                lines.append(f"{param}: {values}")
        lines.append("")

        # Best configuration
        lines.append("🏆 Best Configuration")
        lines.append("-" * 30)
        for param, value in result.best_config.items():
            lines.append(f"{param}: {value}")
        lines.append("")

        return "\n".join(lines)


def create_quick_plot(result: OptimizationResult, plot_type: str = "progress") -> str:
    """Quick function to create a plot from optimization result.

    Args:
        result: Optimization result
        plot_type: Type of plot ("progress", "pareto", "importance")

    Returns:
        Plot as string
    """
    plotter = PlotGenerator(use_matplotlib=False)  # Use ASCII for quick plots

    if plot_type == "progress":
        return plotter.plot_optimization_progress(result)
    elif plot_type == "pareto" and len(result.objectives) >= 2:
        return plotter.plot_pareto_front(
            result, result.objectives[0], result.objectives[1]
        )
    elif plot_type == "report":
        return plotter.generate_optimization_report(result, save_plots=False)
    else:
        return f"Unknown plot type: {plot_type}"
