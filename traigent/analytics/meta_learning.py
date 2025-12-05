"""Meta-learning engine for TraiGent optimization history analysis."""

# Traceability: CONC-Layer-Core CONC-Quality-Observability CONC-Quality-Performance FUNC-ANALYTICS REQ-ANLY-011 SYNC-Observability

from __future__ import annotations

import json
import statistics
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from ..core.constants import HISTORY_PRUNE_RATIO, MAX_OPTIMIZATION_HISTORY_SIZE
from ..utils.logging import get_logger

logger = get_logger(__name__)


class OptimizationStatus(Enum):
    """Optimization status."""

    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class AlgorithmType(Enum):
    """Algorithm types for meta-learning."""

    GRID = "grid"
    RANDOM = "random"
    BAYESIAN = "bayesian"
    GENETIC = "genetic"
    PARTICLE_SWARM = "particle_swarm"
    SIMULATED_ANNEALING = "simulated_annealing"


@dataclass
class OptimizationRecord:
    """Single optimization run record."""

    optimization_id: str
    function_name: str
    algorithm: AlgorithmType
    configuration_space: dict[str, Any]
    objectives: list[str]
    dataset_size: int
    best_score: float
    total_trials: int
    duration_seconds: float
    status: OptimizationStatus
    timestamp: datetime
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "optimization_id": self.optimization_id,
            "function_name": self.function_name,
            "algorithm": self.algorithm.value,
            "configuration_space": self.configuration_space,
            "objectives": self.objectives,
            "dataset_size": self.dataset_size,
            "best_score": self.best_score,
            "total_trials": self.total_trials,
            "duration_seconds": self.duration_seconds,
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OptimizationRecord:
        """Create from dictionary."""
        return cls(
            optimization_id=data["optimization_id"],
            function_name=data["function_name"],
            algorithm=AlgorithmType(data["algorithm"]),
            configuration_space=data["configuration_space"],
            objectives=data["objectives"],
            dataset_size=data["dataset_size"],
            best_score=data["best_score"],
            total_trials=data["total_trials"],
            duration_seconds=data["duration_seconds"],
            status=OptimizationStatus(data["status"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ProblemCharacteristics:
    """Characteristics of an optimization problem."""

    config_space_size: int
    param_types: dict[str, str]  # param_name -> type (categorical, continuous, integer)
    num_objectives: int
    problem_complexity: float  # 0-1 scale
    estimated_evaluation_time: float  # seconds

    def similarity(self, other: ProblemCharacteristics) -> float:
        """Calculate similarity to another problem (0-1)."""
        similarities = []

        # Configuration space size similarity
        size_diff = abs(self.config_space_size - other.config_space_size)
        max_size = max(self.config_space_size, other.config_space_size, 1)
        size_sim = 1.0 - (size_diff / max_size)
        similarities.append(size_sim)

        # Parameter types similarity
        all_params = set(self.param_types.keys()) | set(other.param_types.keys())
        if all_params:
            common_types = sum(
                1
                for p in all_params
                if self.param_types.get(p) == other.param_types.get(p)
            )
            type_sim = common_types / len(all_params)
            similarities.append(type_sim)

        # Objectives similarity
        obj_sim = 1.0 - abs(self.num_objectives - other.num_objectives) / max(
            self.num_objectives, other.num_objectives, 1
        )
        similarities.append(obj_sim)

        # Complexity similarity
        complexity_sim = 1.0 - abs(self.problem_complexity - other.problem_complexity)
        similarities.append(complexity_sim)

        return statistics.mean(similarities)


class OptimizationHistory:
    """Manages optimization history storage and retrieval.

    Thread-safe: All mutable state access is protected by a lock.
    """

    def __init__(self, storage_path: str | None = None) -> None:
        """Initialize optimization history."""
        self.storage_path = storage_path or "optimization_history.jsonl"
        self._lock = threading.Lock()
        self.records: list[OptimizationRecord] = []
        self._load_history()

    def add_record(self, record: OptimizationRecord) -> None:
        """Add optimization record to history."""
        with self._lock:
            self.records.append(record)
            # Enforce memory limits
            if len(self.records) > MAX_OPTIMIZATION_HISTORY_SIZE:
                items_to_keep = int(MAX_OPTIMIZATION_HISTORY_SIZE * (1 - HISTORY_PRUNE_RATIO))
                self.records = self.records[-items_to_keep:]
                logger.debug(
                    f"Pruned optimization history to stay within {MAX_OPTIMIZATION_HISTORY_SIZE} limit"
                )
        self._save_record(record)
        logger.info(f"Added optimization record: {record.optimization_id}")

    def get_records(
        self,
        function_name: str | None = None,
        algorithm: AlgorithmType | None = None,
        status: OptimizationStatus | None = None,
        days_back: int | None = None,
    ) -> list[OptimizationRecord]:
        """Get filtered optimization records."""
        with self._lock:
            filtered_records = list(self.records)

        if function_name:
            filtered_records = [
                r for r in filtered_records if r.function_name == function_name
            ]

        if algorithm:
            filtered_records = [r for r in filtered_records if r.algorithm == algorithm]

        if status:
            filtered_records = [r for r in filtered_records if r.status == status]

        if days_back:
            cutoff_date = datetime.utcnow() - timedelta(days=days_back)
            filtered_records = [
                r for r in filtered_records if r.timestamp >= cutoff_date
            ]

        return filtered_records

    def get_problem_characteristics(
        self, function_name: str
    ) -> ProblemCharacteristics | None:
        """Extract problem characteristics from historical data."""
        records = self.get_records(
            function_name=function_name, status=OptimizationStatus.COMPLETED
        )

        if not records:
            return None

        # Use most recent record as representative
        latest_record = max(records, key=lambda r: r.timestamp)

        # Analyze configuration space
        config_space = latest_record.configuration_space
        config_size = len(config_space)

        param_types = {}
        for param, value in config_space.items():
            if isinstance(value, list):
                param_types[param] = "categorical"
            elif isinstance(value, tuple) and len(value) == 2:
                param_types[param] = "continuous"
            else:
                param_types[param] = "unknown"

        # Estimate complexity based on historical performance
        avg_duration = statistics.mean([r.duration_seconds for r in records])
        avg_trials = statistics.mean([r.total_trials for r in records])
        complexity = min(
            1.0, (avg_duration * avg_trials) / 10000
        )  # Normalized complexity score

        return ProblemCharacteristics(
            config_space_size=config_size,
            param_types=param_types,
            num_objectives=len(latest_record.objectives),
            problem_complexity=complexity,
            estimated_evaluation_time=avg_duration / max(avg_trials, 1),
        )

    def _load_history(self) -> None:
        """Load optimization history from storage."""
        try:
            with open(self.storage_path) as f:
                for line in f:
                    if line.strip():
                        record_data = json.loads(line)
                        record = OptimizationRecord.from_dict(record_data)
                        self.records.append(record)
            logger.info(f"Loaded {len(self.records)} optimization records")
        except FileNotFoundError:
            logger.info("No existing optimization history found")
        except Exception as e:
            logger.error(f"Error loading optimization history: {e}")

    def _save_record(self, record: OptimizationRecord) -> None:
        """Save single record to storage."""
        try:
            with open(self.storage_path, "a") as f:
                f.write(json.dumps(record.to_dict()) + "\n")
        except Exception as e:
            logger.error(f"Error saving optimization record: {e}")


class AlgorithmSelector:
    """Selects optimal algorithms based on problem characteristics and history."""

    def __init__(self, history: OptimizationHistory) -> None:
        """Initialize algorithm selector."""
        self.history = history
        self.algorithm_performance = self._analyze_algorithm_performance()

    def recommend_algorithm(
        self,
        function_name: str,
        configuration_space: dict[str, Any],
        objectives: list[str],
        budget_seconds: float | None = None,
    ) -> tuple[AlgorithmType, float]:
        """Recommend best algorithm with confidence score."""

        # Get problem characteristics
        problem_chars = self._extract_problem_characteristics(
            function_name, configuration_space, objectives
        )

        # Find similar problems
        similar_problems = self._find_similar_problems(problem_chars)

        if not similar_problems:
            # No historical data, use heuristics
            return self._heuristic_selection(problem_chars, budget_seconds)

        # Analyze performance of algorithms on similar problems
        algorithm_scores = defaultdict(list)

        for record in similar_problems:
            if record.status == OptimizationStatus.COMPLETED:
                # Score based on performance and efficiency
                efficiency_score = record.best_score / max(record.duration_seconds, 1)
                algorithm_scores[record.algorithm].append(efficiency_score)

        # Calculate average performance for each algorithm
        algorithm_performance = {}
        for algorithm, scores in algorithm_scores.items():
            if scores:
                algorithm_performance[algorithm] = statistics.mean(scores)

        if not algorithm_performance:
            return self._heuristic_selection(problem_chars, budget_seconds)

        # Select best performing algorithm
        best_algorithm = max(algorithm_performance.items(), key=lambda x: x[1])

        # Calculate confidence based on number of similar cases
        total_similar = len(similar_problems)
        confidence = min(
            1.0, total_similar / 10
        )  # Full confidence with 10+ similar cases

        logger.info(
            f"Recommended {best_algorithm[0].value} with confidence {confidence:.2f}"
        )
        return best_algorithm[0], confidence

    def get_algorithm_rankings(
        self,
        function_name: str,
        configuration_space: dict[str, Any],
        objectives: list[str],
    ) -> list[tuple[AlgorithmType, float, str]]:
        """Get ranked list of algorithms with scores and reasons."""

        problem_chars = self._extract_problem_characteristics(
            function_name, configuration_space, objectives
        )

        similar_problems = self._find_similar_problems(problem_chars)
        rankings = []

        for algorithm in AlgorithmType:
            score, reason = self._score_algorithm(
                algorithm, problem_chars, similar_problems
            )
            rankings.append((algorithm, score, reason))

        # Sort by score descending
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings

    def _extract_problem_characteristics(
        self,
        function_name: str,
        configuration_space: dict[str, Any],
        objectives: list[str],
    ) -> ProblemCharacteristics:
        """Extract characteristics from problem definition."""
        config_size = len(configuration_space)

        param_types = {}
        for param, value in configuration_space.items():
            if isinstance(value, list):
                param_types[param] = "categorical"
            elif isinstance(value, tuple) and len(value) == 2:
                param_types[param] = "continuous"
            else:
                param_types[param] = "unknown"

        # Estimate complexity (simplified)
        complexity = min(1.0, config_size / 20)  # Normalize to 0-1

        return ProblemCharacteristics(
            config_space_size=config_size,
            param_types=param_types,
            num_objectives=len(objectives),
            problem_complexity=complexity,
            estimated_evaluation_time=1.0,  # Default estimate
        )

    def _find_similar_problems(
        self, target_chars: ProblemCharacteristics, similarity_threshold: float = 0.6
    ) -> list[OptimizationRecord]:
        """Find similar problems from history."""
        similar_records = []

        for record in self.history.records:
            record_chars = self.history.get_problem_characteristics(
                record.function_name
            )
            if (
                record_chars
                and record_chars.similarity(target_chars) >= similarity_threshold
            ):
                similar_records.append(record)

        return similar_records

    def _heuristic_selection(
        self, problem_chars: ProblemCharacteristics, budget_seconds: float | None
    ) -> tuple[AlgorithmType, float]:
        """Select algorithm using heuristics when no historical data available."""

        # Simple heuristics based on problem characteristics
        if problem_chars.config_space_size <= 5:
            return AlgorithmType.GRID, 0.7  # Small spaces suit grid search
        elif problem_chars.config_space_size <= 20:
            if budget_seconds and budget_seconds < 300:  # 5 minutes
                return AlgorithmType.RANDOM, 0.6
            else:
                return AlgorithmType.BAYESIAN, 0.8
        else:
            # Large spaces
            if any("categorical" in t for t in problem_chars.param_types.values()):
                return AlgorithmType.GENETIC, 0.6
            else:
                return AlgorithmType.BAYESIAN, 0.7

    def _score_algorithm(
        self,
        algorithm: AlgorithmType,
        problem_chars: ProblemCharacteristics,
        similar_problems: list[OptimizationRecord],
    ) -> tuple[float, str]:
        """Score algorithm for given problem characteristics."""

        # Base score from historical performance
        historical_scores = []
        for record in similar_problems:
            if (
                record.algorithm == algorithm
                and record.status == OptimizationStatus.COMPLETED
            ):
                efficiency = record.best_score / max(record.duration_seconds, 1)
                historical_scores.append(efficiency)

        if historical_scores:
            base_score = statistics.mean(historical_scores)
            reason = f"Based on {len(historical_scores)} similar cases"
        else:
            # Use heuristic scoring
            base_score = self._heuristic_score(algorithm, problem_chars)
            reason = "Based on problem characteristics (no historical data)"

        return base_score, reason

    def _heuristic_score(
        self, algorithm: AlgorithmType, problem_chars: ProblemCharacteristics
    ) -> float:
        """Heuristic scoring when no historical data available."""

        config_size = problem_chars.config_space_size
        complexity = problem_chars.problem_complexity

        if algorithm == AlgorithmType.GRID:
            # Good for small spaces, bad for large ones
            return max(0.1, 1.0 - (config_size / 10))

        elif algorithm == AlgorithmType.RANDOM:
            # Decent baseline for most problems
            return 0.5

        elif algorithm == AlgorithmType.BAYESIAN:
            # Good for medium to complex problems
            return 0.3 + 0.5 * complexity

        elif algorithm == AlgorithmType.GENETIC:
            # Good for large, complex spaces
            return min(0.9, 0.2 + 0.6 * complexity + 0.2 * (config_size / 20))

        else:
            return 0.4  # Default score for other algorithms

    def _analyze_algorithm_performance(self) -> dict[AlgorithmType, dict[str, float]]:
        """Analyze overall algorithm performance across all problems."""
        performance: dict[AlgorithmType, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

        for record in self.history.records:
            if record.status == OptimizationStatus.COMPLETED:
                alg = record.algorithm
                performance[alg]["scores"].append(record.best_score)
                performance[alg]["durations"].append(record.duration_seconds)
                performance[alg]["trials"].append(record.total_trials)

        # Calculate statistics
        stats = {}
        for algorithm, metrics in performance.items():
            if metrics["scores"]:
                stats[algorithm] = {
                    "avg_score": statistics.mean(metrics["scores"]),
                    "avg_duration": statistics.mean(metrics["durations"]),
                    "avg_trials": statistics.mean(metrics["trials"]),
                    "total_runs": len(metrics["scores"]),
                }

        return stats


class PerformancePredictor:
    """Predicts optimization performance based on historical data."""

    def __init__(self, history: OptimizationHistory) -> None:
        """Initialize performance predictor."""
        self.history = history

    def predict_optimization_outcome(
        self,
        function_name: str,
        algorithm: AlgorithmType,
        configuration_space: dict[str, Any],
        max_trials: int,
    ) -> dict[str, Any]:
        """Predict optimization outcome."""

        # Find similar historical optimizations
        similar_records = []
        target_chars = self._get_problem_characteristics(
            function_name, configuration_space
        )

        for record in self.history.records:
            if (
                record.algorithm == algorithm
                and record.status == OptimizationStatus.COMPLETED
            ):
                record_chars = self.history.get_problem_characteristics(
                    record.function_name
                )
                if record_chars and record_chars.similarity(target_chars) > 0.5:
                    similar_records.append(record)

        if not similar_records:
            return self._default_prediction(max_trials)

        # Analyze similar records
        scores = [r.best_score for r in similar_records]
        durations = [r.duration_seconds for r in similar_records]
        trial_counts = [r.total_trials for r in similar_records]

        # Scale predictions based on max_trials
        avg_trials = statistics.mean(trial_counts)
        trial_scale_factor = max_trials / max(avg_trials, 1)

        predicted_score = statistics.mean(scores)
        predicted_duration = statistics.mean(durations) * trial_scale_factor

        # Calculate confidence intervals
        if len(scores) > 1:
            score_std = statistics.stdev(scores)
            duration_std = statistics.stdev(durations)
        else:
            score_std = predicted_score * 0.1
            duration_std = predicted_duration * 0.2

        return {
            "predicted_best_score": predicted_score,
            "score_confidence_interval": (
                predicted_score - score_std,
                predicted_score + score_std,
            ),
            "predicted_duration_seconds": predicted_duration,
            "duration_confidence_interval": (
                max(0, predicted_duration - duration_std),
                predicted_duration + duration_std,
            ),
            "success_probability": len(similar_records)
            / max(len(similar_records) + 1, 10),
            "based_on_cases": len(similar_records),
        }

    def estimate_convergence_curve(
        self, function_name: str, algorithm: AlgorithmType, max_trials: int
    ) -> list[tuple[int, float]]:
        """Estimate convergence curve (trial_number, expected_score)."""

        # Find similar optimizations
        similar_records = self.history.get_records(
            function_name=function_name,
            algorithm=algorithm,
            status=OptimizationStatus.COMPLETED,
        )

        if not similar_records:
            # Default convergence curve
            return [
                (i, 0.5 + 0.4 * (1 - 1 / ((i / 10) + 1)))
                for i in range(1, max_trials + 1)
            ]

        # Analyze convergence patterns (simplified)
        avg_final_score = statistics.mean([r.best_score for r in similar_records])
        avg_trials = statistics.mean([r.total_trials for r in similar_records])
        if avg_trials <= 0:
            avg_trials = 1.0

        # Generate expected convergence curve
        curve = []
        for trial in range(1, max_trials + 1):
            # Simple convergence model: logarithmic improvement
            progress = min(1.0, trial / avg_trials)
            expected_score = avg_final_score * (
                0.3 + 0.7 * (1 - 1 / (progress * 5 + 1))
            )
            curve.append((trial, expected_score))

        return curve

    def _get_problem_characteristics(
        self, function_name: str, configuration_space: dict[str, Any]
    ) -> ProblemCharacteristics:
        """Get problem characteristics for similarity comparison."""
        config_size = len(configuration_space)

        param_types = {}
        for param, value in configuration_space.items():
            if isinstance(value, list):
                param_types[param] = "categorical"
            elif isinstance(value, tuple):
                param_types[param] = "continuous"
            else:
                param_types[param] = "unknown"

        return ProblemCharacteristics(
            config_space_size=config_size,
            param_types=param_types,
            num_objectives=1,  # Default
            problem_complexity=min(1.0, config_size / 20),
            estimated_evaluation_time=1.0,
        )

    def _default_prediction(self, max_trials: int) -> dict[str, Any]:
        """Default prediction when no historical data available."""
        return {
            "predicted_best_score": 0.5,
            "score_confidence_interval": (0.3, 0.7),
            "predicted_duration_seconds": max_trials * 2.0,
            "duration_confidence_interval": (max_trials * 1.0, max_trials * 4.0),
            "success_probability": 0.5,
            "based_on_cases": 0,
        }


class MetaLearningEngine:
    """Main meta-learning engine that coordinates all components."""

    def __init__(self, storage_path: str | None = None) -> None:
        """Initialize meta-learning engine."""
        self.history = OptimizationHistory(storage_path)
        self.algorithm_selector = AlgorithmSelector(self.history)
        self.performance_predictor = PerformancePredictor(self.history)
        logger.info("MetaLearningEngine initialized")

    def record_optimization(
        self,
        optimization_id: str,
        function_name: str,
        algorithm: str,
        configuration_space: dict[str, Any],
        objectives: list[str],
        dataset_size: int,
        best_score: float,
        total_trials: int,
        duration_seconds: float,
        status: str = "completed",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record optimization result for meta-learning."""

        try:
            algorithm_enum = AlgorithmType(algorithm.lower())
        except ValueError:
            logger.warning(f"Unknown algorithm type: {algorithm}, using RANDOM")
            algorithm_enum = AlgorithmType.RANDOM

        try:
            status_enum = OptimizationStatus(status.lower())
        except ValueError:
            logger.warning(f"Unknown status: {status}, using COMPLETED")
            status_enum = OptimizationStatus.COMPLETED

        record = OptimizationRecord(
            optimization_id=optimization_id,
            function_name=function_name,
            algorithm=algorithm_enum,
            configuration_space=configuration_space,
            objectives=objectives,
            dataset_size=dataset_size,
            best_score=best_score,
            total_trials=total_trials,
            duration_seconds=duration_seconds,
            status=status_enum,
            timestamp=datetime.utcnow(),
            metadata=metadata or {},
        )

        self.history.add_record(record)

    def get_optimization_recommendations(
        self,
        function_name: str,
        configuration_space: dict[str, Any],
        objectives: list[str],
        budget_seconds: float | None = None,
        max_trials: int | None = None,
    ) -> dict[str, Any]:
        """Get comprehensive optimization recommendations."""

        # Get algorithm recommendation
        recommended_algorithm, confidence = self.algorithm_selector.recommend_algorithm(
            function_name, configuration_space, objectives, budget_seconds
        )

        # Get all algorithm rankings
        algorithm_rankings = self.algorithm_selector.get_algorithm_rankings(
            function_name, configuration_space, objectives
        )

        # Get performance prediction
        max_trials = max_trials or 50
        performance_prediction = (
            self.performance_predictor.predict_optimization_outcome(
                function_name, recommended_algorithm, configuration_space, max_trials
            )
        )

        # Get convergence curve
        convergence_curve = self.performance_predictor.estimate_convergence_curve(
            function_name, recommended_algorithm, max_trials
        )

        return {
            "recommended_algorithm": recommended_algorithm.value,
            "recommendation_confidence": confidence,
            "algorithm_rankings": [
                {"algorithm": alg.value, "score": score, "reason": reason}
                for alg, score, reason in algorithm_rankings
            ],
            "performance_prediction": performance_prediction,
            "convergence_curve": convergence_curve,
            "historical_data_points": len(self.history.records),
            "similar_problems_found": len(
                self.algorithm_selector._find_similar_problems(
                    self.algorithm_selector._extract_problem_characteristics(
                        function_name, configuration_space, objectives
                    )
                )
            ),
        }

    def get_insights(self, function_name: str | None = None) -> dict[str, Any]:
        """Get optimization insights and patterns."""

        records = self.history.get_records(function_name=function_name)

        if not records:
            return {"message": "No optimization history available"}

        # Algorithm performance analysis
        algorithm_stats = defaultdict(list)
        for record in records:
            if record.status == OptimizationStatus.COMPLETED:
                algorithm_stats[record.algorithm].append(
                    {
                        "score": record.best_score,
                        "duration": record.duration_seconds,
                        "trials": record.total_trials,
                    }
                )

        algorithm_performance = {}
        for algorithm, stats in algorithm_stats.items():
            if stats:
                algorithm_performance[algorithm.value] = {
                    "average_score": statistics.mean([s["score"] for s in stats]),
                    "average_duration": statistics.mean([s["duration"] for s in stats]),
                    "success_rate": len(stats)
                    / len([r for r in records if r.algorithm == algorithm]),
                    "total_runs": len(stats),
                }

        # Trend analysis
        recent_records = sorted(records, key=lambda r: r.timestamp)[-10:]
        if len(recent_records) >= 2:
            recent_scores = [
                r.best_score
                for r in recent_records
                if r.status == OptimizationStatus.COMPLETED
            ]
            if len(recent_scores) >= 2:
                trend = (
                    "improving" if recent_scores[-1] > recent_scores[0] else "declining"
                )
            else:
                trend = "insufficient_data"
        else:
            trend = "insufficient_data"

        return {
            "total_optimizations": len(records),
            "successful_optimizations": len(
                [r for r in records if r.status == OptimizationStatus.COMPLETED]
            ),
            "algorithm_performance": algorithm_performance,
            "recent_trend": trend,
            "most_common_algorithm": (
                Counter([r.algorithm.value for r in records]).most_common(1)[0]
                if records
                else None
            ),
            "average_optimization_time": statistics.mean(
                [r.duration_seconds for r in records]
            ),
            "functions_optimized": len({r.function_name for r in records}),
        }
