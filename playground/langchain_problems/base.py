"""
Base classes for LangChain optimization problems.

This module defines the interface that all optimization problems must implement,
providing a standardized way to define datasets, evaluation metrics, and optimization targets.
"""

import json
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from traigent.evaluators.base import Dataset


@dataclass
class ProblemMetric:
    """Definition of a problem-specific evaluation metric."""

    name: str
    description: str
    higher_is_better: bool = True
    weight: float = 1.0
    display_format: str = ".3f"
    unit: str = ""


@dataclass
class ProblemConfig:
    """Configuration for a specific optimization problem."""

    name: str
    description: str
    difficulty_level: str
    dataset_size: int
    model_configurations: Dict[str, Any]
    metrics: List[ProblemMetric]
    optimization_objectives: List[str]
    expected_model_ranking: List[str] = field(
        default_factory=list
    )  # Expected performance order


class ProblemDefinition(ABC):
    """
    Base class for all LangChain optimization problems.

    Each problem defines:
    1. A dataset generation method
    2. A function to be optimized
    3. Custom evaluation metrics
    4. Expected configuration space
    """

    def __init__(self, config: ProblemConfig):
        """Initialize problem with configuration."""
        self.config = config
        self._dataset_cache: Optional[Dataset] = None

    @property
    def name(self) -> str:
        """Get problem name."""
        return self.config.name

    @property
    def description(self) -> str:
        """Get problem description."""
        return self.config.description

    @abstractmethod
    def create_dataset(self) -> Dataset:
        """
        Create the evaluation dataset for this problem.

        Returns:
            Dataset with evaluation examples
        """
        pass

    @abstractmethod
    def create_function(self) -> Callable:
        """
        Create the function to be optimized.

        Returns:
            Function that will be wrapped with @traigent.optimize
        """
        pass

    @abstractmethod
    def create_optimized_function(self) -> Callable:
        """
        Create the optimized version of the function with Traigent decorator.

        Returns:
            Function decorated with @traigent.optimize
        """
        pass

    @abstractmethod
    def evaluate_custom_metrics(
        self,
        outputs: List[Any],
        expected_outputs: List[Any],
        errors: List[Optional[str]],
    ) -> Dict[str, float]:
        """
        Compute problem-specific evaluation metrics.

        Args:
            outputs: Actual outputs from function
            expected_outputs: Expected outputs
            errors: Error messages (None for successful evaluations)

        Returns:
            Dictionary of metric name to value
        """
        pass

    def get_dataset(self) -> Dataset:
        """Get dataset, creating and caching if needed."""
        if self._dataset_cache is None:
            self._dataset_cache = self.create_dataset()
        return self._dataset_cache

    def get_configuration_space(self) -> Dict[str, Any]:
        """Get the configuration space for optimization."""
        return self.config.model_configurations

    def get_metrics_info(self) -> List[ProblemMetric]:
        """Get information about problem-specific metrics."""
        return self.config.metrics

    def get_optimization_objectives(self) -> List[str]:
        """Get list of objectives for optimization."""
        return self.config.optimization_objectives

    def create_temporary_dataset_file(self, dataset: Optional[Dataset] = None) -> str:
        """Create temporary file for dataset."""
        if dataset is None:
            dataset = self.get_dataset()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for example in dataset.examples:
                json.dump(
                    {"input": example.input_data, "output": example.expected_output}, f
                )
                f.write("\n")
            return f.name

    def format_metric_value(self, metric_name: str, value: float) -> str:
        """Format metric value for display."""
        metric_info = next(
            (m for m in self.config.metrics if m.name == metric_name), None
        )
        if metric_info:
            formatted = f"{value:{metric_info.display_format}}"
            if metric_info.unit:
                formatted += f" {metric_info.unit}"
            return formatted
        return f"{value:.3f}"

    def print_problem_summary(self):
        """Print summary information about this problem."""
        print(f"🎯 Problem: {self.config.name}")
        print(f"📝 Description: {self.config.description}")
        print(f"🔥 Difficulty: {self.config.difficulty_level}")
        print(f"📊 Dataset size: {self.config.dataset_size} examples")
        print(f"📈 Metrics: {', '.join(m.name for m in self.config.metrics)}")
        print(f"🎯 Objectives: {', '.join(self.config.optimization_objectives)}")

        if self.config.expected_model_ranking:
            print(
                f"🏆 Expected ranking: {' > '.join(self.config.expected_model_ranking)}"
            )
        print()


class BaseLangChainProblem(ProblemDefinition):
    """
    Base class specifically for LangChain optimization problems.

    Provides common LangChain utilities and patterns.
    """

    def __init__(self, config: ProblemConfig):
        super().__init__(config)
        self.model_costs = {
            "gpt-3.5-turbo": 0.002,
            "gpt-4o-mini": 0.00075,
            "gpt-4o": 0.01,
            "gpt-4-turbo": 0.015,
        }

    def clean_llm_output(
        self, raw_output: str, valid_options: Optional[List[str]] = None
    ) -> str:
        """
        Clean and standardize LLM output.

        Args:
            raw_output: Raw output from LLM
            valid_options: List of valid output options for validation

        Returns:
            Cleaned output string
        """
        if not raw_output:
            return ""

        # Basic cleaning
        result = raw_output.strip().lower()

        # Remove common prefixes/suffixes
        prefixes_to_remove = [
            "answer:",
            "result:",
            "output:",
            "response:",
            "classification:",
            "category:",
            "the answer is",
            "the result is",
            "i would say",
            "in my opinion",
            "i think",
            "the classification is",
        ]

        for prefix in prefixes_to_remove:
            result = result.replace(prefix, "")
        result = result.strip()

        # Remove punctuation and extra whitespace
        import re

        result = re.sub(r"[^\w\s_-]", "", result)
        result = result.strip()

        # If valid options provided, try to find best match
        if valid_options:
            valid_lower = [opt.lower() for opt in valid_options]

            # Exact match
            if result in valid_lower:
                return valid_options[valid_lower.index(result)]

            # Partial match
            for i, option in enumerate(valid_lower):
                if option in result or result in option:
                    return valid_options[i]

        # Return first word if no specific match
        return result.split()[0] if result else ""

    def estimate_cost(self, model: str, num_tokens: int = 100) -> float:
        """Estimate cost for model and token count."""
        cost_per_1k = self.model_costs.get(model, 0.01)
        return (num_tokens / 1000.0) * cost_per_1k

    def get_standard_model_configurations(self) -> Dict[str, Any]:
        """Get standard model configuration space for LangChain problems."""
        return {
            "model": ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"],
            "temperature": [0.1, 0.7],
            "max_tokens": [100, 200],
        }

    def create_base_metrics(self) -> List[ProblemMetric]:
        """Create standard metrics that all LangChain problems should have."""
        return [
            ProblemMetric(
                name="accuracy",
                description="Exact match accuracy",
                higher_is_better=True,
                weight=1.0,
                display_format=".1%",
            ),
            ProblemMetric(
                name="success_rate",
                description="Rate of successful completions",
                higher_is_better=True,
                weight=0.5,
                display_format=".1%",
            ),
            ProblemMetric(
                name="avg_response_time",
                description="Average response time",
                higher_is_better=False,
                weight=0.3,
                display_format=".2f",
                unit="s",
            ),
        ]
