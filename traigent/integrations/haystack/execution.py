"""Pipeline execution with configuration for Haystack optimization.

This module provides functions to execute a Haystack pipeline with a specific
configuration against an evaluation dataset, collecting results for scoring.

Example usage:
    from traigent.integrations.haystack import (
        EvaluationDataset,
        execute_with_config,
        apply_config,
    )

    # Prepare dataset
    dataset = EvaluationDataset.from_dicts([
        {"input": {"query": "What is AI?"}, "expected": "Artificial Intelligence..."},
    ])

    # Execute with configuration
    config = {"generator.temperature": 0.8, "retriever.top_k": 10}
    result = execute_with_config(pipeline, config, dataset)

    if result.success:
        for output in result.outputs:
            print(output)
"""

from __future__ import annotations

import copy
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from traigent.integrations.haystack.evaluation import EvaluationDataset

if TYPE_CHECKING:
    pass  # For future type hints

logger = logging.getLogger(__name__)


@dataclass
class ExampleResult:
    """Result for a single evaluation example.

    Attributes:
        example_index: Index of this example in the dataset
        input: The input dict passed to the pipeline
        output: Pipeline output (None if failed)
        success: Whether execution succeeded
        error: Error message if failed
        execution_time: Time taken for this example in seconds
    """

    example_index: int
    input: dict[str, Any]
    output: Any | None
    success: bool
    error: str | None = None
    execution_time: float = 0.0

    def __repr__(self) -> str:
        """Return a human-readable representation."""
        status = "success" if self.success else f"failed: {self.error}"
        return f"ExampleResult(index={self.example_index}, {status})"


@dataclass
class RunResult:
    """Result of running a pipeline with a configuration against a dataset.

    RunResult captures all outputs from executing a pipeline with a specific
    configuration. It tracks per-example results, overall success, and timing.

    Attributes:
        config: The configuration used for this run
        example_results: Per-example results with outputs or errors
        success: True if all examples succeeded
        total_execution_time: Total time for the run in seconds
        error: Overall error message if run failed catastrophically
        stopped_early: True if execution was stopped early (e.g., constraint violation)
    """

    config: dict[str, Any]
    example_results: list[ExampleResult] = field(default_factory=list)
    success: bool = True
    total_execution_time: float = 0.0
    error: str | None = None
    stopped_early: bool = False

    @property
    def outputs(self) -> list[Any]:
        """Get all successful outputs.

        Returns:
            List of outputs from successful examples only.
        """
        return [r.output for r in self.example_results if r.success]

    @property
    def all_outputs(self) -> list[Any | None]:
        """Get all outputs including None for failed examples.

        Returns:
            List of outputs (None for failed examples).
        """
        return [r.output for r in self.example_results]

    @property
    def failed_count(self) -> int:
        """Count of failed examples."""
        return sum(1 for r in self.example_results if not r.success)

    @property
    def success_count(self) -> int:
        """Count of successful examples."""
        return sum(1 for r in self.example_results if r.success)

    @property
    def success_rate(self) -> float:
        """Fraction of successful examples.

        Returns:
            Float between 0.0 and 1.0, or 0.0 if no examples.
        """
        if not self.example_results:
            return 0.0
        return self.success_count / len(self.example_results)

    def __repr__(self) -> str:
        """Return a human-readable representation."""
        status = "success" if self.success else "failed"
        return (
            f"RunResult({status}, "
            f"examples={len(self.example_results)}, "
            f"success_rate={self.success_rate:.1%}, "
            f"time={self.total_execution_time:.2f}s)"
        )

    def __len__(self) -> int:
        """Return the number of example results."""
        return len(self.example_results)


def apply_config(pipeline: Any, config: dict[str, Any]) -> Any:
    """Apply a configuration to a pipeline.

    Modifies the pipeline's component parameters according to the config.
    Uses qualified names like "generator.temperature" to identify parameters.

    Args:
        pipeline: Haystack Pipeline instance
        config: Dict mapping qualified names (e.g., "generator.temperature") to values

    Returns:
        The same pipeline instance (mutated in place)

    Raises:
        KeyError: If a component doesn't exist in the pipeline
        KeyError: If a parameter doesn't exist on the component
        KeyError: If the qualified name format is invalid

    Example:
        >>> config = {"generator.temperature": 0.8, "retriever.top_k": 10}
        >>> apply_config(pipeline, config)
        >>> # pipeline.get_component("generator").temperature is now 0.8
    """
    for qualified_name, value in config.items():
        _apply_single_param(pipeline, qualified_name, value)

    return pipeline


def _apply_single_param(pipeline: Any, qualified_name: str, value: Any) -> None:
    """Apply a single parameter to a pipeline component.

    Args:
        pipeline: Haystack Pipeline instance
        qualified_name: Parameter name like "generator.temperature"
        value: Value to set

    Raises:
        KeyError: If component or parameter not found
    """
    parts = qualified_name.split(".", 1)
    if len(parts) != 2:
        raise KeyError(
            f"Invalid qualified name '{qualified_name}': "
            f"expected 'component.parameter' format (e.g., 'generator.temperature')"
        )

    component_name, param_name = parts

    # Get component from pipeline
    component = pipeline.get_component(component_name)
    if component is None:
        raise KeyError(
            f"Component '{component_name}' not found in pipeline. "
            f"Check that the component name matches a component in your pipeline."
        )

    # Set parameter on component
    if not hasattr(component, param_name):
        raise KeyError(
            f"Parameter '{param_name}' not found on component '{component_name}'. "
            f"Available attributes: {[a for a in dir(component) if not a.startswith('_')]}"
        )

    setattr(component, param_name, value)


EarlyStopCallback = Callable[[list["ExampleResult"]], bool]


def execute_with_config(
    pipeline: Any,
    config: dict[str, Any],
    dataset: EvaluationDataset,
    *,
    copy_pipeline: bool = True,
    abort_on_error: bool = False,
    early_stop_callback: EarlyStopCallback | None = None,
) -> RunResult:
    """Execute a pipeline with a configuration against an evaluation dataset.

    This is the core execution function for optimization. It applies a
    configuration to the pipeline and runs it against each example in the
    dataset, collecting outputs and handling errors.

    Args:
        pipeline: Haystack Pipeline instance
        config: Configuration dict mapping qualified names to values
        dataset: Evaluation dataset to run against
        copy_pipeline: If True (default), deep copy pipeline before modifying.
            Set to False if you want to mutate the original pipeline.
        abort_on_error: If True, stop execution on first error.
            If False (default), continue with remaining examples.
        early_stop_callback: Optional callback called after each example.
            Receives list of ExampleResult so far. Returns True to stop early.
            Use this for constraint-based early stopping.

    Returns:
        RunResult with outputs, success status, and timing information.

    Example:
        >>> dataset = EvaluationDataset.from_dicts([
        ...     {"input": {"query": "Q1"}, "expected": "A1"},
        ...     {"input": {"query": "Q2"}, "expected": "A2"},
        ... ])
        >>> config = {"generator.temperature": 0.8}
        >>> result = execute_with_config(pipeline, config, dataset)
        >>> if result.success:
        ...     print(f"Got {len(result.outputs)} outputs")
    """
    start_time = time.time()

    # Optionally copy pipeline to avoid mutating original
    if copy_pipeline:
        try:
            pipeline = copy.deepcopy(pipeline)
        except Exception as e:
            logger.warning(
                f"Could not deep copy pipeline: {e}. Using original instance."
            )

    # Apply configuration
    try:
        apply_config(pipeline, config)
    except KeyError as e:
        return RunResult(
            config=config,
            success=False,
            error=f"Configuration error: {e}",
            total_execution_time=time.time() - start_time,
        )

    # Run pipeline for each example
    example_results: list[ExampleResult] = []
    overall_success = True
    stopped_early = False

    for i, example in enumerate(dataset):
        example_result = _execute_single_example(pipeline, i, example.input)
        example_results.append(example_result)

        if not example_result.success:
            overall_success = False
            if abort_on_error:
                logger.info(f"Aborting execution after error at example {i}")
                break

        # Check early stop callback
        if early_stop_callback is not None and early_stop_callback(example_results):
            logger.info(f"Early stopping triggered after example {i}")
            stopped_early = True
            break

    return RunResult(
        config=config,
        example_results=example_results,
        success=overall_success,
        total_execution_time=time.time() - start_time,
        stopped_early=stopped_early,
    )


def _execute_single_example(
    pipeline: Any,
    index: int,
    input_data: dict[str, Any],
) -> ExampleResult:
    """Execute pipeline for a single example.

    Args:
        pipeline: Haystack Pipeline instance (already configured)
        index: Example index for tracking
        input_data: Input dict to pass to pipeline.run()

    Returns:
        ExampleResult with output or error
    """
    example_start = time.time()

    try:
        output = pipeline.run(**input_data)
        return ExampleResult(
            example_index=index,
            input=input_data,
            output=output,
            success=True,
            execution_time=time.time() - example_start,
        )
    except Exception as e:
        logger.warning(
            f"Pipeline execution failed for example {index}: {e}",
            exc_info=True,
        )
        return ExampleResult(
            example_index=index,
            input=input_data,
            output=None,
            success=False,
            error=str(e),
            execution_time=time.time() - example_start,
        )
