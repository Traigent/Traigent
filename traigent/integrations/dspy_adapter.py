"""DSPy integration adapter for prompt optimization.

This module provides an adapter for DSPy's prompt optimization capabilities,
allowing Traigent users to leverage DSPy's MIPRO and BootstrapFewShot optimizers
for automatic prompt engineering.

Example:
    >>> from traigent.integrations.dspy_adapter import DSPyPromptOptimizer  # doctest: +SKIP
    >>> optimizer = DSPyPromptOptimizer(method="mipro")  # doctest: +SKIP
    >>> optimized_module = optimizer.optimize_prompt(  # doctest: +SKIP
    ...     module=my_dspy_module,
    ...     trainset=train_examples,
    ...     metric=accuracy_metric,
    ... )
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Check DSPy availability
try:
    import dspy

    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    dspy = None  # type: ignore[assignment]


@dataclass
class PromptOptimizationResult:
    """Result of DSPy prompt optimization.

    Attributes:
        optimized_module: The optimized DSPy module with tuned prompts
        method: The optimization method used
        num_demos: Number of demonstrations in the optimized prompt
        trainset_size: Size of the training set used
        best_score: Best metric score achieved during optimization
        metadata: Additional metadata from the optimization process
    """

    optimized_module: Any
    method: str
    num_demos: int = 0
    trainset_size: int = 0
    best_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class DSPyPromptOptimizer:
    """Adapter for DSPy prompt optimization.

    Wraps DSPy's MIPROv2 and BootstrapFewShot optimizers for use with Traigent.
    Enables automatic prompt engineering as part of the optimization workflow.

    Example:
        >>> optimizer = DSPyPromptOptimizer(method="mipro")  # doctest: +SKIP
        >>> result = optimizer.optimize_prompt(  # doctest: +SKIP
        ...     module=qa_module,
        ...     trainset=train_data,
        ...     metric=exact_match,
        ... )
        >>> optimized_qa = result.optimized_module  # doctest: +SKIP
    """

    def __init__(
        self,
        method: Literal["mipro", "bootstrap"] = "mipro",
        *,
        teacher_model: str | None = None,
        auto_setting: Literal["light", "medium", "heavy"] = "medium",
    ):
        """Initialize the DSPy prompt optimizer.

        Args:
            method: Optimization method - "mipro" for MIPROv2 or "bootstrap"
                   for BootstrapFewShot
            teacher_model: Optional teacher model for generating demonstrations.
                          If not specified, uses the default DSPy LM.
            auto_setting: Auto setting for MIPRO - "light", "medium", or "heavy"
                         Controls the thoroughness of instruction optimization.

        Raises:
            ImportError: If DSPy is not installed
        """
        if not DSPY_AVAILABLE:
            raise ImportError(
                "DSPy is required for prompt optimization. "
                "Install with: pip install dspy-ai"
            )

        self.method = method
        self.teacher_model = teacher_model
        self.auto_setting = auto_setting

    def optimize_prompt(
        self,
        module: Any,
        trainset: list[Any],
        metric: Callable[[Any, Any], float],
        *,
        max_bootstrapped_demos: int = 4,
        max_labeled_demos: int = 16,
        num_candidates: int = 10,
        requires_permission_to_run: bool = False,
    ) -> PromptOptimizationResult:
        """Optimize a DSPy module's prompts.

        Uses either MIPROv2 or BootstrapFewShot to optimize the prompts
        in a DSPy module based on the provided training data and metric.

        Args:
            module: DSPy module to optimize
            trainset: Training examples as a list of dspy.Example objects
            metric: Evaluation metric function(example, prediction) -> float
            max_bootstrapped_demos: Maximum bootstrapped demonstrations (bootstrap only)
            max_labeled_demos: Maximum labeled demonstrations (bootstrap only)
            num_candidates: Number of candidate prompts to evaluate (mipro only)
            requires_permission_to_run: Whether to require user confirmation

        Returns:
            PromptOptimizationResult containing the optimized module and metadata

        Example:
            >>> def accuracy(example, pred):  # doctest: +SKIP
            ...     return float(example.answer == pred.answer)
            >>> result = optimizer.optimize_prompt(  # doctest: +SKIP
            ...     module=QAModule(),
            ...     trainset=examples,
            ...     metric=accuracy,
            ... )
        """
        if not DSPY_AVAILABLE:
            raise ImportError("DSPy is not available")

        # Set up teacher model if specified
        if self.teacher_model:
            teacher = dspy.LM(self.teacher_model)
        else:
            teacher = None

        if self.method == "mipro":
            optimized, metadata = self._run_mipro(
                module=module,
                trainset=trainset,
                metric=metric,
                num_candidates=num_candidates,
                requires_permission_to_run=requires_permission_to_run,
                teacher=teacher,
            )
        else:
            optimized, metadata = self._run_bootstrap(
                module=module,
                trainset=trainset,
                metric=metric,
                max_bootstrapped_demos=max_bootstrapped_demos,
                max_labeled_demos=max_labeled_demos,
                teacher=teacher,
            )

        # Count demonstrations in optimized module
        num_demos = self._count_demos(optimized)

        return PromptOptimizationResult(
            optimized_module=optimized,
            method=self.method,
            num_demos=num_demos,
            trainset_size=len(trainset),
            best_score=metadata.get("best_score"),
            metadata=metadata,
        )

    def _compile_with_teacher(
        self,
        optimizer: Any,
        module: Any,
        trainset: list[Any],
        teacher: Any | None,
    ) -> Any:
        """Compile module with optional teacher context.

        Helper to reduce repetition when compiling with/without teacher model.
        """
        if teacher:
            with dspy.context(lm=teacher):
                return optimizer.compile(module, trainset=trainset)
        return optimizer.compile(module, trainset=trainset)

    def _run_mipro(
        self,
        module: Any,
        trainset: list[Any],
        metric: Callable[[Any, Any], float],
        num_candidates: int,
        requires_permission_to_run: bool,
        teacher: Any | None,
    ) -> tuple[Any, dict[str, Any]]:
        """Run MIPROv2 optimization."""
        optimizer = dspy.MIPROv2(
            metric=metric,
            auto=self.auto_setting,
            num_candidates=num_candidates,
            requires_permission_to_run=requires_permission_to_run,
        )

        optimized = self._compile_with_teacher(optimizer, module, trainset, teacher)
        best_score = self._compute_best_score(optimized, trainset, metric)

        return optimized, {
            "method": "mipro",
            "auto_setting": self.auto_setting,
            "num_candidates": num_candidates,
            "best_score": best_score,
        }

    def _run_bootstrap(
        self,
        module: Any,
        trainset: list[Any],
        metric: Callable[[Any, Any], float],
        max_bootstrapped_demos: int,
        max_labeled_demos: int,
        teacher: Any | None,
    ) -> tuple[Any, dict[str, Any]]:
        """Run BootstrapFewShot optimization."""
        optimizer = dspy.BootstrapFewShot(
            metric=metric,
            max_bootstrapped_demos=max_bootstrapped_demos,
            max_labeled_demos=max_labeled_demos,
        )

        optimized = self._compile_with_teacher(optimizer, module, trainset, teacher)
        best_score = self._compute_best_score(optimized, trainset, metric)

        return optimized, {
            "method": "bootstrap",
            "max_bootstrapped_demos": max_bootstrapped_demos,
            "max_labeled_demos": max_labeled_demos,
            "best_score": best_score,
        }

    def _compute_best_score(
        self,
        optimized_module: Any,
        trainset: list[Any],
        metric: Callable[[Any, Any], float],
    ) -> float | None:
        """Compute best score by evaluating optimized module on trainset.

        Args:
            optimized_module: The optimized DSPy module
            trainset: Training examples to evaluate on
            metric: The metric function to use

        Returns:
            Average metric score, or None if evaluation fails
        """
        if not trainset:
            return None

        try:
            scores = []
            for example in trainset:
                # Get prediction from optimized module
                # DSPy modules typically take keyword args matching input fields
                inputs = {k: v for k, v in example.items() if k in example.inputs()}
                prediction = optimized_module(**inputs)
                score = metric(example, prediction)
                scores.append(score)

            return sum(scores) / len(scores) if scores else None
        except Exception as e:
            logger.debug(f"Could not compute best_score: {e}")
            return None

    def _count_demos(self, module: Any) -> int:
        """Count the number of demonstrations in an optimized module."""
        count = 0
        try:
            # Try to access demos from the module
            for attr_name in dir(module):
                attr = getattr(module, attr_name, None)
                if hasattr(attr, "demos"):
                    demos = getattr(attr, "demos", [])
                    if isinstance(demos, list):
                        count += len(demos)
        except Exception:
            pass
        return count

    @classmethod
    def create_prompt_choices(
        cls,
        base_prompts: list[str],
        trainset: list[Any] | None = None,
        metric: Callable[[Any, Any], float] | None = None,
        *,
        method: Literal["mipro", "bootstrap"] = "bootstrap",
        name: str = "prompt",
        return_choices: bool = True,
    ) -> Any:
        """Generate prompt choices for Traigent configuration space.

        This method creates a Choices parameter range from a list of prompt
        templates that can be used directly in Traigent's @optimize decorator.

        When DSPy is available and trainset/metric are provided, this method
        can optionally optimize the prompts first (planned feature).

        Args:
            base_prompts: List of prompt templates to use as choices
            trainset: Optional training examples for future optimization
            metric: Optional evaluation metric for future optimization
            method: Optimization method when optimization is enabled
            name: Name for the Choices parameter (default: "prompt")
            return_choices: If True, returns a Choices object; if False,
                returns a plain list

        Returns:
            Choices object or list of prompt strings

        Example:
            >>> from traigent.integrations.dspy_adapter import DSPyPromptOptimizer
            >>> import traigent
            >>>
            >>> # Use as Choices in configuration space
            >>> @traigent.optimize(
            ...     prompt=DSPyPromptOptimizer.create_prompt_choices([
            ...         "You are a helpful assistant. Answer concisely.",
            ...         "You are an expert. Provide detailed answers.",
            ...         "Think step by step before answering.",
            ...     ]),
            ...     objectives=["accuracy"],
            ... )
            ... def qa_agent(question: str) -> str:
            ...     prompt = traigent.get_config()["prompt"]
            ...     return llm(prompt + "\\n" + question)
        """
        from traigent.api.parameter_ranges import Choices

        if not base_prompts:
            raise ValueError("base_prompts must not be empty")

        # Log DSPy availability for optimization
        if trainset is not None and metric is not None:
            if DSPY_AVAILABLE:
                # Future: Could optimize prompts here
                logger.info(
                    f"DSPy available for {method} optimization. "
                    "Prompt optimization during create_prompt_choices is planned "
                    "for a future release. Using base prompts as-is."
                )
            else:
                logger.debug(
                    "DSPy not available. Returning base prompts as choices. "
                    "Install with: pip install traigent[dspy]"
                )

        if return_choices:
            return Choices(base_prompts, name=name)
        return list(base_prompts)


def create_dspy_integration(
    method: Literal["mipro", "bootstrap"] = "mipro",
    **kwargs: Any,
) -> DSPyPromptOptimizer:
    """Factory function to create a DSPy prompt optimizer.

    Args:
        method: Optimization method - "mipro" or "bootstrap"
        **kwargs: Additional arguments passed to DSPyPromptOptimizer

    Returns:
        Configured DSPyPromptOptimizer instance

    Raises:
        ImportError: If DSPy is not installed

    Example:
        >>> optimizer = create_dspy_integration(method="mipro", auto_setting="light")  # doctest: +SKIP
    """
    return DSPyPromptOptimizer(method=method, **kwargs)


__all__ = [
    "DSPyPromptOptimizer",
    "PromptOptimizationResult",
    "create_dspy_integration",
    "DSPY_AVAILABLE",
]
