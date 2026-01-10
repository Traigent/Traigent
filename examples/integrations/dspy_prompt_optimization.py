"""DSPy Integration Example: Combining DSPy Prompt Optimization with Traigent.

This example demonstrates how to leverage DSPy's prompt optimization capabilities
within Traigent's hyperparameter optimization framework.

DSPy excels at:
- Automatic prompt/instruction optimization
- Few-shot example selection (bootstrapping)
- Programmatic LLM development

Traigent excels at:
- Zero-code-change hyperparameter optimization
- Multi-objective optimization (accuracy + cost + latency)
- Framework-agnostic parameter injection
- Enterprise deployment modes

Together, they provide best-of-both-worlds LLM optimization.

Requirements:
    pip install traigent[dspy]

Usage:
    # Set environment variables
    export OPENAI_API_KEY=your-key-here

    # Run the example
    python dspy_prompt_optimization.py
"""

from __future__ import annotations

from typing import Any

# Check for required dependencies
try:
    import dspy

    DSPY_INSTALLED = True
except ImportError:
    DSPY_INSTALLED = False
    print("DSPy not installed. Install with: pip install traigent[dspy]")

import traigent
from traigent import Choices, Range

# =============================================================================
# Example 1: Use DSPy to Pre-Optimize Prompts, Then Traigent for Hyperparameters
# =============================================================================


def example_1_dspy_then_traigent():
    """Two-stage optimization: DSPy optimizes prompts, Traigent optimizes infra."""

    if not DSPY_INSTALLED:
        print("Skipping Example 1: DSPy not installed")
        return

    from traigent.integrations import DSPyPromptOptimizer

    print("\n" + "=" * 60)
    print("Example 1: DSPy Prompt Optimization + Traigent Hyperparameters")
    print("=" * 60)

    # Stage 1: Define a DSPy module
    class QAModule(dspy.Module):
        """Simple QA module using chain-of-thought."""

        def __init__(self):
            super().__init__()
            self.predict = dspy.ChainOfThought("question -> answer")

        def forward(self, question: str) -> str:
            return self.predict(question=question).answer

    # Stage 1: Optimize prompts with DSPy
    print("\nStage 1: Optimizing prompts with DSPy...")

    # Create sample training data
    trainset = [
        dspy.Example(question="What is 2+2?", answer="4").with_inputs("question"),
        dspy.Example(
            question="What is the capital of France?", answer="Paris"
        ).with_inputs("question"),
        dspy.Example(question="What color is the sky?", answer="Blue").with_inputs(
            "question"
        ),
    ]

    def accuracy_metric(example, pred):
        """Simple exact match metric."""
        return float(example.answer.lower() == pred.answer.lower())

    # Use Traigent's DSPy adapter
    optimizer = DSPyPromptOptimizer(method="bootstrap")  # bootstrap is faster for demo

    try:
        result = optimizer.optimize_prompt(
            module=QAModule(),
            trainset=trainset,
            metric=accuracy_metric,
            max_bootstrapped_demos=2,
        )
        optimized_module = result.optimized_module
        print(f"  DSPy optimization complete: {result.num_demos} demos selected")
    except Exception as e:
        print(f"  DSPy optimization skipped (requires LLM): {e}")
        optimized_module = QAModule()

    # Stage 2: Use the optimized module in a Traigent-decorated function
    print("\nStage 2: Optimizing hyperparameters with Traigent...")

    @traigent.optimize(
        objectives=["accuracy", "cost"],
        configuration_space={
            "model": Choices(["gpt-3.5-turbo", "gpt-4"]),
            "temperature": Range(0.0, 0.7),
        },
        execution={"execution_mode": "mock"},  # Use mock mode for demo
    )
    def answer_question(question: str) -> str:
        """QA function using DSPy-optimized module."""
        # The DSPy module has optimized prompts
        # Traigent injects optimal model/temperature
        return optimized_module(question)

    print("  Traigent decorator applied successfully")
    print("  Run: await answer_question.optimize() to start optimization")


# =============================================================================
# Example 2: DSPy Prompt Variants as Traigent Configuration Choices
# =============================================================================


def example_2_prompt_variants_as_choices():
    """Use DSPy to generate prompt variants, let Traigent search over them."""

    print("\n" + "=" * 60)
    print("Example 2: Prompt Variants as Configuration Choices")
    print("=" * 60)

    # Define multiple prompt strategies
    prompt_strategies = {
        "direct": "Answer the question: {question}",
        "cot": "Think step by step, then answer: {question}",
        "expert": "As an expert, provide a detailed answer: {question}",
    }

    @traigent.optimize(
        objectives=["accuracy", "cost", "latency"],
        configuration_space={
            # Traigent searches over prompt strategies
            "prompt_strategy": Choices(list(prompt_strategies.keys())),
            # Plus standard hyperparameters
            "model": Choices(["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]),
            "temperature": Range(0.0, 1.0),
            "max_tokens": Choices([256, 512, 1024]),
        },
        constraints=[
            # Cost constraint
            lambda cfg, metrics: metrics.get("cost", 0)
            <= 0.05,
        ],
        execution={"execution_mode": "mock"},
    )
    def smart_qa(
        question: str, *, traigent_config: dict[str, Any] | None = None
    ) -> str:
        """QA with searchable prompt strategies."""
        config = traigent_config or {}
        strategy = config.get("prompt_strategy", "direct")
        prompt_template = prompt_strategies[strategy]

        # In real usage, this would call an LLM
        _prompt = prompt_template.format(question=question)  # noqa: F841
        return f"[{strategy}] Answer to: {question}"

    print("  Configuration space includes prompt_strategy as searchable param")
    print(f"  Strategies: {list(prompt_strategies.keys())}")
    print("  Traigent will find optimal (strategy, model, temperature) combo")


# =============================================================================
# Example 3: Using the DSPy Adapter Directly
# =============================================================================


def example_3_adapter_direct_usage():
    """Direct usage of the DSPy adapter without Traigent decorator."""

    if not DSPY_INSTALLED:
        print("\nSkipping Example 3: DSPy not installed")
        return

    from traigent.integrations import (
        DSPY_AVAILABLE,
        create_dspy_integration,
    )

    print("\n" + "=" * 60)
    print("Example 3: Direct DSPy Adapter Usage")
    print("=" * 60)

    print(f"\n  DSPY_AVAILABLE: {DSPY_AVAILABLE}")

    # Factory function usage
    optimizer = create_dspy_integration(
        method="mipro",
        auto_setting="light",  # light/medium/heavy
    )

    print(f"  Optimizer method: {optimizer.method}")
    print(f"  Auto setting: {optimizer.auto_setting}")

    # Result dataclass
    print("\n  PromptOptimizationResult fields:")
    print("    - optimized_module: The DSPy module with tuned prompts")
    print("    - method: 'mipro' or 'bootstrap'")
    print("    - num_demos: Number of few-shot examples")
    print("    - trainset_size: Training set size used")
    print("    - best_score: Best metric achieved")
    print("    - metadata: Additional optimization info")


# =============================================================================
# Main
# =============================================================================


def main():
    """Run all examples."""
    print("=" * 60)
    print("DSPy + Traigent Integration Examples")
    print("=" * 60)

    print("\nThis example demonstrates three integration patterns:")
    print("1. Two-stage: DSPy optimizes prompts, Traigent optimizes hyperparams")
    print("2. Unified: Prompt variants as Traigent configuration choices")
    print("3. Direct: Using the DSPy adapter API directly")

    # Run examples
    example_1_dspy_then_traigent()
    example_2_prompt_variants_as_choices()
    example_3_adapter_direct_usage()

    print("\n" + "=" * 60)
    print("Examples Complete")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Install DSPy: pip install traigent[dspy]")
    print("  2. Set OPENAI_API_KEY environment variable")
    print("  3. Run actual optimization with real LLM calls")
    print("\nSee also:")
    print("  - traigent/integrations/dspy_adapter.py")
    print("  - https://dspy.ai for DSPy documentation")


if __name__ == "__main__":
    main()
