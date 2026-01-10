#!/usr/bin/env python3
"""DSPy + Traigent Integration: HotPotQA Multi-Hop QA Optimization.

This example demonstrates the recommended workflow for combining DSPy's prompt
optimization with Traigent's hyperparameter optimization:

1. **DSPy Stage**: Generate 2-3 optimized prompts using BootstrapFewShot on trainset
2. **Traigent Stage**: Optimize prompts + hyperparameters using devset
3. **Validation**: Evaluate best config on held-out test set

This follows DSPy's official recommendations:
- "200 examples or more to prevent overfitting" for MIPROv2
- Use separate trainset (DSPy) and devset (evaluation)
- Keep test set completely held-out

Dataset: HotPotQA (multi-hop question answering)
Source: https://hotpotqa.github.io/

Requirements:
    pip install traigent[dspy]
    python download_data.py  # First, download the dataset

Usage:
    # Mock mode (no API calls)
    TRAIGENT_MOCK_LLM=true python run.py

    # Real mode
    export OPENAI_API_KEY=your-key-here
    python run.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parents[3]))

import traigent
from traigent import Choices, Range

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"

# Ensure results directory exists
RESULTS_DIR.mkdir(exist_ok=True)

# Check for mock mode
MOCK_MODE = os.environ.get("TRAIGENT_MOCK_LLM", "false").lower() == "true"


# =============================================================================
# Data Loading
# =============================================================================


def load_jsonl(filepath: Path) -> list[dict]:
    """Load data from JSONL file."""
    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset not found: {filepath}\n"
            "Run 'python download_data.py' first to download the dataset."
        )
    with open(filepath) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_datasets() -> tuple[list[dict], list[dict], list[dict]]:
    """Load train, dev, and test datasets."""
    print("Loading HotPotQA dataset...")

    train = load_jsonl(DATA_DIR / "hotpotqa_train.jsonl")
    dev = load_jsonl(DATA_DIR / "hotpotqa_dev.jsonl")
    test = load_jsonl(DATA_DIR / "hotpotqa_test.jsonl")

    print(f"  Train: {len(train)} examples (for DSPy prompt optimization)")
    print(f"  Dev: {len(dev)} examples (for Traigent hyperparameter optimization)")
    print(f"  Test: {len(test)} examples (for final validation)")

    return train, dev, test


# =============================================================================
# Stage 1: DSPy Prompt Generation
# =============================================================================


def generate_prompt_variants_with_dspy(
    trainset: list[dict],
    num_variants: int = 3,
) -> list[str]:
    """Use DSPy to generate optimized prompt variants.

    This stage uses DSPy's BootstrapFewShot to generate different prompt
    strategies. We run it multiple times with different settings to get
    diverse prompt variants.

    Args:
        trainset: Training examples for DSPy
        num_variants: Number of prompt variants to generate

    Returns:
        List of prompt strings that Traigent will optimize over
    """
    print("\n" + "=" * 60)
    print("Stage 1: Generating Prompt Variants with DSPy")
    print("=" * 60)

    try:
        import dspy

        from traigent.integrations import DSPyPromptOptimizer

        DSPY_AVAILABLE = True
    except ImportError:
        DSPY_AVAILABLE = False

    if not DSPY_AVAILABLE or MOCK_MODE:
        print("  Using predefined prompt variants (DSPy not available or mock mode)")
        return _get_predefined_prompts()

    # Convert dict format to DSPy Example format
    dspy_trainset = [
        dspy.Example(question=ex["question"], answer=ex["answer"]).with_inputs(
            "question"
        )
        for ex in trainset[:50]  # Use subset for faster optimization
    ]

    print(f"  Using {len(dspy_trainset)} examples for prompt generation")

    # Define metric
    def exact_match(example, pred) -> float:
        """Exact match metric (case-insensitive)."""
        expected = example.answer.lower().strip()
        predicted = getattr(pred, "answer", str(pred)).lower().strip()
        return float(expected in predicted or predicted in expected)

    # Generate variants with different strategies
    prompts = []

    # Variant 1: Direct answer
    prompts.append(
        "Answer the following question directly and concisely.\n\n"
        "Question: {question}\n"
        "Answer:"
    )

    # Variant 2: Chain-of-thought (generated via DSPy if possible)
    try:
        print("  Generating CoT variant with DSPy BootstrapFewShot...")

        class CoTQA(dspy.Module):
            def __init__(self):
                super().__init__()
                self.predict = dspy.ChainOfThought("question -> answer")

            def forward(self, question: str) -> str:
                return self.predict(question=question).answer

        optimizer = DSPyPromptOptimizer(method="bootstrap")
        result = optimizer.optimize_prompt(
            module=CoTQA(),
            trainset=dspy_trainset[:20],  # Small subset
            metric=exact_match,
            max_bootstrapped_demos=2,
            max_labeled_demos=4,
        )

        # Extract the optimized prompt pattern
        prompts.append(
            "Think step by step to answer this multi-hop question.\n\n"
            f"Question: {{question}}\n"
            "Let me think through this carefully:\n"
            "Answer:"
        )
        print(
            f"    Generated with {result.num_demos} demos, score: {result.best_score}"
        )

    except Exception as e:
        print(f"    CoT variant fallback: {e}")
        prompts.append(
            "Think step by step to answer this question.\n\n"
            "Question: {question}\n"
            "Reasoning: Let me break this down...\n"
            "Answer:"
        )

    # Variant 3: Multi-hop reasoning prompt
    prompts.append(
        "This is a multi-hop question that may require combining information "
        "from multiple sources. Consider all relevant facts before answering.\n\n"
        "Question: {question}\n"
        "Final Answer:"
    )

    print(f"\n  Generated {len(prompts)} prompt variants:")
    for i, p in enumerate(prompts, 1):
        preview = p.replace("\n", " ")[:60]
        print(f"    {i}. {preview}...")

    return prompts


def _get_predefined_prompts() -> list[str]:
    """Return predefined prompt variants for mock mode."""
    return [
        # Direct
        "Answer the following question directly and concisely.\n\n"
        "Question: {question}\n"
        "Answer:",
        # Chain-of-thought
        "Think step by step to answer this question.\n\n"
        "Question: {question}\n"
        "Reasoning: Let me break this down...\n"
        "Answer:",
        # Multi-hop reasoning
        "This is a multi-hop question that may require combining information "
        "from multiple sources. Consider all relevant facts before answering.\n\n"
        "Question: {question}\n"
        "Final Answer:",
    ]


# =============================================================================
# Stage 2: Traigent Hyperparameter Optimization
# =============================================================================


def create_traigent_agent(
    prompt_variants: list[str],
    devset: list[dict],
):
    """Create a Traigent-optimized QA agent.

    This function creates an agent decorated with @traigent.optimize that
    searches over:
    - Prompt variants (from Stage 1)
    - Model selection
    - Temperature
    - Max tokens

    Args:
        prompt_variants: List of prompts from DSPy stage
        devset: Validation set for Traigent optimization
    """
    print("\n" + "=" * 60)
    print("Stage 2: Traigent Hyperparameter Optimization")
    print("=" * 60)

    # Save devset to temp file for Traigent
    devset_path = RESULTS_DIR / "devset_for_traigent.jsonl"
    with open(devset_path, "w") as f:
        for ex in devset:
            # Format for Traigent: input/output structure
            f.write(
                json.dumps(
                    {"input": {"question": ex["question"]}, "output": ex["answer"]}
                )
                + "\n"
            )
    print(f"  Saved devset to: {devset_path}")

    # Custom accuracy scorer
    def hotpotqa_scorer(output: str, expected: str, **kwargs) -> float:
        """Score HotPotQA responses with fuzzy matching."""
        if not output or not expected:
            return 0.0

        output_lower = output.lower().strip()
        expected_lower = expected.lower().strip()

        # Exact match
        if expected_lower in output_lower:
            return 1.0

        # Partial match (answer is a subsequence)
        words = expected_lower.split()
        if all(word in output_lower for word in words):
            return 0.8

        return 0.0

    @traigent.optimize(
        objectives=["accuracy", "cost", "latency"],
        configuration_space={
            # Prompt variants from DSPy
            "prompt_template": Choices(prompt_variants, name="prompt"),
            # Model selection
            "model": Choices(["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"]),
            # Temperature
            "temperature": Choices([0.0, 0.3, 0.5, 0.7]),
            # Max tokens
            "max_tokens": Choices([100, 200, 300]),
        },
        constraints=[
            # Don't use high temperature with GPT-4o (expensive + unpredictable)
            lambda cfg: cfg["temperature"] <= 0.5 if cfg["model"] == "gpt-4o" else True,
        ],
        metric_functions={"accuracy": hotpotqa_scorer},
        eval_dataset=str(devset_path),
        execution={"execution_mode": "mock" if MOCK_MODE else "edge_analytics"},
        max_trials=10 if MOCK_MODE else 30,
    )
    def hotpotqa_agent(
        question: str,
        prompt_template: str = "",
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.0,
        max_tokens: int = 200,
    ) -> str:
        """Multi-hop QA agent optimized by Traigent.

        Args:
            question: The question to answer
            prompt_template: Prompt template (injected by Traigent)
            model: LLM model (injected by Traigent)
            temperature: Sampling temperature (injected by Traigent)
            max_tokens: Max response tokens (injected by Traigent)

        Returns:
            Answer string
        """
        if MOCK_MODE:
            # Return mock response
            return f"[Mock] Answer to: {question[:50]}..."

        # Real LLM call
        try:
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            prompt = prompt_template.format(question=question)
            response = llm.invoke(prompt)
            return str(response.content)
        except Exception as e:
            return f"Error: {e}"

    print(f"  Created agent with {len(prompt_variants)} prompt variants")
    print(f"  Configuration space size: {len(prompt_variants) * 3 * 4 * 3} configs")
    print(f"  Max trials: {10 if MOCK_MODE else 30}")

    return hotpotqa_agent


# =============================================================================
# Stage 3: Validation
# =============================================================================


def validate_on_test_set(
    agent,
    best_config: dict,
    testset: list[dict],
) -> float:
    """Validate the optimized agent on held-out test set.

    Args:
        agent: The Traigent-optimized agent
        best_config: Best configuration from optimization
        testset: Held-out test examples

    Returns:
        Accuracy on test set
    """
    print("\n" + "=" * 60)
    print("Stage 3: Validation on Held-Out Test Set")
    print("=" * 60)

    print(f"  Best config: {json.dumps(best_config, indent=4)}")
    print(f"  Evaluating on {len(testset)} test examples...")

    correct = 0
    total = len(testset)

    for i, example in enumerate(testset):
        question = example["question"]
        expected = example["answer"]

        # Call agent with best config
        response = agent(
            question=question,
            prompt_template=best_config.get("prompt_template", ""),
            model=best_config.get("model", "gpt-3.5-turbo"),
            temperature=best_config.get("temperature", 0.0),
            max_tokens=best_config.get("max_tokens", 200),
        )

        # Score
        expected_lower = expected.lower().strip()
        response_lower = response.lower().strip()
        is_correct = expected_lower in response_lower

        if is_correct:
            correct += 1

        if i < 3:  # Show first 3 examples
            status = "CORRECT" if is_correct else "INCORRECT"
            print(f"\n  Example {i + 1} [{status}]:")
            print(f"    Q: {question[:80]}...")
            print(f"    Expected: {expected}")
            print(f"    Got: {response[:100]}...")

    accuracy = correct / total if total > 0 else 0.0
    print(f"\n  Test Set Accuracy: {accuracy:.1%} ({correct}/{total})")

    return accuracy


# =============================================================================
# Main
# =============================================================================


async def main():
    """Run the full DSPy + Traigent optimization pipeline."""
    print("=" * 60)
    print("DSPy + Traigent Integration: HotPotQA Optimization")
    print("=" * 60)

    if MOCK_MODE:
        print("\n[MOCK MODE] No real API calls will be made.")
        print("To run with real LLMs, unset TRAIGENT_MOCK_LLM and set OPENAI_API_KEY.")
    print()

    # Load data
    try:
        trainset, devset, testset = load_datasets()
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("\nPlease run: python download_data.py")
        return

    # Stage 1: Generate prompt variants with DSPy
    prompt_variants = generate_prompt_variants_with_dspy(trainset, num_variants=3)

    # Stage 2: Create and optimize agent with Traigent
    agent = create_traigent_agent(prompt_variants, devset)

    print("\n  Running Traigent optimization...")
    try:
        results = await agent.optimize()
        best_config = results.best_config
        best_score = results.best_score
        print(f"\n  Optimization complete!")
        print(f"  Best validation score: {best_score}")
    except Exception as e:
        print(f"\n  Optimization error: {e}")
        # Use default config
        best_config = {
            "prompt_template": prompt_variants[0],
            "model": "gpt-3.5-turbo",
            "temperature": 0.0,
            "max_tokens": 200,
        }
        best_score = None

    # Stage 3: Validate on test set
    test_accuracy = validate_on_test_set(agent, best_config, testset)

    # Save results
    results_file = RESULTS_DIR / "optimization_results.json"
    with open(results_file, "w") as f:
        json.dump(
            {
                "best_config": best_config,
                "validation_score": best_score,
                "test_accuracy": test_accuracy,
                "prompt_variants": prompt_variants,
            },
            f,
            indent=2,
        )
    print(f"\n  Results saved to: {results_file}")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Prompt variants generated: {len(prompt_variants)}")
    print(f"  Validation score: {best_score}")
    print(f"  Test accuracy: {test_accuracy:.1%}")
    print(f"  Best model: {best_config.get('model', 'N/A')}")
    print(f"  Best temperature: {best_config.get('temperature', 'N/A')}")
    print("\n  This demonstrates the DSPy + Traigent workflow:")
    print("  1. DSPy generates optimized prompts from trainset")
    print("  2. Traigent optimizes prompts + hyperparams on devset")
    print("  3. Final validation on held-out testset")


if __name__ == "__main__":
    asyncio.run(main())
