#!/usr/bin/env python3
"""Test Edge Analytics mode with per-example measures.

Note: This test runs in Edge Analytics mode and will attempt to connect to a backend at http://localhost:5000.
If the backend is not available, optimization will still work but backend submission will fail.
"""

import asyncio

import traigent
from traigent.evaluators.base import Dataset, EvaluationExample


# Create a simple function to optimize
@traigent.optimize(
    configuration_space={
        "threshold": [0.3, 0.5, 0.7],
    },
    objectives=["accuracy"],
    execution_mode="edge_analytics",  # Use Edge Analytics mode to get per-example measures
)
async def classify_text(text: str, threshold: float = 0.5) -> str:
    """Simple text classifier."""
    # Mock classification based on text length
    score = len(text) / 100.0
    return "positive" if score > threshold else "negative"


async def main():
    """Test Edge Analytics mode submission."""
    print("Testing Edge Analytics mode with per-example measures...")
    print("=" * 60)
    print(
        "⚠️  Note: Backend errors are expected if server is not running at http://localhost:5000"
    )
    print("    The optimization will still work in Edge Analytics mode.")
    print("=" * 60)

    # Create test dataset with multiple examples
    dataset = Dataset(
        examples=[
            EvaluationExample(
                input_data="This is a short text", expected_output="negative"
            ),
            EvaluationExample(
                input_data="This is a medium length text that should be classified",
                expected_output="positive",
            ),
            EvaluationExample(input_data="Short", expected_output="negative"),
            EvaluationExample(
                input_data="This is a very long text with many words that should definitely be classified as positive due to its length",
                expected_output="positive",
            ),
            EvaluationExample(
                input_data="Medium text here", expected_output="negative"
            ),
            EvaluationExample(
                input_data="Another long text example with enough words to trigger positive classification based on the threshold",
                expected_output="positive",
            ),
            EvaluationExample(input_data="Brief", expected_output="negative"),
            EvaluationExample(
                input_data="Extended text content that contains multiple sentences and should be positive",
                expected_output="positive",
            ),
        ]
    )

    print(f"Dataset: {len(dataset.examples)} examples")

    # Set the dataset for the function
    classify_text.eval_dataset = dataset

    # Run optimization
    print("\nRunning optimization in Edge Analytics mode...")
    print("Backend should receive per-example measures (array of 8 scores)")

    results = await classify_text.optimize(
        max_trials=3,
    )

    print("\n" + "=" * 60)
    print("✅ Optimization complete!")
    print(f"Best config: {results.best_config}")
    print(f"Best score: {results.best_score:.3f}")

    print("\n📊 Expected backend submission format:")
    print("For Edge Analytics mode, backend should receive:")
    print("- metrics.measures: [score1, score2, ..., score8]  # Array of 8 scores")
    print("- metadata.mode: 'edge_analytics'")
    print("\nNOT aggregated statistics like mean, std, etc.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\n❌ Test failed unexpectedly: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
