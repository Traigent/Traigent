"""Quick E2E test for nested measures format using mock LLM."""

import asyncio
import os
import sys

# Add project root to path
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# Use mock LLM - still sends to backend but doesn't need real API key
os.environ["TRAIGENT_MOCK_LLM"] = "true"
os.environ["TRAIGENT_LOG_LEVEL"] = "DEBUG"

import traigent  # noqa: E402

# Use existing dataset file
DATASET_PATH = os.path.join(
    os.path.dirname(__file__),
    "../../walkthrough/examples/datasets/simple_questions.jsonl",
)


@traigent.optimize(
    eval_dataset=DATASET_PATH,
    objectives=["accuracy"],
    configuration_space={
        "model": ["gpt-4o-mini"],
        "temperature": [0.1, 0.5],
    },
    execution_mode="edge_analytics",
)
def answer_question(question: str) -> str:
    # Mock function - doesn't call real LLM
    return "mock answer"


async def main():
    print("\n" + "=" * 60)
    print("NESTED MEASURES E2E TEST (MOCK MODE)")
    print("=" * 60)

    # Run only 2 trials with subset of dataset
    results = await answer_question.optimize(
        algorithm="grid",
        max_trials=2,
        max_examples=3,  # Only use 3 examples for quick test
    )

    print(f"\nCompleted {len(results.trials)} trials")
    print(f"Best score: {results.best_score}")

    # Show measures structure
    for i, trial in enumerate(results.trials):
        print(f"\n--- Trial {i + 1} ---")
        measures = trial.metadata.get("measures", [])
        if measures:
            print(f"Measures count: {len(measures)}")
            for j, m in enumerate(measures[:3]):
                print(f"  Example {j}:")
                print(f"    example_id: {m.get('example_id')}")
                print(f"    metrics: {m.get('metrics')}")
        else:
            print("  No measures found in trial metadata")

    print("\n" + "=" * 60)
    print("CHECK FRONTEND AT: http://localhost:3000")
    print("Look for the experiment run and verify per-example measures")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
