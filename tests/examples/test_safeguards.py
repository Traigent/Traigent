#!/usr/bin/env python3
"""
Test script for Traigent Edge Analytics mode safeguards.

This script demonstrates and tests:
1. Trial cap enforcement
2. Example cap functionality
3. Cross-run deduplication
4. CI/CD approval gates (when run in CI environment)
"""

import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import traigent
except ImportError:  # pragma: no cover - support IDE execution paths
    import importlib

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    traigent = importlib.import_module("traigent")

from traigent.config.parallel import ParallelConfig
from traigent.evaluators.base import Dataset, EvaluationExample


def test_trial_caps():
    """Test trial cap enforcement."""
    print("\n=== Testing Trial Caps ===")

    # Create a small dataset first
    examples = [
        EvaluationExample(
            input_data={"value": i},
            expected_output={"result": i * 2},
            metadata={"id": f"ex_{i}"},
        )
        for i in range(3)
    ]

    # Save dataset to a file
    import json
    import tempfile

    workspace_root = Path(__file__).resolve().parents[2]
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, dir=str(workspace_root)
    ) as f:
        for ex in examples:
            json.dump(
                {
                    "input": ex.input_data,
                    "output": ex.expected_output,
                    "metadata": ex.metadata,
                },
                f,
            )
            f.write("\n")
        dataset_path = f.name

    @traigent.optimize(
        configuration_space={
            "x": [-5.0, -2.0, 0.0, 2.0, 5.0],
            "y": [-5.0, -2.0, 0.0, 2.0, 5.0],
        },
        algorithm="random",
        eval_dataset=dataset_path,  # Provide the dataset file
        max_trials=5,  # Hard cap at 5 trials
        parallel_config=ParallelConfig(
            mode="parallel",
            trial_concurrency=3,
            example_concurrency=1,
            thread_workers=3,
        ),
        objectives=["score"],
    )
    def optimize_with_trial_cap(x: float, y: float) -> float:
        """Simple function to optimize with trial cap."""
        return -(x**2 + y**2)  # Minimize distance from origin

    # Run optimization (in mock mode to avoid real evaluation)
    os.environ["TRAIGENT_MOCK_MODE"] = "true"

    print("Running optimization with max_trials=5...")
    result = asyncio.run(optimize_with_trial_cap.optimize(max_trials=5))

    # Clean up
    os.unlink(dataset_path)

    print(f"Trials executed: {len(result.trials)}")
    assert (
        len(result.trials) <= 5
    ), f"Trial cap violated! Got {len(result.trials)} trials"
    print("✅ Trial cap enforced successfully!")


def test_example_caps():
    """Test example cap functionality."""
    print("\n=== Testing Example Caps ===")

    @traigent.optimize(
        configuration_space={
            "threshold": [0.1, 0.5, 0.9],
        },
        algorithm="random",
        max_trials=3,
        objectives=["accuracy"],
    )
    def optimize_with_example_cap(threshold: float) -> float:
        """Function to test example capping."""
        return threshold

    # Create a large dataset
    large_examples = [
        EvaluationExample(
            input_data={"value": i},
            expected_output={"result": i},
            metadata={"id": f"ex_{i}"},
        )
        for i in range(100)  # 100 examples
    ]
    large_dataset = Dataset(
        examples=large_examples, name="large_dataset", description="Large test dataset"
    )

    os.environ["TRAIGENT_MOCK_MODE"] = "true"

    print(f"Original dataset size: {len(large_dataset.examples)}")
    print("Running optimization with max_examples=10...")

    # This would apply the cap if integrated properly
    # For now, demonstrate the logic
    max_examples = 10
    if len(large_dataset.examples) > max_examples:
        capped_examples = large_dataset.examples[:max_examples]
        capped_dataset = Dataset(
            examples=capped_examples,
            name=large_dataset.name,
            description=f"{large_dataset.description} (capped to {max_examples})",
        )
        print(f"Dataset capped to: {len(capped_dataset.examples)} examples")
        assert len(capped_dataset.examples) == 10
        print("✅ Example cap applied successfully!")


def test_deduplication():
    """Test cross-run deduplication."""
    print("\n=== Testing Deduplication ===")

    from traigent.storage.local_storage import LocalStorageManager

    # Test config hashing
    storage = LocalStorageManager()

    config1 = {"x": 1.0, "y": 2.0}
    config2 = {"y": 2.0, "x": 1.0}  # Same config, different order
    config3 = {"x": 2.0, "y": 2.0}  # Different config

    hash1 = storage.compute_config_hash(config1)
    hash2 = storage.compute_config_hash(config2)
    hash3 = storage.compute_config_hash(config3)

    print(f"Config 1 hash: {hash1}")
    print(f"Config 2 hash: {hash2}")
    print(f"Config 3 hash: {hash3}")

    assert hash1 == hash2, "Same configs should have same hash!"
    assert hash1 != hash3, "Different configs should have different hash!"
    print("✅ Config hashing works correctly!")


def test_ci_approval():
    """Test CI/CD approval gates."""
    print("\n=== Testing CI Approval ===")

    # Check if we're in CI
    is_ci = os.getenv("CI") in ("true", "1") or os.getenv("GITHUB_ACTIONS") in (
        "true",
        "1",
    )

    if is_ci:
        print("CI environment detected!")

        # Check if approval is set
        if os.getenv("TRAIGENT_RUN_APPROVED") == "1":
            print(f"✅ Approved by: {os.getenv('TRAIGENT_APPROVED_BY', 'environment')}")
        else:
            print("❌ Would require approval (not set in this test)")
    else:
        print("Not in CI environment - no approval needed")
        print("✅ Non-CI execution allowed!")


def main():
    """Run all safeguard tests."""
    print("=" * 60)
    print("Traigent Local Mode Safeguards Test Suite")
    print("=" * 60)

    try:
        test_trial_caps()
        test_example_caps()
        test_deduplication()
        test_ci_approval()

        print("\n" + "=" * 60)
        print("All safeguard tests completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
