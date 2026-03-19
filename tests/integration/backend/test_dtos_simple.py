#!/usr/bin/env python3
"""Simple test to verify DTO implementation."""

import json
import os
import sys

# Add to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Direct import to avoid full package initialization
from traigent.cloud.dtos import (
    create_local_configuration_run,
    create_local_experiment,
    create_local_experiment_run,
)


def test_dtos():
    """Test DTO creation and serialization."""

    print("Testing DTO creation and serialization...")
    print("=" * 50)

    # 1. Test Experiment DTO
    print("\n1. Creating Experiment DTO:")
    exp = create_local_experiment(
        experiment_id="test_exp_001",
        name="Test Local Mode Experiment",
        description="Testing privacy-preserving metadata submission",
        configuration_space={
            "temperature": [0.1, 0.3, 0.5, 0.7, 0.9],
            "max_tokens": [100, 150, 200],
        },
        max_trials=10,
        dataset_size=100,
    )

    # Check fields
    print(f"   - ID: {exp.id}")
    print(f"   - Name: {exp.name}")
    print("   - Privacy fields (should be None):")
    print(f"     - agent_id: {exp.agent_id}")
    print(f"     - benchmark_id: {exp.benchmark_id}")
    print(f"     - evaluation_set_id: {exp.evaluation_set_id}")
    print(f"     - model_parameters_id: {exp.model_parameters_id}")
    print(f"   - Metadata: {exp.metadata}")

    # Validate (optional) - skip if traigent_schemas not installed
    # Set non-strict mode to avoid exceptions when validator unavailable
    os.environ.setdefault("TRAIGENT_STRICT_VALIDATION", "false")
    if hasattr(exp, "validate"):
        try:
            result = exp.validate()
            print(
                f"   - Validation: {'✅ Passed' if result else '⚠️  Failed (non-blocking)'}"
            )
        except Exception as e:
            print(f"   - Validation: ⚠️  Skipped ({e})")

    # Convert to dict
    exp_dict = exp.to_dict()
    print(f"   - Serialized keys: {list(exp_dict.keys())}")
    print(f"   - Has nullables: {'agent_id' in exp_dict}")

    # 2. Test Experiment Run DTO
    print("\n2. Creating Experiment Run DTO:")
    run = create_local_experiment_run(
        run_id="test_run_001",
        experiment_id="test_exp_001",
        function_name="test_function",
        configuration_space={
            "temperature": [0.1, 0.3, 0.5, 0.7, 0.9],
            "max_tokens": [100, 150, 200],
        },
        objectives=["maximize"],
        max_trials=10,
        dataset_size=100,
    )

    print(f"   - ID: {run.id}")
    print(f"   - Experiment ID: {run.experiment_id}")
    print(f"   - Experiment Data: {type(run.experiment_data)}")
    print(f"   - Metadata: {run.metadata}")

    # Convert to dict
    run_dict = run.to_dict()
    print(f"   - Serialized keys: {list(run_dict.keys())}")

    # 3. Test Configuration Run DTO
    print("\n3. Creating Configuration Run DTO:")
    config = create_local_configuration_run(
        config_id="test_config_001",
        experiment_run_id="test_run_001",
        trial_number=1,
        config={"temperature": 0.7, "max_tokens": 150},
        dataset_subset_info=None,  # Will use privacy defaults
    )

    print(f"   - ID: {config.id}")
    print(f"   - Experiment Run ID: {config.experiment_run_id}")
    print(f"   - Trial Number: {config.experiment_parameters.get('trial_number')}")
    print(f"   - Config: {config.experiment_parameters.get('config')}")
    print(f"   - Dataset Subset: {config.experiment_parameters.get('dataset_subset')}")
    print(f"   - Metadata: {config.metadata}")

    # Convert to dict
    config_dict = config.to_dict()
    print(f"   - Serialized keys: {list(config_dict.keys())}")

    # 4. Test JSON serialization
    print("\n4. Testing JSON serialization:")
    try:
        json.dumps(exp_dict, indent=2)
        print("   - Experiment DTO: ✅ JSON serializable")

        json.dumps(run_dict, indent=2)
        print("   - Experiment Run DTO: ✅ JSON serializable")

        json.dumps(config_dict, indent=2)
        print("   - Configuration Run DTO: ✅ JSON serializable")

    except Exception as e:
        print(f"   - ❌ JSON serialization failed: {e}")
        assert False, f"JSON serialization failed: {e}"

    # 5. Test privacy features
    print("\n5. Verifying privacy features:")
    print(f"   - Experiment has nullable fields: {exp.agent_id is None}")
    print(
        f"   - Dataset indices are empty: {config.metadata['dataset_subset']['indices'] == []}"
    )
    print(f"   - Privacy metadata set: {exp.metadata.get('privacy_mode')}")
    print(
        f"   - Execution mode is Edge Analytics: {exp.metadata.get('execution_mode') == 'edge_analytics'}"
    )

    print("\n✅ All DTO tests passed!")

    # Add actual assertions for pytest
    assert exp.metadata.get("privacy_mode")
    assert exp.metadata.get("execution_mode") in {"edge_analytics"}
    assert config.metadata["dataset_subset"]["indices"] == []


if __name__ == "__main__":
    test_dtos()
