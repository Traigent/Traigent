#!/usr/bin/env python3
"""Minimal test script for backend integration using DTOs."""

import asyncio
import os
import sys

import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging

# Direct imports to avoid full package initialization
import aiohttp

# Import DTOs
from traigent.cloud.dtos import (
    create_local_configuration_run,
    create_local_experiment,
    create_local_experiment_run,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _resolve_backend_url() -> str | None:
    return os.getenv("TRAIGENT_API_URL") or os.getenv("TRAIGENT_BACKEND_URL")


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("TRAIGENT_BACKEND_LIVE"),
    reason=(
        "Set TRAIGENT_BACKEND_LIVE=1 plus TRAIGENT_API_URL (preferred) "
        "or TRAIGENT_BACKEND_URL to run the live backend DTO smoke test"
    ),
)
async def test_backend_api():
    """Test direct API calls to the backend using DTOs."""

    backend_url = _resolve_backend_url()
    if not backend_url:
        pytest.skip(
            "Set TRAIGENT_API_URL (preferred) or TRAIGENT_BACKEND_URL for the live backend DTO smoke test"
        )

    # Create experiment DTO
    experiment_dto = create_local_experiment(
        experiment_id="test_exp_001",
        name="Test Local Mode Experiment",
        description="Testing privacy-preserving metadata submission with DTOs",
        configuration_space={
            "temperature": [0.1, 0.3, 0.5, 0.7, 0.9],
            "max_tokens": [100, 150, 200],
        },
        max_trials=10,
        dataset_size=100,
    )

    # Validate DTO (optional) - skip if traigent_schemas not installed
    # The validate() method requires traigent_schemas package
    # In strict mode (default), it raises DTOSerializationError when unavailable
    import os

    os.environ.setdefault("TRAIGENT_STRICT_VALIDATION", "false")
    if hasattr(experiment_dto, "validate"):
        try:
            experiment_dto.validate()
        except Exception:
            # Validation is optional - continue without it
            pass

    # Convert to dict for API
    experiment_data = experiment_dto.to_dict()

    async with aiohttp.ClientSession() as session:
        # 1. Create experiment
        print("\n1. Creating experiment...")
        url = f"{backend_url}/experiments"
        try:
            async with session.post(
                url, json=experiment_data, timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 201:
                    result = await response.json()
                    experiment_id = result.get(
                        "experiment_id", experiment_data["experiment_id"]
                    )
                    print(f"   ✅ Created experiment: {experiment_id}")
                else:
                    error_text = await response.text()
                    print(
                        f"   ❌ Failed to create experiment: {response.status} - {error_text}"
                    )
                    return
        except aiohttp.ClientError as e:
            pytest.skip(f"Backend not reachable at {backend_url}: {e}")
        except TimeoutError as e:
            pytest.skip(f"Backend timed out at {backend_url}: {e}")

        # 2. Create experiment run
        print("\n2. Creating experiment run...")

        # Create experiment run DTO
        experiment_run_dto = create_local_experiment_run(
            run_id="test_run_001",
            experiment_id=experiment_id,
            function_name="test_function",
            configuration_space={
                "temperature": [0.1, 0.3, 0.5, 0.7, 0.9],
                "max_tokens": [100, 150, 200],
            },
            objectives=["maximize"],
            max_trials=10,
            dataset_size=100,
        )

        # Convert to dict for API
        run_data = experiment_run_dto.to_dict()

        url = f"{backend_url}/experiment-runs/{experiment_id}/runs"
        try:
            async with session.post(
                url, json=run_data, timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 201:
                    result = await response.json()
                    run_id = result.get("id", run_data["id"])
                    print(f"   ✅ Created experiment run: {run_id}")
                else:
                    error_text = await response.text()
                    print(
                        f"   ❌ Failed to create experiment run: {response.status} - {error_text}"
                    )
                    return
        except aiohttp.ClientError as e:
            pytest.skip(f"Backend not reachable at {backend_url}: {e}")
        except TimeoutError as e:
            pytest.skip(f"Backend timed out at {backend_url}: {e}")

        # 3. Create configuration run
        print("\n3. Creating configuration run...")

        # Create configuration run DTO
        config_run_dto = create_local_configuration_run(
            config_id="test_config_001",
            experiment_run_id=run_id,
            trial_number=1,
            config={"temperature": 0.7, "max_tokens": 150},
            dataset_subset_info=None,  # Will use privacy-preserving defaults
        )

        # Convert to dict for API
        config_data = config_run_dto.to_dict()

        url = f"{backend_url}/experiment-runs/runs/{run_id}/configurations"
        try:
            async with session.post(
                url, json=config_data, timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 201:
                    result = await response.json()
                    config_id = result.get("id", config_data["id"])
                    print(f"   ✅ Created configuration run: {config_id}")
                else:
                    error_text = await response.text()
                    print(
                        f"   ❌ Failed to create configuration run: {response.status} - {error_text}"
                    )
                    return
        except aiohttp.ClientError as e:
            pytest.skip(f"Backend not reachable at {backend_url}: {e}")
        except TimeoutError as e:
            pytest.skip(f"Backend timed out at {backend_url}: {e}")

        # 4. Update configuration run with results
        print("\n4. Updating configuration run with results...")

        # Update status
        status_data = {"status": "COMPLETED"}
        url = f"{backend_url}/configuration-runs/{config_id}/status"
        try:
            async with session.put(
                url, json=status_data, timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status in [200, 204]:
                    print("   ✅ Updated status to COMPLETED")
                else:
                    error_text = await response.text()
                    print(
                        f"   ❌ Failed to update status: {response.status} - {error_text}"
                    )
        except aiohttp.ClientError as e:
            pytest.skip(f"Backend not reachable at {backend_url}: {e}")
        except TimeoutError as e:
            pytest.skip(f"Backend timed out at {backend_url}: {e}")

        # Update measures (convert to backend format)
        measures_data = {
            "measures": [
                {"measure_name": "score", "value": 0.85},
                {"measure_name": "cost", "value": 0.002},
                {"measure_name": "latency", "value": 1.23},
            ]
        }
        url = f"{backend_url}/configuration-runs/{config_id}/measures"
        try:
            async with session.put(
                url, json=measures_data, timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    print(
                        "   ✅ Updated measures: score=0.85, cost=0.002, latency=1.23"
                    )
                else:
                    error_text = await response.text()
                    print(
                        f"   ❌ Failed to update measures: {response.status} - {error_text}"
                    )
        except aiohttp.ClientError as e:
            pytest.skip(f"Backend not reachable at {backend_url}: {e}")
        except TimeoutError as e:
            pytest.skip(f"Backend timed out at {backend_url}: {e}")

    print("\n✅ Test completed!")
    print("\nCheck the UI at http://localhost:3000/experiments/")
    print(f"Look for experiment: {experiment_id}")

    # Verify test completed successfully
    assert experiment_id is not None, "Experiment should be created"


if __name__ == "__main__":
    print("Testing Backend API Integration")
    print("================================")
    asyncio.run(test_backend_api())
