#!/usr/bin/env python3
"""Check what measures are actually stored in the backend."""

import asyncio
import json

import aiohttp


async def check_backend_measures():
    """Fetch and examine configuration runs from the backend."""

    backend_url = "http://localhost:5000"

    async with aiohttp.ClientSession() as session:
        # Get experiments
        async with session.get(f"{backend_url}/experiments") as response:
            if response.status == 200:
                experiments = await response.json()
                print(f"Found {len(experiments.get('experiments', []))} experiments")

                if experiments.get("experiments"):
                    # Take the first experiment
                    exp = experiments["experiments"][0]
                    exp_id = exp["id"]
                    print(f"\nChecking experiment: {exp_id}")

                    # Get experiment runs
                    async with session.get(
                        f"{backend_url}/experiment-runs/{exp_id}/runs"
                    ) as runs_response:
                        if runs_response.status == 200:
                            runs = await runs_response.json()
                            if runs.get("experiment_runs"):
                                run = runs["experiment_runs"][0]
                                run_id = run["id"]
                                print(f"Checking run: {run_id}")

                                # Get configuration runs
                                async with session.get(
                                    f"{backend_url}/experiment-runs/{run_id}/configuration-runs"
                                ) as config_response:
                                    if config_response.status == 200:
                                        config_runs = await config_response.json()
                                        configs = config_runs.get(
                                            "configuration_runs", []
                                        )
                                        print(
                                            f"\nFound {len(configs)} configuration runs"
                                        )

                                        # Check first few configuration runs
                                        for i, config in enumerate(configs[:5]):
                                            print(f"\n--- Config Run {i+1} ---")
                                            print(f"ID: {config.get('id')}")
                                            print(f"Status: {config.get('status')}")

                                            measures = config.get("measures", {})
                                            print(f"Measures type: {type(measures)}")
                                            print(
                                                f"Measures content: {json.dumps(measures, indent=2)}"
                                            )

                                            # Check for null values
                                            if isinstance(measures, dict):
                                                metrics = measures.get("metrics", {})
                                                for key, value in metrics.items():
                                                    if value is None:
                                                        print(
                                                            f"⚠️  NULL VALUE found in {key}"
                                                        )

                                            # Check individual measure endpoint
                                            config_id = config.get("id")
                                            if config_id:
                                                async with session.get(
                                                    f"{backend_url}/configuration-runs/{config_id}"
                                                ) as detail_response:
                                                    if detail_response.status == 200:
                                                        detail = (
                                                            await detail_response.json()
                                                        )
                                                        detail_measures = detail.get(
                                                            "measures", {}
                                                        )
                                                        print(
                                                            f"Direct endpoint measures: {json.dumps(detail_measures, indent=2)}"
                                                        )


if __name__ == "__main__":
    asyncio.run(check_backend_measures())
