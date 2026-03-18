#!/usr/bin/env python3
"""Check if measures are saved in the database."""

import asyncio
import json
import os
import sys

try:
    import asyncpg
except ImportError:
    print("asyncpg not installed, trying alternative approach...")
    import subprocess
    import sys

    # Use psql command if available
    def check_with_psql():
        try:
            cmd = [
                "psql",
                os.environ.get("DB_URL", "postgresql://localhost:5432/traigent"),
                "-c",
                "SELECT id, name, created_at FROM experiments ORDER BY created_at DESC LIMIT 5",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            print("Latest experiments:")
            print(result.stdout)

            # Also check configuration runs
            cmd2 = [
                "psql",
                os.environ.get("DB_URL", "postgresql://localhost:5432/traigent"),
                "-c",
                "SELECT id, score, jsonb_array_length(measures) as measure_count FROM configuration_runs ORDER BY created_at DESC LIMIT 5",
            ]
            result2 = subprocess.run(cmd2, capture_output=True, text=True)
            print("\nLatest configuration runs:")
            print(result2.stdout)
        except Exception as e:
            print(f"Error running psql: {e}")
            print("Please check database connection manually")

    check_with_psql()
    sys.exit(0)

# Database connection parameters
DB_URL = os.environ.get("DB_URL", "postgresql://localhost:5432/traigent")


async def check_measures():
    """Check the latest experiment's measures."""
    conn = None
    try:
        # Connect to database
        conn = await asyncpg.connect(DB_URL)

        # Get the latest experiment
        experiment = await conn.fetchrow(
            """
            SELECT id, name, created_at, metadata
            FROM experiments
            ORDER BY created_at DESC
            LIMIT 1
        """
        )

        if not experiment:
            print("No experiments found")
            return

        exp_id = experiment["id"]
        exp_name = experiment["name"]
        created_at = experiment["created_at"]
        experiment["metadata"]

        print("Latest Experiment:")
        print(f"  ID: {exp_id}")
        print(f"  Name: {exp_name}")
        print(f"  Created: {created_at}")

        # Get the configuration runs for this experiment
        runs = await conn.fetch(
            """
            SELECT id, config_params, measures, score, created_at
            FROM configuration_runs
            WHERE experiment_id = $1
            ORDER BY score DESC
            LIMIT 5
        """,
            exp_id,
        )

        print(f"\nFound {len(runs)} configuration runs")

        for i, run in enumerate(runs, 1):
            run_id = run["id"]
            config_params = run["config_params"]
            measures = run["measures"]
            score = run["score"]
            created_at = run["created_at"]

            print(f"\n  Run #{i} (ID: {run_id}):")
            print(
                f"    Config: {json.dumps(config_params, indent=6) if config_params else 'None'}"
            )
            print(f"    Score: {score}")
            print(f"    Measures count: {len(measures) if measures else 0}")
            if measures and len(measures) > 0:
                print(
                    f"    First measure: {json.dumps(measures[0], indent=8) if measures else 'None'}"
                )
                print(f"    Total measures: {len(measures)}")
            else:
                print(f"    Measures: {measures}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if conn:
            await conn.close()


if __name__ == "__main__":
    asyncio.run(check_measures())
