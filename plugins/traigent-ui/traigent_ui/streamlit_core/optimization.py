"""
Optimization Execution Module
============================

This module contains the core optimization execution logic, including
the main run_optimization function and related utilities.
"""

import asyncio
import os
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import streamlit as st


def _get_api_key():
    """Get API key from environment or session state."""
    openai_key = os.environ.get("OPENAI_API_KEY", "") or st.session_state.get(
        "openai_api_key", ""
    )
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "") or st.session_state.get(
        "anthropic_api_key", ""
    )

    # Return the first valid key found
    if openai_key and openai_key.startswith("sk-"):
        return openai_key
    elif anthropic_key and anthropic_key.startswith("sk-ant-"):
        return anthropic_key

    return ""


# Import Traigent modules and utilities
try:
    # Import availability checks only
    import optimization_callbacks  # noqa: F401 - Import check only
    import optimization_storage  # noqa: F401 - Import check only
    from langchain_problems import get_problem_class

    import traigent  # noqa: F401 - Import check only
except ImportError as e:
    st.error(f"Import error: {e}")


async def run_optimization(
    problem_name: str,
    strategy: str,
    models: List[str],
    max_iterations: int = 10,
    subset_size: Optional[int] = None,
    temperature_range: Optional[List[float]] = None,
    dry_run: bool = False,
    mock: bool = False,
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Run agent comparison for a problem to find the best AI configuration."""
    try:
        # Normalize inputs for downstream logic
        models = models or ["gpt-3.5-turbo"]
        temperature_range = temperature_range or [0.3, 0.5, 0.7]

        # Check API key for non-mock runs
        if not mock and not dry_run:
            api_key = _get_api_key()
            if not api_key:
                return {
                    "success": False,
                    "error": "API key not configured. Please set an OpenAI or Anthropic API key in the Settings tab.",
                }
            elif not (api_key.startswith("sk-") or api_key.startswith("sk-ant-")):
                return {
                    "success": False,
                    "error": "Invalid API key format. Please check your API key in the Settings tab.",
                }

            # Set API key in environment for LangChain to access
            if api_key.startswith("sk-"):
                os.environ["OPENAI_API_KEY"] = api_key
                if progress_callback:
                    progress_callback(0.05, "✅ OpenAI API key configured")
            elif api_key.startswith("sk-ant-"):
                os.environ["ANTHROPIC_API_KEY"] = api_key
                if progress_callback:
                    progress_callback(0.05, "✅ Anthropic API key configured")

        # If mock or dry_run is specified, use simulation mode
        if mock or dry_run:
            if progress_callback:
                progress_callback(
                    0.05, f"SIMULATION MODE: mock={mock}, dry_run={dry_run}"
                )
            return await _run_simulation(
                problem_name,
                strategy,
                models,
                max_iterations,
                subset_size,
                temperature_range,
                progress_callback,
            )

        # Real Traigent optimization
        start_time = datetime.now(timezone.utc)

        if progress_callback:
            progress_callback(0.1, f"REAL MODE: Loading problem '{problem_name}'...")

        # Add debug logging to understand what's happening

        # Set Traigent context for quiet mode but allow some debug output
        os.environ["TRAIGENT_QUIET"] = "1"
        os.environ["TRAIGENT_VERBOSE"] = "0"
        os.environ["TRAIGENT_DEBUG"] = "0"

        print(f"\n[DEBUG] Starting optimization with subset_size={subset_size}")
        print(f"[DEBUG] Models: {models}")
        print(f"[DEBUG] Max iterations: {max_iterations}")
        print(f"[DEBUG] Strategy: {strategy}")

        # Get the problem class and create instance
        try:
            # Load all problems first to populate registry
            from langchain_problems import get_available_problems, load_all_problems

            load_all_problems()

            if progress_callback:
                available = get_available_problems()
                progress_callback(0.12, f"Available problems: {available}")

            problem_class = get_problem_class(problem_name)
            problem = problem_class()
            if progress_callback:
                progress_callback(
                    0.15, f"Loaded problem class: {problem_class.__name__}"
                )
        except Exception as e:
            error_msg = f"Failed to load problem '{problem_name}': {str(e)}"
            print(f"[DEBUG] Problem loading error: {error_msg}")
            if progress_callback:
                progress_callback(0.1, f"ERROR: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
            }

        if progress_callback:
            progress_callback(0.2, "Creating optimized function...")

        # Create the Traigent optimized function
        optimized_function = problem.create_optimized_function()

        # Update configuration space if models are specified
        if models:
            # Override the default configuration space
            new_config_space = problem.get_configuration_space().copy()
            new_config_space["model"] = models
            if temperature_range:
                new_config_space["temperature"] = list(temperature_range)

            # Update the optimized function's configuration space
            optimized_function.configuration_space = new_config_space

        # Handle subset size by modifying the dataset for faster testing
        original_dataset = problem.get_dataset()
        print(
            f"[DEBUG] Original dataset size: {len(original_dataset.examples) if hasattr(original_dataset, 'examples') else 'Unknown'}"
        )

        if subset_size and subset_size > 0:
            if hasattr(original_dataset, "examples"):
                print(
                    f"[DEBUG] Limiting dataset from {len(original_dataset.examples)} to {subset_size} examples"
                )
                if progress_callback:
                    progress_callback(
                        0.25,
                        f"Limiting dataset to {subset_size} examples for faster testing...",
                    )
                original_dataset.examples = original_dataset.examples[:subset_size]
                print(
                    f"[DEBUG] Dataset limited to {len(original_dataset.examples)} examples"
                )
            # Update the dataset file path
            optimized_function.eval_dataset = problem.create_temporary_dataset_file(
                original_dataset
            )
            print(f"[DEBUG] Updated eval_dataset: {optimized_function.eval_dataset}")
        else:
            # Default to smaller dataset for speed if not specified
            if (
                hasattr(original_dataset, "examples")
                and len(original_dataset.examples) > 10
            ):
                print("[DEBUG] No subset_size specified, limiting to 10 examples")
                if progress_callback:
                    progress_callback(
                        0.25, "Using first 10 examples for faster testing..."
                    )
                original_dataset.examples = original_dataset.examples[:10]
                optimized_function.eval_dataset = problem.create_temporary_dataset_file(
                    original_dataset
                )
                print(
                    f"[DEBUG] Updated eval_dataset: {optimized_function.eval_dataset}"
                )

        if progress_callback:
            progress_callback(0.3, "Starting optimization...")

        # Convert strategy name to algorithm
        algorithm_map = {
            "grid": "grid",
            "random": "random",
            "smart_exploration_conservative": "bayesian",
            "smart_exploration_aggressive": "bayesian",
            "adaptive_batch": "random",
            "parallel_batch": "random",
        }
        algorithm = algorithm_map.get(strategy, "random")

        # Run the actual Traigent optimization
        try:
            if progress_callback:
                progress_callback(
                    0.4,
                    f"Starting {algorithm} optimization with {max_iterations} trials...",
                )

            # Prepare callbacks for progress tracking
            callbacks = []

            # Add progress callback wrapper if provided
            if progress_callback:
                from traigent.utils.callbacks import OptimizationCallback

                class ProgressWrapper(OptimizationCallback):
                    def __init__(self, callback, max_trials):
                        self.callback = callback
                        self.total_trials = max_trials

                    def on_optimization_start(
                        self, config_space, objectives, algorithm
                    ):
                        self.callback(0.45, f"Starting {algorithm} optimization...")

                    def on_trial_start(self, trial_number, config):
                        progress = 0.45 + (0.4 * trial_number / self.total_trials)
                        self.callback(
                            progress,
                            f"Running trial {trial_number + 1}/{self.total_trials}",
                        )

                    def on_trial_complete(self, trial, progress_info):
                        progress = 0.45 + (
                            0.4 * progress_info.completed_trials / self.total_trials
                        )
                        self.callback(
                            progress,
                            f"Completed trial {progress_info.completed_trials}/{self.total_trials}",
                        )

                    def on_optimization_complete(self, result):
                        self.callback(0.9, "Processing results...")

                callbacks.append(ProgressWrapper(progress_callback, max_iterations))

            # Guard against long-running live calls by enforcing an upper bound.
            try:
                optimization_result = await asyncio.wait_for(
                    optimized_function.optimize(
                        algorithm=algorithm,
                        max_trials=max_iterations,
                        callbacks=callbacks,
                        timeout=60.0,  # 1 minute timeout for faster feedback
                        algorithm_params=(
                            {"n_initial_points": 2}
                            if algorithm == "bayesian"
                            else None
                        ),
                    ),
                    timeout=90.0,
                )
            except asyncio.TimeoutError:
                # Fallback to simulation to keep the UI responsive.
                if progress_callback:
                    progress_callback(
                        0.5,
                        "Optimization timed out. Falling back to simulation to keep things moving...",
                    )
                return await _run_simulation(
                    problem_name,
                    strategy,
                    models,
                    max_iterations,
                    subset_size,
                    temperature_range,
                    progress_callback,
                )

            print(f"[DEBUG] Optimization result type: {type(optimization_result)}")
            print(f"[DEBUG] Has trials: {hasattr(optimization_result, 'trials')}")
            if hasattr(optimization_result, "trials"):
                print(f"[DEBUG] Number of trials: {len(optimization_result.trials)}")
                for i, trial in enumerate(optimization_result.trials):
                    print(
                        f"[DEBUG] Trial {i}: status={trial.status.value if hasattr(trial, 'status') else 'Unknown'}, metrics={trial.metrics}"
                    )

            # Check if optimization actually succeeded
            if not optimization_result or not hasattr(optimization_result, "trials"):
                error_msg = "Optimization completed but returned invalid results"
                if progress_callback:
                    progress_callback(0.5, f"ERROR: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg,
                }

            # Check if we have any successful trials
            successful_trials = [
                trial
                for trial in optimization_result.trials
                if hasattr(trial, "status") and trial.status.value == "completed"
            ]

            print(
                f"[DEBUG] Successful trials: {len(successful_trials)}/{len(optimization_result.trials)}"
            )

            if not successful_trials:
                error_msg = f"Optimization completed but all {len(optimization_result.trials)} trials failed. Check API key and problem configuration."
                if progress_callback:
                    progress_callback(0.8, f"WARNING: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg,
                }

            print(
                f"[DEBUG] Best config: {optimization_result.best_config if hasattr(optimization_result, 'best_config') else 'None'}"
            )
            print(
                f"[DEBUG] Best score: {optimization_result.best_score if hasattr(optimization_result, 'best_score') else 'None'}"
            )

            if progress_callback:
                progress_callback(
                    0.9,
                    f"✅ Optimization completed: {len(successful_trials)}/{len(optimization_result.trials)} trials successful",
                )

        except Exception as e:
            error_msg = f"Optimization execution failed: {str(e)}"
            print(f"[DEBUG] Optimization execution error: {error_msg}")
            import traceback

            traceback.print_exc()
            if progress_callback:
                progress_callback(0.5, f"❌ ERROR: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
            }

        if progress_callback:
            progress_callback(1.0, "Optimization complete!")

        # Extract results and format for UI
        duration = (datetime.now(timezone.utc) - start_time).total_seconds() / 60.0

        # Extract best trial information for UI display
        best_config = (
            optimization_result.best_config
            if hasattr(optimization_result, "best_config")
            else {}
        )
        best_score = (
            optimization_result.best_score
            if hasattr(optimization_result, "best_score")
            else 0
        )

        # Find the best trial to get model and other metrics
        best_trial = None
        if optimization_result.trials:
            # Find trial with highest score/accuracy
            for trial in optimization_result.trials:
                if (
                    trial.metrics
                    and hasattr(trial, "status")
                    and trial.status.value == "completed"
                ):
                    if not best_trial or (
                        trial.metrics.get("accuracy", 0)
                        > best_trial.metrics.get("accuracy", 0)
                    ):
                        best_trial = trial

        # Extract performance metrics
        best_model = best_config.get("model", "N/A") if best_config else "N/A"
        accuracy = (
            best_score
            if best_score
            else (
                best_trial.metrics.get("accuracy", 0)
                if best_trial and best_trial.metrics
                else 0
            )
        )

        # Calculate cost estimate (per call, not per token)
        cost = 0.002  # Default cost estimate per call
        if best_trial and best_trial.config:
            model = best_trial.config.get("model", "gpt-3.5-turbo")
            # Cost per call estimates (not per token)
            cost_per_call = {
                "gpt-3.5-turbo": 0.002,
                "gpt-4o-mini": 0.0005,
                "gpt-4o": 0.03,
                "claude-3-haiku": 0.0025,
                "claude-3-sonnet": 0.015,
            }
            cost = cost_per_call.get(model, 0.002)

        print(
            f"[DEBUG] Final results - best_model: {best_model}, accuracy: {accuracy}, cost: {cost}"
        )

        return {
            "success": True,
            "problem": problem_name,
            "strategy": strategy,
            "timestamp": start_time.isoformat(),
            "duration_minutes": duration,
            "configurations_tested": len(optimization_result.trials),
            "performance": {
                "best_model": best_model,
                "best_config": best_config,
                "best_score": best_score,
                "accuracy": accuracy,
                "cost": cost,
                "latency": (
                    best_trial.duration
                    if best_trial and hasattr(best_trial, "duration")
                    else 1.0
                ),
                "success_rate": accuracy * 0.95 if accuracy else 0,
                "optimization_id": (
                    optimization_result.optimization_id
                    if hasattr(optimization_result, "optimization_id")
                    else None
                ),
            },
            "all_results": [
                {
                    "trial_id": (
                        trial.trial_id if hasattr(trial, "trial_id") else f"trial_{i}"
                    ),
                    "config": trial.config,
                    "metrics": trial.metrics,
                    "status": (
                        trial.status.value if hasattr(trial, "status") else "unknown"
                    ),
                    "duration": trial.duration if hasattr(trial, "duration") else 0,
                }
                for i, trial in enumerate(optimization_result.trials)
            ],
            "mock_run": False,
        }

    except Exception as e:
        print(f"[DEBUG] Top-level optimization error: {str(e)}")
        import traceback

        traceback.print_exc()
        return {
            "success": False,
            "error": f"Optimization failed: {str(e)}",
        }


async def _run_simulation(
    problem_name: str,
    strategy: str,
    models: List[str],
    max_iterations: int,
    subset_size: Optional[int],
    temperature_range: Optional[List[float]],
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Run simulation mode (for mock/dry-run)."""
    # Ensure we have a usable temperature list for the simulation loop
    safe_temperatures = (temperature_range or [0.3, 0.5, 0.7])[:3]

    if progress_callback:
        progress_callback(0.2, "Running in simulation mode (no LLM calls)...")

    await asyncio.sleep(1)  # Simulate initial setup

    if progress_callback:
        progress_callback(0.4, "Simulating agent exploration...")

    # Generate mock results for all requested models
    all_results = []
    for i, model in enumerate(models):
        for temp in safe_temperatures:  # Simulate testing 3 temperatures
            if progress_callback:
                progress = 0.4 + (
                    0.4
                    * (i * len(safe_temperatures) + safe_temperatures.index(temp))
                    / (len(models) * max(len(safe_temperatures), 1))
                )
                progress_callback(
                    progress, f"Testing {model} at temperature {temp:.1f}..."
                )

            await asyncio.sleep(0.2)  # Simulate processing time

            # Generate varied mock metrics
            base_accuracy = {
                "gpt-3.5-turbo": 0.85,
                "gpt-4o-mini": 0.88,
                "gpt-4o": 0.92,
                "claude-3-sonnet": 0.87,
                "claude-3-opus": 0.91,
            }.get(model, 0.80)

            # Add temperature variation
            temp_factor = 1.0 - (abs(temp - 0.7) * 0.1)  # Optimal around 0.7
            accuracy = base_accuracy * temp_factor

            # Add some randomness
            import random

            accuracy += random.uniform(-0.05, 0.05)
            accuracy = max(0.5, min(1.0, accuracy))

            # Cost calculation (mock)
            base_costs = {
                "gpt-3.5-turbo": 0.002,
                "gpt-4o-mini": 0.0005,
                "gpt-4o": 0.03,
                "claude-3-sonnet": 0.015,
                "claude-3-opus": 0.075,
            }.get(model, 0.01)

            cost = base_costs * (1 + temp * 0.1)  # Temperature affects cost slightly

            all_results.append(
                {
                    "model": model,
                    "temperature": temp,
                    "accuracy": accuracy,
                    "cost": cost,
                    "latency": random.uniform(0.8, 2.5),  # Mock latency
                    "success_rate": accuracy * 0.95,  # Success rate related to accuracy
                }
            )

    if progress_callback:
        progress_callback(0.8, "Finding best configuration...")

    await asyncio.sleep(0.5)

    # Find best result (highest accuracy)
    best_result = max(all_results, key=lambda x: x["accuracy"])

    if progress_callback:
        progress_callback(
            1.0,
            f"Complete! Best: {best_result['model']} (accuracy: {best_result['accuracy']:.1%})",
        )

    # Calculate configurations tested
    configurations_tested = len(all_results)

    return {
        "success": True,
        "problem": problem_name,
        "strategy": strategy,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "duration_minutes": 2.0,  # Mock duration
        "configurations_tested": configurations_tested,
        "performance": {
            "best_model": best_result["model"],
            "best_config": {
                "model": best_result["model"],
                "temperature": best_result["temperature"],
            },
            "accuracy": best_result["accuracy"],
            "cost": best_result["cost"],
            "latency": best_result["latency"],
            "success_rate": best_result["success_rate"],
        },
        "all_results": all_results,
        "mock_run": True,
    }


def calculate_configuration_count(
    models: List[str], temperature_points: int = 3
) -> int:
    """Calculate the total number of configurations to be tested."""
    return len(models) * temperature_points


def get_optimization_strategies() -> Dict[str, str]:
    """Get available optimization strategies."""
    return {
        "🔍 Systematic Exploration": "grid",
        "🎲 Random Search": "random",
        "🧠 Smart Exploration (Conservative)": "smart_exploration_conservative",
        "🚀 Smart Exploration (Aggressive)": "smart_exploration_aggressive",
        "⚡ Adaptive Batch": "adaptive_batch",
        "🔄 Parallel Batch": "parallel_batch",
    }


def validate_optimization_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate optimization configuration and return validation result."""
    errors = []
    warnings = []

    # Check models
    if not config.get("selected_models"):
        errors.append("At least one model must be selected")

    # Check temperature range
    temp_range = config.get("temperature_range", [0.0, 1.0])
    if temp_range[0] >= temp_range[1]:
        errors.append("Temperature range minimum must be less than maximum")

    # Check max trials
    max_trials = config.get("max_trials", 10)
    if max_trials < 1:
        errors.append("Max trials must be at least 1")
    elif max_trials > 100:
        warnings.append("Large number of trials may take a long time")

    return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}
