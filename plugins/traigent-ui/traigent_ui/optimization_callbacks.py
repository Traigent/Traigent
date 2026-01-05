"""
Optimization Callbacks for Traigent Playground
=================================================

This module provides custom callbacks for the Traigent optimization process,
including detailed logging of all function invocations for debugging and analysis.

Features:
- Logs every function call with input, output, config, and timing
- Saves logs in JSONL format for easy parsing
- Provides summary statistics
- Integrates seamlessly with Traigent's callback system
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from traigent.api.types import ExampleResult, OptimizationResult, TrialResult
from traigent.utils.callbacks import OptimizationCallback, ProgressInfo


class DeferredLoggingCallback(OptimizationCallback):
    """
    Callback that accumulates logs in memory and saves them after optimization completes.

    This approach works better with Streamlit's file writing limitations.
    """

    def __init__(
        self,
        run_id: str,
        problem_id: Optional[str] = None,
        base_dir: Optional[Path] = None,
    ):
        """Initialize the deferred logging callback."""
        self.run_id = run_id
        self.problem_id = problem_id or "unknown_problem"
        self.base_dir = base_dir or Path(__file__).parent / "optimization_logs"
        self.log_data = {
            "run_id": run_id,
            "problem_id": self.problem_id,
            "trials": [],
            "metadata": {},
            "metrics": {},
        }
        self.start_time = None

    def on_optimization_start(
        self, config_space: Dict[str, Any], objectives: List[str], algorithm: str
    ) -> None:
        """Called when optimization starts."""
        print("\n[DeferredLoggingCallback] Optimization started")
        self.start_time = time.time()
        self.log_data["metadata"] = {
            "start_time": datetime.now().isoformat(),
            "algorithm": algorithm,
            "objectives": objectives,
            "config_space": config_space,
            "status": "started",
        }

    def on_trial_start(self, trial_number: int, config: Dict[str, Any]) -> None:
        """Called when a trial starts."""
        pass  # We'll log on completion

    def on_trial_complete(self, trial: TrialResult, progress: ProgressInfo) -> None:
        """Called when a trial completes."""
        trial_data = {
            "trial_id": trial.trial_id,
            "trial_number": len(self.log_data["trials"]),
            "timestamp": datetime.now().isoformat(),
            "config": trial.config,
            "metrics": trial.metrics,
            "status": trial.status.value,
            "duration": trial.duration,
            "error_message": trial.error_message,
        }
        self.log_data["trials"].append(trial_data)
        print(f"[DeferredLoggingCallback] Logged trial {trial.trial_id}")

    def on_optimization_complete(self, result: OptimizationResult) -> None:
        """Called when optimization completes - saves all logs to disk."""
        print("\n[DeferredLoggingCallback] Optimization complete, saving logs...")

        # Update metadata
        self.log_data["metadata"]["end_time"] = datetime.now().isoformat()
        self.log_data["metadata"]["duration_seconds"] = (
            time.time() - self.start_time if self.start_time else 0
        )
        self.log_data["metadata"]["total_trials"] = len(result.trials)
        self.log_data["metadata"]["successful_trials"] = len(result.successful_trials)
        self.log_data["metadata"]["best_config"] = result.best_config
        self.log_data["metadata"]["best_score"] = result.best_score

        # Save logs
        self._save_logs()

    def _save_logs(self):
        """Save accumulated logs to disk with problem_id/run_id structure."""
        try:
            # Create log directory with problem_id/run_id structure
            log_dir = self.base_dir / self.problem_id / self.run_id
            log_dir.mkdir(parents=True, exist_ok=True)

            # Save main log file
            log_file = log_dir / "optimization_log.json"
            with open(log_file, "w") as f:
                json.dump(self.log_data, f, indent=2, default=str)

            print(f"[DeferredLoggingCallback] Logs saved to: {log_file}")

            # Also save individual trial files for compatibility
            for trial in self.log_data["trials"]:
                trial_file = log_dir / f"trial_{trial['trial_id']}.json"
                with open(trial_file, "w") as f:
                    json.dump(trial, f, indent=2, default=str)

        except Exception as e:
            print(f"[DeferredLoggingCallback] ERROR saving logs: {e}")
            import traceback

            traceback.print_exc()


class InvocationLoggingCallback(OptimizationCallback):
    """
    Callback that logs all function invocations during optimization.

    This callback captures detailed information about each function call,
    including inputs, outputs, configurations, timing, and metrics.
    """

    def __init__(self, run_id: str, base_dir: Optional[Path] = None):
        """
        Initialize the logging callback.

        Args:
            run_id: Unique identifier for this optimization run
            base_dir: Base directory for logs (defaults to examples/optimization_logs)
        """
        self.run_id = run_id
        self.base_dir = base_dir or Path(__file__).parent / "optimization_logs"
        self.log_dir = self.base_dir / run_id

        # Create directories with better error handling
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            print(f"[CALLBACK] Created log directory: {self.log_dir}")
            print(f"[CALLBACK] Directory exists: {self.log_dir.exists()}")
            print(
                f"[CALLBACK] Directory is writable: {os.access(self.log_dir, os.W_OK)}"
            )
        except Exception as e:
            print(f"[CALLBACK] ERROR creating log directory: {e}")
            import traceback

            traceback.print_exc()

        # File paths
        self.invocations_file = self.log_dir / "invocations.jsonl"
        self.summary_file = self.log_dir / "summary.json"
        self.metrics_file = self.log_dir / "metrics.json"

        # State tracking
        self.start_time = None
        self.invocation_count = 0
        self.trial_count = 0
        self.config_space = {}
        self.objectives = []
        self.algorithm = ""

        # Metrics aggregation
        self.all_metrics = []
        self.config_performance = {}  # Track performance by config

    def on_optimization_start(
        self, config_space: Dict[str, Any], objectives: List[str], algorithm: str
    ) -> None:
        """Called when optimization starts."""
        print("\n[CALLBACK] InvocationLoggingCallback.on_optimization_start called!")
        print(f"  Log directory: {self.log_dir}")
        print(f"  Log directory absolute path: {self.log_dir.absolute()}")
        print(f"  Algorithm: {algorithm}")
        print(f"  Objectives: {objectives}")

        # Test file creation with a simple test file
        test_file = self.log_dir / "test.txt"
        try:
            with open(test_file, "w") as f:
                f.write("Test file created successfully\n")
            print(f"  TEST: Successfully created test file: {test_file}")
        except Exception as e:
            print(f"  TEST: Failed to create test file: {e}")
            import traceback

            traceback.print_exc()

        self.start_time = time.time()
        self.config_space = config_space
        self.objectives = objectives
        self.algorithm = algorithm

        # Log optimization metadata
        metadata = {
            "run_id": self.run_id,
            "start_time": datetime.now().isoformat(),
            "algorithm": algorithm,
            "objectives": objectives,
            "config_space": config_space,
            "status": "started",
        }

        try:
            with open(self.summary_file, "w") as f:
                json.dump(metadata, f, indent=2)
            print(f"  Summary file created: {self.summary_file}")
            print(f"  Summary file exists: {self.summary_file.exists()}")
        except Exception as e:
            print(f"  ERROR creating summary file: {e}")
            import traceback

            traceback.print_exc()

    def on_trial_start(self, trial_number: int, config: Dict[str, Any]) -> None:
        """Called when a trial starts."""
        self.trial_count = trial_number
        # Log trial start (optional)

    def on_trial_complete(self, trial: TrialResult, progress: ProgressInfo) -> None:
        """Called when a trial completes."""
        print(f"\n[CALLBACK DEBUG] Trial {trial.trial_id} completed:")
        print(f"  Has example_results: {hasattr(trial, 'example_results')}")
        if hasattr(trial, "example_results"):
            print(
                f"  Example results count: {len(trial.example_results) if trial.example_results else 0}"
            )
        print(f"  Trial attributes: {list(trial.__dict__.keys())}")

        # Log trial-level information even if no example_results
        self._log_trial_summary(trial)

        # Log each example evaluation in the trial
        if hasattr(trial, "example_results") and trial.example_results:
            for example_result in trial.example_results:
                self._log_invocation(trial, example_result)
        else:
            print("  [CALLBACK] No example_results found, logging trial summary only")

        # Track config performance
        config_key = self._config_to_key(trial.config)
        if config_key not in self.config_performance:
            self.config_performance[config_key] = {
                "config": trial.config,
                "metrics": [],
                "avg_metrics": {},
                "invocation_count": 0,
            }

        self.config_performance[config_key]["metrics"].append(trial.metrics)
        self.config_performance[config_key]["invocation_count"] += (
            len(trial.example_results)
            if hasattr(trial, "example_results") and trial.example_results
            else 1
        )

        # Update metrics file periodically
        if self.trial_count % 5 == 0:  # Every 5 trials
            self._update_metrics_file(progress)

    def on_optimization_complete(self, result: OptimizationResult) -> None:
        """Called when optimization completes."""
        end_time = time.time()
        duration = end_time - self.start_time if self.start_time else 0

        # Final summary
        summary = {
            "run_id": self.run_id,
            "start_time": (
                datetime.fromtimestamp(self.start_time).isoformat()
                if self.start_time
                else None
            ),
            "end_time": datetime.now().isoformat(),
            "duration_seconds": duration,
            "algorithm": self.algorithm,
            "objectives": self.objectives,
            "config_space": self.config_space,
            "status": "completed",
            "total_trials": len(result.trials),
            "successful_trials": len(result.successful_trials),
            "failed_trials": len(result.failed_trials),
            "total_invocations": self.invocation_count,
            "best_config": result.best_config,
            "best_score": result.best_score,
            "best_metrics": (
                result.best_metrics if hasattr(result, "best_metrics") else {}
            ),
        }

        with open(self.summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        # Final metrics update
        self._update_metrics_file(None, final=True)

        print("\n📊 Optimization logging complete:")
        print(f"   • Total invocations logged: {self.invocation_count}")
        print(f"   • Logs saved to: {self.log_dir}")
        print("   • Files: invocations.jsonl, summary.json, metrics.json")

    def _log_invocation(
        self, trial: TrialResult, example_result: ExampleResult
    ) -> None:
        """Log a single function invocation."""
        self.invocation_count += 1

        # Prepare invocation record
        invocation = {
            "invocation_id": self.invocation_count,
            "timestamp": datetime.now().isoformat(),
            "trial_id": trial.trial_id,
            "trial_number": self.trial_count,
            "config": trial.config,
            "input_data": example_result.input_data,
            "expected_output": example_result.expected_output,
            "actual_output": example_result.actual_output,
            "success": example_result.success,
            "error_message": example_result.error_message,
            "execution_time": example_result.execution_time,
            "metrics": example_result.metrics,
            "example_id": example_result.example_id,
        }

        # Append to JSONL file
        with open(self.invocations_file, "a") as f:
            f.write(json.dumps(invocation) + "\n")

        # Aggregate metrics
        self.all_metrics.append(example_result.metrics)

    def _log_trial_summary(self, trial: TrialResult) -> None:
        """Log a trial summary when detailed example results aren't available."""
        self.invocation_count += 1

        # Prepare trial summary record
        trial_summary = {
            "invocation_id": self.invocation_count,
            "timestamp": datetime.now().isoformat(),
            "trial_id": trial.trial_id,
            "trial_number": self.trial_count,
            "config": trial.config,
            "metrics": trial.metrics,
            "status": trial.status.value,
            "duration": trial.duration,
            "error_message": trial.error_message,
            "type": "trial_summary",  # Distinguish from individual invocations
        }

        # Append to JSONL file
        with open(self.invocations_file, "a") as f:
            f.write(json.dumps(trial_summary) + "\n")

        # Aggregate metrics
        if trial.metrics:
            self.all_metrics.append(trial.metrics)

    def _config_to_key(self, config: Dict[str, Any]) -> str:
        """Convert config dict to a hashable key."""
        items = sorted(config.items())
        return json.dumps(items)

    def _update_metrics_file(
        self, progress: Optional[ProgressInfo], final: bool = False
    ) -> None:
        """Update the metrics analysis file."""
        metrics_data = {
            "last_updated": datetime.now().isoformat(),
            "total_invocations": self.invocation_count,
            "trials_completed": self.trial_count,
            "is_final": final,
        }

        # Add progress info if available
        if progress:
            metrics_data["progress"] = {
                "percent_complete": progress.progress_percent,
                "success_rate": progress.success_rate,
                "elapsed_time": progress.elapsed_time,
                "best_score": progress.best_score,
                "best_config": progress.best_config,
            }

        # Analyze metrics by objective
        if self.all_metrics:
            objective_stats = {}
            for obj in self.objectives:
                values = [m.get(obj, 0) for m in self.all_metrics if obj in m]
                if values:
                    objective_stats[obj] = {
                        "count": len(values),
                        "mean": sum(values) / len(values),
                        "min": min(values),
                        "max": max(values),
                        "std": self._calculate_std(values),
                    }
            metrics_data["objective_statistics"] = objective_stats

        # Configuration performance analysis
        config_analysis = []
        for _config_key, perf_data in self.config_performance.items():
            # Calculate average metrics for this config
            avg_metrics = {}
            for obj in self.objectives:
                obj_values = [m.get(obj, 0) for m in perf_data["metrics"] if obj in m]
                if obj_values:
                    avg_metrics[obj] = sum(obj_values) / len(obj_values)

            config_analysis.append(
                {
                    "config": perf_data["config"],
                    "invocation_count": perf_data["invocation_count"],
                    "trial_count": len(perf_data["metrics"]),
                    "average_metrics": avg_metrics,
                }
            )

        # Sort by primary objective
        if config_analysis and self.objectives:
            primary_obj = self.objectives[0]
            config_analysis.sort(
                key=lambda x: x["average_metrics"].get(primary_obj, 0), reverse=True
            )

        metrics_data["configuration_analysis"] = config_analysis[:10]  # Top 10 configs

        # Write metrics file
        with open(self.metrics_file, "w") as f:
            json.dump(metrics_data, f, indent=2)

    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance**0.5


class EnhancedDeferredLoggingCallback(OptimizationCallback):
    """
    Enhanced callback that captures detailed LLM invocation data and organizes logs by problem_id/run_id.

    Features:
    - Organizes logs in optimization_logs/problem_id/run_id/ structure
    - Captures comprehensive LLM invocation details (model, temperature, input/output, tokens, cost, time)
    - Safe logging that handles missing data gracefully
    - Deferred file writing for Streamlit compatibility
    """

    def __init__(self, run_id: str, problem_id: str, base_dir: Optional[Path] = None):
        """Initialize the enhanced logging callback."""
        self.run_id = run_id
        self.problem_id = problem_id
        self.base_dir = base_dir or Path(__file__).parent / "optimization_logs"

        # Initialize accumulated data
        self.log_data = {
            "run_id": run_id,
            "problem_id": problem_id,
            "trials": [],
            "invocations": [],
            "metadata": {},
            "metrics": {},
            "llm_usage_summary": {
                "total_tokens_in": 0,
                "total_tokens_out": 0,
                "total_cost": 0.0,
                "total_invocations": 0,
                "models_used": set(),
                "avg_response_time": 0.0,
            },
        }
        self.start_time = None
        self.invocation_count = 0

    def on_optimization_start(
        self, config_space: Dict[str, Any], objectives: List[str], algorithm: str
    ) -> None:
        """Called when optimization starts."""
        print(f"\n[EnhancedLogging] Optimization started for {self.problem_id}")
        self.start_time = time.time()
        self.log_data["metadata"] = {
            "start_time": datetime.now().isoformat(),
            "algorithm": algorithm,
            "objectives": objectives,
            "config_space": config_space,
            "status": "started",
        }

    def on_trial_start(self, trial_number: int, config: Dict[str, Any]) -> None:
        """Called when a trial starts."""
        pass  # We'll log on completion

    def on_trial_complete(self, trial: TrialResult, progress: ProgressInfo) -> None:
        """Called when a trial completes."""
        # Basic trial data
        trial_data = {
            "trial_id": trial.trial_id,
            "trial_number": len(self.log_data["trials"]),
            "timestamp": datetime.now().isoformat(),
            "config": trial.config,
            "metrics": trial.metrics,
            "status": trial.status.value,
            "duration": trial.duration,
            "error_message": trial.error_message,
            "llm_invocations": [],
        }

        # Extract LLM invocation details if available - check both direct attribute and metadata
        example_results = None

        # Strategy 1: Check direct attribute (legacy)
        if hasattr(trial, "example_results") and trial.example_results:
            example_results = trial.example_results
            print(
                f"[ComprehensiveLogging] Found example_results as direct attribute: {len(example_results)} examples"
            )

        # Strategy 2: Check trial metadata (from DetailedLocalEvaluator)
        elif (
            hasattr(trial, "metadata")
            and trial.metadata
            and "example_results" in trial.metadata
        ):
            example_results = trial.metadata["example_results"]
            print(
                f"[ComprehensiveLogging] Found example_results in metadata: {len(example_results)} examples"
            )

        if example_results:
            for example_result in example_results:
                invocation_data = self._extract_llm_invocation_data(
                    trial, example_result
                )
                trial_data["llm_invocations"].append(invocation_data)
                self.log_data["invocations"].append(invocation_data)
                self._update_usage_summary(invocation_data)

        self.log_data["trials"].append(trial_data)
        print(
            f"[EnhancedLogging] Logged trial {trial.trial_id} with {len(trial_data['llm_invocations'])} invocations"
        )

    def on_optimization_complete(self, result: OptimizationResult) -> None:
        """Called when optimization completes - saves all logs to disk."""
        print("\n[EnhancedLogging] Optimization complete, saving enhanced logs...")

        # Update metadata
        self.log_data["metadata"]["end_time"] = datetime.now().isoformat()
        self.log_data["metadata"]["duration_seconds"] = (
            time.time() - self.start_time if self.start_time else 0
        )
        self.log_data["metadata"]["total_trials"] = len(result.trials)
        self.log_data["metadata"]["successful_trials"] = len(result.successful_trials)
        self.log_data["metadata"]["best_config"] = result.best_config
        self.log_data["metadata"]["best_score"] = result.best_score

        # Finalize usage summary
        if self.log_data["invocations"]:
            response_times = [
                inv.get("execution_time", 0) for inv in self.log_data["invocations"]
            ]
            self.log_data["llm_usage_summary"]["avg_response_time"] = sum(
                response_times
            ) / len(response_times)
            self.log_data["llm_usage_summary"]["models_used"] = list(
                self.log_data["llm_usage_summary"]["models_used"]
            )

        # Save logs
        self._save_enhanced_logs()

    def _extract_llm_invocation_data(
        self, trial: TrialResult, example_result: ExampleResult
    ) -> Dict[str, Any]:
        """Extract detailed LLM invocation data from an example result."""
        self.invocation_count += 1

        # Base invocation data
        invocation = {
            "invocation_id": self.invocation_count,
            "timestamp": datetime.now().isoformat(),
            "trial_id": trial.trial_id,
            "example_id": example_result.example_id,
            "execution_time": example_result.execution_time,
            "success": example_result.success,
            "error_message": example_result.error_message,
            # LLM Configuration (safe extraction)
            "llm_config": {
                "model": self._safe_get(trial.config, "model", "unknown"),
                "temperature": self._safe_get(trial.config, "temperature", None),
                "max_tokens": self._safe_get(trial.config, "max_tokens", None),
                "top_p": self._safe_get(trial.config, "top_p", None),
                "frequency_penalty": self._safe_get(
                    trial.config, "frequency_penalty", None
                ),
                "presence_penalty": self._safe_get(
                    trial.config, "presence_penalty", None
                ),
                "system_prompt_style": self._safe_get(
                    trial.config, "system_prompt_style", None
                ),
            },
            # Input/Output Data (safe extraction)
            "input_data": self._safe_extract_data(example_result.input_data),
            "expected_output": self._safe_extract_data(example_result.expected_output),
            "actual_output": self._safe_extract_data(example_result.actual_output),
            # Metrics
            "metrics": example_result.metrics or {},
            # Token usage (if available in metadata)
            "token_usage": self._extract_token_usage(example_result),
            # Cost estimation (if available)
            "estimated_cost": self._estimate_cost(trial.config, example_result),
            # Code reference
            "code_reference": {
                "function_name": getattr(trial, "function_name", None),
                "file_path": getattr(trial, "file_path", None),
                "line_number": getattr(trial, "line_number", None),
            },
        }

        return invocation

    def _safe_get(self, data: Dict[str, Any], key: str, default: Any = None) -> Any:
        """Safely get a value from a dictionary."""
        try:
            return data.get(key, default) if isinstance(data, dict) else default
        except Exception:
            return default

    def _safe_extract_data(self, data: Any) -> Any:
        """Safely extract data, handling various types and potential issues."""
        try:
            if data is None:
                return None
            elif isinstance(data, (str, int, float, bool)):
                return data
            elif isinstance(data, dict):
                # Limit dict size for logging
                if len(str(data)) > 1000:
                    return {"_truncated": True, "_preview": str(data)[:500] + "..."}
                return data
            elif isinstance(data, (list, tuple)):
                # Limit list size for logging
                if len(str(data)) > 1000:
                    return {
                        "_truncated": True,
                        "_type": type(data).__name__,
                        "_preview": str(data)[:500] + "...",
                    }
                return list(data) if isinstance(data, tuple) else data
            else:
                # Convert to string for other types
                str_repr = str(data)
                if len(str_repr) > 1000:
                    return {
                        "_truncated": True,
                        "_type": type(data).__name__,
                        "_preview": str_repr[:500] + "...",
                    }
                return str_repr
        except Exception as e:
            return {
                "_error": f"Failed to extract data: {str(e)}",
                "_type": type(data).__name__,
            }

    def _extract_token_usage(self, example_result: ExampleResult) -> Dict[str, Any]:
        """Extract token usage information if available."""
        token_usage = {"tokens_in": None, "tokens_out": None, "total_tokens": None}

        try:
            # Check various places where token usage might be stored
            metadata = getattr(example_result, "metadata", {}) or {}

            # Common locations for token usage
            if "token_usage" in metadata:
                usage = metadata["token_usage"]
                token_usage.update(
                    {
                        "tokens_in": usage.get(
                            "prompt_tokens", usage.get("input_tokens")
                        ),
                        "tokens_out": usage.get(
                            "completion_tokens", usage.get("output_tokens")
                        ),
                        "total_tokens": usage.get("total_tokens"),
                    }
                )
            elif "usage" in metadata:
                usage = metadata["usage"]
                token_usage.update(
                    {
                        "tokens_in": usage.get(
                            "prompt_tokens", usage.get("input_tokens")
                        ),
                        "tokens_out": usage.get(
                            "completion_tokens", usage.get("output_tokens")
                        ),
                        "total_tokens": usage.get("total_tokens"),
                    }
                )

            # Calculate total if not provided
            if (
                token_usage["tokens_in"]
                and token_usage["tokens_out"]
                and not token_usage["total_tokens"]
            ):
                token_usage["total_tokens"] = (
                    token_usage["tokens_in"] + token_usage["tokens_out"]
                )

        except Exception as e:
            token_usage["_extraction_error"] = str(e)

        return token_usage

    def _estimate_cost(
        self, config: Dict[str, Any], example_result: ExampleResult
    ) -> Dict[str, Any]:
        """Estimate the cost of the LLM invocation."""
        cost_info = {
            "estimated_cost": None,
            "currency": "USD",
            "pricing_model": "estimated",
        }

        try:
            model = config.get("model", "").lower()
            token_usage = self._extract_token_usage(example_result)

            # Simple cost estimation (these are rough estimates)
            cost_per_1k_tokens = {
                "gpt-4": {"input": 0.03, "output": 0.06},
                "gpt-4o": {"input": 0.005, "output": 0.015},
                "claude-3": {"input": 0.015, "output": 0.075},
                "claude-3-sonnet": {"input": 0.003, "output": 0.015},
                "o4-mini": {"input": 0.0001, "output": 0.0004},
            }

            # Find matching model pricing
            pricing = None
            for model_key, model_pricing in cost_per_1k_tokens.items():
                if model_key in model:
                    pricing = model_pricing
                    break

            if (
                pricing
                and token_usage.get("tokens_in")
                and token_usage.get("tokens_out")
            ):
                input_cost = (token_usage["tokens_in"] / 1000) * pricing["input"]
                output_cost = (token_usage["tokens_out"] / 1000) * pricing["output"]
                cost_info["estimated_cost"] = input_cost + output_cost
                cost_info["breakdown"] = {
                    "input_tokens": token_usage["tokens_in"],
                    "output_tokens": token_usage["tokens_out"],
                    "input_cost": input_cost,
                    "output_cost": output_cost,
                }

        except Exception as e:
            cost_info["_estimation_error"] = str(e)

        return cost_info

    def _update_usage_summary(self, invocation_data: Dict[str, Any]) -> None:
        """Update the overall usage summary."""
        try:
            summary = self.log_data["llm_usage_summary"]
            summary["total_invocations"] += 1

            # Update token counts
            token_usage = invocation_data.get("token_usage", {})
            if token_usage.get("tokens_in"):
                summary["total_tokens_in"] += token_usage["tokens_in"]
            if token_usage.get("tokens_out"):
                summary["total_tokens_out"] += token_usage["tokens_out"]

            # Update cost
            cost_info = invocation_data.get("estimated_cost", {})
            if cost_info.get("estimated_cost"):
                summary["total_cost"] += cost_info["estimated_cost"]

            # Track models used
            model = invocation_data.get("llm_config", {}).get("model")
            if model and model != "unknown":
                summary["models_used"].add(model)

        except Exception as e:
            print(f"[EnhancedLogging] Warning: Failed to update usage summary: {e}")

    def _save_enhanced_logs(self):
        """Save accumulated logs to disk with problem_id/run_id structure."""
        try:
            # Create log directory with problem_id/run_id structure
            log_dir = self.base_dir / self.problem_id / self.run_id
            log_dir.mkdir(parents=True, exist_ok=True)

            # Save comprehensive log file
            main_log_file = log_dir / "optimization_log.json"
            with open(main_log_file, "w") as f:
                json.dump(self.log_data, f, indent=2, default=str)

            # Save individual trial files with detailed invocation data
            for trial in self.log_data["trials"]:
                trial_file = log_dir / f"trial_{trial['trial_id']}.json"
                with open(trial_file, "w") as f:
                    json.dump(trial, f, indent=2, default=str)

            # Save invocations in JSONL format for easy processing
            invocations_file = log_dir / "invocations.jsonl"
            with open(invocations_file, "w") as f:
                for invocation in self.log_data["invocations"]:
                    f.write(json.dumps(invocation, default=str) + "\n")

            # Save usage summary
            summary_file = log_dir / "llm_usage_summary.json"
            with open(summary_file, "w") as f:
                json.dump(self.log_data["llm_usage_summary"], f, indent=2, default=str)

            print(f"[EnhancedLogging] Enhanced logs saved to: {log_dir}")
            print(f"  - Main log: {main_log_file.name}")
            print(f"  - Invocations: {invocations_file.name}")
            print(f"  - Usage summary: {summary_file.name}")
            print(f"  - Individual trial files: {len(self.log_data['trials'])} files")

        except Exception as e:
            print(f"[EnhancedLogging] ERROR saving enhanced logs: {e}")
            import traceback

            traceback.print_exc()


class ComprehensiveTrialLoggingCallback(OptimizationCallback):
    """
    Comprehensive callback that embeds detailed LLM invocation data directly in trial files.

    Features:
    - Organizes logs in optimization_logs/problem_id/run_id/ structure
    - Embeds all LLM invocation details within each trial file (no separate invocations.jsonl)
    - Extensive debugging to understand data availability
    - Smart data extraction with multiple fallback sources
    - Deferred file writing for Streamlit compatibility
    - Safe logging that handles missing data gracefully
    """

    def __init__(self, run_id: str, problem_id: str, base_dir: Optional[Path] = None):
        """Initialize the comprehensive trial logging callback."""
        self.run_id = run_id
        self.problem_id = problem_id
        self.base_dir = base_dir or Path(__file__).parent / "optimization_logs"

        # Initialize accumulated data
        self.log_data = {
            "run_id": run_id,
            "problem_id": problem_id,
            "trials": [],
            "metadata": {},
            "llm_usage_summary": {
                "total_tokens_in": 0,
                "total_tokens_out": 0,
                "total_cost": 0.0,
                "total_invocations": 0,
                "models_used": set(),
                "avg_response_time": 0.0,
            },
        }
        self.start_time = None
        self.invocation_count = 0

    def on_optimization_start(
        self, config_space: Dict[str, Any], objectives: List[str], algorithm: str
    ) -> None:
        """Called when optimization starts."""
        print(f"\n[ComprehensiveLogging] Optimization started for {self.problem_id}")
        self.start_time = time.time()
        self.log_data["metadata"] = {
            "start_time": datetime.now().isoformat(),
            "algorithm": algorithm,
            "objectives": objectives,
            "config_space": config_space,
            "status": "started",
        }

    def on_trial_start(self, trial_number: int, config: Dict[str, Any]) -> None:
        """Called when a trial starts."""
        print(
            f"[ComprehensiveLogging] Trial {trial_number} starting with config: {config}"
        )

    def on_trial_complete(self, trial: TrialResult, progress: ProgressInfo) -> None:
        """Called when a trial completes - with extensive debugging."""
        print("\n[ComprehensiveLogging] === TRIAL COMPLETE DEBUG ===")
        print(f"Trial ID: {trial.trial_id}")
        print(f"Trial type: {type(trial)}")
        print(f"Trial attributes: {list(trial.__dict__.keys())}")

        # Check for example_results
        has_example_results = hasattr(trial, "example_results")
        print(f"Has example_results attribute: {has_example_results}")

        if has_example_results:
            example_results = trial.example_results
            print(f"example_results type: {type(example_results)}")
            print(f"example_results value: {example_results}")
            if example_results:
                print(f"example_results length: {len(example_results)}")
                if len(example_results) > 0:
                    print(f"First example_result type: {type(example_results[0])}")
                    print(
                        f"First example_result attributes: {list(example_results[0].__dict__.keys())}"
                    )

        # Check trial metadata
        if hasattr(trial, "metadata"):
            print(f"Trial metadata: {trial.metadata}")

        # Basic trial data
        trial_data = {
            "trial_id": trial.trial_id,
            "trial_number": len(self.log_data["trials"]),
            "timestamp": datetime.now().isoformat(),
            "config": trial.config,
            "metrics": trial.metrics,
            "status": trial.status.value,
            "duration": trial.duration,
            "error_message": trial.error_message,
            "llm_invocations": [],
            "trial_usage_summary": {
                "total_invocations": 0,
                "total_tokens_in": 0,
                "total_tokens_out": 0,
                "total_cost": 0.0,
                "avg_execution_time": 0.0,
            },
        }

        # Try to extract detailed invocation data
        invocations = self._extract_trial_invocations(trial)
        trial_data["llm_invocations"] = invocations

        # Calculate trial-level usage summary
        if invocations:
            trial_data["trial_usage_summary"] = self._calculate_trial_summary(
                invocations
            )
            # Update global usage summary
            for invocation in invocations:
                self._update_global_usage_summary(invocation)

        self.log_data["trials"].append(trial_data)
        print(
            f"[ComprehensiveLogging] Logged trial {trial.trial_id} with {len(invocations)} invocations"
        )

    def on_optimization_complete(self, result: OptimizationResult) -> None:
        """Called when optimization completes - saves all logs to disk."""
        print(
            "\n[ComprehensiveLogging] Optimization complete, saving comprehensive logs..."
        )

        # Update metadata
        self.log_data["metadata"]["end_time"] = datetime.now().isoformat()
        self.log_data["metadata"]["duration_seconds"] = (
            time.time() - self.start_time if self.start_time else 0
        )
        self.log_data["metadata"]["total_trials"] = len(result.trials)
        self.log_data["metadata"]["successful_trials"] = len(result.successful_trials)
        self.log_data["metadata"]["best_config"] = result.best_config
        self.log_data["metadata"]["best_score"] = result.best_score

        # Finalize usage summary
        if self.log_data["llm_usage_summary"]["total_invocations"] > 0:
            all_execution_times = []
            for trial in self.log_data["trials"]:
                for invocation in trial.get("llm_invocations", []):
                    if invocation.get("execution_time"):
                        all_execution_times.append(invocation["execution_time"])

            if all_execution_times:
                self.log_data["llm_usage_summary"]["avg_response_time"] = sum(
                    all_execution_times
                ) / len(all_execution_times)

            self.log_data["llm_usage_summary"]["models_used"] = list(
                self.log_data["llm_usage_summary"]["models_used"]
            )

        # Save logs
        self._save_comprehensive_logs()

    def _extract_trial_invocations(self, trial: TrialResult) -> List[Dict[str, Any]]:
        """Extract all available invocation data from a trial using multiple strategies."""
        invocations = []

        print(
            f"\n[ComprehensiveLogging] Extracting invocations for trial {trial.trial_id}"
        )

        # Strategy 1: Check for example_results (detailed evaluation results)
        example_results = None

        # Check direct attribute first
        if hasattr(trial, "example_results") and trial.example_results:
            example_results = trial.example_results
            print(
                f"Strategy 1a: Found {len(example_results)} example_results as direct attribute"
            )

        # Check trial metadata (from DetailedLocalEvaluator)
        elif (
            hasattr(trial, "metadata")
            and trial.metadata
            and "example_results" in trial.metadata
        ):
            example_results = trial.metadata["example_results"]
            print(
                f"Strategy 1b: Found {len(example_results)} example_results in trial metadata"
            )

        if example_results:
            for i, example_result in enumerate(example_results):
                invocation = self._extract_from_example_result(trial, example_result, i)
                invocations.append(invocation)

        # Strategy 2: Check trial metadata for invocation data
        elif hasattr(trial, "metadata") and trial.metadata:
            print("Strategy 2: Extracting from trial metadata")
            invocation = self._extract_from_trial_metadata(trial)
            invocations.append(invocation)

        # Strategy 3: Create synthetic invocation from trial-level data
        else:
            print("Strategy 3: Creating synthetic invocation from trial data")
            invocation = self._create_synthetic_invocation(trial)
            invocations.append(invocation)

        print(f"Extracted {len(invocations)} invocations")
        return invocations

    def _extract_from_example_result(
        self, trial: TrialResult, example_result: ExampleResult, index: int
    ) -> Dict[str, Any]:
        """Extract detailed invocation data from an ExampleResult."""
        self.invocation_count += 1

        print(f"  Extracting from example_result {index}:")
        print(f"    Type: {type(example_result)}")
        print(f"    Attributes: {list(example_result.__dict__.keys())}")
        print(f"    input_data: {getattr(example_result, 'input_data', 'NOT_FOUND')}")
        print(
            f"    expected_output: {getattr(example_result, 'expected_output', 'NOT_FOUND')}"
        )
        print(
            f"    actual_output: {getattr(example_result, 'actual_output', 'NOT_FOUND')}"
        )
        print(f"    Raw data dump: {example_result.__dict__}")

        invocation = {
            "invocation_id": self.invocation_count,
            "timestamp": datetime.now().isoformat(),
            "trial_id": trial.trial_id,
            "example_index": index,
            "example_id": getattr(example_result, "example_id", f"example_{index}"),
            "execution_time": getattr(example_result, "execution_time", 0.0),
            "success": getattr(example_result, "success", True),
            "error_message": getattr(example_result, "error_message", None),
            # LLM Configuration
            "llm_config": {
                "model": self._safe_get(trial.config, "model", "unknown"),
                "temperature": self._safe_get(trial.config, "temperature", None),
                "max_tokens": self._safe_get(trial.config, "max_tokens", None),
                "top_p": self._safe_get(trial.config, "top_p", None),
                "frequency_penalty": self._safe_get(
                    trial.config, "frequency_penalty", None
                ),
                "presence_penalty": self._safe_get(
                    trial.config, "presence_penalty", None
                ),
                "system_prompt_style": self._safe_get(
                    trial.config, "system_prompt_style", None
                ),
            },
            # Input/Output Data
            "input_data": self._safe_extract_data(
                getattr(example_result, "input_data", None)
            ),
            "expected_output": self._safe_extract_data(
                getattr(example_result, "expected_output", None)
            ),
            "actual_output": self._safe_extract_data(
                getattr(example_result, "actual_output", None)
            ),
            # Metrics
            "metrics": getattr(example_result, "metrics", {}),
            # Token usage and cost
            "token_usage": self._extract_token_usage_from_example(example_result),
            "estimated_cost": self._estimate_cost_from_config_and_tokens(
                trial.config, example_result
            ),
        }

        return invocation

    def _extract_from_trial_metadata(self, trial: TrialResult) -> Dict[str, Any]:
        """Extract invocation data from trial metadata."""
        self.invocation_count += 1

        print(f"  Extracting from trial metadata: {trial.metadata}")

        invocation = {
            "invocation_id": self.invocation_count,
            "timestamp": datetime.now().isoformat(),
            "trial_id": trial.trial_id,
            "execution_time": trial.duration,
            "success": trial.status.value == "completed",
            "error_message": trial.error_message,
            # LLM Configuration
            "llm_config": {
                "model": self._safe_get(trial.config, "model", "unknown"),
                "temperature": self._safe_get(trial.config, "temperature", None),
                "max_tokens": self._safe_get(trial.config, "max_tokens", None),
                "top_p": self._safe_get(trial.config, "top_p", None),
                "frequency_penalty": self._safe_get(
                    trial.config, "frequency_penalty", None
                ),
                "presence_penalty": self._safe_get(
                    trial.config, "presence_penalty", None
                ),
                "system_prompt_style": self._safe_get(
                    trial.config, "system_prompt_style", None
                ),
            },
            # Data from metadata if available
            "input_data": self._safe_get(trial.metadata, "input_data", None),
            "expected_output": self._safe_get(trial.metadata, "expected_output", None),
            "actual_output": self._safe_get(trial.metadata, "actual_output", None),
            # Metrics
            "metrics": trial.metrics or {},
            # Token usage from metadata
            "token_usage": self._extract_token_usage_from_metadata(trial.metadata),
            "estimated_cost": self._estimate_cost_from_metadata(
                trial.config, trial.metadata
            ),
            "data_source": "trial_metadata",
        }

        return invocation

    def _create_synthetic_invocation(self, trial: TrialResult) -> Dict[str, Any]:
        """Create a synthetic invocation record from basic trial data."""
        self.invocation_count += 1

        print("  Creating synthetic invocation from trial data")
        print(
            f"    Trial has {len(trial.__dict__)} attributes: {list(trial.__dict__.keys())}"
        )
        print(f"    Trial metadata: {getattr(trial, 'metadata', 'NOT_FOUND')}")

        # Try to extract any available input/output data from trial metadata or other sources
        input_data = None
        expected_output = None
        actual_output = None

        # Check if trial has metadata with input/output information
        if hasattr(trial, "metadata") and trial.metadata:
            input_data = trial.metadata.get("input_data")
            expected_output = trial.metadata.get("expected_output")
            actual_output = trial.metadata.get("actual_output")
            print(
                f"    Found in metadata - input: {input_data}, expected: {expected_output}, actual: {actual_output}"
            )

        # For customer support, we know the structure - try to infer from available data
        if not input_data and trial.metrics:
            # Sometimes the function parameters might be stored elsewhere
            print(f"    Trial metrics: {trial.metrics}")

        invocation = {
            "invocation_id": self.invocation_count,
            "timestamp": datetime.now().isoformat(),
            "trial_id": trial.trial_id,
            "execution_time": trial.duration,
            "success": trial.status.value == "completed",
            "error_message": trial.error_message,
            # LLM Configuration
            "llm_config": {
                "model": self._safe_get(trial.config, "model", "unknown"),
                "temperature": self._safe_get(trial.config, "temperature", None),
                "max_tokens": self._safe_get(trial.config, "max_tokens", None),
                "top_p": self._safe_get(trial.config, "top_p", None),
                "frequency_penalty": self._safe_get(
                    trial.config, "frequency_penalty", None
                ),
                "presence_penalty": self._safe_get(
                    trial.config, "presence_penalty", None
                ),
                "system_prompt_style": self._safe_get(
                    trial.config, "system_prompt_style", None
                ),
            },
            # Data from metadata or None if not available
            "input_data": self._safe_extract_data(input_data),
            "expected_output": self._safe_extract_data(expected_output),
            "actual_output": self._safe_extract_data(actual_output),
            # Metrics from trial
            "metrics": trial.metrics or {},
            # Try to estimate token usage from metrics if available
            "token_usage": self._extract_token_usage_from_metrics(trial.metrics),
            "estimated_cost": self._estimate_cost_from_trial_data(
                trial.config, trial.metrics
            ),
            "data_source": "synthetic_from_trial",
            "note": "Limited data available - extracted from trial-level information",
        }

        return invocation

    def _extract_token_usage_from_example(
        self, example_result: ExampleResult
    ) -> Dict[str, Any]:
        """Extract token usage from ExampleResult metadata."""
        token_usage = {"tokens_in": None, "tokens_out": None, "total_tokens": None}

        try:
            if hasattr(example_result, "metadata") and example_result.metadata:
                metadata = example_result.metadata

                # Common token usage locations
                for usage_key in ["token_usage", "usage", "llm_usage"]:
                    if usage_key in metadata:
                        usage = metadata[usage_key]
                        token_usage.update(
                            {
                                "tokens_in": usage.get(
                                    "prompt_tokens", usage.get("input_tokens")
                                ),
                                "tokens_out": usage.get(
                                    "completion_tokens", usage.get("output_tokens")
                                ),
                                "total_tokens": usage.get("total_tokens"),
                            }
                        )
                        break

                # Calculate total if not provided
                if (
                    token_usage["tokens_in"]
                    and token_usage["tokens_out"]
                    and not token_usage["total_tokens"]
                ):
                    token_usage["total_tokens"] = (
                        token_usage["tokens_in"] + token_usage["tokens_out"]
                    )

        except Exception as e:
            token_usage["_extraction_error"] = str(e)

        return token_usage

    def _extract_token_usage_from_metadata(
        self, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract token usage from trial metadata."""
        token_usage = {"tokens_in": None, "tokens_out": None, "total_tokens": None}

        try:
            if metadata:
                for usage_key in ["token_usage", "usage", "llm_usage"]:
                    if usage_key in metadata:
                        usage = metadata[usage_key]
                        token_usage.update(
                            {
                                "tokens_in": usage.get(
                                    "prompt_tokens", usage.get("input_tokens")
                                ),
                                "tokens_out": usage.get(
                                    "completion_tokens", usage.get("output_tokens")
                                ),
                                "total_tokens": usage.get("total_tokens"),
                            }
                        )
                        break

        except Exception as e:
            token_usage["_extraction_error"] = str(e)

        return token_usage

    def _estimate_cost_from_config_and_tokens(
        self, config: Dict[str, Any], example_result: ExampleResult
    ) -> Dict[str, Any]:
        """Estimate cost from config and token usage."""
        cost_info = {
            "estimated_cost": None,
            "currency": "USD",
            "pricing_model": "estimated",
        }

        try:
            model = config.get("model", "").lower()
            token_usage = self._extract_token_usage_from_example(example_result)
            cost_info.update(self._calculate_cost_from_tokens(model, token_usage))
        except Exception as e:
            cost_info["_estimation_error"] = str(e)

        return cost_info

    def _estimate_cost_from_metadata(
        self, config: Dict[str, Any], metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Estimate cost from config and metadata."""
        cost_info = {
            "estimated_cost": None,
            "currency": "USD",
            "pricing_model": "estimated",
        }

        try:
            model = config.get("model", "").lower()
            token_usage = self._extract_token_usage_from_metadata(metadata)
            cost_info.update(self._calculate_cost_from_tokens(model, token_usage))
        except Exception as e:
            cost_info["_estimation_error"] = str(e)

        return cost_info

    def _calculate_cost_from_tokens(
        self, model: str, token_usage: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate cost from model and token usage."""
        cost_info = {}

        # Pricing per 1k tokens (rough estimates)
        cost_per_1k_tokens = {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4o": {"input": 0.005, "output": 0.015},
            "claude-3": {"input": 0.015, "output": 0.075},
            "claude-3-sonnet": {"input": 0.003, "output": 0.015},
            "o4-mini": {"input": 0.0001, "output": 0.0004},
        }

        # Find matching model pricing
        pricing = None
        for model_key, model_pricing in cost_per_1k_tokens.items():
            if model_key in model:
                pricing = model_pricing
                break

        if pricing and token_usage.get("tokens_in") and token_usage.get("tokens_out"):
            input_cost = (token_usage["tokens_in"] / 1000) * pricing["input"]
            output_cost = (token_usage["tokens_out"] / 1000) * pricing["output"]
            cost_info["estimated_cost"] = input_cost + output_cost
            cost_info["breakdown"] = {
                "input_tokens": token_usage["tokens_in"],
                "output_tokens": token_usage["tokens_out"],
                "input_cost": input_cost,
                "output_cost": output_cost,
            }

        return cost_info

    def _extract_token_usage_from_metrics(
        self, metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Try to extract token usage from trial metrics."""
        token_usage = {"tokens_in": None, "tokens_out": None, "total_tokens": None}

        try:
            if metrics:
                # Look for token information in metrics
                for key in ["tokens_used", "token_count", "llm_tokens"]:
                    if key in metrics:
                        token_data = metrics[key]
                        if isinstance(token_data, dict):
                            token_usage.update(
                                {
                                    "tokens_in": token_data.get(
                                        "input_tokens", token_data.get("prompt_tokens")
                                    ),
                                    "tokens_out": token_data.get(
                                        "output_tokens",
                                        token_data.get("completion_tokens"),
                                    ),
                                    "total_tokens": token_data.get("total_tokens"),
                                }
                            )
                        elif isinstance(token_data, (int, float)):
                            token_usage["total_tokens"] = token_data
                        break

        except Exception as e:
            token_usage["_extraction_error"] = str(e)

        return token_usage

    def _estimate_cost_from_trial_data(
        self, config: Dict[str, Any], metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Estimate cost from trial config and metrics."""
        cost_info = {
            "estimated_cost": None,
            "currency": "USD",
            "pricing_model": "estimated",
        }

        try:
            model = config.get("model", "").lower()
            token_usage = self._extract_token_usage_from_metrics(metrics)
            cost_info.update(self._calculate_cost_from_tokens(model, token_usage))
        except Exception as e:
            cost_info["_estimation_error"] = str(e)

        return cost_info

    def _calculate_trial_summary(
        self, invocations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate usage summary for a single trial."""
        summary = {
            "total_invocations": len(invocations),
            "total_tokens_in": 0,
            "total_tokens_out": 0,
            "total_cost": 0.0,
            "avg_execution_time": 0.0,
        }

        execution_times = []
        for invocation in invocations:
            # Token counts
            token_usage = invocation.get("token_usage", {})
            if token_usage.get("tokens_in"):
                summary["total_tokens_in"] += token_usage["tokens_in"]
            if token_usage.get("tokens_out"):
                summary["total_tokens_out"] += token_usage["tokens_out"]

            # Cost
            cost_info = invocation.get("estimated_cost", {})
            if cost_info.get("estimated_cost"):
                summary["total_cost"] += cost_info["estimated_cost"]

            # Execution time
            if invocation.get("execution_time"):
                execution_times.append(invocation["execution_time"])

        if execution_times:
            summary["avg_execution_time"] = sum(execution_times) / len(execution_times)

        return summary

    def _update_global_usage_summary(self, invocation: Dict[str, Any]) -> None:
        """Update the global usage summary."""
        try:
            summary = self.log_data["llm_usage_summary"]
            summary["total_invocations"] += 1

            # Update token counts
            token_usage = invocation.get("token_usage", {})
            if token_usage.get("tokens_in"):
                summary["total_tokens_in"] += token_usage["tokens_in"]
            if token_usage.get("tokens_out"):
                summary["total_tokens_out"] += token_usage["tokens_out"]

            # Update cost
            cost_info = invocation.get("estimated_cost", {})
            if cost_info.get("estimated_cost"):
                summary["total_cost"] += cost_info["estimated_cost"]

            # Track models used
            model = invocation.get("llm_config", {}).get("model")
            if model and model != "unknown":
                summary["models_used"].add(model)

        except Exception as e:
            print(
                f"[ComprehensiveLogging] Warning: Failed to update usage summary: {e}"
            )

    def _safe_get(self, data: Dict[str, Any], key: str, default: Any = None) -> Any:
        """Safely get a value from a dictionary."""
        try:
            return data.get(key, default) if isinstance(data, dict) else default
        except Exception:
            return default

    def _safe_extract_data(self, data: Any) -> Any:
        """Safely extract data, handling various types and potential issues."""
        try:
            if data is None:
                return None
            elif isinstance(data, (str, int, float, bool)):
                return data
            elif isinstance(data, dict):
                # Limit dict size for logging
                if len(str(data)) > 1000:
                    return {"_truncated": True, "_preview": str(data)[:500] + "..."}
                return data
            elif isinstance(data, (list, tuple)):
                # Limit list size for logging
                if len(str(data)) > 1000:
                    return {
                        "_truncated": True,
                        "_type": type(data).__name__,
                        "_preview": str(data)[:500] + "...",
                    }
                return list(data) if isinstance(data, tuple) else data
            else:
                # Convert to string for other types
                str_repr = str(data)
                if len(str_repr) > 1000:
                    return {
                        "_truncated": True,
                        "_type": type(data).__name__,
                        "_preview": str_repr[:500] + "...",
                    }
                return str_repr
        except Exception as e:
            return {
                "_error": f"Failed to extract data: {str(e)}",
                "_type": type(data).__name__,
            }

    def _save_comprehensive_logs(self):
        """Save accumulated logs to disk with problem_id/run_id structure."""
        try:
            # Create log directory with problem_id/run_id structure
            log_dir = self.base_dir / self.problem_id / self.run_id
            log_dir.mkdir(parents=True, exist_ok=True)

            # Save comprehensive main log file
            main_log_file = log_dir / "optimization_log.json"
            with open(main_log_file, "w") as f:
                json.dump(self.log_data, f, indent=2, default=str)

            # Save individual trial files with embedded invocation data
            for trial in self.log_data["trials"]:
                trial_file = log_dir / f"trial_{trial['trial_id']}.json"
                with open(trial_file, "w") as f:
                    json.dump(trial, f, indent=2, default=str)

            # Save usage summary
            summary_file = log_dir / "llm_usage_summary.json"
            with open(summary_file, "w") as f:
                json.dump(self.log_data["llm_usage_summary"], f, indent=2, default=str)

            print(f"[ComprehensiveLogging] Comprehensive logs saved to: {log_dir}")
            print(f"  - Main log: {main_log_file.name}")
            print(f"  - Usage summary: {summary_file.name}")
            print(
                f"  - Trial files with embedded invocations: {len(self.log_data['trials'])} files"
            )

        except Exception as e:
            print(f"[ComprehensiveLogging] ERROR saving comprehensive logs: {e}")
            import traceback

            traceback.print_exc()


class StreamlitProgressCallback(OptimizationCallback):
    """
    Callback for updating Streamlit UI during optimization.

    This callback updates a Streamlit progress bar and status text
    to show real-time optimization progress.
    """

    def __init__(self, progress_bar, status_text):
        """
        Initialize Streamlit progress callback.

        Args:
            progress_bar: Streamlit progress bar widget
            status_text: Streamlit text widget for status updates
        """
        self.progress_bar = progress_bar
        self.status_text = status_text
        self.start_time = None

    def on_optimization_start(
        self, config_space: Dict[str, Any], objectives: List[str], algorithm: str
    ) -> None:
        """Called when optimization starts."""
        self.start_time = time.time()
        self.status_text.text(f"Starting {algorithm} optimization...")
        self.progress_bar.progress(0.0)

    def on_trial_start(self, trial_number: int, config: Dict[str, Any]) -> None:
        """Called when a trial starts."""
        model = config.get("model", "unknown")
        self.status_text.text(f"Testing configuration {trial_number}: {model}")

    def on_trial_complete(self, trial: TrialResult, progress: ProgressInfo) -> None:
        """Called when a trial completes."""
        # Update progress bar
        self.progress_bar.progress(progress.progress_percent / 100.0)

        # Update status with more detail
        status = f"Trial {progress.completed_trials}/{progress.total_trials} | "
        status += f"Success rate: {progress.success_rate:.1f}% | "
        if progress.best_score is not None:
            status += f"Best score: {progress.best_score:.3f}"

        self.status_text.text(status)

    def on_optimization_complete(self, result: OptimizationResult) -> None:
        """Called when optimization completes."""
        self.progress_bar.progress(1.0)
        duration = time.time() - self.start_time if self.start_time else 0
        self.status_text.text(f"Optimization complete! ({duration:.1f}s)")
