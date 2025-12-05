"""Execution adapters for different execution modes."""

# Traceability: CONC-Layer-Integration CONC-Quality-Compatibility FUNC-AGENTS REQ-AGNT-013

import inspect
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, cast

from traigent.config.types import ExecutionMode

logger = logging.getLogger(__name__)


class ExecutionAdapter(ABC):
    """Base class for execution adapters."""

    @abstractmethod
    async def execute_configuration(
        self, agent_spec: dict[str, Any], dataset: dict[str, Any], trial_id: str
    ) -> dict[str, Any]:
        """Execute a configuration and return metrics.

        Args:
            agent_spec: Agent specification
            dataset: Dataset to evaluate on
            trial_id: Trial identifier

        Returns:
            Execution results with metrics
        """
        pass

    @abstractmethod
    async def get_execution_mode(self) -> str:
        """Get the current execution mode.

        Returns:
            Execution mode string
        """
        pass


class LocalExecutionAdapter(ExecutionAdapter):
    """Adapter for Edge Analytics execution (formerly "local" mode)."""

    def __init__(self, agent_builder: Any, progress_interval: int = 10) -> None:
        """Initialize local execution adapter.

        Args:
            agent_builder: Agent builder instance
            progress_interval: Log progress every N examples (debug level). 0 disables.
        """
        self.agent_builder = agent_builder
        self.progress_interval = max(0, int(progress_interval))

    async def execute_configuration(
        self, agent_spec: dict[str, Any], dataset: Any, trial_id: str
    ) -> dict[str, Any]:
        """Execute configuration on the client (Edge Analytics).

        Args:
            agent_spec: Agent specification
            dataset: Dataset to evaluate on
            trial_id: Trial identifier

        Returns:
            Execution results with metrics
        """
        try:
            # Build agent from specification
            agent = self.agent_builder.build_agent(agent_spec)

            # Normalize dataset/examples (support dict or Dataset from core types)
            examples: list[dict[str, Any]] = self._normalize_examples(dataset)

            # Execute on dataset
            results = []
            execution_times = []
            total_examples = len(examples)

            for idx, example in enumerate(examples):
                start_time = time.perf_counter()

                try:
                    # Execute agent on example
                    inp = example.get("input")
                    # Prefer awaitable detection over function-declaration detection
                    res = None
                    try:
                        res = agent.execute(inp)
                    except TypeError:
                        # Some agents may expect unpacked inputs (dict)
                        res = (
                            agent.execute(**inp)
                            if isinstance(inp, dict)
                            else agent.execute(inp)
                        )

                    output = await res if inspect.isawaitable(res) else res

                    # Evaluate result
                    evaluation = self._evaluate_output(
                        output,
                        example.get("expected_output"),
                        example.get("metadata", {}),
                    )

                    results.append(evaluation)
                    execution_times.append(time.perf_counter() - start_time)

                    # Log progress periodically
                    if (
                        self.progress_interval
                        and (idx + 1) % self.progress_interval == 0
                    ):
                        logger.debug(
                            "Processed %d/%d examples", idx + 1, total_examples
                        )

                except Exception as e:
                    logger.error(f"Failed to execute example {idx}: {str(e)}")
                    results.append(
                        {
                            "success": False,
                            "error": str(e),
                            "example_id": example.get("id", idx),
                        }
                    )
                    execution_times.append(time.perf_counter() - start_time)

            # Calculate metrics
            metrics = self._calculate_metrics(results)

            return {
                "trial_id": trial_id,
                "metrics": metrics,
                "execution_time": sum(execution_times),
                "metadata": {
                    "examples_processed": len(results),
                    "failures": sum(1 for r in results if not r.get("success", True)),
                    "average_time_per_example": (
                        sum(execution_times) / len(execution_times)
                        if execution_times
                        else 0
                    ),
                    "agent_spec": self._sanitize_agent_spec(agent_spec),
                },
            }

        except Exception as e:
            logger.error(f"Edge Analytics execution failed: {str(e)}")
            raise

    async def get_execution_mode(self) -> str:
        """Get execution mode."""
        return ExecutionMode.EDGE_ANALYTICS.value

    def _evaluate_output(
        self, output: Any, expected: Any, metadata: dict[str, Any]
    ) -> dict[str, Any]:
        """Evaluate output against expected result.

        Args:
            output: Agent output
            expected: Expected output
            metadata: Example metadata

        Returns:
            Evaluation result
        """
        # Handle different evaluation types
        eval_type = metadata.get("evaluation_type", "exact_match")

        if expected is None:
            # No ground truth available
            return {"success": True, "output": output, "evaluation_type": eval_type}

        result = {"success": True, "output": output, "expected": expected}
        result["evaluation_type"] = eval_type

        if eval_type == "exact_match":
            # Simple exact match
            is_correct = str(output).strip() == str(expected).strip()
            result["correct"] = is_correct

        elif eval_type == "contains":
            # Check if output contains expected
            is_correct = str(expected) in str(output)
            result["correct"] = is_correct

        elif eval_type == "numeric":
            # Numeric comparison with tolerance
            try:
                output_num = float(output)
                expected_num = float(expected)
                tolerance = metadata.get("tolerance", 0.01)
                is_correct = abs(output_num - expected_num) <= tolerance
                result["correct"] = is_correct
                result["difference"] = abs(output_num - expected_num)
            except (ValueError, TypeError):
                result["correct"] = False
                result["error"] = "Failed to parse numeric values"

        elif eval_type == "semantic":
            # Semantic similarity (placeholder)
            # In production, this would use embeddings or LLM evaluation
            result["correct"] = None
            result["requires_semantic_eval"] = True

        else:
            # Unknown evaluation type
            result["correct"] = None
            result["unknown_eval_type"] = eval_type

        return result

    # -------------------- helpers --------------------
    def _normalize_examples(self, dataset: Any) -> list[dict[str, Any]]:
        """Normalize incoming dataset to a list of dict examples with validation.

        Supports dict with 'examples' key or Dataset from core evaluators.
        Raises ValueError with a clear message on schema errors.
        """
        examples: list[dict[str, Any]] = []
        if isinstance(dataset, dict):
            raw = dataset.get("examples")
            if raw is None or not isinstance(raw, list):
                raise ValueError(
                    "dataset['examples'] must be a non-empty list"
                ) from None
            for idx, ex in enumerate(raw):
                if not isinstance(ex, dict):
                    raise ValueError(f"example at index {idx} must be a dict")
                if "input" not in ex:
                    raise ValueError(f"example at index {idx} missing 'input'")
                meta = ex.get("metadata") or {}
                if not isinstance(meta, dict):
                    raise ValueError(f"example at index {idx} has non-dict 'metadata'")
                examples.append(
                    {
                        "id": ex.get("id", idx),
                        "input": ex["input"],
                        "expected_output": ex.get("expected_output"),
                        "metadata": meta,
                    }
                )
            return examples

        # Lazy import Dataset to avoid hard dependency
        try:
            from traigent.evaluators.base import Dataset as _Dataset
        except Exception:  # pragma: no cover
            _Dataset = None  # type: ignore

        if _Dataset is not None and isinstance(dataset, _Dataset):
            for idx, ex in enumerate(dataset.examples):
                meta = ex.metadata or {}
                if not isinstance(meta, dict):
                    raise ValueError(f"example at index {idx} has non-dict metadata")
                examples.append(
                    {
                        "id": meta.get("example_id", idx),
                        "input": ex.input_data,
                        "expected_output": ex.expected_output,
                        "metadata": meta,
                    }
                )
            return examples

        raise TypeError("dataset must be a dict with 'examples' or a Dataset instance")

    def _sanitize_agent_spec(self, agent_spec: dict[str, Any]) -> dict[str, Any]:
        """Redact potentially sensitive fields in agent_spec.

        Redacts keys containing: 'api', 'key', 'secret', 'token' (case-insensitive).
        Applies recursively for nested dicts.
        """

        def _redact(d: Any) -> Any:
            if isinstance(d, dict):
                out: dict[str, Any] = {}
                for k, v in d.items():
                    if any(s in k.lower() for s in ("api", "key", "secret", "token")):
                        out[k] = "***REDACTED***"
                    else:
                        out[k] = _redact(v)
                return out
            elif isinstance(d, list):
                return [_redact(x) for x in d]
            return d

        try:
            return cast(dict[str, Any], _redact(dict(agent_spec)))
        except Exception as e:
            logger.debug(f"Could not fully sanitize agent spec (using fallback): {e}")
            return (
                {"keys": list(agent_spec.keys())}
                if isinstance(agent_spec, dict)
                else {"summary": "unsupported"}
            )

    def _calculate_metrics(self, results: list[dict[str, Any]]) -> dict[str, float]:
        """Calculate metrics from results.

        Args:
            results: List of evaluation results

        Returns:
            Calculated metrics
        """
        total = len(results)
        if total == 0:
            return {"accuracy": 0.0, "success_rate": 0.0, "total_examples": 0.0}

        successful = sum(1 for r in results if r.get("success", False))
        correct = sum(1 for r in results if r.get("correct", False))

        # Count examples with ground truth
        with_ground_truth = sum(1 for r in results if "correct" in r)

        metrics = {
            "accuracy": correct / with_ground_truth if with_ground_truth > 0 else 0.0,
            "success_rate": successful / total if total > 0 else 0.0,
            "total_examples": float(total),
            "examples_with_ground_truth": float(with_ground_truth),
            "successful_executions": float(successful),
        }

        # Add evaluation type breakdown
        eval_types = {}
        for result in results:
            if "correct" in result:
                eval_type = result.get("evaluation_type", "unknown")
                if eval_type not in eval_types:
                    eval_types[eval_type] = {"total": 0, "correct": 0}
                eval_types[eval_type]["total"] += 1
                if result.get("correct"):
                    eval_types[eval_type]["correct"] += 1

        # Add per-type accuracy
        for eval_type, counts in eval_types.items():
            if counts["total"] > 0:
                metrics[f"accuracy_{eval_type}"] = counts["correct"] / counts["total"]

        return metrics


class RemoteExecutionAdapter(ExecutionAdapter):
    """Adapter for remote execution in SaaS mode."""

    def __init__(self, backend_client: Any) -> None:
        """Initialize remote execution adapter.

        Args:
            backend_client: Backend client instance
        """
        self.backend_client = backend_client

    async def execute_configuration(
        self, agent_spec: dict[str, Any], dataset: dict[str, Any], trial_id: str
    ) -> dict[str, Any]:
        """Execute configuration remotely.

        In SaaS mode, the actual execution happens on the backend.
        This adapter just handles the communication.

        Args:
            agent_spec: Agent specification
            dataset: Dataset (already uploaded)
            trial_id: Trial identifier

        Returns:
            Execution results from backend
        """
        # In SaaS mode, execution is handled by the backend
        # The dataset should already be uploaded and we just reference it
        dataset_id = dataset.get("dataset_id")
        if not dataset_id:
            raise ValueError("Dataset ID required for remote execution") from None

        # Submit configuration for execution
        result = await self.backend_client.execute_configuration(
            trial_id=trial_id, agent_spec=agent_spec, dataset_id=dataset_id
        )

        return result  # type: ignore[no-any-return]

    async def get_execution_mode(self) -> str:
        """Get execution mode."""
        return "remote"


class HybridPlatformAdapter(ExecutionAdapter):
    """Adapter for hybrid platform execution (client-side with platform-specific features)."""

    def __init__(self, platform_client: Any, agent_builder: Any) -> None:
        """Initialize hybrid platform adapter.

        Args:
            platform_client: Platform-specific client (e.g., OpenAI, Anthropic)
            agent_builder: Agent builder instance
        """
        self.platform_client = platform_client
        self.agent_builder = agent_builder

    async def execute_configuration(
        self, agent_spec: dict[str, Any], dataset: dict[str, Any], trial_id: str
    ) -> dict[str, Any]:
        """Execute configuration using platform-specific features.

        This adapter allows using platform-specific features like:
        - OpenAI's function calling
        - Anthropic's constitutional AI
        - Custom evaluation metrics

        While still running locally and only submitting metrics.

        Args:
            agent_spec: Agent specification
            dataset: Dataset to evaluate on
            trial_id: Trial identifier

        Returns:
            Execution results with metrics
        """
        try:
            # Build agent with platform-specific features
            agent = self.agent_builder.build_platform_agent(
                agent_spec, self.platform_client
            )

            # Use local execution logic but with platform features
            local_adapter = LocalExecutionAdapter(self.agent_builder)

            # Override the agent builder temporarily
            original_builder = local_adapter.agent_builder
            # Provide stub builder that always returns the prebuilt platform agent
            local_adapter.agent_builder = type(
                "HybridAgentBuilderOverride",
                (object,),
                {"build_agent": lambda self, *_args, **_kwargs: agent},
            )()

            try:
                # Execute using local adapter logic
                result = await local_adapter.execute_configuration(
                    agent_spec, dataset, trial_id
                )

                # Add platform-specific metadata
                result["metadata"]["platform"] = agent_spec.get("platform", "unknown")
                result["metadata"]["platform_features"] = agent_spec.get(
                    "platform_features", []
                )

                return result

            finally:
                # Restore original builder
                local_adapter.agent_builder = original_builder

        except Exception as e:
            logger.error(f"Hybrid platform execution failed: {str(e)}")
            raise

    async def get_execution_mode(self) -> str:
        """Get execution mode."""
        return "hybrid_platform"
