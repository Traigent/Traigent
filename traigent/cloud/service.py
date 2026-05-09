"""In-process optimization service scaffold.

This helper is not the SDK ``execution_mode="cloud"`` remote execution path.
It runs locally inside the current process and exists for service experiments
around subset selection, billing, and orchestration shapes.
"""

# Traceability: CONC-Layer-Infra CONC-Quality-Reliability FUNC-CLOUD-HYBRID REQ-CLOUD-009 SYNC-CloudHybrid

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.evaluators.local import LocalEvaluator
from traigent.optimizers.registry import get_optimizer
from traigent.utils.exceptions import ValidationError as ValidationException
from traigent.utils.logging import get_logger
from traigent.utils.validation import CoreValidators, validate_or_raise

from .billing import BillingManager, UsageTracker
from .subset_selection import SmartSubsetSelector

logger = get_logger(__name__)


@dataclass
class OptimizationRequest:
    """Request for the in-process optimization service scaffold."""

    function_name: str
    dataset: Dataset
    configuration_space: dict[str, Any]
    objectives: list[str]
    max_trials: int = 50
    target_cost_reduction: float = 0.65
    user_id: str | None = None
    billing_tier: str = "standard"


@dataclass
class OptimizationResponse:
    """Response shape for the in-process optimization service scaffold."""

    request_id: str
    best_config: dict[str, Any]
    best_metrics: dict[str, float]
    trials_count: int
    optimization_time: float
    cost_reduction: float
    subset_used: bool
    billing_info: dict[str, Any]
    status: str = "completed"


class TraigentCloudService:
    """Local scaffold for service-style optimization experiments."""

    def __init__(self) -> None:
        """Initialize the local service scaffold."""
        self.subset_selector = SmartSubsetSelector()
        self.usage_tracker = UsageTracker()
        self.billing_manager = BillingManager(self.usage_tracker)

        # Service statistics
        self.total_optimizations = 0
        self.total_cost_savings = 0.0
        self.uptime_start = time.time()

    @staticmethod
    def _allowed_billing_tiers() -> set[str]:
        return {"free", "standard", "professional", "enterprise"}

    @staticmethod
    def _validate_request(request: OptimizationRequest) -> None:
        if not isinstance(request, OptimizationRequest):
            raise ValidationException("request must be an OptimizationRequest instance")

        validate_or_raise(
            CoreValidators.validate_string_non_empty(
                request.function_name, "function_name"
            )
        )

        if not isinstance(request.dataset, Dataset):
            raise ValidationException("dataset must be a Dataset instance")

        validate_or_raise(
            CoreValidators.validate_dict(
                request.configuration_space, "configuration_space"
            )
        )
        if not request.configuration_space:
            raise ValidationException("configuration_space must not be empty")

        validate_or_raise(
            CoreValidators.validate_list(
                request.objectives,
                "objectives",
                min_length=1,
                item_validator=CoreValidators.validate_string_non_empty,
            )
        )

        validate_or_raise(
            CoreValidators.validate_positive_int(request.max_trials, "max_trials")
        )

        validate_or_raise(
            CoreValidators.validate_number(
                request.target_cost_reduction,
                "target_cost_reduction",
                min_value=0.0,
                max_value=1.0,
            )
        )

        if request.billing_tier not in TraigentCloudService._allowed_billing_tiers():
            raise ValidationException(
                f"billing_tier '{request.billing_tier}' is not supported"
            )

    async def process_optimization_request(
        self, request: OptimizationRequest
    ) -> OptimizationResponse:
        """Process an in-process optimization request with cost controls.

        Args:
            request: OptimizationRequest with all parameters

        Returns:
            OptimizationResponse with results
        """
        self._validate_request(request)
        start_time = time.time()
        request_id = f"opt_{int(time.time() * 1000)}"

        logger.info(
            f"Processing optimization request {request_id} for {request.function_name}"
        )

        try:
            # Check billing limits
            limits_check = await self.billing_manager.check_usage_limits(
                request.max_trials, len(request.dataset.examples)
            )

            if not limits_check["allowed"]:
                return OptimizationResponse(
                    request_id=request_id,
                    best_config={},
                    best_metrics={},
                    trials_count=0,
                    optimization_time=0.0,
                    cost_reduction=0.0,
                    subset_used=False,
                    billing_info=limits_check,
                    status="failed_limits",
                )

            # Smart dataset subset selection
            original_size = len(request.dataset.examples)
            subset_used = False

            if original_size == 0:
                logger.warning(
                    "Received empty dataset for request %s; skipping subset selection",
                    request_id,
                )
                optimized_dataset = request.dataset
                subset_size = 0
                cost_reduction = 0.0
            else:
                optimized_dataset = await self.subset_selector.select_optimal_subset(
                    request.dataset, target_reduction=request.target_cost_reduction
                )
                subset_size = len(optimized_dataset.examples)
                subset_used = subset_size != original_size
                cost_reduction = max(0.0, 1 - (subset_size / original_size))

                logger.info(
                    f"Dataset optimization: {original_size} → {subset_size} examples "
                    f"({cost_reduction * 100:.1f}% reduction)"
                )

            # Run optimization with enhanced algorithms
            optimization_result = await self._run_enhanced_optimization(
                optimized_dataset,
                request.configuration_space,
                request.objectives,
                request.max_trials,
                request.billing_tier,
            )

            # Record usage for billing
            await self.usage_tracker.record_optimization(
                function_name=request.function_name,
                trials_count=optimization_result["trials_count"],
                dataset_size=subset_size,
                optimization_time=time.time() - start_time,
                billing_tier=request.billing_tier,
            )

            # Update service statistics
            self.total_optimizations += 1
            self.total_cost_savings += cost_reduction

            billing_info = {
                "credits_used": limits_check.get("estimated_cost", 0.0),
                "remaining_credits": limits_check.get("remaining_credits", 0),
                "billing_tier": request.billing_tier,
            }

            return OptimizationResponse(
                request_id=request_id,
                best_config=optimization_result["best_config"],
                best_metrics=optimization_result["best_metrics"],
                trials_count=optimization_result["trials_count"],
                optimization_time=time.time() - start_time,
                cost_reduction=cost_reduction,
                subset_used=subset_used,
                billing_info=billing_info,
                status="completed",
            )

        except Exception as e:
            logger.error(f"Optimization failed for request {request_id}: {e}")
            return OptimizationResponse(
                request_id=request_id,
                best_config={},
                best_metrics={},
                trials_count=0,
                optimization_time=time.time() - start_time,
                cost_reduction=0.0,
                subset_used=False,
                billing_info={"error": str(e)},
                status="failed",
            )

    async def _run_enhanced_optimization(
        self,
        dataset: Dataset,
        configuration_space: dict[str, Any],
        objectives: list[str],
        max_trials: int,
        billing_tier: str,
    ) -> dict[str, Any]:
        """Run optimization with enhanced algorithms based on billing tier.

        Args:
            dataset: Evaluation dataset
            configuration_space: Parameter search space
            objectives: Optimization objectives
            max_trials: Maximum number of trials
            billing_tier: User's billing tier

        Returns:
            Dict with optimization results
        """
        # Adjust max_trials based on billing tier before optimizer selection
        tier_multipliers = {
            "free": 0.5,
            "standard": 1.0,
            "professional": 1.5,
            "enterprise": 2.0,
        }

        multiplier = tier_multipliers.get(billing_tier)
        if multiplier is None:
            logger.warning("Unknown billing tier; falling back to standard multiplier")
            multiplier = 1.0

        adjusted_max_trials = int(max_trials * multiplier)

        # Choose optimizer based on billing tier
        if billing_tier in ["professional", "enterprise"]:
            # Use advanced algorithms for premium tiers
            try:
                optimizer = get_optimizer(
                    "bayesian",
                    configuration_space,
                    objectives,
                    max_trials=adjusted_max_trials,
                )
                logger.info("Using Bayesian optimization for premium tier")
            except (ImportError, ValueError, Exception):
                optimizer = get_optimizer(
                    "random",
                    configuration_space,
                    objectives,
                    max_trials=adjusted_max_trials,
                )
                logger.info("Fallback to random optimization")
        else:
            # Use standard algorithms for free/standard tiers
            optimizer = get_optimizer(
                "random",
                configuration_space,
                objectives,
                max_trials=adjusted_max_trials,
            )
            logger.info("Using random optimization for standard tier")

        # Enhanced evaluator with cloud-specific optimizations
        evaluator = LocalEvaluator()

        # Run optimization
        optimization_result = await optimizer.optimize(  # type: ignore[attr-defined]
            configuration_space=configuration_space,
            evaluator=evaluator,
            dataset=dataset,
            objectives=objectives,
            max_trials=adjusted_max_trials,
        )

        return {
            "best_config": optimization_result.best_config,
            "best_metrics": optimization_result.best_metrics,
            "trials_count": len(optimization_result.trials),
        }

    async def get_service_health(self) -> dict[str, Any]:
        """Get service health and status information.

        Returns:
            Dict with service health metrics
        """
        uptime_seconds = time.time() - self.uptime_start
        uptime_hours = uptime_seconds / 3600

        avg_cost_reduction = (
            self.total_cost_savings / self.total_optimizations
            if self.total_optimizations > 0
            else 0.0
        )

        return {
            "status": "healthy",
            "uptime_hours": round(uptime_hours, 2),
            "total_optimizations": self.total_optimizations,
            "average_cost_reduction": round(avg_cost_reduction * 100, 1),
            "service_version": "1.0.0",
            "available_algorithms": ["random", "grid", "bayesian"],
            "supported_objectives": [
                "accuracy",
                "success_rate",
                "error_rate",
                "avg_execution_time",
            ],
            "max_dataset_size": 10000,
            "max_trials": 1000,
        }

    async def get_optimization_history(
        self, user_id: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Get optimization history for a user.

        Args:
            user_id: User ID (optional for demo)
            limit: Maximum number of records to return

        Returns:
            List of optimization records
        """
        # In real implementation, this would query a database
        # For demo, return mock data
        mock_history = []
        for i in range(min(limit, 10)):
            mock_history.append(
                {
                    "request_id": f"opt_{int(time.time() * 1000) - i * 60000}",
                    "function_name": f"function_{i % 3 + 1}",
                    "timestamp": time.time() - i * 3600,
                    "trials_count": 25 + i * 5,
                    "cost_reduction": 0.6 + (i % 3) * 0.1,
                    "status": "completed",
                }
            )

        return mock_history

    def create_optimization_request(
        self,
        function_name: str,
        dataset_data: dict[str, Any],
        configuration_space: dict[str, Any],
        objectives: list[str],
        max_trials: int = 50,
        target_cost_reduction: float = 0.65,
        user_id: str | None = None,
        billing_tier: str = "standard",
    ) -> OptimizationRequest:
        """Create optimization request from raw data.

        Args:
            function_name: Name of function being optimized
            dataset_data: Dataset in dictionary format
            configuration_space: Parameter search space
            objectives: Optimization objectives
            max_trials: Maximum optimization trials
            target_cost_reduction: Target cost reduction ratio
            user_id: User identifier
            billing_tier: User's billing tier

        Returns:
            OptimizationRequest object
        """
        # Convert dataset data to Dataset object
        examples = []
        for example_data in dataset_data.get("examples", []):
            example = EvaluationExample(
                input_data=example_data["input_data"],
                expected_output=example_data["expected_output"],
                metadata=example_data.get("metadata", {}),
            )
            examples.append(example)

        dataset = Dataset(
            examples=examples, name=dataset_data.get("name", "cloud_dataset")
        )

        return OptimizationRequest(
            function_name=function_name,
            dataset=dataset,
            configuration_space=configuration_space,
            objectives=objectives,
            max_trials=max_trials,
            target_cost_reduction=target_cost_reduction,
            user_id=user_id,
            billing_tier=billing_tier,
        )
