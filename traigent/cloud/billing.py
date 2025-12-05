"""Unified Billing and Cost Tracking for TraiGent Cloud Service.

This module provides comprehensive billing and cost tracking functionality,
combining usage tracking, billing management, and detailed cost monitoring
for TraiGent SDK and OptiGen Backend integration.
"""

# Traceability: CONC-Layer-Infra CONC-Quality-Reliability FUNC-CLOUD-HYBRID REQ-CLOUD-009 SYNC-CloudHybrid

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from traigent.utils.logging import get_logger

logger = get_logger(__name__)


# Cost Tracking Enums and Classes (from cost_tracking.py)


class CostCategory(Enum):
    """Cost categories for tracking."""

    OPTIMIZATION = "optimization"
    INFERENCE = "inference"
    DATA_TRANSFER = "data_transfer"
    STORAGE = "storage"
    COMPUTE = "compute"


class BillingTier(Enum):
    """Billing tiers."""

    FREE = "free"
    STANDARD = "standard"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"


@dataclass
class CostItem:
    """Individual cost item tracking."""

    item_id: str
    category: CostCategory
    description: str
    quantity: float
    unit_cost: float
    total_cost: float
    currency: str = "USD"
    timestamp: float = field(default_factory=time.time)
    session_id: str | None = None
    trial_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class UsageMetrics:
    """Usage metrics for cost calculation."""

    tokens_processed: int = 0
    api_calls: int = 0
    dataset_size: int = 0
    trials_executed: int = 0
    compute_minutes: float = 0.0
    storage_gb_hours: float = 0.0
    bandwidth_gb: float = 0.0
    custom_metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class CostReport:
    """Comprehensive cost report."""

    report_id: str
    start_time: float
    end_time: float
    total_cost: float
    currency: str
    items: list[CostItem]
    usage_metrics: UsageMetrics
    billing_tier: BillingTier
    session_summary: dict[str, Any] = field(default_factory=dict)
    cost_breakdown: dict[str, float] = field(default_factory=dict)


@dataclass
class CostTrackingConfig:
    """Configuration for cost tracking."""

    enable_client_tracking: bool = True
    enable_server_sync: bool = True
    sync_interval: float = 60.0  # Sync every minute
    cache_costs_locally: bool = True
    cost_cache_file: str | None = None
    billing_tier: BillingTier = BillingTier.STANDARD
    currency: str = "USD"
    enable_real_time_alerts: bool = False
    cost_alert_threshold: float = 50.0  # Alert if costs exceed $50


# Original Billing Classes


@dataclass
class UsageRecord:
    """Record of TraiGent Cloud Service usage."""

    timestamp: datetime
    function_name: str
    trials_count: int
    dataset_size: int
    optimization_time: float
    cost_credits: float
    billing_tier: str = "standard"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "function_name": self.function_name,
            "trials_count": self.trials_count,
            "dataset_size": self.dataset_size,
            "optimization_time": self.optimization_time,
            "cost_credits": self.cost_credits,
            "billing_tier": self.billing_tier,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UsageRecord:
        """Create from dictionary."""
        timestamp = datetime.fromisoformat(data["timestamp"])
        # Ensure timezone-aware datetime
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=UTC)

        return cls(
            timestamp=timestamp,
            function_name=data["function_name"],
            trials_count=data["trials_count"],
            dataset_size=data["dataset_size"],
            optimization_time=data["optimization_time"],
            cost_credits=data["cost_credits"],
            billing_tier=data.get("billing_tier", "standard"),
        )


@dataclass
class BillingPlan:
    """TraiGent Cloud Service billing plan."""

    name: str
    monthly_credits: int
    cost_per_credit: float
    max_trials_per_optimization: int
    max_dataset_size: int
    priority_support: bool = False
    advanced_algorithms: bool = False

    def calculate_monthly_cost(self) -> float:
        """Calculate monthly cost for this plan."""
        return self.monthly_credits * self.cost_per_credit


class UsageTracker:
    """Tracks usage for billing and analytics."""

    def __init__(self, storage_path: str | None = None) -> None:
        """Initialize usage tracker.

        Args:
            storage_path: Path to store usage data (defaults to ~/.traigent/usage.json)
        """
        if storage_path:
            self.storage_path = Path(storage_path)
        else:
            self.storage_path = Path.home() / ".traigent" / "usage.json"

        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._usage_records: list[UsageRecord] = []
        self._load_usage_data()

        # Billing plans
        self.billing_plans: dict[str, Any] = {
            "free": BillingPlan(
                name="Free",
                monthly_credits=100,
                cost_per_credit=0.0,
                max_trials_per_optimization=20,
                max_dataset_size=100,
            ),
            "standard": BillingPlan(
                name="Standard",
                monthly_credits=1000,
                cost_per_credit=0.01,
                max_trials_per_optimization=100,
                max_dataset_size=1000,
                priority_support=True,
            ),
            "professional": BillingPlan(
                name="Professional",
                monthly_credits=10000,
                cost_per_credit=0.008,
                max_trials_per_optimization=500,
                max_dataset_size=10000,
                priority_support=True,
                advanced_algorithms=True,
            ),
            "enterprise": BillingPlan(
                name="Enterprise",
                monthly_credits=100000,
                cost_per_credit=0.005,
                max_trials_per_optimization=-1,  # Unlimited
                max_dataset_size=-1,  # Unlimited
                priority_support=True,
                advanced_algorithms=True,
            ),
        }

    async def record_optimization(
        self,
        function_name: str,
        trials_count: int,
        dataset_size: int,
        optimization_time: float,
        billing_tier: str = "standard",
    ) -> UsageRecord:
        """Record optimization usage.

        Args:
            function_name: Name of optimized function
            trials_count: Number of optimization trials
            dataset_size: Size of evaluation dataset
            optimization_time: Time spent on optimization
            billing_tier: User's billing tier

        Returns:
            UsageRecord for the optimization
        """
        # Calculate cost in credits
        cost_credits = self._calculate_cost_credits(
            trials_count, dataset_size, billing_tier
        )

        usage_record = UsageRecord(
            timestamp=datetime.now(UTC),
            function_name=function_name,
            trials_count=trials_count,
            dataset_size=dataset_size,
            optimization_time=optimization_time,
            cost_credits=cost_credits,
            billing_tier=billing_tier,
        )

        self._usage_records.append(usage_record)
        await self._save_usage_data()

        logger.info(
            f"Recorded optimization: {function_name} "
            f"({trials_count} trials, {dataset_size} examples, {cost_credits:.2f} credits)"
        )

        return usage_record

    def _calculate_cost_credits(
        self, trials_count: int, dataset_size: int, billing_tier: str
    ) -> float:
        """Calculate cost in credits for optimization.

        Args:
            trials_count: Number of optimization trials
            dataset_size: Size of evaluation dataset
            billing_tier: User's billing tier

        Returns:
            Cost in credits
        """
        base_cost = 0.1  # Base cost per optimization
        trial_cost = trials_count * 0.01  # Cost per trial
        data_cost = dataset_size * 0.001  # Cost per example

        total_cost = base_cost + trial_cost + data_cost

        # Apply tier discounts
        tier_discounts = {
            "free": 1.0,  # No discount
            "standard": 0.9,  # 10% discount
            "professional": 0.8,  # 20% discount
            "enterprise": 0.7,  # 30% discount
        }

        discount = tier_discounts.get(billing_tier, 1.0)
        return total_cost * discount

    async def get_usage_stats(
        self, start_date: datetime | None = None, end_date: datetime | None = None
    ) -> dict[str, Any]:
        """Get usage statistics for a time period.

        Args:
            start_date: Start of time period (defaults to current month)
            end_date: End of time period (defaults to now)

        Returns:
            Dict with usage statistics
        """
        if start_date is None:
            start_date = datetime.now(UTC).replace(
                day=1, hour=0, minute=0, second=0, microsecond=0
            )
        if end_date is None:
            end_date = datetime.now(UTC)

        # Filter records by date range
        filtered_records = [
            record
            for record in self._usage_records
            if start_date <= record.timestamp <= end_date
        ]

        if not filtered_records:
            return {
                "total_optimizations": 0,
                "total_trials": 0,
                "total_credits": 0.0,
                "total_time": 0.0,
                "avg_trials_per_optimization": 0.0,
                "avg_dataset_size": 0.0,
                "functions_optimized": [],
                "period": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat(),
                },
            }

        # Calculate statistics
        total_optimizations = len(filtered_records)
        total_trials = sum(record.trials_count for record in filtered_records)
        total_credits = sum(record.cost_credits for record in filtered_records)
        total_time = sum(record.optimization_time for record in filtered_records)

        avg_trials = total_trials / total_optimizations
        avg_dataset_size = (
            sum(record.dataset_size for record in filtered_records)
            / total_optimizations
        )

        # Get unique functions
        functions_optimized = list(
            {record.function_name for record in filtered_records}
        )

        # Function-specific stats
        function_stats = {}
        for func_name in functions_optimized:
            func_records = [r for r in filtered_records if r.function_name == func_name]
            function_stats[func_name] = {
                "optimizations": len(func_records),
                "total_trials": sum(r.trials_count for r in func_records),
                "total_credits": sum(r.cost_credits for r in func_records),
                "avg_optimization_time": sum(r.optimization_time for r in func_records)
                / len(func_records),
            }

        return {
            "total_optimizations": total_optimizations,
            "total_trials": total_trials,
            "total_credits": round(total_credits, 2),
            "total_time": round(total_time, 2),
            "avg_trials_per_optimization": round(avg_trials, 1),
            "avg_dataset_size": round(avg_dataset_size, 1),
            "functions_optimized": functions_optimized,
            "function_stats": function_stats,
            "period": {"start": start_date.isoformat(), "end": end_date.isoformat()},
        }

    def get_billing_plan(self, plan_name: str) -> BillingPlan | None:
        """Get billing plan details.

        Args:
            plan_name: Name of billing plan

        Returns:
            BillingPlan object or None if not found
        """
        return self.billing_plans.get(plan_name.lower())

    def list_billing_plans(self) -> dict[str, BillingPlan]:
        """List all available billing plans.

        Returns:
            Dict mapping plan names to BillingPlan objects
        """
        return self.billing_plans.copy()

    async def estimate_monthly_cost(
        self,
        optimizations_per_month: int,
        avg_trials: int = 50,
        avg_dataset_size: int = 100,
        billing_tier: str = "standard",
    ) -> dict[str, float | str]:
        """Estimate monthly cost based on usage patterns.

        Args:
            optimizations_per_month: Expected optimizations per month
            avg_trials: Average trials per optimization
            avg_dataset_size: Average dataset size
            billing_tier: Billing tier

        Returns:
            Dict with cost estimates
        """
        cost_per_optimization = self._calculate_cost_credits(
            avg_trials, avg_dataset_size, billing_tier
        )

        total_credits = optimizations_per_month * cost_per_optimization

        plan = self.get_billing_plan(billing_tier)
        if not plan:
            raise ValueError(f"Unknown billing tier: {billing_tier}")

        monthly_cost = total_credits * plan.cost_per_credit

        return {
            "optimizations_per_month": optimizations_per_month,
            "credits_per_optimization": round(cost_per_optimization, 3),
            "total_credits_per_month": round(total_credits, 2),
            "monthly_cost_usd": round(monthly_cost, 2),
            "billing_tier": billing_tier,
            "cost_per_credit": plan.cost_per_credit,
        }

    def _load_usage_data(self) -> None:
        """Load usage data from storage."""
        try:
            if self.storage_path.exists():
                with open(self.storage_path) as f:
                    data = json.load(f)
                    self._usage_records = [
                        UsageRecord.from_dict(record_data)
                        for record_data in data.get("usage_records", [])
                    ]
                logger.debug(f"Loaded {len(self._usage_records)} usage records")
            else:
                self._usage_records = []
        except Exception as e:
            logger.warning(f"Failed to load usage data: {e}")
            self._usage_records = []

    async def _save_usage_data(self) -> None:
        """Save usage data to storage."""
        try:
            data = {
                "usage_records": [record.to_dict() for record in self._usage_records],
                "last_updated": datetime.now(UTC).isoformat(),
            }
            with open(self.storage_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save usage data: {e}")


class BillingManager:
    """Manages billing operations and plan management."""

    def __init__(self, usage_tracker: UsageTracker) -> None:
        """Initialize billing manager.

        Args:
            usage_tracker: UsageTracker instance
        """
        self.usage_tracker = usage_tracker
        self.current_plan = "free"  # Default plan

    async def check_usage_limits(
        self, trials_count: int, dataset_size: int
    ) -> dict[str, Any]:
        """Check if operation is within usage limits.

        Args:
            trials_count: Requested number of trials
            dataset_size: Size of dataset

        Returns:
            Dict with limit check results
        """
        plan = self.usage_tracker.get_billing_plan(self.current_plan)
        if not plan:
            return {"allowed": False, "reason": "Invalid billing plan"}

        # Check trial limits
        if (
            plan.max_trials_per_optimization != -1
            and trials_count > plan.max_trials_per_optimization
        ):
            return {
                "allowed": False,
                "reason": f"Trials limit exceeded (max: {plan.max_trials_per_optimization})",
                "suggested_action": "Reduce trials count or upgrade plan",
            }

        # Check dataset size limits
        if plan.max_dataset_size != -1 and dataset_size > plan.max_dataset_size:
            return {
                "allowed": False,
                "reason": f"Dataset size limit exceeded (max: {plan.max_dataset_size})",
                "suggested_action": "Reduce dataset size or upgrade plan",
            }

        # Check monthly credit limits
        current_month_stats = await self.usage_tracker.get_usage_stats()
        estimated_cost = self.usage_tracker._calculate_cost_credits(
            trials_count, dataset_size, self.current_plan
        )

        if current_month_stats["total_credits"] + estimated_cost > plan.monthly_credits:
            return {
                "allowed": False,
                "reason": f"Monthly credit limit would be exceeded (limit: {plan.monthly_credits})",
                "suggested_action": "Wait for next billing cycle or upgrade plan",
            }

        return {
            "allowed": True,
            "estimated_cost": estimated_cost,
            "remaining_credits": plan.monthly_credits
            - current_month_stats["total_credits"],
        }

    def upgrade_plan(self, new_plan: str) -> bool:
        """Upgrade to a new billing plan.

        Args:
            new_plan: Name of new billing plan

        Returns:
            True if upgrade successful
        """
        if new_plan not in self.usage_tracker.billing_plans:
            logger.error(f"Unknown billing plan: {new_plan}")
            return False

        old_plan = self.current_plan
        self.current_plan = new_plan

        logger.info(f"Upgraded billing plan: {old_plan} → {new_plan}")
        return True

    def get_current_plan_info(self) -> dict[str, Any]:
        """Get current billing plan information.

        Returns:
            Dict with current plan details
        """
        plan = self.usage_tracker.get_billing_plan(self.current_plan)
        if not plan:
            return {}

        return {
            "name": plan.name,
            "monthly_credits": plan.monthly_credits,
            "cost_per_credit": plan.cost_per_credit,
            "monthly_cost": plan.calculate_monthly_cost(),
            "max_trials_per_optimization": plan.max_trials_per_optimization,
            "max_dataset_size": plan.max_dataset_size,
            "priority_support": plan.priority_support,
            "advanced_algorithms": plan.advanced_algorithms,
        }


# Cost Tracking Classes (from cost_tracking.py)


class CostTracker:
    """Client-side cost tracking with server synchronization.

    This tracker monitors costs locally while synchronizing with the backend
    server to ensure accurate billing and usage reporting.
    """

    def __init__(self, config: CostTrackingConfig | None = None) -> None:
        """Initialize cost tracker.

        Args:
            config: Cost tracking configuration
        """
        self.config = config or CostTrackingConfig()

        # Cost tracking state with bounded collections
        self._cost_items: list[CostItem] = []
        self._session_costs: dict[str, list[CostItem]] = {}

        # Memory bounds to prevent unbounded growth
        self._max_cost_items = 10000  # Maximum items to keep in memory
        self._max_items_per_session = 1000  # Maximum items per session
        self._max_sessions = 100  # Maximum concurrent sessions to track
        self._usage_metrics = UsageMetrics()
        self._total_cost = 0.0

        # Synchronization state
        self._sync_lock = asyncio.Lock()
        self._sync_task: asyncio.Task[None] | None = None
        self._last_sync_time = 0.0
        self._pending_sync_items: list[CostItem] = []

        # Statistics
        self._stats: dict[str, Any] = {
            "total_items_tracked": 0,
            "successful_syncs": 0,
            "failed_syncs": 0,
            "cache_hits": 0,
            "alerts_triggered": 0,
        }

        # Track background tasks so we can surface failures and avoid leaks.
        self._background_tasks: set[asyncio.Task[Any]] = set()

        # Cost rate configuration (would normally come from server)
        self._cost_rates = self._initialize_cost_rates()

    def _initialize_cost_rates(self) -> dict[str, dict[str, float]]:
        """Initialize cost rates by tier and category."""
        return {
            BillingTier.FREE.value: {
                "token_rate": 0.0,  # Free tier
                "api_call_rate": 0.0,
                "compute_rate": 0.0,
                "storage_rate": 0.0,
                "bandwidth_rate": 0.0,
            },
            BillingTier.STANDARD.value: {
                "token_rate": 0.0001,  # $0.0001 per token
                "api_call_rate": 0.001,  # $0.001 per API call
                "compute_rate": 0.02,  # $0.02 per compute minute
                "storage_rate": 0.001,  # $0.001 per GB-hour
                "bandwidth_rate": 0.01,  # $0.01 per GB
            },
            BillingTier.PREMIUM.value: {
                "token_rate": 0.00008,  # Discounted rates
                "api_call_rate": 0.0008,
                "compute_rate": 0.018,
                "storage_rate": 0.0008,
                "bandwidth_rate": 0.008,
            },
            BillingTier.ENTERPRISE.value: {
                "token_rate": 0.00005,  # Further discounted
                "api_call_rate": 0.0005,
                "compute_rate": 0.015,
                "storage_rate": 0.0005,
                "bandwidth_rate": 0.005,
            },
        }

    # Cost Tracking Methods

    def track_optimization_cost(
        self,
        session_id: str,
        trial_id: str | None = None,
        tokens_used: int = 0,
        compute_time: float = 0.0,
        api_calls: int = 1,
        metadata: dict[str, Any] | None = None,
    ) -> CostItem:
        """Track costs for optimization operations.

        Args:
            session_id: Optimization session ID
            trial_id: Optional trial ID
            tokens_used: Number of tokens processed
            compute_time: Compute time in minutes
            api_calls: Number of API calls made
            metadata: Additional metadata

        Returns:
            CostItem for the tracked operation
        """
        rates = self._cost_rates[self.config.billing_tier.value]

        # Calculate costs
        token_cost = tokens_used * rates["token_rate"]
        compute_cost = compute_time * rates["compute_rate"]
        api_cost = api_calls * rates["api_call_rate"]
        total_cost = token_cost + compute_cost + api_cost

        # Create cost item
        cost_item = CostItem(
            item_id=str(uuid.uuid4()),
            category=CostCategory.OPTIMIZATION,
            description=f"Optimization trial {trial_id or 'unknown'} for session {session_id}",
            quantity=1,
            unit_cost=total_cost,
            total_cost=total_cost,
            currency=self.config.currency,
            session_id=session_id,
            trial_id=trial_id,
            metadata={
                "tokens_used": tokens_used,
                "compute_time": compute_time,
                "api_calls": api_calls,
                "billing_tier": self.config.billing_tier.value,
                **(metadata or {}),
            },
        )

        # Track the cost
        self._add_cost_item(cost_item)

        # Update usage metrics
        self._usage_metrics.tokens_processed += tokens_used
        self._usage_metrics.api_calls += api_calls
        self._usage_metrics.trials_executed += 1
        self._usage_metrics.compute_minutes += compute_time

        logger.debug(
            f"Tracked optimization cost: ${total_cost:.4f} for session {session_id}"
        )
        return cost_item

    def track_inference_cost(
        self,
        session_id: str,
        input_tokens: int,
        output_tokens: int,
        model_name: str,
        metadata: dict[str, Any] | None = None,
    ) -> CostItem:
        """Track costs for inference operations.

        Args:
            session_id: Session ID
            input_tokens: Input tokens processed
            output_tokens: Output tokens generated
            model_name: Model used for inference
            metadata: Additional metadata

        Returns:
            CostItem for the inference operation
        """
        rates = self._cost_rates[self.config.billing_tier.value]

        # Different rates for input vs output tokens
        input_cost = input_tokens * rates["token_rate"]
        output_cost = (
            output_tokens * rates["token_rate"] * 1.5
        )  # Output typically costs more
        total_cost = input_cost + output_cost

        cost_item = CostItem(
            item_id=str(uuid.uuid4()),
            category=CostCategory.INFERENCE,
            description=f"Inference using {model_name}",
            quantity=input_tokens + output_tokens,
            unit_cost=rates["token_rate"],
            total_cost=total_cost,
            currency=self.config.currency,
            session_id=session_id,
            metadata={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "model_name": model_name,
                "billing_tier": self.config.billing_tier.value,
                **(metadata or {}),
            },
        )

        self._add_cost_item(cost_item)
        self._usage_metrics.tokens_processed += input_tokens + output_tokens
        self._usage_metrics.api_calls += 1

        logger.debug(
            f"Tracked inference cost: ${total_cost:.4f} for {input_tokens + output_tokens} tokens"
        )
        return cost_item

    def _register_background_task(self, task: asyncio.Task[Any]) -> None:
        """Track background tasks and surface unexpected failures."""

        def _on_task_done(fut: asyncio.Task[Any]) -> None:
            self._background_tasks.discard(fut)
            try:
                fut.result()
            except asyncio.CancelledError:  # pragma: no cover - cancellation path
                logger.debug("Cost tracker background task cancelled")
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.error("❌ Cost tracker background task failed", exc_info=exc)

        self._background_tasks.add(task)
        task.add_done_callback(_on_task_done)

    def track_data_transfer_cost(
        self,
        session_id: str,
        data_size_gb: float,
        transfer_type: str = "upload",
        metadata: dict[str, Any] | None = None,
    ) -> CostItem:
        """Track costs for data transfer operations.

        Args:
            session_id: Session ID
            data_size_gb: Data size in GB
            transfer_type: Type of transfer (upload/download)
            metadata: Additional metadata

        Returns:
            CostItem for the data transfer
        """
        rates = self._cost_rates[self.config.billing_tier.value]
        total_cost = data_size_gb * rates["bandwidth_rate"]

        cost_item = CostItem(
            item_id=str(uuid.uuid4()),
            category=CostCategory.DATA_TRANSFER,
            description=f"Data {transfer_type} ({data_size_gb:.3f} GB)",
            quantity=data_size_gb,
            unit_cost=rates["bandwidth_rate"],
            total_cost=total_cost,
            currency=self.config.currency,
            session_id=session_id,
            metadata={
                "data_size_gb": data_size_gb,
                "transfer_type": transfer_type,
                "billing_tier": self.config.billing_tier.value,
                **(metadata or {}),
            },
        )

        self._add_cost_item(cost_item)
        self._usage_metrics.bandwidth_gb += data_size_gb

        logger.debug(
            f"Tracked data transfer cost: ${total_cost:.4f} for {data_size_gb:.3f} GB"
        )
        return cost_item

    def track_custom_cost(
        self,
        session_id: str,
        category: CostCategory,
        description: str,
        cost: float,
        metadata: dict[str, Any] | None = None,
    ) -> CostItem:
        """Track custom cost item.

        Args:
            session_id: Session ID
            category: Cost category
            description: Cost description
            cost: Total cost
            metadata: Additional metadata

        Returns:
            CostItem for the custom cost
        """
        cost_item = CostItem(
            item_id=str(uuid.uuid4()),
            category=category,
            description=description,
            quantity=1,
            unit_cost=cost,
            total_cost=cost,
            currency=self.config.currency,
            session_id=session_id,
            metadata={
                "billing_tier": self.config.billing_tier.value,
                **(metadata or {}),
            },
        )

        self._add_cost_item(cost_item)

        logger.debug(f"Tracked custom cost: ${cost:.4f} - {description}")
        return cost_item

    def _add_cost_item(self, cost_item: CostItem) -> None:
        """Add cost item to tracking with memory bounds."""
        # Enforce global cost items limit (FIFO)
        if len(self._cost_items) >= self._max_cost_items:
            # Remove oldest items (10% at a time for efficiency)
            items_to_remove = max(1, self._max_cost_items // 10)
            self._cost_items = self._cost_items[items_to_remove:]
            logger.debug(
                f"Pruned {items_to_remove} old cost items to stay within memory limit"
            )

        self._cost_items.append(cost_item)
        self._total_cost += cost_item.total_cost
        self._stats["total_items_tracked"] += 1

        # Add to session costs with bounds
        if cost_item.session_id:
            if cost_item.session_id not in self._session_costs:
                # Enforce max sessions limit
                if len(self._session_costs) >= self._max_sessions:
                    # Remove oldest session (based on oldest cost item)
                    oldest_session = min(
                        self._session_costs.keys(),
                        key=lambda s: (
                            self._session_costs[s][0].timestamp
                            if self._session_costs[s]
                            else float("inf")
                        ),
                    )
                    del self._session_costs[oldest_session]
                    logger.debug(
                        f"Removed oldest session {oldest_session} to stay within session limit"
                    )

                self._session_costs[cost_item.session_id] = []

            # Enforce per-session items limit
            session_items = self._session_costs[cost_item.session_id]
            if len(session_items) >= self._max_items_per_session:
                # Remove oldest items from this session
                items_to_remove = max(1, self._max_items_per_session // 10)
                self._session_costs[cost_item.session_id] = session_items[
                    items_to_remove:
                ]

            self._session_costs[cost_item.session_id].append(cost_item)

        # Add to pending sync
        if self.config.enable_server_sync:
            self._pending_sync_items.append(cost_item)

        # Cache locally if enabled
        if self.config.cache_costs_locally:
            cache_task = asyncio.create_task(self._cache_cost_item(cost_item))
            self._register_background_task(cache_task)

        # Check for cost alerts
        if self.config.enable_real_time_alerts:
            self._check_cost_alerts()

    # Reporting Methods

    def get_session_costs(self, session_id: str) -> list[CostItem]:
        """Get all costs for a specific session.

        Args:
            session_id: Session ID

        Returns:
            List of cost items for the session
        """
        return self._session_costs.get(session_id, [])

    def get_session_total_cost(self, session_id: str) -> float:
        """Get total cost for a specific session.

        Args:
            session_id: Session ID

        Returns:
            Total cost for the session
        """
        return sum(item.total_cost for item in self.get_session_costs(session_id))

    def generate_cost_report(
        self,
        start_time: float | None = None,
        end_time: float | None = None,
        session_id: str | None = None,
    ) -> CostReport:
        """Generate comprehensive cost report.

        Args:
            start_time: Report start time (timestamp)
            end_time: Report end time (timestamp)
            session_id: Optional session ID filter

        Returns:
            CostReport with detailed cost breakdown
        """
        # Filter cost items
        filtered_items = self._cost_items

        if session_id:
            filtered_items = [
                item for item in filtered_items if item.session_id == session_id
            ]

        if start_time:
            filtered_items = [
                item for item in filtered_items if item.timestamp >= start_time
            ]

        if end_time:
            filtered_items = [
                item for item in filtered_items if item.timestamp <= end_time
            ]

        # Calculate totals
        total_cost = sum(item.total_cost for item in filtered_items)

        # Cost breakdown by category
        cost_breakdown = {}
        for category in CostCategory:
            category_items = [
                item for item in filtered_items if item.category == category
            ]
            cost_breakdown[category.value] = sum(
                item.total_cost for item in category_items
            )

        # Session summary
        session_summary = {}
        if session_id:
            session_items = self.get_session_costs(session_id)
            session_summary = {
                "total_items": len(session_items),
                "total_cost": sum(item.total_cost for item in session_items),
                "trials_executed": len(
                    [item for item in session_items if item.trial_id]
                ),
                "avg_cost_per_trial": sum(item.total_cost for item in session_items)
                / max(1, len([item for item in session_items if item.trial_id])),
            }

        return CostReport(
            report_id=str(uuid.uuid4()),
            start_time=start_time
            or (
                min(item.timestamp for item in filtered_items)
                if filtered_items
                else time.time()
            ),
            end_time=end_time
            or (
                max(item.timestamp for item in filtered_items)
                if filtered_items
                else time.time()
            ),
            total_cost=total_cost,
            currency=self.config.currency,
            items=filtered_items,
            usage_metrics=self._usage_metrics,
            billing_tier=self.config.billing_tier,
            session_summary=session_summary,
            cost_breakdown=cost_breakdown,
        )

    # Server Synchronization

    async def start_sync(self) -> bool:
        """Start automatic synchronization with server.

        Returns:
            True if sync started successfully
        """
        if not self.config.enable_server_sync:
            logger.info("Server sync disabled")
            return False

        if self._sync_task and not self._sync_task.done():
            logger.warning("Sync already running")
            return False

        logger.info(
            f"Starting cost tracking sync (interval: {self.config.sync_interval}s)"
        )
        self._sync_task = asyncio.create_task(self._sync_loop())
        return True

    async def stop_sync(self) -> None:
        """Stop automatic synchronization."""
        if self._sync_task and not self._sync_task.done():
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass

        logger.info("Cost tracking sync stopped")

    async def manual_sync(self) -> bool:
        """Manually trigger synchronization with server.

        Returns:
            True if sync successful
        """
        if not self.config.enable_server_sync:
            return False

        return await self._sync_with_server()

    async def _sync_loop(self) -> None:
        """Background sync loop."""
        while True:
            try:
                await asyncio.sleep(self.config.sync_interval)
                await self._sync_with_server()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Sync loop error: {e}")
                await asyncio.sleep(5)  # Wait before retrying

    async def _sync_with_server(self) -> bool:
        """Synchronize pending cost items with server."""
        async with self._sync_lock:
            if not self._pending_sync_items:
                return True

            try:
                # Prepare sync payload
                sync_payload: dict[str, Any] = {
                    "cost_items": [
                        {
                            "item_id": item.item_id,
                            "category": item.category.value,
                            "description": item.description,
                            "quantity": item.quantity,
                            "unit_cost": item.unit_cost,
                            "total_cost": item.total_cost,
                            "currency": item.currency,
                            "timestamp": item.timestamp,
                            "session_id": item.session_id,
                            "trial_id": item.trial_id,
                            "metadata": item.metadata,
                        }
                        for item in self._pending_sync_items
                    ],
                    "usage_metrics": {
                        "tokens_processed": self._usage_metrics.tokens_processed,
                        "api_calls": self._usage_metrics.api_calls,
                        "dataset_size": self._usage_metrics.dataset_size,
                        "trials_executed": self._usage_metrics.trials_executed,
                        "compute_minutes": self._usage_metrics.compute_minutes,
                        "storage_gb_hours": self._usage_metrics.storage_gb_hours,
                        "bandwidth_gb": self._usage_metrics.bandwidth_gb,
                        "custom_metrics": self._usage_metrics.custom_metrics,
                    },
                    "billing_tier": self.config.billing_tier.value,
                    "sync_timestamp": time.time(),
                }

                # Simulate server sync (would be actual HTTP call in production)
                success = await self._send_costs_to_server(sync_payload)

                if success:
                    self._pending_sync_items.clear()
                    self._last_sync_time = time.time()
                    self._stats["successful_syncs"] += 1
                    logger.debug(
                        f"Successfully synced {len(sync_payload['cost_items'])} cost items"
                    )
                    return True
                else:
                    self._stats["failed_syncs"] += 1
                    logger.warning("Failed to sync cost items with server")
                    return False

            except Exception as e:
                logger.error(f"Cost sync failed: {e}")
                self._stats["failed_syncs"] += 1
                return False

    async def _send_costs_to_server(self, payload: dict[str, Any]) -> bool:
        """Send cost data to server.

        Args:
            payload: Cost data payload

        Returns:
            True if successful
        """
        # Placeholder for actual server communication
        # In production, this would make HTTP requests to the backend
        logger.debug(f"Simulating server sync of {len(payload['cost_items'])} items")
        await asyncio.sleep(0.1)  # Simulate network delay
        return True

    # Utility Methods

    async def _cache_cost_item(self, cost_item: CostItem) -> None:
        """Cache cost item locally."""
        if not self.config.cost_cache_file:
            return

        try:
            cache_path = Path(self.config.cost_cache_file)
            cache_path.parent.mkdir(parents=True, exist_ok=True)

            # Load existing cache
            cached_items = []
            if cache_path.exists():
                with open(cache_path) as f:
                    cached_items = json.load(f)

            # Add new item
            cached_items.append(
                {
                    "item_id": cost_item.item_id,
                    "category": cost_item.category.value,
                    "description": cost_item.description,
                    "quantity": cost_item.quantity,
                    "unit_cost": cost_item.unit_cost,
                    "total_cost": cost_item.total_cost,
                    "currency": cost_item.currency,
                    "timestamp": cost_item.timestamp,
                    "session_id": cost_item.session_id,
                    "trial_id": cost_item.trial_id,
                    "metadata": cost_item.metadata,
                }
            )

            # Save cache
            with open(cache_path, "w") as f:
                json.dump(cached_items, f, indent=2)

        except Exception as e:
            logger.warning(f"Failed to cache cost item: {e}")

    def _check_cost_alerts(self) -> None:
        """Check if cost alerts should be triggered."""
        if self._total_cost > self.config.cost_alert_threshold:
            self._stats["alerts_triggered"] += 1
            logger.warning(
                f"Cost alert: Total costs (${self._total_cost:.2f}) "
                f"exceeded threshold (${self.config.cost_alert_threshold:.2f})"
            )

    def get_statistics(self) -> dict[str, Any]:
        """Get cost tracking statistics."""
        return {
            **self._stats,
            "total_cost": self._total_cost,
            "pending_sync_items": len(self._pending_sync_items),
            "last_sync_time": self._last_sync_time,
            "tracked_sessions": len(self._session_costs),
            "billing_tier": self.config.billing_tier.value,
        }

    def reset_costs(self) -> None:
        """Reset all cost tracking (for testing/development)."""
        logger.warning("Resetting all cost tracking data")
        self._cost_items.clear()
        self._session_costs.clear()
        self._usage_metrics = UsageMetrics()
        self._total_cost = 0.0
        self._pending_sync_items.clear()


# Global cost tracker instance
_cost_tracker: CostTracker | None = None


def get_cost_tracker(config: CostTrackingConfig | None = None) -> CostTracker:
    """Get or create global cost tracker.

    Args:
        config: Optional cost tracking configuration

    Returns:
        CostTracker instance
    """
    global _cost_tracker
    if _cost_tracker is None:
        _cost_tracker = CostTracker(config)
    return _cost_tracker


def set_cost_tracker(tracker: CostTracker) -> None:
    """Set global cost tracker.

    Args:
        tracker: CostTracker instance
    """
    global _cost_tracker
    _cost_tracker = tracker
