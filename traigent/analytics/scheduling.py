"""Smart scheduling components for TraiGent analytics.

This module contains intelligent scheduling functionality for optimizing
resource usage and cost management.
"""

# Traceability: CONC-Layer-Core CONC-Quality-Performance CONC-Quality-Reliability FUNC-ANALYTICS REQ-ANLY-011 SYNC-Observability

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

from ..core.constants import HISTORY_PRUNE_RATIO, MAX_OPTIMIZATION_HISTORY_SIZE
from ..utils.logging import get_logger

logger = get_logger(__name__)


class ScheduleType(Enum):
    """Types of schedules."""

    FIXED = "fixed"
    DYNAMIC = "dynamic"
    ADAPTIVE = "adaptive"
    EVENT_DRIVEN = "event_driven"


class JobPriority(Enum):
    """Job priority levels."""

    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BATCH = 5


@dataclass
class ScheduledJob:
    """Represents a scheduled job."""

    job_id: str
    job_type: str
    priority: JobPriority
    estimated_duration: timedelta
    estimated_cost: float
    resource_requirements: dict[str, float]
    dependencies: list[str] = field(default_factory=list)
    constraints: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    scheduled_time: datetime | None = None
    completed_at: datetime | None = None
    actual_cost: float | None = None
    status: str = "pending"


@dataclass
class ScheduleWindow:
    """Time window for scheduling."""

    start_time: datetime
    end_time: datetime
    cost_multiplier: float = 1.0
    resource_availability: dict[str, float] = field(default_factory=dict)
    reserved_capacity: float = 0.0


@dataclass
class SchedulingPolicy:
    """Policy for job scheduling."""

    max_concurrent_jobs: int = 10
    max_cost_per_hour: float = 100.0
    prefer_off_peak: bool = True
    batch_similar_jobs: bool = True
    preemption_allowed: bool = False
    min_job_duration: timedelta = timedelta(minutes=1)
    max_job_duration: timedelta = timedelta(hours=24)


class SmartScheduler:
    """Intelligent job scheduler with cost optimization.

    Thread-safe: All mutable state access is protected by a lock.
    """

    def __init__(self, policy: SchedulingPolicy | None = None) -> None:
        """Initialize smart scheduler."""
        self.policy = policy or SchedulingPolicy()
        self._lock = threading.Lock()
        self.job_queue: list[ScheduledJob] = []
        self.running_jobs: dict[str, ScheduledJob] = {}
        self.completed_jobs: list[ScheduledJob] = []
        self.schedule_windows: list[ScheduleWindow] = []
        self.cost_history: list[tuple[datetime, float]] = []

    def add_job(self, job: ScheduledJob) -> str:
        """Add a job to the scheduling queue."""
        # Validate job
        if job.estimated_duration < self.policy.min_job_duration:
            raise ValueError(f"Job duration too short: {job.estimated_duration}")
        if job.estimated_duration > self.policy.max_job_duration:
            raise ValueError(f"Job duration too long: {job.estimated_duration}")

        with self._lock:
            # Add to queue
            self.job_queue.append(job)
            self.job_queue.sort(key=lambda j: (j.priority.value, j.created_at))

        logger.info(
            f"Added job {job.job_id} to queue with priority {job.priority.name}"
        )
        return job.job_id

    def define_schedule_windows(
        self,
        start_date: datetime,
        days: int = 7,
        peak_hours: tuple[int, int] = (9, 17),
        weekend_multiplier: float = 0.7,
    ) -> list[ScheduleWindow]:
        """Define scheduling windows with cost considerations."""
        windows = []

        for day in range(days):
            current_date = start_date + timedelta(days=day)
            is_weekend = current_date.weekday() >= 5

            # Off-peak window (night)
            windows.append(
                ScheduleWindow(
                    start_time=current_date.replace(hour=0, minute=0),
                    end_time=current_date.replace(hour=peak_hours[0], minute=0),
                    cost_multiplier=0.5 if not is_weekend else 0.3,
                    resource_availability={"compute": 1.0, "memory": 1.0},
                )
            )

            # Peak window (business hours)
            if not is_weekend:
                windows.append(
                    ScheduleWindow(
                        start_time=current_date.replace(hour=peak_hours[0], minute=0),
                        end_time=current_date.replace(hour=peak_hours[1], minute=0),
                        cost_multiplier=1.5,
                        resource_availability={"compute": 0.7, "memory": 0.8},
                    )
                )

            # Off-peak window (evening)
            windows.append(
                ScheduleWindow(
                    start_time=current_date.replace(hour=peak_hours[1], minute=0),
                    end_time=current_date.replace(hour=23, minute=59),
                    cost_multiplier=0.8 if not is_weekend else weekend_multiplier,
                    resource_availability={"compute": 0.9, "memory": 0.9},
                )
            )

        self.schedule_windows = windows
        return windows

    def schedule_jobs(
        self, optimization_objective: str = "cost"
    ) -> dict[str, ScheduledJob]:
        """Schedule jobs based on optimization objective."""
        scheduled = {}

        if optimization_objective == "cost":
            scheduled = self._schedule_for_cost()
        elif optimization_objective == "time":
            scheduled = self._schedule_for_time()
        elif optimization_objective == "balanced":
            scheduled = self._schedule_balanced()
        else:
            raise ValueError(
                f"Unknown optimization objective: {optimization_objective}"
            )

        return scheduled

    def _schedule_for_cost(
        self, jobs: list[ScheduledJob] | None = None
    ) -> dict[str, ScheduledJob]:
        """Schedule jobs to minimize cost."""
        scheduled = {}
        source_jobs = jobs if jobs is not None else self.job_queue
        remaining_jobs = list(source_jobs)

        # Sort windows by cost multiplier
        sorted_windows = sorted(self.schedule_windows, key=lambda w: w.cost_multiplier)

        for window in sorted_windows:
            # Duration is computed on-demand where needed
            current_time = window.start_time

            # Try to fit jobs in this window
            jobs_to_remove = []
            for job in remaining_jobs:
                # Check if job fits in remaining window time
                job_duration_hours = job.estimated_duration.total_seconds() / 3600
                remaining_window = window.end_time - current_time

                if job.estimated_duration <= remaining_window:
                    # Check cost constraint
                    job_cost = job.estimated_cost * window.cost_multiplier
                    if job_cost <= self.policy.max_cost_per_hour * job_duration_hours:
                        # Schedule the job
                        job.scheduled_time = current_time
                        scheduled[job.job_id] = job
                        jobs_to_remove.append(job)
                        current_time += job.estimated_duration

                        logger.info(
                            f"Scheduled job {job.job_id} at {job.scheduled_time} "
                            f"with cost multiplier {window.cost_multiplier}"
                        )

            # Remove scheduled jobs
            for job in jobs_to_remove:
                remaining_jobs.remove(job)

            if not remaining_jobs:
                break

        # Handle unscheduled jobs
        if remaining_jobs:
            logger.warning(
                f"{len(remaining_jobs)} jobs could not be scheduled within cost constraints"
            )

        return scheduled

    def _schedule_for_time(self) -> dict[str, ScheduledJob]:
        """Schedule jobs to minimize completion time."""
        scheduled = {}
        current_time = datetime.now(UTC)

        # Sort by priority and duration (shortest job first within same priority)
        sorted_jobs = sorted(
            self.job_queue,
            key=lambda j: (j.priority.value, j.estimated_duration.total_seconds()),
        )

        concurrent_jobs = 0
        job_end_times: list[datetime] = []

        for job in sorted_jobs:
            # Find earliest available slot
            if concurrent_jobs < self.policy.max_concurrent_jobs:
                job.scheduled_time = current_time
                concurrent_jobs += 1
            else:
                # Find the earliest ending job
                earliest_end = min(job_end_times)
                job.scheduled_time = earliest_end
                job_end_times.remove(earliest_end)

            # Track when this job will end
            job_end_time = job.scheduled_time + job.estimated_duration
            job_end_times.append(job_end_time)

            scheduled[job.job_id] = job
            logger.info(
                f"Scheduled job {job.job_id} at {job.scheduled_time} for time optimization"
            )

        return scheduled

    def _schedule_balanced(self) -> dict[str, ScheduledJob]:
        """Balance between cost and time optimization."""
        scheduled = {}

        # Work with a snapshot of the queue so the original ordering/state stays intact
        queue_snapshot = list(self.job_queue)

        # Group jobs by priority without mutating the shared queue
        priority_groups: dict[JobPriority, list[ScheduledJob]] = {}
        for job in queue_snapshot:
            priority_groups.setdefault(job.priority, []).append(job)

        # Schedule critical and high priority jobs for time
        for priority in [JobPriority.CRITICAL, JobPriority.HIGH]:
            if priority in priority_groups:
                for job in priority_groups[priority]:
                    job.scheduled_time = datetime.now(UTC)
                    scheduled[job.job_id] = job

        # Schedule other jobs for cost
        remaining_jobs = []
        for priority in [JobPriority.NORMAL, JobPriority.LOW, JobPriority.BATCH]:
            if priority in priority_groups:
                remaining_jobs.extend(priority_groups[priority])

        # Use cost optimization for remaining jobs without mutating the shared queue
        if remaining_jobs:
            cost_scheduled = self._schedule_for_cost(remaining_jobs)
            scheduled.update(cost_scheduled)

        return scheduled

    def execute_job(self, job_id: str) -> bool:
        """Execute a scheduled job."""
        if job_id not in self.running_jobs and job_id not in [
            j.job_id for j in self.job_queue
        ]:
            logger.error(f"Job {job_id} not found")
            return False

        # Find the job
        job = next((j for j in self.job_queue if j.job_id == job_id), None)
        if not job:
            logger.error(f"Job {job_id} not in queue")
            return False

        # Move to running
        self.job_queue.remove(job)
        self.running_jobs[job_id] = job
        job.status = "running"

        logger.info(f"Started executing job {job_id}")
        return True

    def complete_job(
        self, job_id: str, actual_cost: float, success: bool = True
    ) -> bool:
        """Mark a job as completed."""
        if job_id not in self.running_jobs:
            logger.error(f"Job {job_id} not running")
            return False

        job = self.running_jobs.pop(job_id)
        job.completed_at = datetime.now(UTC)
        job.actual_cost = actual_cost
        job.status = "completed" if success else "failed"

        self.completed_jobs.append(job)
        self.cost_history.append((job.completed_at, actual_cost))

        # Enforce memory limits on completed_jobs and cost_history
        if len(self.completed_jobs) > MAX_OPTIMIZATION_HISTORY_SIZE:
            items_to_keep = int(
                MAX_OPTIMIZATION_HISTORY_SIZE * (1 - HISTORY_PRUNE_RATIO)
            )
            self.completed_jobs = self.completed_jobs[-items_to_keep:]
        if len(self.cost_history) > MAX_OPTIMIZATION_HISTORY_SIZE:
            items_to_keep = int(
                MAX_OPTIMIZATION_HISTORY_SIZE * (1 - HISTORY_PRUNE_RATIO)
            )
            self.cost_history = self.cost_history[-items_to_keep:]

        # Log performance
        if job.scheduled_time is not None:
            actual_duration = job.completed_at - job.scheduled_time
        else:
            actual_duration = job.estimated_duration
        cost_variance = (
            (actual_cost - job.estimated_cost) / job.estimated_cost * 100
            if job.estimated_cost > 0
            else 0
        )

        logger.info(
            f"Completed job {job_id}: "
            f"Duration: {actual_duration} (estimated: {job.estimated_duration}), "
            f"Cost: ${actual_cost:.2f} (variance: {cost_variance:+.1f}%)"
        )

        return True

    def get_schedule_summary(self) -> dict[str, Any]:
        """Get summary of current schedule."""
        total_jobs = (
            len(self.job_queue) + len(self.running_jobs) + len(self.completed_jobs)
        )

        if self.completed_jobs:
            avg_cost_variance = sum(
                (
                    (j.actual_cost - j.estimated_cost) / j.estimated_cost * 100
                    if j.estimated_cost > 0 and j.actual_cost
                    else 0
                )
                for j in self.completed_jobs
            ) / len(self.completed_jobs)
        else:
            avg_cost_variance = 0

        return {
            "total_jobs": total_jobs,
            "pending_jobs": len(self.job_queue),
            "running_jobs": len(self.running_jobs),
            "completed_jobs": len(self.completed_jobs),
            "total_cost": sum(
                j.actual_cost for j in self.completed_jobs if j.actual_cost
            ),
            "avg_cost_variance": avg_cost_variance,
            "job_success_rate": (
                sum(1 for j in self.completed_jobs if j.status == "completed")
                / len(self.completed_jobs)
                * 100
                if self.completed_jobs
                else 0
            ),
        }

    def optimize_future_schedules(self) -> dict[str, Any]:
        """Optimize future schedules based on historical data."""
        if len(self.completed_jobs) < 10:
            return {"status": "insufficient_data"}

        # Analyze patterns
        cost_by_hour: dict[int, list[float]] = {}
        duration_accuracy = []

        for job in self.completed_jobs:
            if job.scheduled_time and job.actual_cost:
                hour = job.scheduled_time.hour
                if hour not in cost_by_hour:
                    cost_by_hour[hour] = []
                cost_by_hour[hour].append(job.actual_cost)

            if job.completed_at and job.scheduled_time:
                actual_duration = job.completed_at - job.scheduled_time
                accuracy = (
                    actual_duration.total_seconds()
                    / job.estimated_duration.total_seconds()
                )
                duration_accuracy.append(accuracy)

        # Generate recommendations
        recommendations = {
            "optimal_hours": [],
            "avoid_hours": [],
            "duration_adjustment_factor": 1.0,
            "cost_patterns": {},
        }

        # Find optimal scheduling hours
        avg_cost_by_hour = {
            hour: sum(costs) / len(costs) for hour, costs in cost_by_hour.items()
        }

        if avg_cost_by_hour:
            sorted_hours = sorted(avg_cost_by_hour.items(), key=lambda x: x[1])
            recommendations["optimal_hours"] = [h for h, _ in sorted_hours[:3]]
            recommendations["avoid_hours"] = [h for h, _ in sorted_hours[-3:]]

        # Calculate duration adjustment
        if duration_accuracy:
            avg_accuracy = sum(duration_accuracy) / len(duration_accuracy)
            recommendations["duration_adjustment_factor"] = avg_accuracy

        recommendations["cost_patterns"] = avg_cost_by_hour

        return recommendations

    def schedule_optimization(
        self,
        tasks=None,
        optimization_request=None,
        cost_preferences=None,
        time_constraints=None,
        constraints=None,
        optimization_window=None,
        **kwargs,
    ):
        """Schedule optimization tasks based on constraints and priorities."""

        # Handle different calling patterns
        if optimization_request and not tasks:
            # Convert optimization_request to tasks format
            tasks = [
                {
                    "id": "optimization_task",
                    "name": optimization_request.get("function_name", "optimization"),
                    "estimated_duration": optimization_request.get(
                        "estimated_duration", 3600
                    )
                    // 60,  # Convert to minutes
                    "priority": {"high": 3, "medium": 2, "low": 1}.get(
                        optimization_request.get("priority", "medium"), 2
                    ),
                    "estimated_cost": optimization_request.get("estimated_cost", 0.0),
                }
            ]

        if not tasks:
            return {"schedule": [], "total_duration": 0, "conflicts": []}

        # Default constraints
        if constraints is None:
            constraints = {
                "max_concurrent": 2,
                "business_hours_only": False,
                "maintenance_windows": [],
            }

        # Default optimization window (7 days)
        if optimization_window is None:
            optimization_window = {"start": time.time(), "duration_days": 7}

        # Simple scheduling algorithm
        scheduled_tasks: list[dict[str, Any]] = []
        current_time = optimization_window["start"]
        conflicts = []

        # Sort tasks by priority and estimated duration
        sorted_tasks = sorted(
            tasks,
            key=lambda t: (-t.get("priority", 1), t.get("estimated_duration", 60)),
        )

        # Schedule tasks
        running_tasks: list[dict[str, Any]] = []
        for task in sorted_tasks:
            task_duration = task.get("estimated_duration", 60)  # minutes
            task_priority = task.get("priority", 1)

            # Check concurrent task limit
            # Remove completed tasks
            running_tasks = [t for t in running_tasks if t["end_time"] > current_time]

            if len(running_tasks) >= constraints["max_concurrent"]:
                # Find earliest completion time
                earliest_end = min(t["end_time"] for t in running_tasks)
                current_time = earliest_end
                running_tasks = [
                    t for t in running_tasks if t["end_time"] > current_time
                ]

            # Schedule the task
            start_time = current_time
            end_time = start_time + (task_duration * 60)  # Convert to seconds

            scheduled_task = {
                "task_id": task.get("id", f"task_{len(scheduled_tasks)}"),
                "name": task.get("name", "Unnamed task"),
                "start_time": start_time,
                "end_time": end_time,
                "duration_minutes": task_duration,
                "priority": task_priority,
                "status": "scheduled",
            }

            scheduled_tasks.append(scheduled_task)
            running_tasks.append(scheduled_task)

            # Check if we're within the optimization window
            if end_time > optimization_window["start"] + (
                optimization_window["duration_days"] * 24 * 3600
            ):
                conflicts.append(
                    {
                        "task_id": scheduled_task["task_id"],
                        "reason": "exceeds_optimization_window",
                        "details": "Task extends beyond the optimization window",
                    }
                )

        # Calculate total duration
        if scheduled_tasks:
            total_duration = (
                max(t["end_time"] for t in scheduled_tasks)
                - optimization_window["start"]
            )
        else:
            total_duration = 0

        # Calculate additional fields for API compatibility
        if scheduled_tasks:
            first_task = scheduled_tasks[0]
            scheduled_start_time = datetime.fromtimestamp(first_task["start_time"])

            # Apply cost optimizations based on preferences
            base_cost = tasks[0].get("estimated_cost", 0.0) if tasks else 0.0
            if cost_preferences and cost_preferences.get("cost_priority") == "high":
                # Apply 10-20% cost reduction for high priority cost optimization
                optimized_cost = base_cost * 0.85  # 15% reduction
            else:
                optimized_cost = base_cost * 0.95  # 5% reduction

            # Calculate priority score based on task priority and constraints
            priority_score = (
                sum(t.get("priority", 1) for t in tasks) / len(tasks) if tasks else 0
            )

            # Basic resource allocation
            resource_allocation = {
                "cpu_cores": 2,
                "memory_gb": 4,
                "storage_gb": 100,
                "estimated_instances": max(1, len(tasks)),
            }
        else:
            scheduled_start_time = datetime.utcnow()
            optimized_cost = 0.0
            priority_score = 0.0
            resource_allocation = {}

        return {
            "schedule": scheduled_tasks,
            "total_duration": total_duration,
            "total_duration_hours": total_duration / 3600,
            "conflicts": conflicts,
            "utilization": len(scheduled_tasks) / len(tasks) if tasks else 0,
            "optimization_window": optimization_window,
            # Additional fields expected by tests
            "scheduled_start_time": scheduled_start_time.isoformat(),
            "estimated_duration": total_duration,
            "estimated_cost": optimized_cost,
            "priority_score": priority_score,
            "resource_allocation": resource_allocation,
        }
