"""Dummy server implementation for testing privacy-first mode.

This module provides a complete mock server that simulates the TraiGent Cloud
service behavior for privacy-first optimization mode, ensuring that:
1. No actual data is received from the client
2. Only metadata and aggregated metrics are processed
3. Smart subset selection works based on indices only
"""

import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from traigent.cloud.models import (
    DatasetSubsetIndices,
    NextTrialRequest,
    NextTrialResponse,
    OptimizationFinalizationRequest,
    OptimizationFinalizationResponse,
    OptimizationSessionStatus,
    SessionCreationRequest,
    SessionCreationResponse,
    TrialResultSubmission,
    TrialStatus,
    TrialSuggestion,
)


@dataclass
class MockSession:
    """Internal session state for the mock server."""

    session_id: str
    function_name: str
    configuration_space: dict[str, Any]
    objectives: list[str]
    dataset_metadata: dict[str, Any]
    max_trials: int
    optimization_strategy: dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)
    trials: list[dict[str, Any]] = field(default_factory=list)
    suggestions: list[dict[str, Any]] = field(default_factory=list)
    best_config: dict[str, Any] | None = None
    best_metrics: dict[str, float] | None = None
    status: OptimizationSessionStatus = OptimizationSessionStatus.ACTIVE


class DummyPrivacyServer:
    """Mock server that simulates TraiGent Cloud for privacy-first mode testing."""

    def __init__(self):
        self.sessions: dict[str, MockSession] = {}
        self.trial_counter = 0
        self.session_counter = 0

        # Privacy verification tracking
        self.privacy_violations: list[str] = []
        self.received_data: list[dict[str, Any]] = []

    async def create_session(
        self, request: SessionCreationRequest
    ) -> SessionCreationResponse:
        """Create a new optimization session."""
        # Privacy check: Ensure no actual data is sent
        self._verify_no_data_in_request(request)

        self.session_counter += 1
        session_id = f"mock-session-{self.session_counter}"

        # Extract optimization strategy
        strategy = request.optimization_strategy or {
            "exploration_ratio": 0.3,
            "min_examples_per_trial": 5,
            "adaptive_sampling": True,
            "max_subset_size": 50,
        }

        # Create session
        session = MockSession(
            session_id=session_id,
            function_name=request.function_name,
            configuration_space=request.configuration_space,
            objectives=request.objectives,
            dataset_metadata=request.dataset_metadata,
            max_trials=request.max_trials,
            optimization_strategy=strategy,
        )

        self.sessions[session_id] = session

        return SessionCreationResponse(
            session_id=session_id,
            status=OptimizationSessionStatus.ACTIVE,
            optimization_strategy=strategy,
            estimated_duration=float(
                request.max_trials * 2
            ),  # 2 seconds per trial estimate
            billing_estimate={
                "estimated_cost": 0.10 * request.max_trials,
                "cost_per_trial": 0.10,
            },
        )

    async def get_next_trial(self, request: NextTrialRequest) -> NextTrialResponse:
        """Get next trial suggestion with smart subset selection."""
        # Privacy check
        self._verify_no_data_in_request(request)

        session = self.sessions.get(request.session_id)
        if not session:
            raise ValueError(f"Session {request.session_id} not found")

        # Check if optimization should continue
        if len(session.trials) >= session.max_trials:
            session.status = OptimizationSessionStatus.COMPLETED
            return NextTrialResponse(
                suggestion=None,
                should_continue=False,
                reason="Max trials reached",
                session_status=session.status,
            )

        # Generate configuration based on optimization phase
        self.trial_counter += 1
        trial_number = len(session.trials) + 1

        config = self._generate_smart_config(session, trial_number)

        # Use dataset_size from request metadata if available, otherwise from session
        dataset_size = session.dataset_metadata.get("size", 100)
        if request.request_metadata and "dataset_size" in request.request_metadata:
            dataset_size = request.request_metadata["dataset_size"]

        subset_indices = self._generate_smart_subset(
            session, trial_number, dataset_size
        )

        # Create suggestion
        suggestion = TrialSuggestion(
            trial_id=f"mock-trial-{self.trial_counter}",
            session_id=session.session_id,
            trial_number=trial_number,
            config=config,
            dataset_subset=subset_indices,
            exploration_type=self._determine_exploration_type(session, trial_number),
            priority=self._calculate_priority(session, trial_number),
            estimated_duration=2.0,
        )

        # Store the suggestion for later reference
        session.suggestions.append(
            {
                "trial_id": suggestion.trial_id,
                "config": suggestion.config,
                "dataset_subset": suggestion.dataset_subset,
                "timestamp": datetime.now().isoformat(),
            }
        )

        return NextTrialResponse(
            suggestion=suggestion, should_continue=True, session_status=session.status
        )

    async def submit_result(self, result: TrialResultSubmission) -> None:
        """Process trial results submission."""
        # Privacy check: Ensure only metrics are submitted
        self._verify_privacy_compliant_result(result)

        session = self.sessions.get(result.session_id)
        if not session:
            raise ValueError(f"Session {result.session_id} not found")

        # Store trial data - include subset size from the suggestion
        trial_data = {
            "trial_id": result.trial_id,
            "metrics": result.metrics,
            "duration": result.duration,
            "status": result.status,
            "metadata": result.metadata or {},
        }

        # Find the corresponding suggestion to get subset size
        for suggestion in session.suggestions:
            if suggestion.get("trial_id") == result.trial_id:
                subset_indices = suggestion.get("dataset_subset")
                if subset_indices and hasattr(subset_indices, "indices"):
                    trial_data["metadata"]["subset_size"] = len(subset_indices.indices)
                break

        session.trials.append(trial_data)

        # Update best metrics if improved
        if result.status == TrialStatus.COMPLETED and result.metrics:
            if self._is_better_metrics(
                result.metrics, session.best_metrics, session.objectives
            ):
                session.best_metrics = result.metrics
                # Note: In real implementation, we'd track which config produced these metrics
                session.best_config = self._infer_config_from_trial_id(result.trial_id)

    async def finalize_session(
        self, request: OptimizationFinalizationRequest
    ) -> OptimizationFinalizationResponse:
        """Finalize optimization session."""
        session = self.sessions.get(request.session_id)
        if not session:
            raise ValueError(f"Session {request.session_id} not found")

        session.status = OptimizationSessionStatus.COMPLETED

        # Calculate statistics
        successful_trials = sum(
            1 for t in session.trials if t.get("status") == TrialStatus.COMPLETED
        )

        total_duration = sum(t.get("duration", 0) for t in session.trials)

        # Estimate cost savings from subset selection
        avg_subset_size = np.mean(
            [t.get("metadata", {}).get("subset_size", 50) for t in session.trials]
        )
        dataset_size = session.dataset_metadata.get("size", 100)
        cost_savings = (
            1.0 - (avg_subset_size / dataset_size) if dataset_size > 0 else 0.5
        )

        # Prepare response
        response = OptimizationFinalizationResponse(
            session_id=session.session_id,
            best_config=session.best_config or {"temperature": 0.7, "model": "o4-mini"},
            best_metrics=session.best_metrics or {"accuracy": 0.0},
            total_trials=len(session.trials),
            successful_trials=successful_trials,
            total_duration=total_duration,
            cost_savings=cost_savings,
        )

        if request.include_full_history:
            response.full_history = [
                TrialResultSubmission(
                    session_id=session.session_id,
                    trial_id=t["trial_id"],
                    metrics=t["metrics"],
                    duration=t["duration"],
                    status=t["status"],
                    metadata=t.get("metadata", {}),
                )
                for t in session.trials
            ]

        return response

    def _generate_smart_config(
        self, session: MockSession, trial_number: int
    ) -> dict[str, Any]:
        """Generate configuration based on optimization phase and history."""
        config = {}

        for param, space in session.configuration_space.items():
            if isinstance(space, list):
                # Categorical parameter
                if trial_number <= 3:
                    # Early exploration: try different values
                    config[param] = random.choice(space)
                else:
                    # Later: bias toward better performing values
                    if session.best_config and param in session.best_config:
                        # 70% chance to use best, 30% to explore
                        if random.random() < 0.7:
                            config[param] = session.best_config[param]
                        else:
                            config[param] = random.choice(space)
                    else:
                        config[param] = random.choice(space)

            elif isinstance(space, tuple) and len(space) == 2:
                # Continuous parameter
                min_val, max_val = space
                if trial_number <= 3:
                    # Early exploration: random sampling
                    config[param] = random.uniform(min_val, max_val)
                else:
                    # Later: focus around promising regions
                    if session.best_config and param in session.best_config:
                        # Sample around best value with decreasing variance
                        best_val = session.best_config[param]
                        variance = (max_val - min_val) * 0.2 / (trial_number**0.5)
                        config[param] = np.clip(
                            np.random.normal(best_val, variance), min_val, max_val
                        )
                    else:
                        config[param] = random.uniform(min_val, max_val)

        return config

    def _generate_smart_subset(
        self, session: MockSession, trial_number: int, dataset_size: int
    ) -> DatasetSubsetIndices:
        """Generate smart subset indices based on optimization phase."""
        strategy = session.optimization_strategy

        # Determine subset size based on phase
        if trial_number <= 3:
            # Early exploration: small subsets
            subset_size = strategy.get("min_examples_per_trial", 5)
            selection_strategy = "diverse_sampling"
            confidence = 0.6
        elif trial_number <= 10:
            # Mid exploration: medium subsets
            subset_size = min(20, max(10, dataset_size // 5))
            selection_strategy = "representative_sampling"
            confidence = 0.75
        else:
            # Late exploitation: larger subsets for validation
            max_size = strategy.get("max_subset_size", 50)
            subset_size = min(max_size, max(15, dataset_size // 2))
            selection_strategy = "high_confidence_sampling"
            confidence = 0.9

        # Generate indices - ensure valid bounds
        subset_size = min(subset_size, dataset_size)  # Never exceed dataset size

        if selection_strategy == "diverse_sampling":
            # Select diverse indices across dataset
            if subset_size >= dataset_size:
                indices = list(range(dataset_size))
            else:
                step = max(1, dataset_size // subset_size)
                indices = list(range(0, dataset_size, step))[:subset_size]
        elif selection_strategy == "representative_sampling":
            # Stratified sampling
            indices = sorted(random.sample(range(dataset_size), subset_size))
        else:
            # High confidence: focus on challenging examples (simulate by using middle range)
            start = dataset_size // 4
            end = min(dataset_size, 3 * dataset_size // 4)
            if end - start < subset_size:
                # If middle range is too small, use full range
                indices = sorted(random.sample(range(dataset_size), subset_size))
            else:
                indices = sorted(random.sample(range(start, end), subset_size))

        return DatasetSubsetIndices(
            indices=indices,
            selection_strategy=selection_strategy,
            confidence_level=confidence,
            estimated_representativeness=confidence * 0.95,
            metadata={
                "phase": self._determine_exploration_type(session, trial_number),
                "subset_ratio": len(indices) / dataset_size,
            },
        )

    def _determine_exploration_type(
        self, session: MockSession, trial_number: int
    ) -> str:
        """Determine if trial should explore or exploit."""
        exploration_ratio = session.optimization_strategy.get("exploration_ratio", 0.3)

        if trial_number <= 3:
            return "exploration"
        elif random.random() < exploration_ratio:
            return "exploration"
        else:
            return "exploitation"

    def _calculate_priority(self, session: MockSession, trial_number: int) -> int:
        """Calculate trial priority."""
        if trial_number <= 3:
            return 3  # High priority for initial exploration
        elif self._determine_exploration_type(session, trial_number) == "exploitation":
            return 2  # Medium priority for exploitation
        else:
            return 1  # Lower priority for later exploration

    def _is_better_metrics(
        self,
        new_metrics: dict[str, float],
        best_metrics: dict[str, float] | None,
        objectives: list[str],
    ) -> bool:
        """Check if new metrics are better than current best."""
        if not best_metrics:
            return True

        # Simple comparison: average of all objectives
        new_score = np.mean([new_metrics.get(obj, 0) for obj in objectives])
        best_score = np.mean([best_metrics.get(obj, 0) for obj in objectives])

        return new_score > best_score

    def _infer_config_from_trial_id(self, trial_id: str) -> dict[str, Any]:
        """Infer configuration from trial ID (mock implementation)."""
        # In real implementation, we'd track trial_id -> config mapping
        # For testing, return a mock config
        return {"temperature": 0.7, "model": "GPT-4o", "max_tokens": 200}

    def _verify_no_data_in_request(self, request: Any) -> None:
        """Verify that no actual data is in the request."""
        request_dict = request.__dict__ if hasattr(request, "__dict__") else {}

        # Check for dataset field
        if "dataset" in request_dict and request_dict["dataset"] is not None:
            self.privacy_violations.append(
                f"Dataset found in request: {type(request).__name__}"
            )

        # Check for any field that might contain actual data
        suspicious_fields = ["data", "examples", "inputs", "outputs", "raw_data"]
        for field_name in suspicious_fields:
            if field_name in request_dict and request_dict[field_name] is not None:
                self.privacy_violations.append(
                    f"Suspicious field '{field_name}' found in request: {type(request).__name__}"
                )

        # Record what we received (for verification)
        self.received_data.append(
            {
                "request_type": type(request).__name__,
                "fields": list(request_dict.keys()),
                "metadata": request_dict.get("dataset_metadata", {}),
            }
        )

    def _verify_privacy_compliant_result(self, result: TrialResultSubmission) -> None:
        """Verify that result submission is privacy compliant."""
        # Check outputs_sample is either None or very limited
        if result.outputs_sample is not None and len(result.outputs_sample) > 5:
            self.privacy_violations.append(
                f"Too many output samples in result: {len(result.outputs_sample)}"
            )

        # Verify only aggregated metrics
        if result.metrics:
            for key, value in result.metrics.items():
                if not isinstance(value, (int, float)):
                    self.privacy_violations.append(
                        f"Non-numeric metric found: {key} = {type(value)}"
                    )

    def get_privacy_report(self) -> dict[str, Any]:
        """Get privacy compliance report."""
        return {
            "violations": self.privacy_violations,
            "violation_count": len(self.privacy_violations),
            "requests_received": len(self.received_data),
            "data_fields_seen": {
                field for req in self.received_data for field in req.get("fields", [])
            },
            "compliant": len(self.privacy_violations) == 0,
        }

    def reset(self) -> None:
        """Reset server state for new test."""
        self.sessions.clear()
        self.trial_counter = 0
        self.session_counter = 0
        self.privacy_violations.clear()
        self.received_data.clear()
