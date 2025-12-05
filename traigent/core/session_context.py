"""Session context for backend session management."""

# Traceability: CONC-Layer-Core CONC-Quality-Reliability FUNC-CLOUD-HYBRID FUNC-ORCH-LIFECYCLE REQ-CLOUD-009 REQ-ORCH-003 SYNC-CloudHybrid

from dataclasses import dataclass


@dataclass
class SessionContext:
    """Context for backend session management.

    Bundles session-related parameters to reduce method parameter counts
    and provide clear ownership of session state.

    Attributes:
        session_id: Backend session identifier (None if backend disabled)
        dataset_name: Name of evaluation dataset
        function_name: Fully-qualified identifier for the optimized function
        optimization_id: Unique identifier for this optimization run
        start_time: Timestamp when optimization started
    """

    session_id: str | None
    dataset_name: str
    function_name: str | None
    optimization_id: str
    start_time: float
