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
        agent_id: Agent the backend bound this session to (None when the
            backend did not disclose one)
        head_generation: The agent's head generation read when the session was
            created, i.e. at the start of the optimization step (None when it
            could not be read or the run is offline)
        head_environment: Environment the head generation was read for
        agent_ref: Opaque agent ref the content-less session-create 201 advertised
        content_less: True when the backend negotiated ``content_less_v1``; every
            promotion of such a session is protected (ref + captured generation)
        head_capture_refusal: Closed code (``HeadRefusalCode`` value) when a
            content-less run could not capture its head at start; a later publish
            is then refused locally
    """

    session_id: str | None
    dataset_name: str
    function_name: str | None
    optimization_id: str
    start_time: float
    agent_id: str | None = None
    head_generation: int | None = None
    head_environment: str | None = None
    agent_ref: str | None = None
    content_less: bool = False
    head_capture_refusal: str | None = None
