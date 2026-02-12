"""Workflow trace collection and submission to backend."""

# Traceability: CONC-Layer-Core FUNC-ORCH-LIFECYCLE REQ-ORCH-003

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from traigent.cloud.backend_client import BackendIntegratedClient
    from traigent.integrations.observability.workflow_traces import (
        SpanPayload,
        WorkflowGraphPayload,
        WorkflowTracesTracker,
    )

from traigent.utils.env_config import is_backend_offline
from traigent.utils.function_identity import FunctionDescriptor
from traigent.utils.logging import get_logger

logger = get_logger(__name__)


class WorkflowTraceManager:
    """Manages workflow trace span collection and backend submission.

    Collects workflow spans during optimization trials and submits them
    along with a workflow graph to the backend for visualization.
    """

    def __init__(
        self,
        workflow_traces_tracker: WorkflowTracesTracker | None,
        backend_client: BackendIntegratedClient | None,
        function_descriptor: FunctionDescriptor | None,
        optimizer_config_space: dict[str, Any],
        max_trials: int | None,
        optimizer_class_name: str,
        optimization_id: str,
    ) -> None:
        """Initialize workflow trace manager.

        Args:
            workflow_traces_tracker: Tracker for span ingestion (None disables tracing)
            backend_client: Backend API client for session mapping lookups
            function_descriptor: Descriptor for the optimized function
            optimizer_config_space: Config space keys for graph metadata
            max_trials: Maximum trial count for graph metadata
            optimizer_class_name: Optimizer algorithm name for graph metadata
            optimization_id: Unique optimization run identifier
        """
        self._workflow_traces_tracker = workflow_traces_tracker
        self._backend_client = backend_client
        self._function_descriptor = function_descriptor
        self._optimizer_config_space = optimizer_config_space
        self._max_trials = max_trials
        self._optimizer_class_name = optimizer_class_name
        self._optimization_id = optimization_id
        self._collected_spans: list[SpanPayload] = []

    def collect_span(self, span_data: SpanPayload) -> None:
        """Collect a workflow span for later submission to backend.

        Args:
            span_data: Span payload to collect
        """
        if self._workflow_traces_tracker is not None:
            self._collected_spans.append(span_data)

    async def submit_traces(self, session_id: str | None = None) -> None:
        """Submit collected workflow spans and graph to backend.

        Called during optimization finalization to send all collected spans
        and workflow graph to the backend for visualization in the workflow traces view.

        Spans are grouped by configuration_run_id and submitted separately so each
        trial's ConfigurationRun gets its run_metadata updated with trace_id.

        Args:
            session_id: Backend session ID (used to get experiment_id for graph)
        """
        if self._workflow_traces_tracker is None:
            return

        if not self._collected_spans:
            logger.debug("No workflow spans to submit")
            return

        # Skip trace submission when running in offline mode or with mock sessions
        # Mock IDs (mock_session_, mock_exp_, mock_run_, mock-session-) don't exist
        # in the backend database, so trace ingestion would fail with 404
        if is_backend_offline():
            logger.debug("Skipping workflow trace submission: backend is offline")
            self._collected_spans = []
            return

        # Check if session_id is a mock ID (created when backend was unavailable)
        if session_id and (
            session_id.startswith("mock_session_")
            or session_id.startswith("mock-session-")
        ):
            logger.debug(
                f"Skipping workflow trace submission: mock session '{session_id}'"
            )
            self._collected_spans = []
            return

        try:
            # Get trace_id from first span (all spans share same trace)
            trace_id = self._collected_spans[0].trace_id

            # Try to create workflow graph for visualization (only submit once)
            graph = self._create_optimization_workflow_graph(session_id)

            # Group spans by configuration_run_id so each trial gets trace_id in its metadata
            spans_by_config_run = self._group_spans_by_config_run()

            # Submit each group separately
            total_submitted = 0
            graph_id = None
            for idx, (config_run_id, spans) in enumerate(spans_by_config_run.items()):
                # Only include graph in first request
                include_graph = graph if idx == 0 else None

                response = await self._workflow_traces_tracker.ingest_traces_async(
                    graph=include_graph,
                    spans=spans,
                    trace_id=trace_id,
                    configuration_run_id=config_run_id,
                )

                if response.success:
                    total_submitted += len(spans)
                    if response.graph_id:
                        graph_id = response.graph_id
                else:
                    logger.warning(
                        f"Failed to submit spans for config_run {config_run_id}: {response.error}"
                    )

            if total_submitted > 0:
                graph_msg = f", graph_id={graph_id}" if graph_id else ""
                logger.info(
                    f"Submitted {total_submitted} workflow spans "
                    f"for trace {trace_id[:8]}...{graph_msg}"
                )

        except Exception as exc:
            logger.warning(f"Workflow trace submission failed: {exc}")
        finally:
            # Clear collected spans after submission attempt
            self._collected_spans = []

    def _group_spans_by_config_run(self) -> dict[str, list]:
        """Group collected spans by configuration_run_id."""
        spans_by_config_run: dict[str, list] = defaultdict(list)
        for span in self._collected_spans:
            spans_by_config_run[span.configuration_run_id].append(span)
        return dict(spans_by_config_run)

    def _create_optimization_workflow_graph(
        self, session_id: str | None
    ) -> WorkflowGraphPayload | None:
        """Create a workflow graph representing the optimization flow.

        Creates a simple graph with START -> optimization_run -> END nodes
        to enable workflow trace visualization in the frontend.

        Args:
            session_id: Backend session ID for looking up experiment_id

        Returns:
            WorkflowGraphPayload if experiment_id found, None otherwise
        """
        # Import here to avoid circular dependency
        from traigent.integrations.observability.workflow_traces import (
            WorkflowEdge,
            WorkflowGraphPayload,
            WorkflowNode,
        )

        # Get experiment_id from session mapping
        experiment_id = self._get_experiment_id_from_session(session_id)
        if not experiment_id:
            logger.debug("Cannot create workflow graph: no experiment_id available")
            return None

        # Skip graph creation for mock experiment IDs (they don't exist in backend)
        if experiment_id.startswith("mock_exp_") or experiment_id.startswith(
            "mock-exp-"
        ):
            logger.debug(
                f"Cannot create workflow graph: mock experiment_id '{experiment_id}'"
            )
            return None

        # Get experiment_run_id for linking
        experiment_run_id = self._get_experiment_run_id_from_session(session_id)

        # Get function name for display
        function_name = (
            self._function_descriptor.identifier
            if self._function_descriptor
            else "optimization"
        )

        # Create nodes for the optimization workflow
        nodes = [
            WorkflowNode(
                id="__start__",
                type="entry",
                display_name="Start",
                metadata={"purpose": "Workflow entry point"},
            ),
            WorkflowNode(
                id="optimization_run",
                type="agent",
                display_name=f"Optimize: {function_name}",
                tunable_params=(
                    list(self._optimizer_config_space.keys())
                    if self._optimizer_config_space
                    else []
                ),
                metadata={
                    "purpose": "Execute optimization trials",
                    "function": function_name,
                    "max_trials": self._max_trials,
                    "algorithm": self._optimizer_class_name,
                },
            ),
            WorkflowNode(
                id="__end__",
                type="exit",
                display_name="End",
                metadata={"purpose": "Workflow exit point"},
            ),
        ]

        # Create edges
        edges = [
            WorkflowEdge(from_node="__start__", to_node="optimization_run"),
            WorkflowEdge(from_node="optimization_run", to_node="__end__"),
        ]

        return WorkflowGraphPayload(
            experiment_id=experiment_id,
            experiment_run_id=experiment_run_id,
            nodes=nodes,
            edges=edges,
            metadata={
                "workflow_type": "sdk_optimization",
                "function_name": function_name,
                "optimization_id": self._optimization_id,
            },
        )

    def _get_experiment_id_from_session(self, session_id: str | None) -> str | None:
        """Get experiment_id from backend session mapping.

        Args:
            session_id: Backend session ID

        Returns:
            experiment_id if found, None otherwise
        """
        if not session_id or not self._backend_client:
            return None

        try:
            get_mapping = getattr(self._backend_client, "get_session_mapping", None)
            if callable(get_mapping):
                mapping = get_mapping(session_id)
                if mapping:
                    return getattr(mapping, "experiment_id", None)
        except Exception as exc:
            logger.debug(f"Could not get experiment_id from session mapping: {exc}")

        return None

    def _get_experiment_run_id_from_session(self, session_id: str | None) -> str | None:
        """Get experiment_run_id from backend session mapping.

        Args:
            session_id: Backend session ID

        Returns:
            experiment_run_id if found, None otherwise
        """
        if not session_id or not self._backend_client:
            return None

        try:
            get_mapping = getattr(self._backend_client, "get_session_mapping", None)
            if callable(get_mapping):
                mapping = get_mapping(session_id)
                if mapping:
                    return getattr(mapping, "experiment_run_id", None)
        except Exception as exc:
            logger.debug(f"Could not get experiment_run_id from session mapping: {exc}")

        return None
