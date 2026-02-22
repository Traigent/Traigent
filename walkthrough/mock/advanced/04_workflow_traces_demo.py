#!/usr/bin/env python3
"""Example: Workflow Traces Demo

Demonstrates the new workflow traces integration for multi-agent visualization.
This example simulates a LangGraph-like workflow and sends traces to the backend.

Run with: python 04_workflow_traces_demo.py

Uses TRAIGENT_BACKEND_URL from .env (or defaults to http://localhost:5000)
"""

import os
import uuid
from datetime import UTC, datetime

# Load .env file from project root
from dotenv import load_dotenv

load_dotenv()  # Loads .env from current directory or parent directories

# Import workflow traces components
from traigent.integrations.observability.workflow_traces import (
    SpanPayload,
    SpanStatus,
    SpanType,
    WorkflowEdge,
    WorkflowGraphPayload,
    WorkflowNode,
    WorkflowTracesClient,
    detect_loops_in_graph,
)


def generate_trace_id() -> str:
    """Generate a trace ID in hex format."""
    return uuid.uuid4().hex


def generate_span_id() -> str:
    """Generate a span ID in hex format."""
    return uuid.uuid4().hex[:16]


def simulate_workflow_execution(
    experiment_id: str,
    run_id: str,
    trial_id: int,
) -> tuple[list[SpanPayload], WorkflowGraphPayload, str, str]:
    """Simulate a multi-agent RAG workflow execution.

    Simulates this workflow:
        START -> retrieve -> grade_documents -> generate -> END
                    ^              |
                    |              v (if not relevant)
                    +-------- web_search
    """
    trace_id = generate_trace_id()
    config_run_id = f"config-run-{trial_id}"
    now = datetime.now(UTC)

    # Define workflow nodes (including __start__ and __end__ for edges)
    nodes = [
        WorkflowNode(
            id="__start__",
            type="entry",
            display_name="Start",
            metadata={"purpose": "Workflow entry point"},
        ),
        WorkflowNode(
            id="retrieve",
            type="agent",
            display_name="Retriever Agent",
            metadata={"purpose": "Retrieve relevant documents from vector store"},
        ),
        WorkflowNode(
            id="grade_documents",
            type="agent",
            display_name="Document Grader",
            metadata={"purpose": "Grade document relevance"},
        ),
        WorkflowNode(
            id="web_search",
            type="tool",
            display_name="Web Search Tool",
            metadata={"purpose": "Search web for additional context"},
        ),
        WorkflowNode(
            id="generate",
            type="agent",
            display_name="Generator Agent",
            metadata={"purpose": "Generate final answer"},
        ),
        WorkflowNode(
            id="__end__",
            type="exit",
            display_name="End",
            metadata={"purpose": "Workflow exit point"},
        ),
    ]

    # Define workflow edges
    edges = [
        WorkflowEdge(from_node="__start__", to_node="retrieve"),
        WorkflowEdge(from_node="retrieve", to_node="grade_documents"),
        WorkflowEdge(
            from_node="grade_documents",
            to_node="web_search",
            condition="not_relevant",
        ),
        WorkflowEdge(
            from_node="grade_documents",
            to_node="generate",
            condition="relevant",
        ),
        WorkflowEdge(from_node="web_search", to_node="retrieve"),
        WorkflowEdge(from_node="generate", to_node="__end__"),
    ]

    # Detect loops
    loops = detect_loops_in_graph(edges)

    # Create graph payload
    graph = WorkflowGraphPayload(
        experiment_id=experiment_id,
        experiment_run_id=run_id,
        nodes=nodes,
        edges=edges,
        loops=loops,
        metadata={
            "workflow_type": "rag_with_fallback",
            "total_agents": 3,
            "total_tools": 1,
            "entry_point": "retrieve",
        },
    )

    # Simulate span execution using actual SpanPayload fields
    spans: list[SpanPayload] = []

    # Root span for the workflow
    root_span_id = generate_span_id()

    # 1. Retrieve span
    retrieve_span_id = generate_span_id()
    spans.append(
        SpanPayload(
            span_id=retrieve_span_id,
            trace_id=trace_id,
            configuration_run_id=config_run_id,
            span_name="retrieve",
            span_type=SpanType.NODE.value,
            start_time=now.isoformat() + "Z",
            parent_span_id=root_span_id,
            node_id="retrieve",
            end_time=now.isoformat() + "Z",
            status=SpanStatus.COMPLETED.value,
            input_tokens=0,
            output_tokens=0,
            input_data={"k": 5, "method": "similarity"},
            output_data={"documents_count": 5},
        )
    )

    # 2. Grade documents span
    grade_span_id = generate_span_id()
    spans.append(
        SpanPayload(
            span_id=grade_span_id,
            trace_id=trace_id,
            configuration_run_id=config_run_id,
            span_name="grade_documents",
            span_type=SpanType.NODE.value,
            start_time=now.isoformat() + "Z",
            parent_span_id=root_span_id,
            node_id="grade_documents",
            end_time=now.isoformat() + "Z",
            status=SpanStatus.COMPLETED.value,
            decision_reason="Documents are relevant to the query",
            input_data={"documents_count": 5},
            output_data={"relevant_count": 3, "decision": "relevant"},
        )
    )

    # 3. Generate span (LLM call)
    generate_span_id_val = generate_span_id()
    spans.append(
        SpanPayload(
            span_id=generate_span_id_val,
            trace_id=trace_id,
            configuration_run_id=config_run_id,
            span_name="generate",
            span_type=SpanType.LLM_CALL.value,
            start_time=now.isoformat() + "Z",
            parent_span_id=root_span_id,
            node_id="generate",
            end_time=now.isoformat() + "Z",
            status=SpanStatus.COMPLETED.value,
            input_tokens=1200,
            output_tokens=150,
            cost_usd=0.0018,
            input_data={"model": "gpt-4o-mini", "temperature": 0.7},
            output_data={"response": "Generated answer based on context"},
        )
    )

    # Root span (wraps all)
    spans.append(
        SpanPayload(
            span_id=root_span_id,
            trace_id=trace_id,
            configuration_run_id=config_run_id,
            span_name="rag_workflow",
            span_type=SpanType.NODE.value,
            start_time=now.isoformat() + "Z",
            parent_span_id=None,
            node_id=None,
            end_time=now.isoformat() + "Z",
            status=SpanStatus.COMPLETED.value,
            input_data={"experiment_id": experiment_id, "trial_id": trial_id},
            output_data={"success": True},
        )
    )

    return spans, graph, trace_id, config_run_id


def main() -> None:
    """Run the workflow traces demo."""
    print("=" * 60)
    print("Workflow Traces Demo - Multi-Agent Visualization")
    print("=" * 60)

    # Get backend URL and auth token (uses same env vars as rest of SDK)
    backend_url = os.environ.get("TRAIGENT_BACKEND_URL", "http://localhost:5000")
    auth_token = os.environ.get("TRAIGENT_API_KEY")
    print(f"\nBackend URL: {backend_url}")
    print(f"Auth token: {'set' if auth_token else 'NOT SET (will fail with 401)'}")

    # Create client with authentication
    client = WorkflowTracesClient(
        backend_url=backend_url,
        auth_token=auth_token,
        timeout=30.0,
    )

    # Use existing experiment ID (can be overridden via env var)
    # You can find experiment IDs via: curl -H "X-API-Key: $TRAIGENT_API_KEY" http://localhost:5000/api/v1/experiments
    experiment_id = os.environ.get(
        "DEMO_EXPERIMENT_ID", "405e9344666be9eb9b6624b9c5e3d226"
    )
    run_id = f"demo-run-{uuid.uuid4().hex[:8]}"

    print(f"\nExperiment ID: {experiment_id}")
    print(f"Run ID: {run_id}")

    # Simulate a single trial (spans share trace_id and config_run_id)
    print("\nSimulating workflow execution...")

    spans, graph, trace_id, config_run_id = simulate_workflow_execution(
        experiment_id=experiment_id,
        run_id=run_id,
        trial_id=1,
    )
    print(f"  Generated {len(spans)} spans with trace_id={trace_id[:8]}...")

    print(f"\nTotal spans: {len(spans)}")
    print(f"Workflow graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")

    if graph.loops:
        print(f"Detected loops: {[loop.to_dict() for loop in graph.loops]}")

    # Send to backend
    print("\nSending traces to backend...")
    print(f"  trace_id: {trace_id}")
    print(f"  configuration_run_id: {config_run_id}")
    print("  Note: Spans require a real configuration_run_id from the backend.")
    print("        This demo sends the graph only. For full span ingestion,")
    print("        create experiment runs via the API first.")

    try:
        # Send graph only (spans require valid configuration_run_id from backend)
        response = client.ingest_traces(
            graph=graph,
            # Spans are skipped - would require real configuration_run_id
            # spans=spans,
            # trace_id=trace_id,
            # configuration_run_id=config_run_id,
        )

        if response.success:
            print("\nSuccess!")
            print(f"  Trace ID: {response.trace_id}")
            print(f"  Spans ingested: {response.spans_ingested}")
            print(f"  Graph ID: {response.graph_id}")
        else:
            print(f"\nIngestion failed: {response.error}")

    except Exception as e:
        print(f"\nFailed to send traces: {e}")
        print("\nNote: Make sure the backend is running and accessible.")
        print("Set TRAIGENT_BACKEND_URL in .env to point to your backend.")

    print("\n" + "=" * 60)
    print("Check the frontend to see the workflow graph visualization!")
    print("=" * 60)


if __name__ == "__main__":
    main()
