#!/usr/bin/env python3
"""Example: LangGraph Multi-Agent Workflow with Traigent Optimization

Demonstrates a multi-agent RAG workflow with:
- Multiple agents (retriever, grader, generator, web_search)
- Conditional branching (grade_documents -> generate OR web_search)
- Workflow trace visualization in the frontend
- Real LLM mode with OpenAI (gpt-4o-mini) OR mock mode for testing

Run in REAL mode (uses OpenAI API, incurs costs):
    TRAIGENT_API_KEY=<your_key> OPENAI_API_KEY=<your_key> python 05_langgraph_multiagent_demo.py

Run in MOCK mode (no API costs):
    TRAIGENT_API_KEY=<your_key> TRAIGENT_MOCK_LLM=true python 05_langgraph_multiagent_demo.py

The workflow topology will be automatically extracted and sent to the backend
for visualization in the Traigent frontend.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

import traigent
from traigent.integrations.observability.workflow_traces import (
    SpanStatus,
    SpanType,
    WorkflowEdge,
    WorkflowGraphPayload,
    WorkflowNode,
    WorkflowTracesTracker,
    detect_loops_in_graph,
)

# Load environment variables from .env file
load_dotenv()

# Initialize Traigent
traigent.initialize(execution_mode="edge_analytics")

SCRIPT_DIR = Path(__file__).parent.parent  # Go up to mock/ directory

# Check if we're in mock mode
MOCK_MODE = os.environ.get("TRAIGENT_MOCK_LLM", "false").lower() == "true"

# OpenAI client (lazy init)
_openai_client = None

# Pricing for gpt-4o-mini (per 1M tokens) - as of 2025
GPT4O_MINI_INPUT_PRICE = 0.15 / 1_000_000  # $0.15 per 1M input tokens
GPT4O_MINI_OUTPUT_PRICE = 0.60 / 1_000_000  # $0.60 per 1M output tokens


def get_openai_client():
    """Get or create OpenAI client."""
    global _openai_client
    if _openai_client is None:
        try:
            from openai import OpenAI
            _openai_client = OpenAI()  # Uses OPENAI_API_KEY env var
        except ImportError:
            raise ImportError("openai package required for real mode. Install with: pip install openai")
    return _openai_client


def calculate_cost(input_tokens: int, output_tokens: int, model: str = "gpt-4o-mini") -> float:
    """Calculate cost based on token usage."""
    if model == "gpt-4o-mini":
        return (input_tokens * GPT4O_MINI_INPUT_PRICE) + (output_tokens * GPT4O_MINI_OUTPUT_PRICE)
    # Default fallback
    return (input_tokens + output_tokens) * 0.001 / 1000


# Global tracker for per-node span recording
_workflow_tracker: WorkflowTracesTracker | None = None
_agent_spans: list = []  # Collect spans during workflow execution


def record_agent_span(
    node_id: str,
    display_name: str,
    start_time: float,
    end_time: float,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cost_usd: float = 0.0,
) -> None:
    """Record a span for an agent execution."""
    span_data = {
        "span_id": uuid.uuid4().hex[:16],
        "node_id": node_id,
        "display_name": display_name,
        "start_time": datetime.fromtimestamp(start_time, UTC).isoformat(),
        "end_time": datetime.fromtimestamp(end_time, UTC).isoformat(),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost_usd,
    }
    _agent_spans.append(span_data)


# =============================================================================
# State Definition
# =============================================================================


class RAGState(TypedDict):
    """State for the RAG workflow."""

    question: str
    documents: list[str]
    relevant_docs: list[str]
    web_results: list[str]
    answer: str
    iteration: int


# =============================================================================
# Agent Functions (Mock implementations for demo)
# =============================================================================


def retrieve_documents(state: RAGState) -> RAGState:
    """Retriever agent: Fetch documents from vector store.

    Note: This is typically a vector DB lookup, not an LLM call.
    We keep it mock for simplicity (no vector DB setup required).
    """
    start = time.time()
    question = state["question"]
    # Mock retrieval based on question keywords
    if "capital" in question.lower():
        docs = ["Paris is the capital of France.", "France is in Europe."]
    elif "python" in question.lower():
        docs = ["Python is a programming language.", "Python was created by Guido."]
    else:
        docs = ["Generic document 1.", "Generic document 2."]

    end = time.time()
    # Retrieval is typically free (just vector DB lookup)
    record_agent_span("retrieve", "Retriever Agent", start, end, input_tokens=0, cost_usd=0.0)
    return {**state, "documents": docs}


def grade_documents(state: RAGState) -> RAGState:
    """Document grader: Evaluate relevance of retrieved documents."""
    start = time.time()
    docs = state["documents"]
    question = state["question"]
    input_tokens = 0
    output_tokens = 0
    cost_usd = 0.0

    if MOCK_MODE:
        # Simple relevance check - mock grading
        relevant = []
        for doc in docs:
            doc_lower = doc.lower()
            # Check if any question word appears in document
            if any(word in doc_lower for word in question.lower().split() if len(word) > 3):
                relevant.append(doc)
        input_tokens = 100
        output_tokens = 20
        cost_usd = 0.0002
    else:
        # Real LLM grading
        client = get_openai_client()
        relevant = []

        for doc in docs:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a document relevance grader. Answer only 'yes' or 'no'."},
                    {"role": "user", "content": f"Is this document relevant to the question?\n\nQuestion: {question}\n\nDocument: {doc}\n\nAnswer (yes/no):"}
                ],
                max_tokens=5,
                temperature=0,
            )
            answer = response.choices[0].message.content.strip().lower()
            input_tokens += response.usage.prompt_tokens
            output_tokens += response.usage.completion_tokens

            if "yes" in answer:
                relevant.append(doc)

        cost_usd = calculate_cost(input_tokens, output_tokens)

    end = time.time()
    record_agent_span("grade_documents", "Document Grader", start, end, input_tokens=input_tokens, output_tokens=output_tokens, cost_usd=cost_usd)
    return {**state, "relevant_docs": relevant}


def decide_next_step(state: RAGState) -> str:
    """Router: Decide whether to generate answer or search web."""
    if state["relevant_docs"]:
        return "generate"
    elif state["iteration"] < 2:
        return "web_search"
    else:
        # Give up after 2 iterations
        return "generate"


def web_search(state: RAGState) -> RAGState:
    """Web search agent: Search web for additional context."""
    start = time.time()
    question = state["question"]
    # Mock web search results
    web_results = [f"Web result for: {question}", "Additional context from the web."]

    end = time.time()
    record_agent_span("web_search", "Web Search", start, end, cost_usd=0.0005)
    return {
        **state,
        "web_results": web_results,
        "documents": state["documents"] + web_results,
        "iteration": state["iteration"] + 1,
    }


def generate_answer(state: RAGState) -> RAGState:
    """Generator agent: Produce final answer from context."""
    start = time.time()
    docs = state["relevant_docs"] or state["documents"]
    question = state["question"]
    input_tokens = 0
    output_tokens = 0
    cost_usd = 0.0

    if MOCK_MODE:
        # Mock answer generation
        if docs:
            answer = f"Based on the context: {docs[0]}"
        else:
            answer = f"I don't have enough information to answer: {question}"
        input_tokens = 200
        output_tokens = 50
        cost_usd = 0.001
    else:
        # Real LLM generation
        client = get_openai_client()
        context = "\n".join(docs) if docs else "No relevant documents found."

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Answer the question based on the provided context. Be concise."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"}
            ],
            max_tokens=150,
            temperature=0.7,
        )
        answer = response.choices[0].message.content.strip()
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        cost_usd = calculate_cost(input_tokens, output_tokens)

    end = time.time()
    record_agent_span("generate", "Generator Agent", start, end, input_tokens=input_tokens, output_tokens=output_tokens, cost_usd=cost_usd)
    return {**state, "answer": answer}


# =============================================================================
# Build LangGraph Workflow
# =============================================================================


def build_rag_workflow() -> StateGraph:
    """Build the multi-agent RAG workflow graph."""
    workflow = StateGraph(RAGState)

    # Add nodes (agents)
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("web_search", web_search)
    workflow.add_node("generate", generate_answer)

    # Set entry point
    workflow.set_entry_point("retrieve")

    # Add edges
    workflow.add_edge("retrieve", "grade_documents")

    # Conditional edge: grade_documents -> generate OR web_search
    workflow.add_conditional_edges(
        "grade_documents",
        decide_next_step,
        {
            "generate": "generate",
            "web_search": "web_search",
        },
    )

    # Web search loops back to grade_documents
    workflow.add_edge("web_search", "grade_documents")

    # Generate ends the workflow
    workflow.add_edge("generate", END)

    return workflow


# =============================================================================
# Manual Graph Extraction (for visualization without running LangGraph)
# =============================================================================


def extract_workflow_graph(
    experiment_id: str, experiment_run_id: str | None = None
) -> WorkflowGraphPayload:
    """Extract workflow graph topology for visualization.

    This creates a graph representation that matches the LangGraph workflow
    structure for visualization in the Traigent frontend.
    """
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
            tunable_params=["retrieve.top_k", "retrieve.similarity_threshold"],
            metadata={
                "purpose": "Retrieve documents from vector store",
                "function": "retrieve_documents",
            },
        ),
        WorkflowNode(
            id="grade_documents",
            type="agent",
            display_name="Document Grader",
            tunable_params=["grade_documents.model", "grade_documents.relevance_threshold"],
            metadata={
                "purpose": "Evaluate document relevance",
                "function": "grade_documents",
            },
        ),
        WorkflowNode(
            id="web_search",
            type="tool",
            display_name="Web Search",
            tunable_params=["web_search.max_results", "web_search.search_depth"],
            metadata={
                "purpose": "Search web for additional context",
                "function": "web_search",
            },
        ),
        WorkflowNode(
            id="generate",
            type="agent",
            display_name="Generator Agent",
            tunable_params=["generate.model", "generate.temperature", "generate.max_tokens"],
            metadata={
                "purpose": "Generate final answer from context",
                "function": "generate_answer",
            },
        ),
        WorkflowNode(
            id="__end__",
            type="exit",
            display_name="End",
            metadata={"purpose": "Workflow exit point"},
        ),
    ]

    edges = [
        WorkflowEdge(from_node="__start__", to_node="retrieve"),
        WorkflowEdge(from_node="retrieve", to_node="grade_documents"),
        WorkflowEdge(
            from_node="grade_documents",
            to_node="generate",
            edge_type="conditional",
            condition="documents_relevant",
        ),
        WorkflowEdge(
            from_node="grade_documents",
            to_node="web_search",
            edge_type="conditional",
            condition="not_relevant",
        ),
        WorkflowEdge(from_node="web_search", to_node="grade_documents"),
        WorkflowEdge(from_node="generate", to_node="__end__"),
    ]

    loops = detect_loops_in_graph(edges)

    return WorkflowGraphPayload(
        experiment_id=experiment_id,
        experiment_run_id=experiment_run_id,
        nodes=nodes,
        edges=edges,
        loops=loops,
        metadata={
            "workflow_type": "multi_agent_rag",
            "agents": ["retrieve", "grade_documents", "generate"],
            "tools": ["web_search"],
            "has_conditional_branching": True,
            "has_loops": True,
        },
    )


# =============================================================================
# Optimized Function
# =============================================================================


@traigent.optimize(
    eval_dataset=str(SCRIPT_DIR / "simple_questions.jsonl"),
    objectives=["accuracy", "cost"],
    configuration_space={
        # Generator Agent parameters (namespaced)
        # Note: In real mode, we always use gpt-4o-mini for simplicity
        # These params demonstrate what Traigent CAN optimize
        "generate.model": ["gpt-4o-mini"],  # Cheap model only
        "generate.temperature": [0.3, 0.7],  # Reduced options to minimize API calls
        # Retriever Agent parameters
        "retrieve.top_k": [3, 5],  # Reduced options
    },
    execution_mode="edge_analytics",
)
def run_rag_workflow(question: str) -> str:
    """Run the RAG workflow on a question.

    This function is optimized by Traigent to find the best model/temperature
    combination for the RAG workflow.
    """
    global _agent_spans
    _agent_spans = []  # Clear spans for this execution

    # Build and compile the workflow
    workflow = build_rag_workflow()
    app = workflow.compile()

    # Initial state
    initial_state: RAGState = {
        "question": question,
        "documents": [],
        "relevant_docs": [],
        "web_results": [],
        "answer": "",
        "iteration": 0,
    }

    # Run the workflow
    result = app.invoke(initial_state)

    return result["answer"]


# =============================================================================
# Main
# =============================================================================


async def main() -> None:
    """Run the LangGraph multi-agent demo with Traigent optimization."""
    print("=" * 70)
    print("LangGraph Multi-Agent RAG Workflow with Traigent Optimization")
    print("=" * 70)

    # Check environment
    backend_url = os.environ.get("TRAIGENT_BACKEND_URL", "http://localhost:5000")
    api_key = os.environ.get("TRAIGENT_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    print(f"\nBackend URL: {backend_url}")
    print(f"Traigent API Key: {'set' if api_key else 'NOT SET'}")
    print(f"OpenAI API Key: {'set' if openai_key else 'NOT SET'}")
    print(f"\n*** MODE: {'MOCK (no real API costs)' if MOCK_MODE else 'REAL (using OpenAI gpt-4o-mini)'} ***")

    if not MOCK_MODE and not openai_key:
        print("\nERROR: OPENAI_API_KEY not set for real mode.")
        print("Either set OPENAI_API_KEY or run with TRAIGENT_MOCK_LLM=true")
        return

    if not api_key:
        print("\nERROR: TRAIGENT_API_KEY not set. Please set it and retry.")
        return

    # First, send the workflow graph for visualization
    print("\n" + "-" * 70)
    print("Step 1: Sending workflow graph topology to backend...")
    print("-" * 70)

    tracker = WorkflowTracesTracker(
        backend_url=backend_url,
        auth_token=api_key,
    )

    # We'll get the experiment_id after optimization starts
    # For now, use a placeholder that will be updated
    print("  Graph will be sent during optimization...")

    # Run optimization
    print("\n" + "-" * 70)
    print("Step 2: Running Traigent optimization...")
    print("-" * 70)
    print("\nWorkflow structure:")
    print("  START -> retrieve -> grade_documents --(relevant)--> generate -> END")
    print("                            |")
    print("                            +---(not relevant)--> web_search --+")
    print("                            ^                                   |")
    print("                            +-----------------------------------+")

    # Use fewer trials in real mode to save API costs
    max_trials = 4 if MOCK_MODE else 2
    print(f"\nOptimizing parameters ({max_trials} trials)...")

    results = await run_rag_workflow.optimize(
        algorithm="grid",
        max_trials=max_trials,
        random_seed=42,
    )

    # Send the workflow graph to backend for visualization
    # Note: The experiment_id comes from the backend session
    print("\n" + "-" * 70)
    print("Step 3: Sending workflow graph topology...")
    print("-" * 70)

    # Get experiment_id and experiment_run_id directly from results metadata
    experiment_id = None
    experiment_run_id = None
    if results.metadata:
        experiment_id = results.metadata.get("experiment_id")
        experiment_run_id = results.metadata.get("experiment_run_id")

    if experiment_id:
        graph_payload = extract_workflow_graph(experiment_id, experiment_run_id)
        response = tracker.client.ingest_traces(graph=graph_payload)
        if response.success:
            print(f"  Graph sent successfully (ID: {response.graph_id})")
            print(f"  Nodes: {len(graph_payload.nodes)}, Edges: {len(graph_payload.edges)}")
            if graph_payload.loops:
                print(f"  Detected loops: {len(graph_payload.loops)}")
        else:
            print(f"  Failed to send graph: {response.error}")

        # Step 4: Submit per-agent spans for attribution
        print("\n" + "-" * 70)
        print("Step 4: Submitting per-agent spans for attribution...")
        print("-" * 70)

        # Import SpanPayload for creating per-agent spans
        from traigent.integrations.observability.workflow_traces import SpanPayload

        # Get trial IDs from results
        trial_ids = []
        for trial in results.trials:
            if hasattr(trial, "trial_id"):
                trial_ids.append(trial.trial_id)

        if trial_ids:
            # Create per-agent spans for the last trial (as a demo)
            # In production, you'd create spans during each trial execution
            trace_id = results.metadata.get("optimization_id", uuid.uuid4().hex)
            config_run_id = trial_ids[-1]  # Use last trial

            # Create spans for each agent node with realistic durations
            # Simulating: retrieve=50ms, grade=30ms, generate=200ms
            from datetime import timedelta

            base_time = datetime.now(UTC)
            agent_spans = [
                SpanPayload(
                    span_id=uuid.uuid4().hex[:16],
                    trace_id=trace_id,
                    configuration_run_id=config_run_id,
                    span_name="retrieve_documents",
                    span_type=SpanType.NODE.value,
                    start_time=base_time.isoformat(),
                    end_time=(base_time + timedelta(milliseconds=50)).isoformat(),
                    status=SpanStatus.COMPLETED.value,
                    node_id="retrieve",  # Links to graph node
                    input_tokens=50,
                    output_tokens=0,
                    cost_usd=0.0001,
                ),
                SpanPayload(
                    span_id=uuid.uuid4().hex[:16],
                    trace_id=trace_id,
                    configuration_run_id=config_run_id,
                    span_name="grade_documents",
                    span_type=SpanType.NODE.value,
                    start_time=(base_time + timedelta(milliseconds=50)).isoformat(),
                    end_time=(base_time + timedelta(milliseconds=80)).isoformat(),
                    status=SpanStatus.COMPLETED.value,
                    node_id="grade_documents",
                    input_tokens=100,
                    output_tokens=20,
                    cost_usd=0.0002,
                ),
                SpanPayload(
                    span_id=uuid.uuid4().hex[:16],
                    trace_id=trace_id,
                    configuration_run_id=config_run_id,
                    span_name="generate_answer",
                    span_type=SpanType.NODE.value,
                    start_time=(base_time + timedelta(milliseconds=80)).isoformat(),
                    end_time=(base_time + timedelta(milliseconds=280)).isoformat(),
                    status=SpanStatus.COMPLETED.value,
                    node_id="generate",
                    input_tokens=200,
                    output_tokens=50,
                    cost_usd=0.001,
                ),
            ]

            # Submit per-agent spans
            span_response = tracker.client.ingest_traces(
                spans=agent_spans,
                trace_id=trace_id,
                configuration_run_id=config_run_id,
            )
            if span_response.success:
                print(f"  Submitted {len(agent_spans)} per-agent spans")
                print("  Agent nodes: retrieve, grade_documents, generate")
            else:
                print(f"  Failed to submit spans: {span_response.error}")
        else:
            print("  No trial IDs available for per-agent span submission")
    else:
        print("  Note: Workflow graph visualization requires experiment_id.")
        print("  Trial execution spans (6) were submitted successfully.")
        print("  To see the graph, use the 04_workflow_traces_demo.py example.")

    # Display results
    print("\n" + "-" * 70)
    print("Optimization Results")
    print("-" * 70)

    print("\nBest Configuration:")
    print(f"  Model: {results.best_config.get('model')}")
    print(f"  Temperature: {results.best_config.get('temperature')}")

    print("\nPerformance:")
    print(f"  Accuracy: {results.best_metrics.get('accuracy', 0):.2%}")
    print(f"  Cost: ${results.best_metrics.get('cost', 0):.6f}")

    print("\n" + "=" * 70)
    print("Check the Traigent frontend to see:")
    print("  1. Workflow graph visualization (nodes, edges, loops)")
    print("  2. Trial execution traces (spans for each trial)")
    print("  3. Agent performance breakdown")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
