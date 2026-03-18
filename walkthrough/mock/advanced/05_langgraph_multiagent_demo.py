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
import os
import random
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from opentelemetry import trace

import traigent
from traigent.integrations.observability.workflow_traces import (
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

SCRIPT_DIR = Path(__file__).parent  # Current directory (advanced/)

# Check if we're in mock mode
MOCK_MODE = os.environ.get("TRAIGENT_MOCK_LLM", "false").lower() == "true"

# Model configuration
DEFAULT_MODEL = "gpt-4.1-nano"

# NOTE: LangGraphAdapter will auto-instrument workflows when available.
# This demo shows MANUAL workarounds until LangGraphAdapter is implemented.
# Ideal experience: users just define agents, SDK handles instrumentation.

# OpenTelemetry tracer for span recording
# NOTE: This is temporary - should be handled by LangGraphAdapter
tracer = trace.get_tracer(__name__)

# Global tracker for workflow topology extraction
_workflow_tracker: WorkflowTracesTracker | None = None


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
    In production, this would use a vector database (Pinecone, Weaviate, etc.).

    NOTE: LangGraphAdapter will auto-instrument this node with OTEL spans.
    """
    question = state["question"]

    # Mock retrieval based on question keywords
    if "capital" in question.lower():
        docs = ["Paris is the capital of France.", "France is in Europe."]
    elif "python" in question.lower():
        docs = ["Python is a programming language.", "Python was created by Guido."]
    else:
        docs = ["Generic document 1.", "Generic document 2."]

    return {**state, "documents": docs}


def grade_documents(state: RAGState) -> RAGState:
    """Document grader: Evaluate relevance of retrieved documents.

    Note: Uses LangChain ChatOpenAI wrapper for automatic cost tracking.
    The LangChain interceptor automatically captures tokens and cost.

    NOTE: LangGraphAdapter will:
    - Auto-create OTEL spans with node.id="grade_documents"
    - Auto-inject temperature from config_space["grade_documents.temperature"]
    """
    docs = state["documents"]
    question = state["question"]

    if MOCK_MODE:
        # Simple relevance check - mock grading
        relevant = []
        for doc in docs:
            doc_lower = doc.lower()
            # Check if any question word appears in document
            if any(
                word in doc_lower for word in question.lower().split() if len(word) > 3
            ):
                relevant.append(doc)
    else:
        # Real LLM grading - use LangChain wrapper for automatic cost tracking
        llm = ChatOpenAI(
            model=DEFAULT_MODEL,
            max_tokens=5,
            temperature=0,
        )
        relevant = []

        for doc in docs:
            messages = [
                SystemMessage(
                    content="You are a document relevance grader. Answer only 'yes' or 'no'."
                ),
                HumanMessage(
                    content=f"Is this document relevant to the question?\n\nQuestion: {question}\n\nDocument: {doc}\n\nAnswer (yes/no):"
                ),
            ]
            response = llm.invoke(messages)
            answer = response.content.strip().lower()

            if "yes" in answer:
                relevant.append(doc)

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
    """Web search agent: Search web for additional context.

    NOTE: LangGraphAdapter will auto-instrument this node.
    """
    question = state["question"]
    # Mock web search results
    web_results = [f"Web result for: {question}", "Additional context from the web."]

    return {
        **state,
        "web_results": web_results,
        "documents": state["documents"] + web_results,
        "iteration": state["iteration"] + 1,
    }


def generate_answer(state: RAGState) -> RAGState:
    """Generator agent: Produce final answer from context.

    Note: Uses LangChain ChatOpenAI wrapper for automatic cost tracking.
    The LangChain interceptor automatically captures tokens and cost.

    NOTE: LangGraphAdapter will:
    - Auto-inject temperature from config_space["generate.temperature"]
    - Auto-create OTEL spans
    """
    docs = state["relevant_docs"] or state["documents"]
    question = state["question"]

    if MOCK_MODE:
        # Mock answer generation
        if docs:
            answer = f"Based on the context: {docs[0]}"
        else:
            answer = f"I don't have enough information to answer: {question}"
    else:
        # Real LLM generation - use LangChain wrapper for automatic cost tracking
        llm = ChatOpenAI(
            model=DEFAULT_MODEL,
            max_tokens=150,
            temperature=0.7,  # NOTE: Will be injected from config_space when LangGraphAdapter is available
        )
        context = "\n".join(docs) if docs else "No relevant documents found."

        messages = [
            SystemMessage(
                content="You are a helpful assistant. Answer the question based on the provided context. Be concise."
            ),
            HumanMessage(
                content=f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
            ),
        ]
        response = llm.invoke(messages)
        answer = response.content.strip()

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
            tunable_params=[
                "grade_documents.model",
                "grade_documents.relevance_threshold",
            ],
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
            tunable_params=[
                "generate.model",
                "generate.temperature",
                "generate.max_tokens",
            ],
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
# Custom Metrics (Accuracy and Cost simulation for mock mode)
# =============================================================================


def accuracy_metric(output: str, expected: str) -> float:
    """Calculate accuracy by checking if expected output is in generated answer.

    This is a simple substring match metric. For production use cases,
    you might want to use semantic similarity, fuzzy matching, or LLM-as-judge.

    Args:
        output: The generated answer from the workflow (passed by evaluator)
        expected: The expected answer from the dataset

    Returns:
        1.0 if expected output is found in generated answer, 0.0 otherwise
        In mock mode, returns simulated accuracy (0.6-0.95) for realistic demo
    """
    # In mock mode, simulate realistic accuracy scores
    if MOCK_MODE:
        # Simulate varying accuracy based on output content
        # This gives realistic-looking results in the demo
        random.seed(hash(output + expected) % 2**32)  # Deterministic per example
        return random.uniform(0.6, 0.95)

    # Real mode: Normalize both strings for comparison
    output_lower = output.lower().strip()
    expected_lower = expected.lower().strip()

    # Check if expected answer appears in the output
    if expected_lower in output_lower:
        return 1.0

    # Also check individual words for partial credit (e.g., "Paris" in "The capital is Paris, France")
    expected_words = expected_lower.split()
    if len(expected_words) == 1 and expected_words[0] in output_lower:
        return 1.0

    return 0.0


def cost_metric(output: str, expected: str) -> float:
    """Calculate simulated cost for each example.

    In mock mode, simulates realistic LLM costs based on output length.
    In real mode, returns 0.0 (actual costs are tracked by the SDK).

    The simulated costs are based on realistic gpt-4o-mini pricing:
    - ~$0.15 per 1M input tokens
    - ~$0.60 per 1M output tokens

    Args:
        output: The generated answer from the workflow
        expected: The expected answer (unused, but required by evaluator signature)

    Returns:
        Simulated cost in USD for mock mode, 0.0 for real mode
    """
    if MOCK_MODE:
        # Simulate realistic costs based on output length
        # Assuming ~4 chars per token (rough estimate)
        output_tokens = max(len(output) // 4, 10)
        input_tokens = 150  # Approximate prompt size for RAG workflow

        # Pricing: gpt-4o-mini rates (per 1M tokens)
        input_cost_per_1m = 0.15
        output_cost_per_1m = 0.60

        input_cost = (input_tokens / 1_000_000) * input_cost_per_1m
        output_cost = (output_tokens / 1_000_000) * output_cost_per_1m

        # Add some variance for realism
        random.seed(hash(output) % 2**32)
        variance = random.uniform(0.8, 1.2)

        return (input_cost + output_cost) * variance

    # In real mode, SDK tracks costs automatically via LangChain interceptor
    return 0.0


# =============================================================================
# Optimized Function
# =============================================================================


@traigent.optimize(
    eval_dataset=str(
        (SCRIPT_DIR / ".." / ".." / "datasets" / "simple_questions.jsonl").resolve()
    ),
    objectives=["accuracy", "cost"],
    metric_functions={
        "accuracy": accuracy_metric,  # Custom accuracy metric
        "total_cost": cost_metric,  # Simulated cost for mock mode (overrides SDK's 0.0 default)
    },
    configuration_space={
        # Generator Agent parameters (namespaced)
        # Note: In real mode, we use gpt-4.1-nano for speed and low cost
        # These params demonstrate what Traigent CAN optimize
        "generate.model": [DEFAULT_MODEL],  # Fast and cheap model
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

    Cost tracking: The LangChain interceptor automatically captures all LLM calls
    made by the ChatOpenAI instances in grade_documents and generate_answer.
    The SDK's evaluation infrastructure extracts tokens and costs from these
    captured responses, so we just return the answer directly.

    Note: OpenTelemetry spans are automatically recorded for each agent execution
    and exported to the Traigent backend for parameter attribution analysis.
    """
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
    # All LangChain LLM calls are automatically tracked by the interceptor
    result = app.invoke(initial_state)

    # Return the answer - SDK will automatically extract cost from captured LangChain responses
    return result["answer"]


# =============================================================================
# Main
# =============================================================================


async def main() -> None:
    """Run the LangGraph multi-agent demo with Traigent optimization."""
    print("Traigent Advanced: LangGraph Multi-Agent RAG Workflow")
    print("=" * 50)

    # Check environment
    backend_url = os.environ.get("TRAIGENT_BACKEND_URL", "http://localhost:5000")
    api_key = os.environ.get("TRAIGENT_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    print(f"\nBackend URL: {backend_url}")
    print(f"Traigent API Key: {'set' if api_key else 'NOT SET'}")
    print(f"OpenAI API Key: {'set' if openai_key else 'NOT SET'}")
    print(
        f"Mode: {'MOCK (no real API costs)' if MOCK_MODE else f'REAL (using OpenAI {DEFAULT_MODEL})'}"
    )

    if not MOCK_MODE and not openai_key:
        print("\nERROR: OPENAI_API_KEY not set for real mode.")
        print("Either set OPENAI_API_KEY or run with TRAIGENT_MOCK_LLM=true")
        return

    if not api_key:
        print("\nERROR: TRAIGENT_API_KEY not set. Please set it and retry.")
        return

    # First, send the workflow graph for visualization
    print("\n" + "-" * 50)
    print("Step 1: Sending workflow graph topology to backend...")
    print("-" * 50)

    tracker = WorkflowTracesTracker(
        backend_url=backend_url,
        auth_token=api_key,
    )

    # We'll get the experiment_id after optimization starts
    # For now, use a placeholder that will be updated
    print("  Graph will be sent during optimization...")

    # Run optimization
    print("\n" + "-" * 50)
    print("Step 2: Running Traigent optimization...")
    print("-" * 50)
    print("\nWorkflow structure:")
    print("  START -> retrieve -> grade_documents --(relevant)--> generate -> END")
    print("                            |")
    print("                            +---(not relevant)--> web_search --+")
    print("                            ^                                   |")
    print("                            +-----------------------------------+")

    # Use fewer trials in real mode to save API costs
    max_trials = 4 if MOCK_MODE else 2
    print(f"\nOptimizing parameters ({max_trials} trials)...")

    # Set a generous timeout to allow all trials to complete
    # Each trial takes ~100s with 20 examples, so timeout should be at least max_trials * 120s
    timeout_seconds = max_trials * 120  # 240s for 2 trials in real mode
    print(
        f"  Timeout: {timeout_seconds}s (allowing ~{timeout_seconds // max_trials}s per trial)"
    )

    results = await run_rag_workflow.optimize(
        algorithm="grid",
        max_trials=max_trials,
        random_seed=42,
        timeout=timeout_seconds,  # Explicit timeout to allow all trials to complete
    )

    # Send the workflow graph to backend for visualization
    # Note: The experiment_id comes from the backend session
    print("\n" + "-" * 50)
    print("Step 3: Sending workflow graph topology...")
    print("-" * 50)

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
            print(
                f"  Nodes: {len(graph_payload.nodes)}, Edges: {len(graph_payload.edges)}"
            )
            if graph_payload.loops:
                print(f"  Detected loops: {len(graph_payload.loops)}")
        else:
            print(f"  Failed to send graph: {response.error}")

        # Note: Per-agent spans are automatically recorded via OpenTelemetry
        # and exported to the Traigent backend by the TraigentSpanExporter.
        # No manual span creation needed - OTEL handles it during workflow execution.
        print("\n" + "-" * 50)
        print("Note: Per-agent spans automatically recorded via OpenTelemetry")
        print("-" * 50)
        print("  Spans are created during workflow execution with:")
        print("  - node.id attribute for parameter attribution")
        print("  - llm.input_tokens, llm.output_tokens for cost tracking")
        print("  - configuration_run_id for linking to specific trials")
    else:
        print("\n  Note: Workflow graph visualization requires experiment_id.")
        print("  Per-agent spans are automatically recorded via OpenTelemetry.")

    # Display results
    print("\n" + "-" * 50)
    print("Optimization Results")
    print("-" * 50)

    print("\nBest Configuration Found:")
    print(f"  Model: {results.best_config.get('model')}")
    print(f"  Temperature: {results.best_config.get('temperature')}")

    print("\nPerformance:")
    print(f"  Accuracy: {results.best_metrics.get('accuracy', 0):.2%}")
    print(
        f"  Cost: ${results.best_metrics.get('total_cost', results.best_metrics.get('cost', 0)):.6f}"
    )

    print("\n" + "-" * 50)
    print("Check the Traigent frontend to see:")
    print("  1. Workflow graph visualization (nodes, edges, loops)")
    print("  2. Trial execution traces (spans for each trial)")
    print("  3. Agent performance breakdown")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        raise SystemExit(130)
