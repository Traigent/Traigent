#!/usr/bin/env python3
"""LangGraph Document Routing Demo with Traigent Optimization.

This example demonstrates multi-agent optimization for document processing
using LangGraph's StateGraph with conditional routing:
- Router node: Classifies documents as legal or financial
- Legal branch: Analyzes legal documents (contracts, agreements, etc.)
- Financial branch: Analyzes financial documents (invoices, statements, etc.)

Each agent has independently optimizable model, temperature, and prompt settings.

Usage:
    # Mock mode (no API costs)
    TRAIGENT_MOCK_LLM=true python run.py

    # Real LLM mode
    OPENAI_API_KEY=sk-... python run.py
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
from pathlib import Path
from typing import Any, Literal, TypedDict

# Mock mode detection - must be before imports that depend on it
MOCK = str(os.getenv("TRAIGENT_MOCK_LLM", "")).lower() in {"1", "true", "yes", "y"}
BASE = Path(__file__).parent


def _setup_mock_environment() -> None:
    """Set up mock environment for testing without API calls."""
    os.environ["HOME"] = str(BASE)
    results_dir = BASE / ".traigent_local"
    results_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TRAIGENT_RESULTS_FOLDER"] = str(results_dir)


if MOCK:
    _setup_mock_environment()

# Import LangChain components
from langchain_core.messages import HumanMessage

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None  # type: ignore

# Import LangGraph
try:
    from langgraph.graph import END, START, StateGraph
except ImportError:
    StateGraph = None  # type: ignore
    START = None
    END = None

try:
    import traigent
except ImportError:  # pragma: no cover - support IDE execution paths
    import importlib

    module_path = Path(__file__).resolve()
    for depth in (2, 3):
        try:
            sys.path.append(str(module_path.parents[depth]))
        except IndexError:
            continue
    traigent = importlib.import_module("traigent")

from evaluators import (
    processing_quality_scorer,
    routing_accuracy_scorer,
)

# Import prompts
from prompts import FINANCIAL_PROMPTS, LEGAL_PROMPTS, ROUTER_PROMPTS

from traigent.api.types import OptimizationResult

# Paths
DATA_ROOT = Path(__file__).resolve().parents[2] / "datasets" / "langgraph-routing"
DATASET = str(DATA_ROOT / "evaluation_set.jsonl")

# Initialize traigent in edge_analytics mode for mock
if MOCK:
    try:
        traigent.initialize(execution_mode="edge_analytics")
    except Exception:
        pass


# ==============================================================================
# State Schema for LangGraph
# ==============================================================================


class GraphState(TypedDict, total=False):
    """State passed through the document processing graph."""

    # Input
    document: str

    # Configuration (passed through state for node access)
    router_model: str
    router_temperature: float
    router_prompt_style: str
    legal_agent_model: str
    legal_agent_temperature: float
    legal_agent_prompt_template: str
    financial_agent_model: str
    financial_agent_temperature: float
    financial_agent_prompt_template: str

    # Routing results
    doc_type: Literal["legal", "financial"]
    routing_confidence: float

    # Analysis results
    analysis_content: str

    # Final output
    result: dict[str, Any]


# ==============================================================================
# Mock Implementations
# ==============================================================================


def _mock_classify(document: str, prompt_style: str) -> tuple[str, float]:
    """Mock classification using keyword matching."""
    doc_lower = document.lower()

    legal_keywords = [
        "agreement",
        "contract",
        "hereby",
        "whereas",
        "parties",
        "clause",
        "liability",
        "indemnify",
        "jurisdiction",
        "termination",
        "confidential",
        "obligations",
        "governing law",
        "witness whereof",
    ]
    financial_keywords = [
        "invoice",
        "payment",
        "amount",
        "balance",
        "revenue",
        "expense",
        "profit",
        "loss",
        "quarter",
        "fiscal",
        "budget",
        "transaction",
        "account",
        "total due",
        "subtotal",
        "tax",
    ]

    legal_score = sum(1 for kw in legal_keywords if kw in doc_lower)
    financial_score = sum(1 for kw in financial_keywords if kw in doc_lower)

    # Add some variation based on prompt style
    style_bonus = {
        "concise": 0,
        "detailed": 0.02,
        "few_shot": 0.03,
        "chain_of_thought": 0.05,
    }
    bonus = style_bonus.get(prompt_style, 0)

    # Use score difference to determine classification (handles ties better)
    if legal_score > financial_score:
        confidence = min(0.95, 0.6 + legal_score * 0.05 + bonus)
        return "legal", confidence
    elif financial_score > legal_score:
        confidence = min(0.95, 0.6 + financial_score * 0.05 + bonus)
        return "financial", confidence
    else:
        # Tie: default to financial if any financial keywords, else legal
        if financial_score > 0:
            return "financial", 0.5
        return "legal", 0.5


def _mock_legal_analysis(document: str, prompt_template: str) -> str:
    """Mock legal document analysis."""
    doc_lower = document.lower()

    # Extract key elements
    parties = []
    if "acme" in doc_lower:
        parties.append("Acme Corporation")
    if "beta" in doc_lower:
        parties.append("Beta Industries")
    if "techstart" in doc_lower:
        parties.append("TechStart Inc.")
    if not parties:
        parties = ["Party A", "Party B"]

    # Find dates
    date_match = re.search(
        r"(january|february|march|april|may|june|july|august|september|"
        r"october|november|december)\s+\d{1,2},?\s+\d{4}",
        doc_lower,
    )
    effective_date = date_match.group(0).title() if date_match else "Not specified"

    # Determine document type
    doc_type = "Agreement"
    if "nda" in doc_lower or "non-disclosure" in doc_lower:
        doc_type = "Non-Disclosure Agreement"
    elif "employment" in doc_lower:
        doc_type = "Employment Agreement"
    elif "license" in doc_lower:
        doc_type = "License Agreement"
    elif "termination" in doc_lower:
        doc_type = "Notice of Termination"
    elif "settlement" in doc_lower:
        doc_type = "Settlement Agreement"

    # Build response based on template
    if prompt_template == "structured_extraction":
        return f"""DOCUMENT_TYPE: {doc_type}
EFFECTIVE_DATE: {effective_date}
PARTIES:
- Party 1: {parties[0]}
- Party 2: {parties[1] if len(parties) > 1 else "Not specified"}
TERM/DURATION: As specified in document
KEY_CLAUSES:
- Confidentiality: Information protection provisions
- Termination: Standard termination terms
OBLIGATIONS:
- Each party has defined responsibilities"""
    elif prompt_template == "risk_focused":
        return f"""RISK ANALYSIS for {doc_type}

1. LIABILITY EXPOSURE
   - Standard liability provisions identified
   - Caps may apply per document terms

2. TERMINATION RISKS
   - Notice period required
   - Obligations upon termination

3. COMPLIANCE CONCERNS
   - Standard regulatory requirements apply

RISK LEVEL: Medium
KEY RECOMMENDATIONS:
- Review termination clauses
- Verify compliance requirements"""
    else:  # standard
        return f"""Document Analysis:

1. Document Type: {doc_type}
2. Parties: {', '.join(parties)}
3. Effective Date: {effective_date}
4. Key Terms: Standard contractual provisions apply
5. Obligations: Each party has defined responsibilities
6. Notable Clauses: Confidentiality, termination, and governing law provisions"""


def _mock_financial_analysis(document: str, prompt_template: str) -> str:
    """Mock financial document analysis."""
    doc_lower = document.lower()

    # Extract amounts
    amounts = re.findall(r"\$[\d,]+(?:\.\d{2})?", document)
    total_amount = amounts[-1] if amounts else "$0.00"

    # Determine document type
    doc_type = "Financial Document"
    if "invoice" in doc_lower:
        doc_type = "Invoice"
    elif "statement" in doc_lower:
        doc_type = "Bank Statement"
    elif "purchase order" in doc_lower or "po-" in doc_lower:
        doc_type = "Purchase Order"
    elif "expense" in doc_lower:
        doc_type = "Expense Report"
    elif "budget" in doc_lower:
        doc_type = "Budget Proposal"
    elif (
        "quarterly" in doc_lower
        or "q1" in doc_lower
        or "q2" in doc_lower
        or "q3" in doc_lower
        or "q4" in doc_lower
    ):
        doc_type = "Quarterly Report"
    elif "profit" in doc_lower and "loss" in doc_lower:
        doc_type = "Profit & Loss Statement"

    # Build response based on template
    if prompt_template == "detailed_breakdown":
        return f"""DETAILED BREAKDOWN:

1. DOCUMENT IDENTIFICATION
   - Type: {doc_type}
   - Reference: As stated in document

2. FINANCIAL SUMMARY
   | Category | Amount |
   |----------|--------|
   | Total    | {total_amount} |

3. LINE ITEM DETAILS
   Multiple items identified in document

4. PAYMENT INFORMATION
   - Terms: As specified
   - Total Due: {total_amount}"""
    elif prompt_template == "metric_extraction":
        return f"""EXTRACTED METRICS:

IDENTIFICATION:
- Document Type: {doc_type}
- Date: As stated

PRIMARY AMOUNTS:
- Total: {total_amount}
- Line Items: Multiple

TRANSACTION COUNT: {len(amounts)} amounts identified"""
    else:  # standard or summary_focused
        return f"""Financial Analysis:

1. Document Type: {doc_type}
2. Total Amount: {total_amount}
3. Key Items: {len(amounts)} monetary values identified
4. Period: As specified in document
5. Status: Review complete"""


# ==============================================================================
# LangGraph Node Functions
# ==============================================================================


def router_node(state: GraphState) -> GraphState:
    """Router node: Classify the document type.

    This node analyzes the document and determines whether it should be
    processed by the legal or financial analysis branch.
    """
    document = state["document"]
    router_model = state.get("router_model", "gpt-4o-mini")
    router_temperature = state.get("router_temperature", 0.0)
    router_prompt_style = state.get("router_prompt_style", "concise")

    if MOCK:
        doc_type, confidence = _mock_classify(document, router_prompt_style)
    else:
        if ChatOpenAI is None:
            raise ImportError("langchain-openai is required for real LLM mode")

        prompt_template = ROUTER_PROMPTS.get(
            router_prompt_style, ROUTER_PROMPTS["concise"]
        )
        prompt = prompt_template.format(document=document[:3000])

        llm = ChatOpenAI(model=router_model, temperature=router_temperature)
        response = llm.invoke([HumanMessage(content=prompt)])
        content = str(response.content).lower().strip()

        # Parse classification - check financial first to avoid "legal" substring issue
        if "financial" in content and "legal" not in content:
            doc_type, confidence = "financial", 0.9
        elif "legal" in content and "financial" not in content:
            doc_type, confidence = "legal", 0.9
        elif "financial" in content:
            # Both present, check which comes first or use keyword density
            financial_pos = content.find("financial")
            legal_pos = content.find("legal")
            if financial_pos < legal_pos:
                doc_type, confidence = "financial", 0.7
            else:
                doc_type, confidence = "legal", 0.7
        elif "legal" in content:
            doc_type, confidence = "legal", 0.9
        else:
            doc_type, confidence = "legal", 0.5

    return {
        **state,
        "doc_type": doc_type,
        "routing_confidence": confidence,
    }


def legal_analyzer_node(state: GraphState) -> GraphState:
    """Legal analyzer node: Process legal documents.

    Extracts parties, clauses, dates, obligations and other legal elements.
    """
    document = state["document"]
    legal_agent_model = state.get("legal_agent_model", "gpt-4o")
    legal_agent_temperature = state.get("legal_agent_temperature", 0.3)
    legal_agent_prompt_template = state.get("legal_agent_prompt_template", "standard")

    if MOCK:
        content = _mock_legal_analysis(document, legal_agent_prompt_template)
    else:
        if ChatOpenAI is None:
            raise ImportError("langchain-openai is required for real LLM mode")

        prompt_template = LEGAL_PROMPTS.get(
            legal_agent_prompt_template, LEGAL_PROMPTS["standard"]
        )
        prompt = prompt_template.format(document=document[:4000])

        llm = ChatOpenAI(model=legal_agent_model, temperature=legal_agent_temperature)
        response = llm.invoke([HumanMessage(content=prompt)])
        content = str(response.content).strip()

    return {
        **state,
        "analysis_content": content,
    }


def financial_analyzer_node(state: GraphState) -> GraphState:
    """Financial analyzer node: Process financial documents.

    Extracts amounts, accounts, transactions, and other financial metrics.
    """
    document = state["document"]
    financial_agent_model = state.get("financial_agent_model", "gpt-4o")
    financial_agent_temperature = state.get("financial_agent_temperature", 0.3)
    financial_agent_prompt_template = state.get(
        "financial_agent_prompt_template", "standard"
    )

    if MOCK:
        content = _mock_financial_analysis(document, financial_agent_prompt_template)
    else:
        if ChatOpenAI is None:
            raise ImportError("langchain-openai is required for real LLM mode")

        prompt_template = FINANCIAL_PROMPTS.get(
            financial_agent_prompt_template, FINANCIAL_PROMPTS["standard"]
        )
        prompt = prompt_template.format(document=document[:4000])

        llm = ChatOpenAI(
            model=financial_agent_model, temperature=financial_agent_temperature
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        content = str(response.content).strip()

    return {
        **state,
        "analysis_content": content,
    }


def output_node(state: GraphState) -> GraphState:
    """Output node: Compile final results.

    Merges routing information and analysis content into structured output.
    """
    return {
        **state,
        "result": {
            "routing": {
                "type": state.get("doc_type", "unknown"),
                "confidence": state.get("routing_confidence", 0.0),
                "strategy": state.get("router_prompt_style", "concise"),
            },
            "content": state.get("analysis_content", ""),
            "config": {
                "router": {
                    "model": state.get("router_model", "gpt-4o-mini"),
                    "temperature": state.get("router_temperature", 0.0),
                    "prompt_style": state.get("router_prompt_style", "concise"),
                },
                "analyzer": {
                    "model": (
                        state.get("legal_agent_model", "gpt-4o")
                        if state.get("doc_type") == "legal"
                        else state.get("financial_agent_model", "gpt-4o")
                    ),
                    "temperature": (
                        state.get("legal_agent_temperature", 0.3)
                        if state.get("doc_type") == "legal"
                        else state.get("financial_agent_temperature", 0.3)
                    ),
                    "template": (
                        state.get("legal_agent_prompt_template", "standard")
                        if state.get("doc_type") == "legal"
                        else state.get("financial_agent_prompt_template", "standard")
                    ),
                },
            },
        },
    }


def route_by_doc_type(state: GraphState) -> str:
    """Conditional edge function: Route based on document type."""
    return state.get("doc_type", "legal")


def build_document_graph() -> Any:
    """Build the LangGraph document processing graph.

    Graph structure:
        START -> router -> [legal_analyzer | financial_analyzer] -> output -> END

    Returns:
        Compiled LangGraph StateGraph
    """
    if StateGraph is None:
        raise ImportError("langgraph is required. Install with: pip install langgraph")

    # Create the graph
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("router", router_node)
    workflow.add_node("legal_analyzer", legal_analyzer_node)
    workflow.add_node("financial_analyzer", financial_analyzer_node)
    workflow.add_node("output", output_node)

    # Add edges
    workflow.add_edge(START, "router")

    # Conditional routing based on document type
    workflow.add_conditional_edges(
        "router",
        route_by_doc_type,
        {
            "legal": "legal_analyzer",
            "financial": "financial_analyzer",
        },
    )

    # Both analyzers lead to output
    workflow.add_edge("legal_analyzer", "output")
    workflow.add_edge("financial_analyzer", "output")

    # Output leads to end
    workflow.add_edge("output", END)

    return workflow.compile()


# ==============================================================================
# Configuration Space
# ==============================================================================

CONFIGURATION_SPACE = {
    # Router configuration
    "router_model": ["gpt-4o-mini", "gpt-4o"],
    "router_temperature": [0.0, 0.1],
    "router_prompt_style": ["concise", "detailed", "few_shot", "chain_of_thought"],
    # Legal agent configuration
    "legal_agent_model": ["gpt-4o", "gpt-4o-mini"],
    "legal_agent_temperature": [0.1, 0.3, 0.5],
    "legal_agent_prompt_template": [
        "standard",
        "structured_extraction",
        "risk_focused",
    ],
    # Financial agent configuration
    "financial_agent_model": ["gpt-4o", "gpt-4o-mini"],
    "financial_agent_temperature": [0.1, 0.3, 0.5],
    "financial_agent_prompt_template": [
        "standard",
        "detailed_breakdown",
        "metric_extraction",
    ],
}


# ==============================================================================
# Main Pipeline with Traigent Optimization
# ==============================================================================

# Build the graph once (reused across invocations)
_GRAPH = None


def get_graph() -> Any:
    """Get or create the document processing graph."""
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_document_graph()
    return _GRAPH


@traigent.optimize(
    eval_dataset=DATASET,
    objectives=["routing_accuracy", "processing_quality"],
    configuration_space=CONFIGURATION_SPACE,
    metric_functions={
        "routing_accuracy": routing_accuracy_scorer,
        "processing_quality": processing_quality_scorer,
        "accuracy": routing_accuracy_scorer,  # Map to standard accuracy field
    },
    execution_mode="edge_analytics",
    injection_mode="seamless",
)
def process_document(
    document: str,
    # Router config
    router_model: str = "gpt-4o-mini",
    router_temperature: float = 0.0,
    router_prompt_style: str = "concise",
    # Legal agent config
    legal_agent_model: str = "gpt-4o",
    legal_agent_temperature: float = 0.3,
    legal_agent_prompt_template: str = "standard",
    # Financial agent config
    financial_agent_model: str = "gpt-4o",
    financial_agent_temperature: float = 0.3,
    financial_agent_prompt_template: str = "standard",
) -> dict[str, Any]:
    """Process a document through the LangGraph routing pipeline.

    This function invokes the LangGraph StateGraph which:
    1. Routes the document through the router node
    2. Conditionally executes legal_analyzer or financial_analyzer
    3. Compiles results in the output node

    Args:
        document: The document text to process
        router_model: Model for classification
        router_temperature: Temperature for router
        router_prompt_style: Prompt style for router
        legal_agent_model: Model for legal analysis
        legal_agent_temperature: Temperature for legal analysis
        legal_agent_prompt_template: Prompt template for legal analysis
        financial_agent_model: Model for financial analysis
        financial_agent_temperature: Temperature for financial analysis
        financial_agent_prompt_template: Prompt template for financial analysis

    Returns:
        Dictionary with routing info and analysis content
    """
    # Build initial state with document and configuration
    initial_state: GraphState = {
        "document": document,
        "router_model": router_model,
        "router_temperature": router_temperature,
        "router_prompt_style": router_prompt_style,
        "legal_agent_model": legal_agent_model,
        "legal_agent_temperature": legal_agent_temperature,
        "legal_agent_prompt_template": legal_agent_prompt_template,
        "financial_agent_model": financial_agent_model,
        "financial_agent_temperature": financial_agent_temperature,
        "financial_agent_prompt_template": financial_agent_prompt_template,
    }

    # Invoke the LangGraph
    graph = get_graph()
    final_state = graph.invoke(initial_state)

    # Extract and return result
    return final_state.get(
        "result",
        {
            "routing": {"type": "unknown", "confidence": 0.0},
            "content": "",
        },
    )


# ==============================================================================
# Results Printing
# ==============================================================================


def _print_results(result: OptimizationResult) -> None:
    """Pretty-print optimization results."""
    print("\n" + "=" * 70)
    print("OPTIMIZATION RESULTS")
    print("=" * 70)

    print(f"\nBest Score: {result.best_score:.4f}")
    print(f"Total Trials: {len(result.trials)}")

    best = result.best_config
    print("\nBest Configuration by Agent:")
    print("-" * 50)
    print("ROUTER:")
    print(f"  Model: {best.get('router_model', 'N/A')}")
    print(f"  Temperature: {best.get('router_temperature', 'N/A')}")
    print(f"  Prompt Style: {best.get('router_prompt_style', 'N/A')}")
    print("\nLEGAL ANALYZER:")
    print(f"  Model: {best.get('legal_agent_model', 'N/A')}")
    print(f"  Temperature: {best.get('legal_agent_temperature', 'N/A')}")
    print(f"  Template: {best.get('legal_agent_prompt_template', 'N/A')}")
    print("\nFINANCIAL ANALYZER:")
    print(f"  Model: {best.get('financial_agent_model', 'N/A')}")
    print(f"  Temperature: {best.get('financial_agent_temperature', 'N/A')}")
    print(f"  Template: {best.get('financial_agent_prompt_template', 'N/A')}")

    # Show aggregated results
    primary = result.objectives[0] if result.objectives else None
    df = result.to_aggregated_dataframe(primary_objective=primary)

    # Select relevant columns
    preferred_cols = [
        "router_prompt_style",
        "legal_agent_prompt_template",
        "financial_agent_prompt_template",
        "routing_accuracy",
        "processing_quality",
        "samples_count",
    ]
    cols = [c for c in preferred_cols if c in df.columns]
    if cols:
        df = df[cols]

    if primary and primary in df.columns:
        df = df.sort_values(by=primary, ascending=False, na_position="last")

    if not df.empty:
        print("\n" + "-" * 50)
        print("Aggregated Performance by Configuration:")
        print("-" * 50)
        print(df.head(10).to_string(index=False))

    print("\n" + "=" * 70)


# ==============================================================================
# Main Entry Point
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("LangGraph Document Routing Demo - Traigent Optimization")
    print("=" * 70)
    print()

    # Handle mock mode detection with proper fallback
    run_mock = MOCK
    if not run_mock:
        if not os.getenv("OPENAI_API_KEY"):
            print("WARNING: OPENAI_API_KEY not set. Falling back to mock mode...")
            run_mock = True
            _setup_mock_environment()

    if run_mock:
        print("Running in MOCK mode (no API calls)")
    else:
        print("Running in REAL LLM mode")

    print()
    print("Configuration Space:")
    print(f"  Router models: {CONFIGURATION_SPACE['router_model']}")
    print(f"  Router prompt styles: {CONFIGURATION_SPACE['router_prompt_style']}")
    print(f"  Legal templates: {CONFIGURATION_SPACE['legal_agent_prompt_template']}")
    print(
        f"  Financial templates: {CONFIGURATION_SPACE['financial_agent_prompt_template']}"
    )
    print()
    print("Objectives: routing_accuracy, processing_quality")
    print()
    print("Starting optimization...")
    print()

    async def main() -> None:
        # Use fewer trials in mock mode
        max_trials = 6 if run_mock else 20
        result = await process_document.optimize(
            algorithm="random",
            max_trials=max_trials,
        )
        _print_results(result)

    asyncio.run(main())
