#!/usr/bin/env python3
"""Example: Unified Platform Campaign Copilot with LangGraph and Traigent

Designed for ad-tech teams that operate across audience discovery, planning,
activation, optimization, and measurement on a unified platform.

Why this example exists:
- Reframes the generic LangGraph demo into a cross-screen media workflow
- Uses language that fits DSP + SSP + data platform conversations
- Shows how Traigent can optimize a multi-agent workflow without changing
  the orchestration model

Run locally in mock mode:
    TRAIGENT_MOCK_LLM=true python walkthrough/mock/advanced/06_unified_platform_campaign_copilot.py

Optional workflow graph export:
    TRAIGENT_API_KEY=<your_key> TRAIGENT_MOCK_LLM=true \
      python walkthrough/mock/advanced/06_unified_platform_campaign_copilot.py
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import random
from pathlib import Path
from typing import TypedDict

try:
    from langgraph.graph import END, StateGraph
except ImportError as exc:  # pragma: no cover - dependency guidance
    raise SystemExit(
        "Missing dependency: langgraph. Install it with `pip install langgraph` "
        "to run this walkthrough."
    ) from exc

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional convenience import

    def load_dotenv() -> bool:
        return False


import traigent
from traigent import TraigentConfig
from traigent.config.backend_config import BackendConfig
from traigent.integrations.observability.workflow_traces import (
    WorkflowEdge,
    WorkflowGraphPayload,
    WorkflowNode,
    WorkflowTracesTracker,
    detect_loops_in_graph,
)

load_dotenv()
os.environ.setdefault("TRAIGENT_MOCK_LLM", "true")
traigent.initialize(
    config=TraigentConfig(execution_mode="edge_analytics", minimal_logging=True)
)

SCRIPT_DIR = Path(__file__).parent
DATASET_PATH = str(
    (SCRIPT_DIR / ".." / ".." / "datasets" / "unified_campaign_briefs.jsonl").resolve()
)

MOCK_MODE = os.environ.get("TRAIGENT_MOCK_LLM", "true").lower() == "true"

# Only display these metrics in output (filter internal SDK metrics)
DISPLAY_METRICS = {"business_fit", "operational_efficiency"}


class CampaignState(TypedDict):
    """State carried through the unified media workflow."""

    campaign_brief: str
    audience_segments: list[str]
    channel_mix: list[str]
    supply_strategy: str
    measurement_plan: str
    review_notes: list[str]
    recommendation: str
    revision_count: int


def _config_value(name: str, default: object) -> object:
    """Read a Traigent config value with a safe default."""
    config = traigent.get_config() or {}
    return config.get(name, default)


def discover_audiences(state: CampaignState) -> CampaignState:
    """Translate the campaign brief into high-value audience segments."""
    brief = state["campaign_brief"].lower()
    lookback_days = int(_config_value("discover.lookback_days", 30))
    segments: list[str] = []

    if "parent" in brief or "family" in brief or "back-to-school" in brief:
        segments.append("parents")
    if "ev" in brief or "electric vehicle" in brief or "automotive" in brief:
        segments.append("in_market_auto")
    if "sports" in brief or "playoff" in brief or "live event" in brief:
        segments.append("live_sports_fans")
    if "travel" in brief or "vacation" in brief:
        segments.append("frequent_travelers")
    if "qsr" in brief or "restaurant" in brief or "meal" in brief:
        segments.append("qsr_value_seekers")
    if "streaming" in brief or "video" in brief or "series" in brief:
        segments.append("premium_video_viewers")

    if _config_value("discover.use_identity_graph", True):
        segments.append("identity_graph")

    if not segments:
        segments.append("cross_screen_prospecting")

    if lookback_days <= 30:
        segments.append("high_intent_signals")
    else:
        segments.append("expanded_prospecting")

    return {**state, "audience_segments": segments}


def plan_channel_mix(state: CampaignState) -> CampaignState:
    """Build a channel plan from the brief and discovered audiences."""
    brief = state["campaign_brief"].lower()
    ctv_share = float(_config_value("plan.ctv_share", 0.6))
    channels: list[str] = []

    if ctv_share >= 0.5 or "ctv" in brief or "streaming" in brief:
        channels.append("ctv")
    if "premium_video_viewers" in state["audience_segments"] or "sports" in brief:
        channels.append("premium_video")
    else:
        channels.append("online_video")

    if ctv_share < 0.7:
        channels.append("display")

    if "native" in brief or "content" in brief:
        channels.append("native")

    return {**state, "channel_mix": channels}


def select_supply_strategy(state: CampaignState) -> CampaignState:
    """Choose how the plan accesses inventory across the media supply path."""
    preferred_path = str(_config_value("supply.path", "full_stack"))
    primary_kpi = str(_config_value("measure.primary_kpi", "completed_events"))

    # If review previously flagged the open exchange path, revise to a safer option.
    if state["revision_count"] > 0 and preferred_path == "open_exchange":
        preferred_path = "curated_marketplace"

    if preferred_path == "full_stack":
        supply_strategy = "full_stack"
    elif preferred_path == "curated_marketplace":
        supply_strategy = "curated_marketplace"
    else:
        supply_strategy = "open_exchange"

    if "identity_graph" in state["audience_segments"]:
        measurement_plan = f"{primary_kpi} + cross_screen_reach"
    else:
        measurement_plan = f"{primary_kpi} + channel_performance"

    return {
        **state,
        "supply_strategy": supply_strategy,
        "measurement_plan": measurement_plan,
    }


def review_activation_plan(state: CampaignState) -> CampaignState:
    """Apply an operational review before recommending activation."""
    brief = state["campaign_brief"].lower()
    strictness = str(_config_value("review.strictness", "balanced"))
    notes: list[str] = []

    if state["supply_strategy"] == "open_exchange" and (
        "ctv" in state["channel_mix"] or "premium" in brief or "sports" in brief
    ):
        notes.append("revise_supply_path")

    if strictness == "strict" and "identity_graph" not in state["audience_segments"]:
        notes.append("identity_signal_missing")

    revision_count = state["revision_count"] + (
        1 if "revise_supply_path" in notes else 0
    )

    return {
        **state,
        "review_notes": notes,
        "revision_count": revision_count,
    }


def decide_after_review(state: CampaignState) -> str:
    """Route the workflow after review."""
    if "revise_supply_path" in state["review_notes"] and state["revision_count"] <= 1:
        return "revise_supply"
    return "generate"


def generate_recommendation(state: CampaignState) -> CampaignState:
    """Compose the final recommendation in either ops or executive style."""
    style = str(_config_value("generate.style", "executive"))
    audience_text = ", ".join(state["audience_segments"])
    channel_text = ", ".join(state["channel_mix"])
    review_text = (
        ", ".join(state["review_notes"]) if state["review_notes"] else "approved"
    )

    if style == "ops":
        recommendation = (
            f"Audience Discovery: {audience_text}\n"
            f"Channel Plan: {channel_text}\n"
            f"Supply Strategy: {state['supply_strategy']}\n"
            f"Measurement: {state['measurement_plan']}\n"
            f"Review: {review_text}\n"
            "Activation Note: route discovery directly into activation and optimization."
        )
    else:
        recommendation = (
            f"Recommend a {state['supply_strategy']} activation anchored in {channel_text}. "
            f"Use {audience_text} for discovery, optimize toward {state['measurement_plan']}, "
            f"and carry review status as {review_text}."
        )

    return {**state, "recommendation": recommendation}


def build_campaign_workflow() -> StateGraph:
    """Build the LangGraph workflow for the campaign copilot."""
    workflow = StateGraph(CampaignState)
    workflow.add_node("discover_audiences", discover_audiences)
    workflow.add_node("plan_channel_mix", plan_channel_mix)
    workflow.add_node("select_supply_strategy", select_supply_strategy)
    workflow.add_node("review_activation_plan", review_activation_plan)
    workflow.add_node("generate_recommendation", generate_recommendation)

    workflow.set_entry_point("discover_audiences")
    workflow.add_edge("discover_audiences", "plan_channel_mix")
    workflow.add_edge("plan_channel_mix", "select_supply_strategy")
    workflow.add_edge("select_supply_strategy", "review_activation_plan")
    workflow.add_conditional_edges(
        "review_activation_plan",
        decide_after_review,
        {
            "revise_supply": "select_supply_strategy",
            "generate": "generate_recommendation",
        },
    )
    workflow.add_edge("generate_recommendation", END)
    return workflow


def extract_workflow_graph(
    experiment_id: str,
    experiment_run_id: str | None = None,
) -> WorkflowGraphPayload:
    """Build a graph payload for UI workflow visualization."""
    nodes = [
        WorkflowNode(
            id="__start__",
            type="entry",
            display_name="Start",
            metadata={"purpose": "Workflow entry point"},
        ),
        WorkflowNode(
            id="discover_audiences",
            type="agent",
            display_name="Audience Discovery",
            tunable_params=["discover.lookback_days", "discover.use_identity_graph"],
            metadata={"purpose": "Turn campaign briefs into audience segments"},
        ),
        WorkflowNode(
            id="plan_channel_mix",
            type="agent",
            display_name="Channel Planner",
            tunable_params=["plan.ctv_share"],
            metadata={"purpose": "Balance CTV, video, display, and native"},
        ),
        WorkflowNode(
            id="select_supply_strategy",
            type="agent",
            display_name="Supply Strategist",
            tunable_params=["supply.path", "measure.primary_kpi"],
            metadata={"purpose": "Map the plan onto DSP, SSP, and data choices"},
        ),
        WorkflowNode(
            id="review_activation_plan",
            type="agent",
            display_name="Activation Review",
            tunable_params=["review.strictness"],
            metadata={"purpose": "Check supply quality and activation readiness"},
        ),
        WorkflowNode(
            id="generate_recommendation",
            type="agent",
            display_name="Campaign Recommendation",
            tunable_params=["generate.style"],
            metadata={"purpose": "Summarize the plan for activation teams"},
        ),
        WorkflowNode(
            id="__end__",
            type="exit",
            display_name="End",
            metadata={"purpose": "Workflow exit point"},
        ),
    ]

    edges = [
        WorkflowEdge(from_node="__start__", to_node="discover_audiences"),
        WorkflowEdge(from_node="discover_audiences", to_node="plan_channel_mix"),
        WorkflowEdge(from_node="plan_channel_mix", to_node="select_supply_strategy"),
        WorkflowEdge(
            from_node="select_supply_strategy", to_node="review_activation_plan"
        ),
        WorkflowEdge(
            from_node="review_activation_plan",
            to_node="select_supply_strategy",
            edge_type="conditional",
            condition="revise_supply_path",
        ),
        WorkflowEdge(
            from_node="review_activation_plan",
            to_node="generate_recommendation",
            edge_type="conditional",
            condition="approved",
        ),
        WorkflowEdge(from_node="generate_recommendation", to_node="__end__"),
    ]

    return WorkflowGraphPayload(
        experiment_id=experiment_id,
        experiment_run_id=experiment_run_id,
        nodes=nodes,
        edges=edges,
        loops=detect_loops_in_graph(edges),
        metadata={
            "workflow_type": "unified_media_platform",
            "workflow_tags": [
                "audience_discovery",
                "planning",
                "activation",
                "optimization",
                "measurement",
            ],
        },
    )


def business_fit_metric(output: str, expected: str) -> float:
    """Score whether the recommendation covers the right business concepts."""
    output_lower = output.lower()
    expected_tokens = [
        token.strip() for token in expected.lower().split(",") if token.strip()
    ]

    if not expected_tokens:
        return 0.0

    matches = sum(1 for token in expected_tokens if token in output_lower)
    return matches / len(expected_tokens)


def operational_efficiency_metric(output: str, expected: str) -> float:
    """Simulate an efficiency score for unified planning workflows."""
    output_lower = output.lower()
    score = 0.15

    if "full_stack" in output_lower:
        score += 0.4
    elif "curated_marketplace" in output_lower:
        score += 0.3
    else:
        score += 0.15

    if "cross_screen_reach" in output_lower:
        score += 0.2
    if "identity_graph" in output_lower:
        score += 0.15
    if "high_intent_signals" in output_lower:
        score += 0.05
    if any(
        kpi in output_lower for kpi in ("completed_events", "viewability", "attention")
    ):
        score += 0.1
    if "revise_supply_path" in output_lower:
        score -= 0.15
    if "identity_signal_missing" in output_lower:
        score -= 0.1

    seed_input = f"{output}|{expected}".encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(seed_input).digest()[:4], "little")
    random.seed(seed)
    score += random.uniform(0.0, 0.05)
    return min(score, 1.0)


@traigent.optimize(
    eval_dataset=DATASET_PATH,
    objectives=["business_fit", "operational_efficiency"],
    metric_functions={
        "business_fit": business_fit_metric,
        "operational_efficiency": operational_efficiency_metric,
    },
    configuration_space={
        "discover.lookback_days": [30, 60],
        "discover.use_identity_graph": [True, False],
        "plan.ctv_share": [0.4, 0.7],
        "supply.path": ["full_stack", "curated_marketplace", "open_exchange"],
        "measure.primary_kpi": ["completed_events", "viewability", "attention"],
        "review.strictness": ["balanced", "strict"],
        "generate.style": ["executive", "ops"],
    },
    execution_mode="edge_analytics",
)
def run_campaign_copilot(campaign_brief: str) -> str:
    """Run the optimized LangGraph workflow for a campaign brief."""
    workflow = build_campaign_workflow()
    app = workflow.compile()

    initial_state: CampaignState = {
        "campaign_brief": campaign_brief,
        "audience_segments": [],
        "channel_mix": [],
        "supply_strategy": "",
        "measurement_plan": "",
        "review_notes": [],
        "recommendation": "",
        "revision_count": 0,
    }
    result = app.invoke(initial_state)
    return result["recommendation"]


async def main() -> None:
    """Run the unified platform campaign copilot example."""
    print("Traigent Advanced: Unified Platform Campaign Copilot")
    print("=" * 50)
    print("Domain: audience discovery, cross-screen planning, supply strategy,")
    print("activation review, and measurement on a unified media platform.")
    print()

    api_key = os.environ.get("TRAIGENT_API_KEY")
    backend_url = (
        os.environ.get("TRAIGENT_BACKEND_URL")
        or BackendConfig.get_cloud_backend_url()
    )
    offline_mode = os.environ.get("TRAIGENT_OFFLINE_MODE", "").lower() == "true"

    print(f"Dataset: {DATASET_PATH}")
    print(f"Mock mode: {MOCK_MODE}")
    print(
        "Workflow graph export: "
        f"{'enabled' if api_key and not offline_mode else 'disabled'}"
    )
    print()

    max_trials = 6 if MOCK_MODE else 3
    print(f"Running optimization with {max_trials} trials...")
    results = await run_campaign_copilot.optimize(
        algorithm="random",
        max_trials=max_trials,
        random_seed=42,
    )

    tracker: WorkflowTracesTracker | None = None
    if api_key and not offline_mode:
        tracker = WorkflowTracesTracker(
            backend_url=backend_url,
            auth_token=api_key,
        )

    experiment_id = None
    experiment_run_id = None
    if results.metadata:
        experiment_id = results.metadata.get("experiment_id")
        experiment_run_id = results.metadata.get("experiment_run_id")

    if tracker and experiment_id:
        graph_payload = extract_workflow_graph(experiment_id, experiment_run_id)
        response = tracker.client.ingest_traces(graph=graph_payload)
        if response.success:
            print("Workflow graph exported to backend.")
        else:
            print(f"Workflow graph export failed: {response.error}")

    print("\nBest Configuration Found:")
    for key, value in results.best_config.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    print("\nPerformance:")
    for key, value in results.best_metrics.items():
        if key not in DISPLAY_METRICS:
            continue
        if isinstance(value, float):
            print(f"  {key}: {value:.2%}")
        else:
            print(f"  {key}: {value}")

    print("\nSuggested client framing:")
    print("  - Discovery agent turns briefs into audience strategy")
    print("  - Supply agent chooses full-stack vs curated access")
    print("  - Review loop adds operational control before activation")
    print("  - Measurement stays attached to planning decisions")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        raise SystemExit(130)
