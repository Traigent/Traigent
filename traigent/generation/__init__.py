"""Client-side guided generation.

Privacy-preserving prompt rewrite and benchmark example synthesis, driven by an
opaque backend GuidancePlan. All generation runs on the USER's own LLM; prompt
text and example content never leave the client. These engines are network-free
— they call only the user-supplied RewriteLLM and never touch Traigent's cloud
client or credentials.
"""

from __future__ import annotations

from .example_synth import ExampleSynthesizer
from .llm_provider import (
    CallbackRewriteLLM,
    GenerationProviderError,
    RewriteLLM,
    resolve_rewrite_llm,
)
from .loop import (
    GuidanceLoop,
    GuidanceLoopResult,
    GuidancePlanProvider,
    GuidanceRoundResult,
)
from .models import (
    CoarsePriority,
    GuidanceAction,
    GuidancePlan,
    GuidancePlanItem,
    GuidancePlanRequest,
    GuidanceResultItem,
    GuidanceResultSubmission,
    PlanKind,
)
from .options import DatasetGrowthOptions, PromptRewriteOptions
from .prompt_rewriter import PromptRewriter, merge_prompt_candidates

__all__ = [
    # models
    "GuidanceAction",
    "CoarsePriority",
    "PlanKind",
    "GuidancePlanItem",
    "GuidancePlan",
    "GuidancePlanRequest",
    "GuidanceResultItem",
    "GuidanceResultSubmission",
    # provider
    "RewriteLLM",
    "CallbackRewriteLLM",
    "GenerationProviderError",
    "resolve_rewrite_llm",
    # engines
    "PromptRewriter",
    "merge_prompt_candidates",
    "ExampleSynthesizer",
    # loop
    "GuidanceLoop",
    "GuidanceLoopResult",
    "GuidanceRoundResult",
    "GuidancePlanProvider",
    # options
    "PromptRewriteOptions",
    "DatasetGrowthOptions",
]
