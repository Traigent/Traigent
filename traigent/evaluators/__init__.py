"""Evaluation strategies for Traigent SDK."""

# Traceability: CONC-Layer-Core CONC-Quality-Reliability FUNC-EVAL-METRICS REQ-EVAL-005 SYNC-OptimizationFlow HYBRID-MODE-OPTIMIZATION

from __future__ import annotations

from traigent.evaluators.base import BaseEvaluator, Dataset, SimpleScoringEvaluator
from traigent.evaluators.hybrid_api import HybridAPIEvaluator
from traigent.evaluators.local import LocalEvaluator
from traigent.evaluators.recommendations import (
    EVAL_RECOMMENDATION_CAVEAT,
    list_eval_recommendation_task_types,
    recommend_evaluator,
    recommend_metrics,
)

__all__ = [
    "BaseEvaluator",
    "Dataset",
    "EVAL_RECOMMENDATION_CAVEAT",
    "HybridAPIEvaluator",
    "LocalEvaluator",
    "SimpleScoringEvaluator",
    "list_eval_recommendation_task_types",
    "recommend_evaluator",
    "recommend_metrics",
]
