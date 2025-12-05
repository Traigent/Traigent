"""Evaluation strategies for TraiGent SDK."""

# Traceability: CONC-Layer-Core CONC-Quality-Reliability FUNC-EVAL-METRICS REQ-EVAL-005 SYNC-OptimizationFlow

from __future__ import annotations

from traigent.evaluators.base import BaseEvaluator, Dataset, SimpleScoringEvaluator
from traigent.evaluators.local import LocalEvaluator

__all__ = [
    "BaseEvaluator",
    "Dataset",
    "LocalEvaluator",
    "SimpleScoringEvaluator",
]
