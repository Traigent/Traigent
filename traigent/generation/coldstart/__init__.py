"""Opaque cold-start client boundary."""

from .executor import build_cold_start_eval_set
from .models import ColdStartResult, DiscoveryGap, Receipt

__all__ = [
    "ColdStartResult",
    "DiscoveryGap",
    "Receipt",
    "build_cold_start_eval_set",
]
