"""TraiGent Cloud Service integration module.

This module provides the commercial cloud service integration for TraiGent SDK,
enabling smart optimization with dataset subset selection and cost reduction.
"""

# Traceability: CONC-Layer-Infra CONC-Quality-Reliability FUNC-CLOUD-HYBRID REQ-CLOUD-009 SYNC-CloudHybrid

from __future__ import annotations

from .auth import APIKey, AuthManager
from .billing import BillingManager, UsageTracker
from .client import TraiGentCloudClient
from .service import TraiGentCloudService
from .subset_selection import (
    DiverseSampling,
    HighConfidenceSampling,
    RepresentativeSampling,
    SmartSubsetSelector,
)

__all__ = [
    "TraiGentCloudClient",
    "TraiGentCloudService",
    "AuthManager",
    "APIKey",
    "BillingManager",
    "UsageTracker",
    "SmartSubsetSelector",
    "DiverseSampling",
    "RepresentativeSampling",
    "HighConfidenceSampling",
]
