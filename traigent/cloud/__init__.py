"""Traigent backend and reserved cloud integration module.

This module provides backend integration for portal-tracked SDK runs. Remote
cloud execution is reserved for a future release and fails closed today.
"""

# Traceability: CONC-Layer-Infra CONC-Quality-Reliability FUNC-CLOUD-HYBRID REQ-CLOUD-009 SYNC-CloudHybrid

from __future__ import annotations

from .auth import APIKey, AuthManager
from .billing import BillingManager, UsageTracker
from .client import TraigentCloudClient
from .service import TraigentCloudService
from .subset_selection import (
    DiverseSampling,
    HighConfidenceSampling,
    RepresentativeSampling,
    SmartSubsetSelector,
)

__all__ = [
    "TraigentCloudClient",
    "TraigentCloudService",
    "AuthManager",
    "APIKey",
    "BillingManager",
    "UsageTracker",
    "SmartSubsetSelector",
    "DiverseSampling",
    "RepresentativeSampling",
    "HighConfidenceSampling",
]
