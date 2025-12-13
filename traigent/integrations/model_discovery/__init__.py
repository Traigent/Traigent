"""Model discovery module for dynamic LLM model validation.

This module provides SDK-based model discovery with config file fallback
and pattern-based validation for LLM providers.
"""

# Traceability: CONC-Layer-Integration FUNC-INTEGRATIONS REQ-INT-008

from traigent.integrations.model_discovery.base import ModelDiscovery
from traigent.integrations.model_discovery.cache import CacheEntry, ModelCache
from traigent.integrations.model_discovery.registry import (
    get_model_discovery,
    register_discovery,
)

__all__ = [
    "ModelDiscovery",
    "ModelCache",
    "CacheEntry",
    "get_model_discovery",
    "register_discovery",
]
