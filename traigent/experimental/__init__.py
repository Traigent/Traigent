"""Experimental Traigent Features - NOT FOR PRODUCTION USE.

⚠️  WARNING: This module contains experimental features that are NOT part
    of the production Traigent SDK.

This module is for:
- Local development and testing
- Prototyping new features
- Educational examples
- Experiments while OptiGen backend is under development

DO NOT use these features in production code.
The real Traigent cloud implementation is proprietary IP in the OptiGen backend.
"""

# Traceability: CONC-Layer-Core CONC-Quality-Reliability FUNC-CLOUD-HYBRID REQ-CLOUD-009 SYNC-CloudHybrid

from __future__ import annotations

import warnings

# Issue warning when experimental module is imported
warnings.warn(
    "You are importing experimental Traigent features. "
    "These are NOT for production use and may change or be removed. "
    "Use only for local development/testing.",
    UserWarning,
    stacklevel=2,
)

# Make it clear this is experimental
__experimental__ = True
__version__ = "experimental"
__status__ = "Not for production use"
