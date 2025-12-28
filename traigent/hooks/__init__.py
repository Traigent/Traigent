"""Traigent Hooks - Agent configuration validation for Git workflows.

This module provides Git hooks that validate agent configurations
against constraints before allowing pushes to production.
"""

# Traceability: CONC-Layer-API CONC-Quality-Reliability CONC-Quality-Usability FUNC-API-ENTRY REQ-API-001

from __future__ import annotations

from traigent.hooks.config import HooksConfig, load_hooks_config
from traigent.hooks.installer import HooksInstaller
from traigent.hooks.validator import AgentValidator, ValidationResult

__all__ = [
    "HooksConfig",
    "load_hooks_config",
    "HooksInstaller",
    "AgentValidator",
    "ValidationResult",
]
