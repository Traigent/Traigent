"""Command-line interface for TraiGent SDK.

Note:
    This module is designed for interactive terminal use only. It uses
    rich.console for formatted terminal output rather than structured logging.
    For programmatic access to TraiGent functionality, use the traigent.api
    module instead, which provides proper logging integration.
"""

# Traceability: CONC-Layer-API CONC-Quality-Usability CONC-Quality-Maintainability FUNC-API-ENTRY REQ-API-001 SYNC-OptimizationFlow

from __future__ import annotations

from traigent.cli.main import cli

__all__ = ["cli"]
