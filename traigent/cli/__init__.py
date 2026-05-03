"""Command-line interface for Traigent SDK.

Note:
    This module is designed for interactive terminal use only. It uses
    rich.console for formatted terminal output rather than structured logging.
    For programmatic access to Traigent functionality, use the traigent.api
    module instead, which provides proper logging integration.
"""

# Traceability: CONC-Layer-API CONC-Quality-Usability CONC-Quality-Maintainability FUNC-API-ENTRY REQ-API-001 SYNC-OptimizationFlow

from __future__ import annotations

from typing import Any

__all__ = ["cli"]


def __getattr__(name: str) -> Any:
    if name != "cli":
        raise AttributeError(name)
    from traigent.cli.main import cli

    return cli
