"""Backward compatibility module - use traigent.traigent_client instead.

.. deprecated:: 2.0.0
    This module is deprecated. Use :mod:`traigent.traigent_client` instead.
    The ``OptiGenClient`` class has been renamed to ``TraigentClient``.
"""

import warnings

from traigent.traigent_client import TraigentClient

# Backward compatibility alias (OptiGenClient -> TraigentClient)
OptiGenClient = TraigentClient

warnings.warn(
    "traigent.traigent_integration is a compatibility shim and is deprecated. "
    "Import directly from traigent.traigent_client instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["OptiGenClient", "TraigentClient"]
