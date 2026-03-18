"""Backward compatibility module - use traigent.traigent_client instead.

.. deprecated:: 2.0.0
    This module is deprecated. Use :mod:`traigent.traigent_client` instead.
    The ``TraigentClient`` class has been renamed to ``TraigentClient``.
"""

import warnings

from traigent.traigent_client import TraigentClient

# Backward compatibility alias
TraigentClient = TraigentClient

warnings.warn(
    "traigent.traigent_integration is deprecated. "
    "Use traigent.traigent_client and TraigentClient instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["TraigentClient", "TraigentClient"]
