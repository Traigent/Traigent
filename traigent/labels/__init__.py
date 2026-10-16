"""Label service client crypto (phase 2, additive; nothing here changes today's wire).

Names (labels, knob names, choices, metric names, templates) are sealed to a project
key before they leave the machine. Contract: TraigentSchema ``docs/labels-v0.md``.
"""

# Traceability: CONC-Layer-Core CONC-Quality-Security FUNC-SECURITY

from ._codec import LabelCryptoError
from .bindings import verify_chain
from .envelope import RevisionGuard, open_label, open_label_guarded, seal_label
from .trust_anchor import TrustAnchor, TrustAnchorError, load_trust_anchor
from .writer import LabelWriter

__all__ = [
    "LabelCryptoError",
    "LabelWriter",
    "RevisionGuard",
    "TrustAnchor",
    "TrustAnchorError",
    "load_trust_anchor",
    "open_label",
    "open_label_guarded",
    "seal_label",
    "verify_chain",
]
