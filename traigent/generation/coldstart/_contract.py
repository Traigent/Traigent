"""Private constants for the cold-start-plan wire contract (``cold-start.v1``).

Mirrors the backend's ``POST /api/v1/guidance/cold-start-plan`` contract
exactly (TraigentSchema is the source of truth; this module is a hand copy
scoped to what the SDK needs). Nothing here is re-exported from the package
-- only the sibling private modules and ``executor.py`` use it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

PROTOCOL_VERSION = "cold-start.v1"
ENDPOINT_PATH = "/api/v1/guidance/cold-start-plan"

# input_kinds / output_kind enum.
INPUT_OUTPUT_KINDS = frozenset(
    {"string", "integer", "number", "boolean", "object", "array", "unknown"}
)

# verifier_kinds enum.
VERIFIER_KINDS = frozenset(
    {
        "deterministic_reference",
        "executable_property",
        "state_transition",
        "human_review",
        "calibrated_judge",
    }
)

# generation_capabilities enum.
GENERATION_CAPABILITIES = frozenset({"deterministic_contract", "customer_llm"})

# Closed vocabulary for ScoreReceipt.provenance. The distinction is the whole
# reason receipts exist, so it must not be a free string:
#
#   oracle_returned        -- the expected output came out of the generation
#                             path itself. A candidate output, NOT verified
#                             truth, however obviously correct it looks.
#   independently_verified -- an authority SEPARATE from generation (a property
#                             check, a reference oracle, a human) confirmed it.
#
# A free-text field here would let a verifier assert any claim it liked, and
# would let arbitrary text into the local manifest. The SDK cannot prove a
# claim of independence is honest -- only the caller knows -- but it can refuse
# to record a claim outside this vocabulary.
PROVENANCE_ORACLE_RETURNED = "oracle_returned"
PROVENANCE_INDEPENDENTLY_VERIFIED = "independently_verified"
PROVENANCE_KINDS = frozenset(
    {PROVENANCE_ORACLE_RETURNED, PROVENANCE_INDEPENDENTLY_VERIFIED}
)

MIN_CANDIDATE_LIMIT = 1
MAX_CANDIDATE_LIMIT = 1000

# The 422 "no local X" reasons the backend defines. descriptor_arity_mismatch
# can also happen server-side, but the SDK catches that one client-side
# first (see _plan.validate_descriptor_arity) since the JSON schema itself
# cannot express "len(input_kinds) == input_arity".
KNOWN_422_REASONS = frozenset(
    {
        "no_local_scoring_authority",
        "no_local_generation_capability",
        "descriptor_arity_mismatch",
    }
)


def compute_descriptor_digest(descriptor: Mapping[str, Any]) -> str:
    """sha256 of the descriptor, computed exactly as the backend computes it.

    Contract: ``sha256(json.dumps(descriptor, sort_keys=True,
    separators=(",", ":")))`` over the descriptor AS SENT. Deliberately does
    NOT reuse ``traigent.knobs.canonical.canonical_hash`` -- that helper
    applies NFC/float normalization the backend's formula does not, which
    would silently desync this digest from the one the server computed for
    the same descriptor.
    """
    payload = json.dumps(descriptor, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
