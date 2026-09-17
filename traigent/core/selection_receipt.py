"""Selection receipt (R3) for the finalize ``session_aggregation`` payload.

The receipt records the SDK's OWN selection decision — the winner, the exact
ranking-eligible trial set it was chosen over, and the winner-vs-runner-up
margin — so the Backend can bind it to the session's trials. It is projected
from the same :class:`~traigent.core.result_selection.SelectionResult` that
produced ``OptimizationResult``; ranking is never recomputed here.

Contract: TraigentSchema ``session_aggregation_schema.json`` ``selection``
(``SelectionAccepted``). The SDK only ever sends ``disposition: accepted`` and
never sends ``attestation`` (the server stamps it). CONTENT-FREE: the receipt
carries only trial ids, a count, a digest, bounded numbers, bounded labels and
closed enums — every field is picked by an explicit allowlist, and the SDK
margin payload's config dict (``runner_up``) and free-text ``reason`` are never
forwarded.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

from traigent.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "MAX_ELIGIBLE_TRIALS",
    "build_selection_receipt",
    "eligible_trial_ids_digest",
    "sanitize_selection_receipt",
]

#: Matches the schema ``eligible_trial_ids.maxItems`` / Backend MAX_TRIALS.
MAX_ELIGIBLE_TRIALS = 10000

# Schema SelectionLabel / SelectionTrialId. ``fullmatch`` with a charset that
# excludes whitespace is equivalent to the schema's whitespace lookahead guard.
_LABEL_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.:-]{0,63}")
_TRIAL_ID_RE = re.compile(r"[A-Za-z0-9._-]{1,128}")
_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}")

_PAIRED_VERDICTS = frozenset({"clear", "statistical_tie"})


def eligible_trial_ids_digest(ids: list[str]) -> str:
    """``sha256:`` + lowercase hex over the canonical (sorted) id list preimage."""
    preimage = json.dumps(sorted(ids), separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(preimage.encode("utf-8")).hexdigest()


def _is_label(value: Any) -> bool:
    return isinstance(value, str) and _LABEL_RE.fullmatch(value) is not None


def _is_trial_id(value: Any) -> bool:
    return isinstance(value, str) and _TRIAL_ID_RE.fullmatch(value) is not None


def _finite_number(value: Any) -> float | int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(value) else None


def _strict_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _project_margin(raw: Any) -> dict[str, Any] | None:
    """Allowlist-project an SDK margin payload onto the schema SelectionMargin.

    Returns ``None`` when ``raw`` cannot be projected into a valid, all-finite
    margin — the caller then OMITS ``margin`` (never sends NaN, never sends
    ``null``, which would claim no comparison was made).
    """
    if not isinstance(raw, dict):
        return None
    winner = raw.get("winner_trial_id")
    runner_up = raw.get("runner_up_trial_id")
    verdict = raw.get("verdict")
    test = raw.get("test")
    if not (_is_trial_id(winner) and _is_trial_id(runner_up) and _is_label(test)):
        return None
    effective_alpha = _finite_number(raw.get("effective_alpha"))
    if effective_alpha is None or not 0 <= effective_alpha <= 1:
        return None
    n_configs = _strict_int(raw.get("n_configs"))
    n_shared = _strict_int(raw.get("n_shared_examples"))
    if n_configs is None or n_configs < 2 or n_shared is None:
        return None

    raw_delta = raw.get("delta")
    raw_ci = raw.get("ci95")
    raw_p = raw.get("p_value")
    if verdict in _PAIRED_VERDICTS:
        delta = _finite_number(raw_delta)
        p_value = _finite_number(raw_p)
        if delta is None or p_value is None or not 0 <= p_value <= 1:
            return None
        if not isinstance(raw_ci, (list, tuple)) or len(raw_ci) != 2:
            return None
        bounds: list[float | int] = []
        for bound in raw_ci:
            finite = _finite_number(bound)
            if finite is None:
                return None
            bounds.append(finite)
        ci95: list[float | int] | None = bounds
        if n_shared < 1:
            return None
    elif verdict == "na":
        if raw_ci is not None or raw_p is not None or n_shared != 0:
            return None
        delta = None
        if raw_delta is not None:
            delta = _finite_number(raw_delta)
            if delta is None:
                return None
        ci95 = None
        p_value = None
    else:
        return None

    return {
        "winner_trial_id": winner,
        "runner_up_trial_id": runner_up,
        "delta": delta,
        "ci95": ci95,
        "p_value": p_value,
        "verdict": verdict,
        "test": test,
        "n_shared_examples": n_shared,
        "effective_alpha": effective_alpha,
        "n_configs": n_configs,
    }


def _assemble(
    *,
    winner: Any,
    ids: list[str],
    reason: Any,
    margin_raw: Any,
    margin_absent: bool = False,
) -> dict[str, Any] | None:
    """Build an accepted receipt from already-canonical ids, or ``None``."""
    if not _is_trial_id(winner) or not ids or len(ids) > MAX_ELIGIBLE_TRIALS:
        return None
    if not all(_is_trial_id(trial_id) for trial_id in ids):
        return None
    if winner not in ids:
        # Content-free: never log the ids themselves.
        logger.warning(
            "selection receipt withheld: the selected winner is not in the "
            "ranking-eligible trial set"
        )
        return None
    receipt: dict[str, Any] = {
        "disposition": "accepted",
        "selection_reason": reason if _is_label(reason) else None,
        "winner_trial_id": winner,
        "eligible_trial_ids": ids,
        "eligible_trial_count": len(ids),
        "eligible_trial_ids_digest": eligible_trial_ids_digest(ids),
    }
    if margin_absent:
        return receipt
    if margin_raw is None:
        # No comparison was made (fewer than two distinct scored configs).
        receipt["margin"] = None
        return receipt
    margin = _project_margin(margin_raw)
    if margin is not None:
        receipt["margin"] = margin
    # else: omit margin entirely (invalid or non-finite), never null.
    return receipt


def build_selection_receipt(selection: Any) -> dict[str, Any] | None:
    """Project a ``SelectionResult`` onto the accepted ``selection`` receipt.

    Returns ``None`` (no ``selection`` key is sent) when there is no winner, no
    eligible set, the ids are not wire-valid, or the winner is not in the
    eligible set. Never raises.
    """
    try:
        winner = getattr(selection, "best_trial_id", None)
        raw_ids = getattr(selection, "ranking_eligible_trial_ids", None)
        if winner is None or not isinstance(raw_ids, (list, tuple)) or not raw_ids:
            return None
        if not all(isinstance(trial_id, str) for trial_id in raw_ids):
            return None
        return _assemble(
            winner=winner,
            ids=sorted(set(raw_ids)),
            reason=getattr(selection, "reason_code", None),
            margin_raw=getattr(selection, "best_config_margin", None),
        )
    except Exception:  # noqa: BLE001 - a receipt must never break finalize
        logger.warning("selection receipt withheld: receipt construction failed")
        return None


def sanitize_selection_receipt(raw: Any) -> dict[str, Any] | None:
    """Rebuild a caller-supplied ``selection`` receipt from the allowlist.

    Egress guard: nothing is passed through. Only an ``accepted`` receipt whose
    ids are already unique and canonically sorted, whose count and digest agree
    with those ids, and whose winner is eligible survives; ``attestation`` and
    any unknown key are dropped, and an invalid margin is omitted. A
    ``rejected_inconsistent`` form is Backend-written and is never sent.
    Never raises.
    """
    try:
        if not isinstance(raw, dict) or raw.get("disposition") != "accepted":
            return None
        ids = raw.get("eligible_trial_ids")
        if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
            return None
        if ids != sorted(set(ids)):
            return None
        if _strict_int(raw.get("eligible_trial_count")) != len(ids):
            return None
        digest = raw.get("eligible_trial_ids_digest")
        if not (
            isinstance(digest, str)
            and _DIGEST_RE.fullmatch(digest)
            and digest == eligible_trial_ids_digest(ids)
        ):
            return None
        return _assemble(
            winner=raw.get("winner_trial_id"),
            ids=list(ids),
            reason=raw.get("selection_reason"),
            margin_raw=raw.get("margin"),
            margin_absent="margin" not in raw,
        )
    except Exception:  # noqa: BLE001 - egress guard must never break finalize
        return None
