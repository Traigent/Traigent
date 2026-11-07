"""Offline evidence bundle and replay for one TVL promotion decision.

A bundle is a JSON record an auditor can recompute without the original run:
the promotion policy and objectives, the paired per-example scores, the gate's
decision and statistics, and a SHA-256 manifest over the canonical JSON.

Privacy (enforced, fails closed with ``ValueError`` at build time): example ids
are replaced by ``HMAC-SHA256(salt, id)`` with a random per-bundle salt that is
generated internally and never stored. Every caller-supplied string (objective,
constraint and evaluator names, policy keys/values) must be a code identifier
matching ``^[A-Za-z0-9_.:/-]{1,64}$``; the dataset hash must be 64 lowercase
hex characters; the sample selection is a closed enum plus optional integer
seed/count. Prose is rejected. Objective, evaluator and constraint names are
developer-chosen code identifiers (allowed under GNFR-PRIV-003); callers must
not use customer example ids or labels as names, because a conforming
identifier such as ``Alice_has_diabetes`` is serialized verbatim. Pseudonymised ids do not hide what other
metadata or the scores themselves reveal.

Manifest: ``manifest_sha256`` is an INTEGRITY check only, not provenance. It
is an unkeyed digest stored next to its payload, so anyone who can edit the
bundle can reseal it. To establish that a bundle is the one originally issued,
compare its digest with an independently retained value:
``replay_evidence_bundle(bundle, expected_manifest=...)`` or the CLI
``--expect-sha256``. No signing is provided. Canonicalization is this Python
serialization (``json.dumps`` with sorted keys, ``(",", ":")`` separators,
ASCII escaping, no NaN/Infinity), not a cross-language standard.

Strict mode: the explicit opt-in for "no fallback to a plain score comparison"
is ``PromotionPolicy.require_calibration`` with ``enabled=True`` (or declaring
``chance_constraints``); see ``TraigentOrchestrator._is_strict_evidence_mode``.
A policy that implies strict mode cannot be bundled or replayed with
``strict_mode=False``. Bundles for a run that used a fallback comparison are
out of scope: build refuses ``fallback_used=True`` and replay rejects it.

CLI: ``python -m traigent.tvl.evidence_bundle replay BUNDLE.json
[--expect-sha256 HEX]``.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import hmac
import json
import math
import re
import secrets
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from traigent.tvl.models import BandTarget, PromotionPolicy
from traigent.tvl.promotion_gate import ObjectiveSpec, PromotionDecision, PromotionGate

SCHEMA_VERSION = 1
_REL_TOL = 1e-9
_ABS_TOL = 1e-12
_IDENT = re.compile(r"[A-Za-z0-9_.:/-]{1,64}")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_OPAQUE_ID = re.compile(r"[0-9a-f]{32}")
SAMPLE_SELECTIONS = (
    "full_dataset",
    "random_seeded",
    "stratified_seeded",
    "holdout_split",
    "other",
)


class MalformedBundleError(ValueError):
    """The bundle does not have the required structure (distinct from mismatch)."""


def _ident(value: Any, what: str) -> str:
    # Messages never echo the offending value (it may be customer content).
    if not isinstance(value, str) or _IDENT.fullmatch(value) is None:
        raise ValueError(f"{what} must match [A-Za-z0-9_.:/-]{{1,64}}")
    return value


def _check_idents_deep(obj: Any, what: str) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            _ident(key, what)
            _check_idents_deep(value, what)
    elif isinstance(obj, list):
        for item in obj:
            _check_idents_deep(item, what)
    elif isinstance(obj, str):
        _ident(obj, what)


def is_strict_policy(policy: PromotionPolicy) -> bool:
    """True when the policy itself declares strict evidence mode."""
    calibration = policy.require_calibration
    return bool(
        (calibration is not None and calibration.enabled) or policy.chance_constraints
    )


def sha256_file(path: str | Path) -> str:
    """SHA-256 hex digest of a dataset file (streamed)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(obj: Any) -> bytes:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def manifest_sha256(bundle: Mapping[str, Any]) -> str:
    """SHA-256 over the canonical JSON of the bundle without its manifest.

    Integrity only, not provenance (see the module docstring).
    """
    body = {k: v for k, v in bundle.items() if k != "manifest_sha256"}
    return hashlib.sha256(_canonical(body)).hexdigest()


def _opaque_id(salt: bytes, example_id: str) -> str:
    return hmac.new(salt, example_id.encode("utf-8"), hashlib.sha256).hexdigest()[:32]


def _git_sha() -> str | None:
    try:
        from traigent.identity.agent_build import git_output

        out = git_output(Path(__file__).resolve().parent, "rev-parse", "HEAD")
    except Exception:  # best effort only
        return None
    return out.strip() if isinstance(out, str) and out.strip() else None


def _sdk_version() -> str | None:
    try:
        from traigent import __version__

        return str(__version__)
    except Exception:
        return None


def _decision_to_dict(decision: PromotionDecision) -> dict[str, Any]:
    return {
        "decision": decision.decision,
        "reason": decision.reason,
        "adjusted_p_values": dict(decision.adjusted_p_values),
        "objective_results": [
            dataclasses.asdict(r) for r in decision.objective_results
        ],
        "chance_results": [dataclasses.asdict(r) for r in decision.chance_results],
    }


def _objective_to_dict(spec: ObjectiveSpec) -> dict[str, Any]:
    band = None
    if spec.band is not None:
        band = {"low": spec.band.low, "high": spec.band.high}
    return {
        "name": spec.name,
        "direction": spec.direction,
        "band": band,
        "band_alpha": spec.band_alpha,
    }


def _objective_from_dict(data: Mapping[str, Any]) -> ObjectiveSpec:
    band = data["band"]
    return ObjectiveSpec(
        name=data["name"],
        direction=data["direction"],
        band=BandTarget(low=band["low"], high=band["high"]) if band else None,
        band_alpha=data["band_alpha"],
    )


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def build_evidence_bundle(
    *,
    policy: PromotionPolicy,
    objectives: Sequence[ObjectiveSpec],
    example_ids: Sequence[str],
    incumbent_scores: Mapping[str, Sequence[float]],
    candidate_scores: Mapping[str, Sequence[float]],
    dataset_sha256: str,
    evaluator_name: str,
    evaluator_version: str,
    sample_selection: str,
    sample_seed: int | None = None,
    sample_count: int | None = None,
    constraint_data: Mapping[str, tuple[int, int]] | None = None,
    strict_mode: bool | None = None,
    fallback_used: bool = False,
    _salt: bytes | None = None,
) -> dict[str, Any]:
    """Run the gate on paired scores and return the evidence bundle (a dict).

    ``incumbent_scores[obj][i]`` and ``candidate_scores[obj][i]`` are the paired
    scores for ``example_ids[i]``. ``sample_selection`` is one of
    ``SAMPLE_SELECTIONS``. ``strict_mode`` defaults to what the policy declares
    (:func:`is_strict_policy`); ``False`` is rejected when the policy implies
    strict. ``fallback_used=True`` is refused. Raises ``ValueError`` on any
    privacy-schema violation (see the module docstring).
    """
    if fallback_used:
        raise ValueError("bundles for runs that used a fallback are out of scope")
    implied_strict = is_strict_policy(policy)
    if strict_mode is None:
        strict_mode = implied_strict
    if implied_strict and not strict_mode:
        raise ValueError("policy implies strict mode; strict_mode=False is rejected")

    if not isinstance(dataset_sha256, str) or not _HEX64.fullmatch(dataset_sha256):
        raise ValueError("dataset_sha256 must be 64 lowercase hex characters")
    _ident(evaluator_name, "evaluator_name")
    _ident(evaluator_version, "evaluator_version")
    if sample_selection not in SAMPLE_SELECTIONS:
        raise ValueError(f"sample_selection must be one of {SAMPLE_SELECTIONS}")
    for label, number in (("sample_seed", sample_seed), ("sample_count", sample_count)):
        if number is not None and not _is_int(number):
            raise ValueError(f"{label} must be an integer or None")
    policy_dict = {k: v for k, v in dataclasses.asdict(policy).items() if v is not None}
    _check_idents_deep(policy_dict, "policy key/value")
    obj_dicts = [_objective_to_dict(o) for o in objectives]
    for obj in obj_dicts:
        _ident(obj["name"], "objective name")
    for name in (*incumbent_scores, *candidate_scores, *(constraint_data or {})):
        _ident(name, "score/constraint name")

    if not obj_dicts:
        raise ValueError("at least one objective is required")
    for series in (*incumbent_scores.values(), *candidate_scores.values()):
        if not all(_is_num(x) for x in series):
            raise ValueError("scores must be finite numbers")
    for obj in obj_dicts:
        if obj["name"] not in incumbent_scores or obj["name"] not in candidate_scores:
            raise ValueError(
                "every declared objective needs incumbent and candidate scores"
            )
    if set(incumbent_scores) != set(candidate_scores) or set(candidate_scores) != {
        o["name"] for o in obj_dicts
    }:
        raise ValueError("scores must cover exactly the declared objectives")
    _check_constraint_counts(policy, constraint_data or {}, ValueError)
    if len(set(example_ids)) != len(example_ids):
        raise ValueError("example_ids must be unique")
    for _name, series in (*incumbent_scores.items(), *candidate_scores.items()):
        if len(series) != len(example_ids):
            raise ValueError("scores must align with example_ids")

    gate = PromotionGate(policy, list(objectives))
    decision = gate.evaluate(
        {k: list(v) for k, v in incumbent_scores.items()},
        {k: list(v) for k, v in candidate_scores.items()},
        dict(constraint_data) if constraint_data else None,
    )

    salt = _salt if _salt is not None else secrets.token_bytes(32)
    ids = [_opaque_id(salt, i) for i in example_ids]
    scores = {
        name: [
            {"id": ids[i], "incumbent": incumbent_scores[name][i], "candidate": cand}
            for i, cand in enumerate(candidate_scores[name])
        ]
        for name in sorted(candidate_scores)
        if name in incumbent_scores
    }
    bundle: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "policy": policy_dict,
        "objectives": obj_dicts,
        "dataset_sha256": dataset_sha256,
        "evaluator": {"name": evaluator_name, "version": evaluator_version},
        "sample_selection": {
            "kind": sample_selection,
            "seed": sample_seed,
            "count": sample_count,
        },
        "scores": scores,
        "constraint_data": {k: list(v) for k, v in (constraint_data or {}).items()},
        "strict_mode": bool(strict_mode),
        "fallback_used": False,
        "decision": _decision_to_dict(decision),
        "sdk_version": _sdk_version(),
        "git_sha": _git_sha(),
    }
    bundle["manifest_sha256"] = manifest_sha256(bundle)
    return bundle


def write_evidence_bundle(bundle: Mapping[str, Any], path: str | Path) -> Path:
    """Write the bundle as canonical-key-ordered JSON and return the path."""
    out = Path(path)
    out.write_text(
        json.dumps(bundle, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return out


@dataclass(slots=True)
class ReplayResult:
    """Outcome of replaying a bundle.

    ``status`` is one of ``match``, ``mismatch`` (recomputed differs from
    recorded), ``manifest_failed`` (digest does not match the content or the
    expected digest) or ``malformed`` (structure invalid; ``error`` says
    where). Only ``match`` has ``match=True``.
    """

    status: str
    manifest_ok: bool = False
    match: bool = False
    recorded_decision: str | None = None
    recomputed_decision: str | None = None
    mismatches: list[str] = field(default_factory=list)
    error: str | None = None


def _require(cond: object, where: str) -> None:
    if not cond:
        raise MalformedBundleError(where)


def _is_num(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _check_constraint_counts(
    policy: PromotionPolicy, data: Mapping[str, Any], error: type[ValueError]
) -> None:
    """Every declared chance constraint needs (violations, trials) counts."""
    for constraint in policy.chance_constraints:
        pair = data.get(constraint.name)
        ok = (
            isinstance(pair, (list, tuple))
            and len(pair) == 2
            and all(_is_int(x) and x >= 0 for x in pair)
            and pair[0] <= pair[1]
        )
        if not ok:
            raise error("constraint_data.counts")


def _validate_structure(bundle: Mapping[str, Any]) -> None:
    """Raise MalformedBundleError unless every required field is well-formed."""
    _require(isinstance(bundle, Mapping), "bundle")
    _require(bundle.get("schema_version") == SCHEMA_VERSION, "schema_version")
    for key in (
        "policy",
        "objectives",
        "dataset_sha256",
        "evaluator",
        "sample_selection",
        "scores",
        "constraint_data",
        "strict_mode",
        "fallback_used",
        "decision",
        "manifest_sha256",
    ):
        _require(key in bundle, key)
    _require(
        isinstance(bundle["dataset_sha256"], str)
        and _HEX64.fullmatch(bundle["dataset_sha256"]) is not None,
        "dataset_sha256",
    )
    ev = bundle["evaluator"]
    _require(isinstance(ev, dict), "evaluator")
    for key in ("name", "version"):
        _require(isinstance(ev.get(key), str) and _IDENT.fullmatch(ev[key]), key)
    sel = bundle["sample_selection"]
    _require(
        isinstance(sel, dict) and sel.get("kind") in SAMPLE_SELECTIONS,
        "sample_selection",
    )
    for key in ("seed", "count"):
        _require(key in sel and (sel[key] is None or _is_int(sel[key])), key)
    for key in ("strict_mode", "fallback_used"):
        _require(isinstance(bundle[key], bool), key)
    _require(isinstance(bundle["policy"], dict), "policy")
    _require(
        isinstance(bundle["objectives"], list) and bundle["objectives"], "objectives"
    )
    names = []
    for obj in bundle["objectives"]:
        _require(isinstance(obj, dict), "objectives")
        _require(
            all(k in obj for k in ("name", "direction", "band", "band_alpha")),
            "objectives.fields",
        )
        _require(isinstance(obj["name"], str) and _IDENT.fullmatch(obj["name"]), "name")
        _require(_is_num(obj["band_alpha"]), "band_alpha")
        band = obj["band"]
        _require(
            band is None
            or (
                isinstance(band, dict)
                and _is_num(band.get("low"))
                and _is_num(band.get("high"))
            ),
            "band",
        )
        names.append(obj["name"])
    scores = bundle["scores"]
    _require(isinstance(scores, dict) and set(scores) == set(names), "scores.keys")
    id_lists = []
    for rows in scores.values():
        _require(isinstance(rows, list) and rows, "scores.rows")
        for row in rows:
            _require(
                isinstance(row, dict)
                and set(row) == {"id", "incumbent", "candidate"}
                and isinstance(row["id"], str)
                and _OPAQUE_ID.fullmatch(row["id"]) is not None
                and _is_num(row["incumbent"])
                and _is_num(row["candidate"]),
                "scores.row",
            )
        id_lists.append([r["id"] for r in rows])
    _require(all(ids == id_lists[0] for ids in id_lists), "scores.id_alignment")
    _require(len(set(id_lists[0])) == len(id_lists[0]), "scores.duplicate_ids")
    cd = bundle["constraint_data"]
    _require(isinstance(cd, dict), "constraint_data")
    for name, pair in cd.items():
        _require(_IDENT.fullmatch(name) is not None, "constraint_data.name")
        _require(
            isinstance(pair, list) and len(pair) == 2 and all(_is_int(x) for x in pair),
            "constraint_data.value",
        )
    dec = bundle["decision"]
    _require(isinstance(dec, dict), "decision")
    _require(
        all(
            k in dec
            for k in (
                "decision",
                "reason",
                "adjusted_p_values",
                "objective_results",
                "chance_results",
            )
        ),
        "decision.fields",
    )
    try:
        _check_idents_deep(bundle["policy"], "policy")
    except ValueError as exc:
        raise MalformedBundleError("policy") from exc


def _close(recorded: Any, recomputed: Any) -> bool:
    """Structural equality; floats compare with rel 1e-9 / abs 1e-12."""
    if isinstance(recorded, bool) or isinstance(recomputed, bool):
        return recorded is recomputed
    if _is_num(recorded) and _is_num(recomputed):
        return math.isclose(recorded, recomputed, rel_tol=_REL_TOL, abs_tol=_ABS_TOL)
    if isinstance(recorded, dict) and isinstance(recomputed, dict):
        return set(recorded) == set(recomputed) and all(
            _close(recorded[k], recomputed[k]) for k in recorded
        )
    if isinstance(recorded, list) and isinstance(recomputed, list):
        return len(recorded) == len(recomputed) and all(
            _close(a, b) for a, b in zip(recorded, recomputed, strict=True)
        )
    return bool(recorded == recomputed)


def replay_evidence_bundle(
    bundle: Mapping[str, Any], *, expected_manifest: str | None = None
) -> ReplayResult:
    """Validate, verify the manifest, and recompute the decision from the bundle.

    ``expected_manifest`` is an independently retained digest; when given, a
    bundle whose digest differs fails with status ``manifest_failed``.
    """
    try:
        return _replay(bundle, expected_manifest)
    except MalformedBundleError as exc:
        return ReplayResult("malformed", error=f"invalid field: {exc}")
    except Exception:  # malformed input (incl. NaN/Infinity) must never crash replay
        return ReplayResult("malformed", error="invalid bundle content")


def _replay(bundle: Mapping[str, Any], expected_manifest: str | None) -> ReplayResult:
    try:
        _validate_structure(bundle)
        policy = PromotionPolicy.from_dict(bundle["policy"])
        gate = PromotionGate(
            policy, [_objective_from_dict(o) for o in bundle["objectives"]]
        )
        _check_constraint_counts(
            policy, bundle["constraint_data"], MalformedBundleError
        )
    except MalformedBundleError as exc:
        return ReplayResult("malformed", error=f"invalid field: {exc}")
    except Exception:  # malformed input must never crash replay
        return ReplayResult("malformed", error="invalid field: policy/objectives")

    if manifest_sha256(bundle) != bundle["manifest_sha256"]:
        return ReplayResult("manifest_failed", mismatches=["manifest_sha256"])
    if expected_manifest is not None and expected_manifest != bundle["manifest_sha256"]:
        return ReplayResult("manifest_failed", mismatches=["expected_manifest"])

    recorded = bundle["decision"]
    result = ReplayResult(
        "match", manifest_ok=True, recorded_decision=recorded["decision"]
    )
    if bundle["fallback_used"]:
        result.mismatches.append("fallback_used")
    if is_strict_policy(policy) and not bundle["strict_mode"]:
        result.mismatches.append("strict_mode")

    incumbent: dict[str, Sequence[float]] = {
        n: [r["incumbent"] for r in rows] for n, rows in bundle["scores"].items()
    }
    candidate: dict[str, Sequence[float]] = {
        n: [r["candidate"] for r in rows] for n, rows in bundle["scores"].items()
    }
    constraints = {k: (v[0], v[1]) for k, v in bundle["constraint_data"].items()}
    try:
        recomputed = _decision_to_dict(
            gate.evaluate(incumbent, candidate, constraints or None)
        )
    except Exception:  # malformed numeric input must not crash replay
        return ReplayResult("malformed", error="evaluation failed on bundle inputs")
    result.recomputed_decision = recomputed["decision"]
    for key, value in recomputed.items():
        if not _close(recorded[key], value):
            result.mismatches.append(f"decision.{key}")
    if result.mismatches:
        result.status = "mismatch"
    result.match = not result.mismatches
    return result


def _reject_constant(_name: str) -> Any:
    raise ValueError("NaN/Infinity are not valid in a bundle")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI; exit 0 on match, 1 on mismatch or manifest failure, 2 on malformed."""
    parser = argparse.ArgumentParser(
        prog="python -m traigent.tvl.evidence_bundle",
        description=(
            "Replay a promotion evidence bundle. The manifest digest is an "
            "integrity check only, not provenance: pass --expect-sha256 with a "
            "digest you retained independently to detect a resealed bundle."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)
    replay = sub.add_parser("replay", help="recompute the decision from a bundle")
    replay.add_argument("bundle", help="path to the bundle JSON")
    replay.add_argument(
        "--expect-sha256",
        help="independently retained manifest digest; fail if it differs",
    )
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    try:
        bundle = json.loads(
            Path(args.bundle).read_text(encoding="utf-8"),
            parse_constant=_reject_constant,
        )
    except (OSError, ValueError):
        print("status=malformed error=unreadable bundle file")
        return 2
    result = replay_evidence_bundle(bundle, expected_manifest=args.expect_sha256)
    print(
        f"status={result.status} manifest_ok={result.manifest_ok} "
        f"recorded={result.recorded_decision} recomputed={result.recomputed_decision} "
        f"mismatches={result.mismatches} error={result.error}"
    )
    if result.status == "malformed":
        return 2
    return 0 if result.match else 1


if __name__ == "__main__":
    raise SystemExit(main())
