"""Client-side characterization egress enforcement (WI-B).

The closed-pipe design promises the client that its own machine can prove the
egress rules BEFORE the payload leaves. This module is that proof. It runs on
the ``characterization`` block of every ``run_economics`` event at build time
and refuses the whole batch locally on any violation, so:

- a value the client's sharing policy WITHHELD never leaves the machine;
- every transmitted characterization value carries exactly ONE ``shared`` field
  report naming it;
- duplicate or contradictory field reports fail locally rather than shipping an
  ambiguous or leaked value.

These mirror the three rules the schema's ``CharacterizationTelemetry`` enforces
structurally (EGRESS / COVERAGE / SUBSTANCE) plus the duplicate-field backend
obligation the schema's ``uniqueItems`` cannot express. Enforcing them here is
not redundant: it is the promise made good on the client, and it turns a leak
into a local refusal instead of a server-side rejection of an already-egressed
payload.

Privacy: errors name the offending FIELD (an allowlisted vocabulary term) and
the rule. A withheld VALUE is never read into a message, logged, or echoed.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from traigent.economics.contract import (
    CHARACTERIZATION_FIELD_NAMES,
    EVIDENCE_STATUSES,
    PROVENANCE_VALUES,
    SHARING_OUTCOMES,
)
from traigent.economics.errors import EgressPolicyError


def enforce_characterization_egress(characterization: Mapping[str, Any]) -> None:
    """Enforce the characterization egress contract, raising on any violation.

    Args:
        characterization: The ``characterization`` object of a ``run_economics``
            event: ``{"bands"?, "overrides"?, "field_reports", ...}``.

    Raises:
        EgressPolicyError: If any egress rule is violated. Never mutates the
            input and never reads a withheld value into the error.
    """
    if not isinstance(characterization, Mapping):
        raise EgressPolicyError("characterization must be an object")

    reports = characterization.get("field_reports")
    if not isinstance(reports, list) or not reports:
        # The schema requires a non-empty field_reports; a settlement with none
        # is a blank characterization, which the contract exists to reject.
        raise EgressPolicyError(
            "characterization.field_reports must be a non-empty list"
        )

    present = _present_field_names(characterization)

    shared_fields: set[str] = set()
    withheld_fields: set[str] = set()
    seen_fields: set[str] = set()

    for report in reports:
        if not isinstance(report, Mapping):
            raise EgressPolicyError("each field report must be an object")

        field = report.get("field")
        if not isinstance(field, str) or field not in CHARACTERIZATION_FIELD_NAMES:
            # Never echo an unknown/foreign field verbatim: it could be a
            # content-shaped string. Name the rule, not the value.
            raise EgressPolicyError(
                "a field report names a field outside the characterization allowlist"
            )

        # Duplicate / contradictory reports fail locally. Two reports for one
        # field — even with differing provenance or sharing outcome — are a
        # contradiction the schema's uniqueItems cannot catch; refuse here.
        if field in seen_fields:
            raise EgressPolicyError(
                f"field '{field}' has more than one field report "
                "(duplicate or contradictory reports are rejected locally)"
            )
        seen_fields.add(field)

        sharing = report.get("sharing_outcome")
        if sharing not in SHARING_OUTCOMES:
            raise EgressPolicyError(
                f"field '{field}' has an invalid or missing sharing_outcome"
            )

        _check_provenance_and_evidence(field, report)

        if sharing == "shared":
            shared_fields.add(field)
        else:  # withheld_by_policy
            withheld_fields.add(field)

    _check_present_iff_shared(present, shared_fields, withheld_fields)


def _present_field_names(characterization: Mapping[str, Any]) -> set[str]:
    """Collect the allowlisted field names present in bands/overrides.

    A value present under a name OUTSIDE the allowlist is itself an egress
    violation: telemetry can only ever carry allowlisted characterization
    fields.
    """
    present: set[str] = set()
    for block_name in ("bands", "overrides"):
        block = characterization.get(block_name)
        if block is None:
            continue
        if not isinstance(block, Mapping):
            raise EgressPolicyError(f"characterization.{block_name} must be an object")
        for name in block:
            if name not in CHARACTERIZATION_FIELD_NAMES:
                raise EgressPolicyError(
                    f"characterization.{block_name} carries a field outside the allowlist"
                )
            present.add(name)
    return present


def _check_provenance_and_evidence(field: str, report: Mapping[str, Any]) -> None:
    """Enforce the per-report provenance/evidence egress rules.

    Inferred values must account for their evidence; asked/defaulted values must
    not claim any. A withheld field — by sharing OR by evidence — must never
    carry the evidence pointer, which is itself derived from the client's traces
    and would leak the withheld magnitude in prose.
    """
    provenance = report.get("provenance")
    if provenance not in PROVENANCE_VALUES:
        raise EgressPolicyError(f"field '{field}' has an invalid or missing provenance")

    has_status = "evidence_status" in report
    has_pointer = "evidence_pointer" in report
    evidence_status = report.get("evidence_status")
    sharing = report.get("sharing_outcome")

    if provenance == "inferred":
        if not has_status:
            raise EgressPolicyError(
                f"inferred field '{field}' must declare an evidence_status"
            )
        if evidence_status not in EVIDENCE_STATUSES:
            raise EgressPolicyError(f"field '{field}' has an invalid evidence_status")
    else:
        # asked / defaulted values have no inference evidence to account for.
        if has_status or has_pointer:
            raise EgressPolicyError(
                f"{provenance} field '{field}' must not claim inference evidence"
            )

    # Evidence egress: a pointer may travel only alongside evidence_status=provided.
    if evidence_status == "provided" and not has_pointer:
        raise EgressPolicyError(
            f"field '{field}' declares provided evidence but carries no pointer"
        )
    if has_pointer and evidence_status != "provided":
        raise EgressPolicyError(
            f"field '{field}' carries an evidence pointer without provided evidence"
        )

    # Withheld means withheld: a withheld field cannot egress its evidence.
    if sharing == "withheld_by_policy":
        if has_pointer:
            raise EgressPolicyError(
                f"withheld field '{field}' must not carry an evidence pointer"
            )
        if has_status and evidence_status != "withheld_by_policy":
            raise EgressPolicyError(
                f"withheld field '{field}' must not declare provided evidence"
            )


def _check_present_iff_shared(
    present: set[str], shared_fields: set[str], withheld_fields: set[str]
) -> None:
    """Enforce present <=> exactly-one-shared-report as a biconditional.

    Duplicate reports are already refused, so each field has at most one report;
    ``shared_fields`` and ``withheld_fields`` are therefore disjoint here and a
    field named ``shared`` has exactly one shared report.
    """
    # EGRESS: a withheld field must be absent from bands/overrides. This is the
    # rule that keeps a withheld value on the machine.
    leaked = present & withheld_fields
    if leaked:
        raise EgressPolicyError(
            f"field '{sorted(leaked)[0]}' is marked withheld but its value is present "
            "(withheld values must never leave the machine)"
        )

    # COVERAGE: every transmitted value must have a shared report naming it.
    unreported = present - shared_fields
    if unreported:
        raise EgressPolicyError(
            f"field '{sorted(unreported)[0]}' is transmitted without a shared field report"
        )

    # SUBSTANCE: a shared report must carry a value. A 'shared, but nothing here'
    # report is an empty alibi.
    empty_shared = shared_fields - present
    if empty_shared:
        raise EgressPolicyError(
            f"field '{sorted(empty_shared)[0]}' has a shared report but no transmitted value"
        )


__all__ = ["enforce_characterization_egress"]
