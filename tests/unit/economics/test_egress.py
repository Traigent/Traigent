"""Client-side characterization egress enforcement (WI-B)."""

from __future__ import annotations

import pytest

from traigent.economics.egress import enforce_characterization_egress
from traigent.economics.errors import EgressPolicyError


def _shared_report(field: str, **extra: object) -> dict[str, object]:
    report = {
        "field": field,
        "provenance": "asked",
        "confidence": 1.0,
        "sharing_outcome": "shared",
    }
    report.update(extra)
    return report


def test_shared_value_with_matching_report_passes() -> None:
    enforce_characterization_egress(
        {
            "bands": {"value_channel": "save_expert_time"},
            "field_reports": [_shared_report("value_channel")],
        }
    )


def test_all_withheld_characterization_is_representable() -> None:
    # The honest all-withheld case: reports are withheld, nothing in bands.
    enforce_characterization_egress(
        {
            "field_reports": [
                {
                    "field": "error_cost_band",
                    "provenance": "asked",
                    "confidence": 1.0,
                    "sharing_outcome": "withheld_by_policy",
                }
            ]
        }
    )


def test_withheld_value_present_is_refused_locally() -> None:
    with pytest.raises(EgressPolicyError, match="never leave the machine"):
        enforce_characterization_egress(
            {
                "bands": {"error_cost_band": "not_measured"},
                "field_reports": [
                    {
                        "field": "error_cost_band",
                        "provenance": "asked",
                        "confidence": 1.0,
                        "sharing_outcome": "withheld_by_policy",
                    }
                ],
            }
        )


def test_transmitted_value_without_shared_report_is_refused() -> None:
    with pytest.raises(EgressPolicyError, match="without a shared field report"):
        enforce_characterization_egress(
            {
                "bands": {"value_channel": "save_expert_time"},
                "field_reports": [_shared_report("daily_volume_band")],
                # daily_volume_band shared-report references a value not present -> also caught,
                # but value_channel's missing report is the leading violation here.
            }
        )


def test_shared_report_without_value_is_refused() -> None:
    with pytest.raises(EgressPolicyError, match="no transmitted value"):
        enforce_characterization_egress(
            {"field_reports": [_shared_report("value_channel")]}
        )


def test_duplicate_field_reports_fail_locally() -> None:
    with pytest.raises(EgressPolicyError, match="more than one field report"):
        enforce_characterization_egress(
            {
                "bands": {"value_channel": "save_expert_time"},
                "field_reports": [
                    _shared_report("value_channel"),
                    _shared_report("value_channel"),
                ],
            }
        )


def test_contradictory_shared_and_withheld_reports_fail_locally() -> None:
    # A value present with BOTH a shared and a withheld report: duplicate `field`
    # is refused before the contradiction can slip through.
    with pytest.raises(EgressPolicyError, match="more than one field report"):
        enforce_characterization_egress(
            {
                "bands": {"value_channel": "save_expert_time"},
                "field_reports": [
                    _shared_report("value_channel"),
                    {
                        "field": "value_channel",
                        "provenance": "asked",
                        "confidence": 1.0,
                        "sharing_outcome": "withheld_by_policy",
                    },
                ],
            }
        )


def test_field_outside_allowlist_is_refused() -> None:
    with pytest.raises(
        EgressPolicyError, match="outside the characterization allowlist"
    ):
        enforce_characterization_egress(
            {"field_reports": [_shared_report("secret_customer_name")]}
        )


def test_band_outside_allowlist_is_refused() -> None:
    with pytest.raises(EgressPolicyError, match="outside the allowlist"):
        enforce_characterization_egress(
            {
                "bands": {"customer_ssn": "123-45-6789"},
                "field_reports": [_shared_report("value_channel")],
            }
        )


def test_inferred_value_requires_evidence_status() -> None:
    with pytest.raises(EgressPolicyError, match="must declare an evidence_status"):
        enforce_characterization_egress(
            {
                "bands": {"daily_volume_band": "1k_to_99k"},
                "field_reports": [
                    {
                        "field": "daily_volume_band",
                        "provenance": "inferred",
                        "confidence": 0.8,
                        "sharing_outcome": "shared",
                    }
                ],
            }
        )


def test_inferred_shared_value_with_evidence_pointer_passes() -> None:
    enforce_characterization_egress(
        {
            "bands": {"daily_volume_band": "1k_to_99k"},
            "field_reports": [
                {
                    "field": "daily_volume_band",
                    "provenance": "inferred",
                    "confidence": 0.8,
                    "sharing_outcome": "shared",
                    "evidence_status": "provided",
                    "evidence_pointer": "traces show ~3.1k runs/day over 14 days",
                }
            ],
        }
    )


def test_asked_value_must_not_claim_evidence() -> None:
    with pytest.raises(EgressPolicyError, match="must not claim inference evidence"):
        enforce_characterization_egress(
            {
                "bands": {"value_channel": "save_expert_time"},
                "field_reports": [
                    _shared_report(
                        "value_channel",
                        evidence_status="provided",
                        evidence_pointer="should not be here",
                    )
                ],
            }
        )


def test_withheld_field_must_not_carry_evidence_pointer() -> None:
    with pytest.raises(EgressPolicyError, match="evidence pointer"):
        enforce_characterization_egress(
            {
                "field_reports": [
                    {
                        "field": "loss_per_bad_output_usd",
                        "provenance": "inferred",
                        "confidence": 0.5,
                        "sharing_outcome": "withheld_by_policy",
                        "evidence_status": "withheld_by_policy",
                        "evidence_pointer": "incident ledger shows $4k median escalation",
                    }
                ]
            }
        )


def test_empty_field_reports_refused() -> None:
    with pytest.raises(EgressPolicyError, match="non-empty list"):
        enforce_characterization_egress({"field_reports": []})


def test_egress_error_never_quotes_the_withheld_value() -> None:
    withheld_value = "severe_harm_above_5k_usd"
    with pytest.raises(EgressPolicyError) as excinfo:
        enforce_characterization_egress(
            {
                "bands": {"error_cost_band": withheld_value},
                "field_reports": [
                    {
                        "field": "error_cost_band",
                        "provenance": "asked",
                        "confidence": 1.0,
                        "sharing_outcome": "withheld_by_policy",
                    }
                ],
            }
        )
    # The message names the field and the rule, never the withheld value.
    assert withheld_value not in str(excinfo.value)
