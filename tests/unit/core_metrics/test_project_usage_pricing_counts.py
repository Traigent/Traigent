"""Pricing-coverage counts on project-usage top experiments."""

from __future__ import annotations

from traigent.core_metrics.dtos import ProjectUsageExperimentDTO

_BASE = {
    "experiment_id": "exp_1",
    "name": "Support Experiment",
    "configuration_runs": 9,
    "total_cost_usd": 0.0,
    "total_tokens": 3210,
    "avg_latency_ms": None,
    "privacy_classification": "aggregate_safe",
}


def test_pricing_counts_absent_default_to_none() -> None:
    dto = ProjectUsageExperimentDTO.from_dict(dict(_BASE))
    assert dto.priced_configuration_runs is None
    assert dto.unpriced_configuration_runs is None


def test_pricing_counts_parsed_when_present() -> None:
    dto = ProjectUsageExperimentDTO.from_dict(
        {**_BASE, "priced_configuration_runs": 0, "unpriced_configuration_runs": 9}
    )
    assert dto.priced_configuration_runs == 0
    assert dto.unpriced_configuration_runs == 9
    assert dto.total_cost_usd == 0.0


def test_pricing_counts_null_stays_none() -> None:
    dto = ProjectUsageExperimentDTO.from_dict(
        {**_BASE, "priced_configuration_runs": None, "unpriced_configuration_runs": 3}
    )
    assert dto.priced_configuration_runs is None
    assert dto.unpriced_configuration_runs == 3
