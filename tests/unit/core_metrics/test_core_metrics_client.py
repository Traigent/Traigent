from __future__ import annotations

from urllib import error

import pytest

import traigent.core_metrics.client as client_module
from traigent.core_metrics import CoreMetricsClient, CoreMetricsConfig
from traigent.utils.exceptions import (
    AuthenticationError,
    ClientError,
    TraigentConnectionError,
)


def test_core_metrics_client_reads_overview_and_trend() -> None:
    calls: list[tuple[str, str, dict | None, str]] = []

    def request_sender(
        method: str,
        path: str,
        payload: dict | None,
        response_kind: str = "json",
    ):
        calls.append((method, path, payload, response_kind))
        if path == "/core-metrics/overview":
            return {
                "data": {
                    "tenant_id": "tenant_acme",
                    "project_id": "project_alpha",
                    "entities": {
                        "agents": 2,
                        "benchmarks": 1,
                        "measures": 3,
                        "experiments": 4,
                        "experiment_runs": 5,
                        "configuration_runs": 6,
                    },
                    "statuses": {
                        "experiments": {"running": 2},
                        "experiment_runs": {"completed": 5},
                        "configuration_runs": {"completed": 6},
                    },
                    "recent_run_volume": [{"date": "2026-03-10", "count": 5}],
                    "measure_summary": {
                        "accuracy": {
                            "count": 5,
                            "mean": 0.91,
                            "min": 0.82,
                            "max": 0.97,
                        }
                    },
                }
            }
        return {
            "data": {
                "experiment_id": "exp/123",
                "experiment_name": "Support Experiment",
                "project_id": "project_alpha",
                "runs_total": 3,
                "configuration_runs_total": 9,
                "daily_runs": [{"date": "2026-03-10", "count": 3}],
                "daily_configuration_runs": [{"date": "2026-03-10", "count": 9}],
                "measure_summary": {},
            }
        }

    client = CoreMetricsClient(request_sender=request_sender)

    overview = client.get_core_metrics_overview()
    trend = client.get_experiment_trend("exp/123")

    assert calls[0] == ("GET", "/core-metrics/overview", None, "json")
    assert calls[1] == ("GET", "/core-metrics/experiments/exp%2F123/trend", None, "json")
    assert overview.tenant_id == "tenant_acme"
    assert overview.project_id == "project_alpha"
    assert overview.entities.experiments == 4
    assert trend.project_id == "project_alpha"
    assert trend.experiment_name == "Support Experiment"


def test_core_metrics_client_reads_project_analytics_shapes() -> None:
    calls: list[tuple[str, str, dict | None, str]] = []

    def request_sender(
        method: str,
        path: str,
        payload: dict | None,
        response_kind: str = "json",
    ):
        calls.append((method, path, payload, response_kind))
        if path == "/analytics/summary?days=7":
            return {
                "data": {
                    "context": {
                        "tenant_id": "tenant_acme",
                        "project_id": "project_alpha",
                        "generated_at": "2026-03-11T10:15:00Z",
                        "privacy_classification": "aggregate_safe",
                    },
                    "range_days": 7,
                    "entity_counts": {
                        "agents": 2,
                        "benchmarks": 1,
                        "measures": 3,
                        "experiments": 4,
                        "experiment_runs": 5,
                        "configuration_runs": 6,
                    },
                    "status_breakdowns": {
                        "experiments": {"running": 1},
                        "experiment_runs": {"completed": 5},
                        "configuration_runs": {"completed": 6},
                    },
                    "usage_summary": {
                        "experiment_runs": 5,
                        "configuration_runs": 6,
                        "priced_configuration_runs": 5,
                        "unpriced_configuration_runs": 1,
                        "total_cost_usd": 1.25,
                        "avg_cost_usd": 0.25,
                        "cost_source_breakdown": {
                            "observed_usage": 4,
                            "recorded_metrics": 1,
                            "catalog_fallback": 0,
                            "unknown_unpriced": 1,
                        },
                        "total_tokens": 1234,
                        "avg_latency_ms": 120.5,
                        "p95_latency_ms": 210.0,
                    },
                    "measure_summaries": [
                        {
                            "measure_key": "accuracy",
                            "measure_id": "measure_accuracy",
                            "label": "Accuracy",
                            "value_type": "numeric",
                            "sample_count": 6,
                            "mean": 0.91,
                            "min": 0.84,
                            "max": 0.97,
                            "privacy_classification": "aggregate_safe",
                        }
                    ],
                }
            }
        if path == "/analytics/pricing-catalog":
            return {
                "data": {
                    "context": {
                        "tenant_id": "tenant_acme",
                        "project_id": "project_alpha",
                        "generated_at": "2026-03-12T09:00:00Z",
                        "privacy_classification": "aggregate_safe",
                    },
                    "catalog_source": "static_catalog",
                    "catalog_last_updated": "2026-03-12T08:55:00Z",
                    "total_providers": 1,
                    "total_models": 1,
                    "providers": [
                        {
                            "provider": "openai",
                            "model_count": 1,
                            "pricing_resolution_mode": "static_catalog",
                            "models": [
                                {
                                    "model": "gpt-4o",
                                    "input_price_per_1k_usd": 0.005,
                                    "output_price_per_1k_usd": 0.015,
                                    "context_window": 128000,
                                    "available_tiers": ["standard", "premium", "enterprise"],
                                    "supports_catalog_fallback": True,
                                }
                            ],
                        }
                    ],
                }
            }
        if path == "/analytics/dashboards/optimization-overview?days=7&limit=3":
            return {
                "data": {
                    "context": {
                        "tenant_id": "tenant_acme",
                        "project_id": "project_alpha",
                        "generated_at": "2026-03-12T09:15:00Z",
                        "privacy_classification": "aggregate_safe",
                    },
                    "range_days": 7,
                    "summary_cards": {
                        "experiments_total": 4,
                        "experiment_runs_in_range": 5,
                        "configuration_runs_in_range": 12,
                        "priced_configuration_runs_in_range": 11,
                        "unpriced_configuration_runs_in_range": 1,
                        "total_cost_usd_in_range": 1.55,
                        "avg_latency_ms_in_range": 118.2,
                        "total_tokens_in_range": 4321,
                    },
                    "cost_source_breakdown": {
                        "observed_usage": 8,
                        "recorded_metrics": 2,
                        "catalog_fallback": 1,
                        "unknown_unpriced": 1,
                    },
                    "recent_experiments": [
                        {
                            "experiment_id": "exp_1",
                            "name": "Support Experiment",
                            "status": "completed",
                            "experiment_run_count": 3,
                            "configuration_run_count": 9,
                            "priced_configuration_runs": 8,
                            "unpriced_configuration_runs": 1,
                            "total_cost_usd": 1.23,
                            "avg_latency_ms": 111.0,
                            "avg_primary_score": 0.93,
                            "total_tokens": 3210,
                            "last_run_at": "2026-03-12T08:30:00Z",
                            "privacy_classification": "aggregate_safe",
                        }
                    ],
                }
            }
        if path == "/analytics/export-jobs?page=1&per_page=10":
            return {
                "data": {
                    "context": {
                        "tenant_id": "tenant_acme",
                        "project_id": "project_alpha",
                        "generated_at": "2026-03-12T09:20:00Z",
                        "privacy_classification": "aggregate_safe",
                    },
                    "items": [
                        {
                            "job_id": "export_job_1",
                            "export_type": "fine_tuning_manifest",
                            "status": "completed",
                            "privacy_classification": "manifest_safe",
                            "export_mode": "manifest",
                            "privacy_mode": True,
                            "include_content": False,
                            "record_count": 12,
                            "artifact_filename": "fine-tuning-manifest.json",
                            "artifact_content_type": "application/json",
                            "experiment_id": "exp_1",
                            "experiment_run_id": None,
                            "limit": 1000,
                            "requested_by": "user_123",
                            "requested_at": "2026-03-12T09:18:00Z",
                            "completed_at": "2026-03-12T09:18:02Z",
                            "error_message": None,
                        }
                    ],
                    "pagination": {
                        "page": 1,
                        "per_page": 10,
                        "total": 1,
                        "total_pages": 1,
                        "has_next": False,
                        "has_prev": False,
                    },
                }
            }
        if path == "/analytics/export-jobs/export_job_1":
            return {
                "data": {
                    "context": {
                        "tenant_id": "tenant_acme",
                        "project_id": "project_alpha",
                        "generated_at": "2026-03-12T09:20:00Z",
                        "privacy_classification": "aggregate_safe",
                    },
                    "job": {
                        "job_id": "export_job_1",
                        "export_type": "fine_tuning_manifest",
                        "status": "completed",
                        "privacy_classification": "manifest_safe",
                        "export_mode": "manifest",
                        "privacy_mode": True,
                        "include_content": False,
                        "record_count": 12,
                        "artifact_filename": "fine-tuning-manifest.json",
                        "artifact_content_type": "application/json",
                        "experiment_id": "exp_1",
                        "experiment_run_id": None,
                        "limit": 1000,
                        "requested_by": "user_123",
                        "requested_at": "2026-03-12T09:18:00Z",
                        "completed_at": "2026-03-12T09:18:02Z",
                        "error_message": None,
                    },
                }
            }
        if path == "/analytics/trends/run-volume?experiment_id=exp_1&days=7&bucket=day":
            return {
                "data": {
                    "context": {
                        "tenant_id": "tenant_acme",
                        "project_id": "project_alpha",
                        "generated_at": "2026-03-11T10:15:00Z",
                        "privacy_classification": "aggregate_safe",
                    },
                    "metric_id": "run_volume",
                    "experiment_id": "exp_1",
                    "range_days": 7,
                    "requested_bucket": "day",
                    "resolved_bucket": "day",
                    "series": [
                        {
                            "series_key": "experiment_runs",
                            "label": "Experiment runs",
                            "unit": "count",
                            "points": [
                                {
                                    "bucket_start": "2026-03-10T00:00:00+00:00",
                                    "bucket_label": "2026-03-10",
                                    "value": 3,
                                }
                            ],
                        }
                    ],
                }
            }
        return {
            "data": {
                "context": {
                    "tenant_id": "tenant_acme",
                    "project_id": "project_alpha",
                    "generated_at": "2026-03-11T10:15:00Z",
                    "privacy_classification": "aggregate_safe",
                },
                "measure_key": "accuracy",
                "measure_id": "measure_accuracy",
                "label": "Accuracy",
                "experiment_id": "exp_1",
                "value_type": "numeric",
                "sample_count": 6,
                "mean": 0.91,
                "min": 0.84,
                "max": 0.97,
                "bucket_count": 5,
                "histogram": [
                    {"lower_bound": 0.8, "upper_bound": 0.84, "count": 1},
                    {"lower_bound": 0.84, "upper_bound": 0.88, "count": 1},
                ],
            }
        }

    client = CoreMetricsClient(request_sender=request_sender)
    summary = client.get_analytics_summary(days=7)
    catalog = client.get_pricing_catalog()
    dashboard = client.get_optimization_overview_dashboard(days=7, limit=3)
    export_jobs = client.list_export_jobs(page=1, per_page=10)
    export_job = client.get_export_job("export_job_1")
    trend = client.get_run_volume_trend(experiment_id="exp_1", days=7, bucket="day")
    distribution = client.get_measure_distribution("accuracy", experiment_id="exp_1", bins=5)

    assert calls[0] == ("GET", "/analytics/summary?days=7", None, "json")
    assert calls[1] == ("GET", "/analytics/pricing-catalog", None, "json")
    assert calls[2] == (
        "GET",
        "/analytics/dashboards/optimization-overview?days=7&limit=3",
        None,
        "json",
    )
    assert calls[3] == ("GET", "/analytics/export-jobs?page=1&per_page=10", None, "json")
    assert calls[4] == ("GET", "/analytics/export-jobs/export_job_1", None, "json")
    assert calls[5] == (
        "GET",
        "/analytics/trends/run-volume?experiment_id=exp_1&days=7&bucket=day",
        None,
        "json",
    )
    assert calls[6] == (
        "GET",
        "/analytics/distributions/measures/accuracy?experiment_id=exp_1&bins=5",
        None,
        "json",
    )
    assert summary.usage_summary.priced_configuration_runs == 5
    assert summary.usage_summary.cost_source_breakdown.observed_usage == 4
    assert summary.usage_summary.total_cost_usd == 1.25
    assert catalog.catalog_source == "static_catalog"
    assert catalog.providers[0].models[0].supports_catalog_fallback is True
    assert dashboard.summary_cards.unpriced_configuration_runs_in_range == 1
    assert dashboard.cost_source_breakdown.catalog_fallback == 1
    assert dashboard.recent_experiments[0].experiment_id == "exp_1"
    assert export_jobs.pagination.total == 1
    assert export_jobs.items[0].job_id == "export_job_1"
    assert export_job.job.requested_by == "user_123"
    assert trend.series[0].points[0].value == 3
    assert distribution.histogram[0].count == 1


def test_core_metrics_client_exports_fine_tuning_jsonl() -> None:
    def request_sender(
        method: str,
        path: str,
        payload: dict | None,
        response_kind: str = "json",
    ):
        assert method == "GET"
        assert payload is None
        assert response_kind == "text"
        assert path == "/core-exports/fine-tuning.jsonl?experiment_id=exp_1&limit=25"
        return (
            '{"messages":[{"role":"user","content":"hello"}]}',
            {
                "content-disposition": "attachment; filename*=UTF-8''ft-export.jsonl",
                "content-type": "application/x-ndjson",
            },
        )

    client = CoreMetricsClient(request_sender=request_sender)
    export = client.export_fine_tuning_jsonl(experiment_id="exp_1", limit=25)

    assert export.filename == "ft-export.jsonl"
    assert export.content_type == "application/x-ndjson"
    assert '"messages"' in export.content


def test_core_metrics_client_exports_privacy_safe_manifest() -> None:
    def request_sender(
        method: str,
        path: str,
        payload: dict | None,
        response_kind: str = "json",
    ):
        assert method == "GET"
        assert payload is None
        assert response_kind == "json"
        assert (
            path
            == "/analytics/exports/fine-tuning.manifest?experiment_id=exp_1&limit=25&include_content=false"
        )
        return {
            "data": {
                "context": {
                    "tenant_id": "tenant_acme",
                    "project_id": "project_alpha",
                    "generated_at": "2026-03-11T10:15:00Z",
                    "privacy_classification": "manifest_safe",
                },
                "export_mode": "manifest",
                "privacy_mode": True,
                "include_content": False,
                "job_id": "export_job_1",
                "record_count": 1,
                "records": [
                    {
                        "record_id": "config_1",
                        "experiment_id": "exp_1",
                        "experiment_run_id": "run_1",
                        "configuration_run_id": "config_1",
                        "input_hash": "abc123",
                        "output_hash": "def456",
                        "input_ref": "configuration_run:config_1:input",
                        "output_ref": "configuration_run:config_1:output",
                        "input_content": None,
                        "output_content": None,
                        "materialization": "local_only",
                        "measure_summary": {"accuracy": 0.91},
                        "metadata": {"privacy_mode": True},
                    }
                ],
            }
        }

    client = CoreMetricsClient(request_sender=request_sender)
    manifest = client.export_fine_tuning_manifest(experiment_id="exp_1", limit=25)

    assert manifest.export_mode == "manifest"
    assert manifest.privacy_mode is True
    assert manifest.job_id == "export_job_1"
    assert manifest.records[0].input_content is None
    assert manifest.records[0].input_hash == "abc123"


def test_core_metrics_client_allows_missing_export_job_id() -> None:
    def request_sender(
        method: str,
        path: str,
        payload: dict | None,
        response_kind: str = "json",
    ):
        assert method == "GET"
        assert payload is None
        assert response_kind == "json"
        assert (
            path
            == "/analytics/exports/fine-tuning.manifest?experiment_id=exp_1&limit=25&include_content=false"
        )
        return {
            "data": {
                "context": {
                    "tenant_id": "tenant_acme",
                    "project_id": "project_alpha",
                    "generated_at": "2026-03-11T10:15:00Z",
                    "privacy_classification": "manifest_safe",
                },
                "export_mode": "manifest",
                "privacy_mode": True,
                "include_content": False,
                "job_id": None,
                "record_count": 1,
                "records": [],
            }
        }

    client = CoreMetricsClient(request_sender=request_sender)
    manifest = client.export_fine_tuning_manifest(experiment_id="exp_1", limit=25)

    assert manifest.job_id is None


def test_core_metrics_client_validates_custom_request_sender_shapes() -> None:
    client = CoreMetricsClient(request_sender=lambda *args, **kwargs: [])

    with pytest.raises(ClientError):
        client.get_analytics_summary()


def test_core_metrics_client_surfaces_missing_required_fields() -> None:
    def request_sender(
        method: str,
        path: str,
        payload: dict | None,
        response_kind: str = "json",
    ):
        assert method == "GET"
        assert payload is None
        assert response_kind == "json"
        assert path == "/analytics/summary?days=30"
        return {
            "data": {
                "range_days": 30,
                "entity_counts": {
                    "agents": 1,
                    "benchmarks": 1,
                    "measures": 1,
                    "experiments": 1,
                    "experiment_runs": 1,
                    "configuration_runs": 1,
                },
                "status_breakdowns": {
                    "experiments": {},
                    "experiment_runs": {},
                    "configuration_runs": {},
                },
                "usage_summary": {
                    "experiment_runs": 1,
                    "configuration_runs": 1,
                    "total_cost_usd": 0.1,
                    "avg_cost_usd": 0.1,
                    "total_tokens": 10,
                    "avg_latency_ms": 1.0,
                    "p95_latency_ms": 1.0,
                },
                "measure_summaries": [],
            }
        }

    client = CoreMetricsClient(request_sender=request_sender)

    with pytest.raises(KeyError, match="context"):
        client.get_analytics_summary()


def test_core_metrics_client_maps_auth_and_network_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    client = CoreMetricsClient(
        config=CoreMetricsConfig(
            backend_origin="https://backend.example",
            api_key="sk-test",  # pragma: allowlist secret
        )
    )

    with pytest.raises(AuthenticationError):
        def _raise_auth(*args, **kwargs):
            raise error.HTTPError(
                url="https://backend.example/api/v1beta/core-metrics/overview",
                code=403,
                msg="forbidden",
                hdrs=None,
                fp=None,
            )

        monkeypatch.setattr(client_module.request, "urlopen", _raise_auth)
        client.get_core_metrics_overview()

    with pytest.raises(TraigentConnectionError):
        def _raise_network(*args, **kwargs):
            raise error.URLError("offline")

        monkeypatch.setattr(client_module.request, "urlopen", _raise_network)
        client.get_core_metrics_overview()
