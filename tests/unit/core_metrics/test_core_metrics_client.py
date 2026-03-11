from __future__ import annotations

from urllib import error

import pytest

import traigent.core_metrics.client as client_module
from traigent.core_metrics import CoreMetricsClient, CoreMetricsConfig
from traigent.utils.exceptions import AuthenticationError, TraigentConnectionError


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
