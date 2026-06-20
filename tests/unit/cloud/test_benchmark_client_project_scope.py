"""BenchmarkClient generation is project-scoped (Traigent/Traigent#1066).

With TRAIGENT_PROJECT_ID set, generation must hit the project-scoped backend
route (POST /api/v1beta/projects/{id}/benchmarks/generate) rather than the bare,
unscoped /api/v1/datasets/generate alias. With no project, behavior is unchanged.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

import traigent.cloud.benchmark_client as bc_module
from traigent.cloud.benchmark_client import BenchmarkClient, BenchmarkClientConfig
from traigent.utils.error_handler import OfflineModeError

ORIGIN = "https://api.example.test"


def _config(project_id):
    return BenchmarkClientConfig(
        backend_origin=ORIGIN, api_key="k", tenant_id=None, project_id=project_id
    )


# --- generate_path ---------------------------------------------------------


def test_generate_path_is_project_scoped_when_project_set():
    assert (
        _config("proj-123").generate_path()
        == "/api/v1beta/projects/proj-123/benchmarks/generate"
    )


def test_generate_path_unchanged_when_no_project():
    assert _config(None).generate_path() == "/api/v1/datasets/generate"


def test_generate_path_url_encodes_project_id():
    assert (
        _config("a/b c").generate_path()
        == "/api/v1beta/projects/a%2Fb%20c/benchmarks/generate"
    )


# --- end-to-end URL (capture the outbound Request) -------------------------


def _capture_urlopen(monkeypatch):
    """Patch urlopen to return a minimal benchmark response and capture the Request."""
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "false")
    captured = {}
    body = json.dumps(
        {
            "data": {
                "examples": [],
                "benchmark": {"id": "b1", "name": "n"},
                "status": "ok",
            }
        }
    ).encode()

    def fake_urlopen(http_req, timeout=None):
        captured["url"] = http_req.full_url
        cm = MagicMock()
        cm.__enter__.return_value = MagicMock(read=lambda: body)
        cm.__exit__.return_value = False
        return cm

    patcher = patch.object(bc_module.request, "urlopen", side_effect=fake_urlopen)
    return captured, patcher


def test_generate_sync_targets_project_scoped_route(monkeypatch):
    captured, patcher = _capture_urlopen(monkeypatch)
    client = BenchmarkClient(_config("proj-123"))
    with patcher:
        client.generate_sync(description="qa", count=5, use_case="question-answering")
    assert (
        captured["url"] == f"{ORIGIN}/api/v1beta/projects/proj-123/benchmarks/generate"
    )


def test_generate_sync_unscoped_route_when_no_project(monkeypatch):
    captured, patcher = _capture_urlopen(monkeypatch)
    client = BenchmarkClient(_config(None))
    with patcher:
        client.generate_sync(description="qa", count=5, use_case="question-answering")
    assert captured["url"] == f"{ORIGIN}/api/v1/datasets/generate"


def test_no_egress_blocks_benchmark_generation(monkeypatch):
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "false")
    client = BenchmarkClient(_config(None), no_egress=True)

    with patch.object(bc_module.request, "urlopen") as urlopen:
        with pytest.raises(OfflineModeError):
            client.generate_sync(
                description="qa", count=5, use_case="question-answering"
            )

    urlopen.assert_not_called()
