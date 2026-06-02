"""Standalone clients must honor TRAIGENT_OFFLINE_MODE (Traigent/Traigent#1068).

`is_backend_offline()` promises that with TRAIGENT_OFFLINE_MODE=true "all
communication with the Traigent backend is skipped". These tests assert each of
the seven standalone backend clients fails closed (raises OfflineModeError) and
issues NO outbound HTTP in offline mode, and that behavior is unchanged when the
switch is off.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from traigent.utils.env_config import raise_if_backend_offline
from traigent.utils.error_handler import OfflineModeError, TraigentError

OFFLINE = "TRAIGENT_OFFLINE_MODE"


# --- guard helper ----------------------------------------------------------


def test_guard_raises_when_offline(monkeypatch):
    monkeypatch.setenv(OFFLINE, "true")
    with pytest.raises(OfflineModeError):
        raise_if_backend_offline("unit op")


def test_guard_noop_when_off(monkeypatch):
    monkeypatch.setenv(OFFLINE, "false")
    assert raise_if_backend_offline("unit op") is None
    monkeypatch.delenv(OFFLINE, raising=False)
    assert raise_if_backend_offline("unit op") is None


def test_offline_error_is_traigent_error():
    assert issubclass(OfflineModeError, TraigentError)


# --- per-client: no egress in offline mode ---------------------------------


def _evaluation():
    from traigent.evaluation.client import EvaluationClient

    return EvaluationClient(), lambda c: c._request_json_sync("GET", "/x")


def _prompts():
    from traigent.prompts.client import PromptManagementClient

    return PromptManagementClient(), lambda c: c._request_json_sync("GET", "/x")


def _admin():
    from traigent.admin.client import EnterpriseAdminClient

    return EnterpriseAdminClient(), lambda c: c._request_json_sync("GET", "/x")


def _projects():
    from traigent.projects.client import ProjectManagementClient

    return ProjectManagementClient(), lambda c: c._request_json_sync("GET", "/x")


def _core_metrics_json():
    from traigent.core_metrics.client import CoreMetricsClient

    return CoreMetricsClient(), lambda c: c._request_json_sync("GET", "/x")


def _core_metrics_text():
    from traigent.core_metrics.client import CoreMetricsClient

    return CoreMetricsClient(), lambda c: c._request_text_sync("GET", "/x")


def _benchmark():
    from traigent.cloud.benchmark_client import BenchmarkClient

    return BenchmarkClient(), lambda c: c._post("/x", {})


# (module path to patch `request.urlopen`, factory) for the urlopen clients
URLOPEN_CLIENTS = [
    ("traigent.evaluation.client", _evaluation),
    ("traigent.prompts.client", _prompts),
    ("traigent.admin.client", _admin),
    ("traigent.projects.client", _projects),
    ("traigent.core_metrics.client", _core_metrics_json),
    ("traigent.core_metrics.client", _core_metrics_text),
    ("traigent.cloud.benchmark_client", _benchmark),
]


@pytest.mark.parametrize("module_path, factory", URLOPEN_CLIENTS)
def test_urlopen_client_fails_closed_offline(monkeypatch, module_path, factory):
    monkeypatch.setenv(OFFLINE, "true")
    client, call = factory()
    with patch(f"{module_path}.request.urlopen") as urlopen:
        with pytest.raises(OfflineModeError):
            call(client)
    urlopen.assert_not_called()


def test_example_insights_fails_closed_offline(monkeypatch):
    monkeypatch.setenv(OFFLINE, "true")
    from traigent.analytics.example_insights import ExampleInsightsClient

    client = ExampleInsightsClient()
    with patch("traigent.analytics.example_insights.httpx.AsyncClient") as async_client:
        with pytest.raises(OfflineModeError):
            client._get_client()
    async_client.assert_not_called()


# --- behavior unchanged when offline mode is off ---------------------------


class _TransportReached(Exception):
    pass


def test_transport_is_reached_when_not_offline(monkeypatch):
    """With offline mode off, the guard is a no-op and the request proceeds to
    the transport (here forced to raise a sentinel so we don't parse a response)."""
    monkeypatch.delenv(OFFLINE, raising=False)
    client, call = _evaluation()
    with patch(
        "traigent.evaluation.client.request.urlopen",
        MagicMock(side_effect=_TransportReached),
    ) as urlopen:
        with pytest.raises(Exception) as excinfo:
            call(client)
    # the guard did NOT block — transport was invoked
    urlopen.assert_called_once()
    assert not isinstance(excinfo.value, OfflineModeError)
