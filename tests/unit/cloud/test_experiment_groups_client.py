"""Unit tests for experiment-group/cohort read DTOs and client methods."""

from __future__ import annotations

import builtins
import importlib.util
from unittest.mock import AsyncMock, MagicMock

import pytest

HTTPX_AVAILABLE = importlib.util.find_spec("httpx") is not None

pytestmark = pytest.mark.skipif(not HTTPX_AVAILABLE, reason="httpx not installed")


def _make_client():
    from traigent.cloud.analytics_client import BackendAnalyticsClient

    return BackendAnalyticsClient(
        backend_url="http://localhost:5000",
        api_key="uk_test_key",  # pragma: allowlist secret
    )


def _success_envelope(data: object) -> dict[str, object]:
    return {"success": True, "message": "ok", "data": data}


def _mock_get_response(client, data: object):
    mock_response = MagicMock()
    mock_response.json.return_value = _success_envelope(data)
    mock_response.raise_for_status = MagicMock()
    mock_http = AsyncMock()
    mock_http.get.return_value = mock_response
    client._client = mock_http
    return mock_http, mock_response


class TestExperimentGroupDTOs:
    def test_overview_preserves_nullable_dataset_and_source_ids(self) -> None:
        from traigent.cloud.dtos import ExperimentGroupOverviewDTO

        dto = ExperimentGroupOverviewDTO.from_dict(
            {
                "group_id": "grp_1",
                "name": "Production prompt cohort",
                "agent_id": "agent_1",
                "dataset_id": None,
                "experiment_count": 1,
                "experiment_run_count": 1,
                "configuration_run_count": 2,
                "first_experiment_created_at": "2026-06-29T10:00:00Z",
                "last_experiment_updated_at": "2026-06-29T10:30:00Z",
                "first_experiment_run_created_at": "2026-06-29T10:01:00Z",
                "last_experiment_run_updated_at": "2026-06-29T10:31:00Z",
                "status_summary": {
                    "experiment_run_status_counts": {"completed": 1},
                    "configuration_run_status_counts": {"completed": 2},
                },
                "source_experiments": [
                    {
                        "experiment_id": "exp_1",
                        "experiment_run_id": "run_1",
                        "name": "first run",
                    }
                ],
                "configuration_runs_count": 2,
                "backend_added": {"kept": True},
            }
        )

        assert dto.dataset_id is None
        assert dto.agent_id == "agent_1"
        assert dto.experiment_run_count == 1
        assert dto.configuration_run_count == 2
        assert dto.configuration_runs_count == 2
        assert dto.first_experiment_created_at == "2026-06-29T10:00:00Z"
        assert dto.last_experiment_run_updated_at == "2026-06-29T10:31:00Z"
        assert dto.status_summary.experiment_run_status_counts == {"completed": 1}
        assert dto.status_summary.configuration_run_status_counts == {"completed": 2}
        assert dto.source_experiments[0].experiment_id == "exp_1"
        assert dto.source_experiments[0].experiment_run_id == "run_1"
        assert dto.to_dict()["backend_added"] == {"kept": True}
        assert dto.to_dict()["configuration_run_count"] == 2

    def test_grouped_configuration_rows_do_not_dedupe_by_config_or_measures(
        self,
    ) -> None:
        from traigent.cloud.dtos import GroupedConfigurationRunsPageDTO

        payload = {
            "configuration_runs": [
                {
                    "configuration_run_id": "cfg_run_a",
                    "experiment_run_id": "exp_run_1",
                    "experiment_id": "exp_1",
                    "configuration": {"temperature": 0.2},
                    "measures": {"quality": 0.9},
                    "started_at": "2026-06-29T10:01:00Z",
                    "completed_at": "2026-06-29T10:02:00Z",
                },
                {
                    "configuration_run_id": "cfg_run_b",
                    "experiment_run_id": "exp_run_2",
                    "experiment_id": "exp_2",
                    "configuration": {"temperature": 0.2},
                    "measures": {"quality": 0.9},
                },
            ],
            "total": 2,
        }

        dto = GroupedConfigurationRunsPageDTO.from_dict(payload)

        assert [row.configuration_run_id for row in dto.items] == [
            "cfg_run_a",
            "cfg_run_b",
        ]
        assert [row.experiment_run_id for row in dto.items] == [
            "exp_run_1",
            "exp_run_2",
        ]
        assert [row.experiment_id for row in dto.items] == ["exp_1", "exp_2"]
        assert dto.items[0].started_at == "2026-06-29T10:01:00Z"
        assert dto.items[0].completed_at == "2026-06-29T10:02:00Z"
        assert dto.items[0].to_dict()["started_at"] == "2026-06-29T10:01:00Z"

    def test_grouped_configuration_rows_require_source_ids(self) -> None:
        from traigent.cloud.dtos import GroupedConfigurationRunRowDTO

        with pytest.raises(ValueError, match="experiment_run_id"):
            GroupedConfigurationRunRowDTO.from_dict(
                {
                    "configuration_run_id": "cfg_run_a",
                    "experiment_id": "exp_1",
                }
            )


class TestExperimentGroupClient:
    @pytest.mark.asyncio
    async def test_list_experiment_groups_calls_endpoint_without_null_dataset_filter(
        self,
    ) -> None:
        client = _make_client()
        mock_http, mock_response = _mock_get_response(
            client,
            {
                "items": [
                    {
                        "group_id": "grp_1",
                        "name": "Production prompt cohort",
                        "dataset_id": None,
                        "source_experiments": [
                            {
                                "experiment_id": "exp_1",
                                "experiment_run_id": "run_1",
                            }
                        ],
                    }
                ],
                "page": 2,
                "page_size": 10,
                "total": 1,
                "total_pages": 1,
            },
        )

        result = await client.list_experiment_groups(
            "proj_abc", dataset_id=None, page=2, page_size=10
        )

        assert result.items[0].dataset_id is None
        assert result.items[0].source_experiments[0].experiment_id == "exp_1"
        mock_response.raise_for_status.assert_called_once()
        mock_http.get.assert_called_once_with(
            "/api/v1/experiment-groups",
            headers={"X-Project-Id": "proj_abc"},
            params={"page": "2", "page_size": "10"},
        )
        mock_http.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_list_experiment_groups_sends_agent_and_dataset_filters_when_present(
        self,
    ) -> None:
        client = _make_client()
        mock_http, _ = _mock_get_response(client, {"experiment_groups": []})

        await client.list_experiment_groups(
            "proj_abc",
            agent_id="agent_1",
            dataset_id="dataset_1",
        )

        mock_http.get.assert_called_once_with(
            "/api/v1/experiment-groups",
            headers={"X-Project-Id": "proj_abc"},
            params={
                "agent_id": "agent_1",
                "dataset_id": "dataset_1",
                "page": "1",
                "page_size": "50",
            },
        )

    @pytest.mark.asyncio
    async def test_get_experiment_group_url_encodes_id_and_preserves_rows(
        self,
    ) -> None:
        client = _make_client()
        mock_http, _ = _mock_get_response(
            client,
            {
                "group_id": "group/with slash",
                "name": "Grouped runs",
                "grouped_configurations": [
                    {
                        "configuration_run_id": "cfg_run_a",
                        "experiment_run_id": "exp_run_1",
                        "experiment_id": "exp_1",
                        "configuration": {"model": "gpt"},
                    },
                    {
                        "configuration_run_id": "cfg_run_b",
                        "experiment_run_id": "exp_run_1",
                        "experiment_id": "exp_1",
                        "configuration": {"model": "gpt"},
                    },
                ],
            },
        )

        result = await client.get_experiment_group("group/with slash", "proj_abc")

        assert [row.configuration_run_id for row in result.grouped_configurations] == [
            "cfg_run_a",
            "cfg_run_b",
        ]
        mock_http.get.assert_called_once_with(
            "/api/v1/experiment-groups/group%2Fwith%20slash",
            headers={"X-Project-Id": "proj_abc"},
            params=None,
        )

    @pytest.mark.asyncio
    async def test_get_experiment_group_parses_canonical_detail_shape(
        self,
    ) -> None:
        client = _make_client()
        _mock_get_response(
            client,
            {
                "group": {
                    "group_id": "grp_1",
                    "name": "Canonical cohort",
                    "agent_id": "agent_1",
                    "dataset_id": None,
                    "experiment_count": 2,
                    "experiment_run_count": 3,
                    "configuration_run_count": 5,
                    "first_experiment_created_at": "2026-06-29T10:00:00Z",
                    "last_experiment_updated_at": "2026-06-29T11:00:00Z",
                    "first_experiment_run_created_at": "2026-06-29T10:05:00Z",
                    "last_experiment_run_updated_at": "2026-06-29T11:05:00Z",
                    "status_summary": {
                        "experiment_run_status_counts": {
                            "completed": 2,
                            "failed": 1,
                        },
                        "configuration_run_status_counts": {
                            "completed": 4,
                            "failed": 1,
                        },
                    },
                },
                "source_experiments": [
                    {
                        "experiment_id": "exp_1",
                        "experiment_run_id": "run_1",
                    },
                    {
                        "experiment_id": "exp_2",
                        "experiment_run_id": "run_2",
                    },
                ],
            },
        )

        result = await client.get_experiment_group("grp_1", "proj_abc")

        assert result.group_id == "grp_1"
        assert result.agent_id == "agent_1"
        assert result.dataset_id is None
        assert result.experiment_count == 2
        assert result.experiment_run_count == 3
        assert result.configuration_run_count == 5
        assert result.first_experiment_created_at == "2026-06-29T10:00:00Z"
        assert result.last_experiment_run_updated_at == "2026-06-29T11:05:00Z"
        assert result.status_summary.experiment_run_status_counts == {
            "completed": 2,
            "failed": 1,
        }
        assert result.status_summary.configuration_run_status_counts == {
            "completed": 4,
            "failed": 1,
        }
        assert [row.experiment_id for row in result.source_experiments] == [
            "exp_1",
            "exp_2",
        ]
        assert result.to_dict()["configuration_run_count"] == 5

    @pytest.mark.asyncio
    async def test_list_group_configuration_runs_calls_endpoint_with_pagination(
        self,
    ) -> None:
        client = _make_client()
        mock_http, _ = _mock_get_response(
            client,
            {
                "configuration_runs": [
                    {
                        "configuration_run_id": "cfg_run_1",
                        "experiment_run_id": "exp_run_1",
                        "experiment_id": "exp_1",
                        "started_at": "2026-06-29T10:01:00Z",
                        "completed_at": "2026-06-29T10:02:00Z",
                    }
                ],
                "page": 3,
                "page_size": 25,
                "total": 1,
            },
        )

        result = await client.list_experiment_group_configuration_runs(
            "grp 1", "proj_abc", page=3, page_size=25
        )

        assert result.items[0].configuration_run_id == "cfg_run_1"
        assert result.items[0].experiment_run_id == "exp_run_1"
        assert result.items[0].experiment_id == "exp_1"
        assert result.items[0].started_at == "2026-06-29T10:01:00Z"
        assert result.items[0].completed_at == "2026-06-29T10:02:00Z"
        mock_http.get.assert_called_once_with(
            "/api/v1/experiment-groups/grp%201/configuration-runs",
            headers={"X-Project-Id": "proj_abc"},
            params={"page": "3", "page_size": "25"},
        )
        mock_http.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_experiment_group_reads_require_project_id_before_http(
        self,
    ) -> None:
        client = _make_client()
        mock_http = AsyncMock()
        client._client = mock_http

        with pytest.raises(TypeError):
            await client.list_experiment_groups()
        with pytest.raises(TypeError):
            await client.get_experiment_group("grp_1")
        with pytest.raises(TypeError):
            await client.list_experiment_group_configuration_runs("grp_1")

        mock_http.get.assert_not_called()
        mock_http.post.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("method_name", "args"),
        [
            ("list_experiment_groups", ("  ",)),
            ("get_experiment_group", ("grp_1", "  ")),
            ("list_experiment_group_configuration_runs", ("grp_1", "  ")),
        ],
    )
    async def test_experiment_group_reads_reject_empty_project_id_before_http(
        self,
        method_name: str,
        args: tuple[str, ...],
    ) -> None:
        client = _make_client()
        mock_http = AsyncMock()
        client._client = mock_http

        with pytest.raises(ValueError, match="project_id"):
            await getattr(client, method_name)(*args)

        mock_http.get.assert_not_called()
        mock_http.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_experiment_group_reads_do_not_import_sync_manager(
        self,
    ) -> None:
        client = _make_client()
        mock_http, _ = _mock_get_response(client, {"items": []})
        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name == "traigent.cloud.sync_manager":
                raise AssertionError("experiment group reads touched sync_manager")
            return original_import(name, *args, **kwargs)

        try:
            builtins.__import__ = guarded_import
            await client.list_experiment_groups("proj_abc")
        finally:
            builtins.__import__ = original_import

        mock_http.get.assert_called_once()
        mock_http.post.assert_not_called()
