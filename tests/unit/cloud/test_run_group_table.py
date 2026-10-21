"""Tests for the run-group table (Traigent#1711 slice 1). HTTP is mocked."""

from __future__ import annotations

import importlib.util
from unittest.mock import AsyncMock, MagicMock

import pytest

HTTPX_AVAILABLE = importlib.util.find_spec("httpx") is not None

pytestmark = [
    pytest.mark.skipif(not HTTPX_AVAILABLE, reason="httpx not installed"),
    pytest.mark.backend_online,
]


def _row(run: str, cfg: str, measures: dict, **extra) -> dict:
    return {
        "configuration_run_id": cfg,
        "experiment_run_id": run,
        "experiment_id": "exp_1",
        "configuration": {"model": "m", "temperature": 0.1},
        "measures": measures,
        "status": "completed",
        "created_at": "2026-10-01T00:00:00Z",
        **extra,
    }


def _page(items: list[dict], page: int, total_pages: int) -> dict:
    return {
        "success": True,
        "data": {
            "items": items,
            "page": page,
            "page_size": 2,
            "total": 99,
            "total_pages": total_pages,
            "has_next": page < total_pages,
        },
    }


def _namespace(responses: list):
    """Namespace whose read client is the real one over a mocked httpx."""
    from traigent.cloud.analytics_client import BackendAnalyticsClient
    from traigent.cloud.backend_client import ExperimentGroupsNamespace

    reader = BackendAnalyticsClient(
        backend_url="http://localhost:5000",
        api_key="uk_test_key",  # pragma: allowlist secret
    )
    mocked = []
    for resp in responses:
        if isinstance(resp, Exception):
            mocked.append(resp)
            continue
        r = MagicMock()
        r.json.return_value = resp
        r.raise_for_status = MagicMock()
        mocked.append(r)
    http = AsyncMock()
    http.get.side_effect = mocked
    reader._client = http
    ns = ExperimentGroupsNamespace(MagicMock())
    ns._new_read_client = AsyncMock(return_value=reader)  # type: ignore[method-assign]
    return ns, http


@pytest.mark.asyncio
async def test_multi_run_group_merges_rows_with_run_ids_and_pages() -> None:
    ns, http = _namespace(
        [
            _page(
                [
                    _row("run_a", "c1", {"accuracy": 0.9, "cost": 0.02}, run_name="A"),
                    _row("run_b", "c2", {"accuracy": 0.8, "cost": 0.01}),
                ],
                1,
                2,
            ),
            _page([_row("run_c", "c3", {"accuracy": 0.7, "cost": 0.5})], 2, 2),
        ]
    )
    table = await ns.table("grp_1", "proj_abc", page_size=2)
    assert [r.run_id for r in table] == ["run_a", "run_b", "run_c"]
    assert table.rows[0].run_name == "A"
    assert table.rows[1].run_name is None
    assert table.rows[0].config == {"model": "m", "temperature": 0.1}
    assert table.rows[0].metrics == {"accuracy": 0.9}
    assert table.rows[0].cost == 0.02
    assert table.truncated is False
    assert http.get.await_count == 2
    assert http.get.await_args_list[1].kwargs["params"]["page"] == "2"


@pytest.mark.asyncio
async def test_unmeasured_cost_is_none_not_zero() -> None:
    ns, _ = _namespace(
        [
            _page(
                [
                    _row(
                        "run_a",
                        "c1",
                        {
                            "accuracy": 0.9,
                            "cost": 0.0,
                            "cost_unmeasured": 1.0,
                            "cost_lower_bound": 0.004,
                        },
                    ),
                    _row("run_b", "c2", {"accuracy": 0.8}),
                ],
                1,
                1,
            )
        ]
    )
    table = await ns.table("grp_1", "proj_abc")
    assert table.rows[0].cost is None
    assert table.rows[0].cost_unmeasured is True
    assert table.rows[0].cost_floor == 0.004
    assert table.rows[1].cost is None
    assert table.rows[1].cost_unmeasured is False


@pytest.mark.asyncio
async def test_empty_group() -> None:
    ns, _ = _namespace([_page([], 1, 0)])
    table = await ns.table("grp_1", "proj_abc")
    assert len(table) == 0
    assert table.to_dicts() == []


@pytest.mark.asyncio
async def test_max_rows_truncates() -> None:
    ns, http = _namespace(
        [
            _page([_row("r1", "c1", {}), _row("r2", "c2", {})], 1, 3),
            _page([_row("r3", "c3", {})], 2, 3),
        ]
    )
    table = await ns.table("grp_1", "proj_abc", page_size=2, max_rows=2)
    assert len(table) == 2
    assert table.truncated is True
    assert http.get.await_count == 1


@pytest.mark.asyncio
async def test_auth_error_is_value_free() -> None:
    import httpx

    from traigent.cloud.analytics_client import AnalyticsClientError

    request = httpx.Request("GET", "http://localhost:5000/secret-path?token=abc")
    err = httpx.HTTPStatusError(
        "401 for secret-path",
        request=request,
        response=httpx.Response(401, request=request),
    )
    ns, _ = _namespace([err])
    with pytest.raises(AnalyticsClientError) as info:
        await ns.table("grp_1", "proj_abc")
    assert "401" in str(info.value)
    assert "secret-path" not in str(info.value)
    assert "abc" not in str(info.value)


@pytest.mark.asyncio
async def test_rejects_blank_group_id() -> None:
    ns, _ = _namespace([])
    with pytest.raises(ValueError, match="group_id"):
        await ns.table("  ", "proj_abc")


def test_to_dataframe_keeps_missing_cost_missing() -> None:
    pd = pytest.importorskip("pandas")
    from traigent.cloud.dtos import GroupedConfigurationRunRowDTO
    from traigent.cloud.run_group_table import RunGroupTable, build_run_group_row

    dto = GroupedConfigurationRunRowDTO.from_dict(
        _row("run_a", "c1", {"accuracy": 0.9, "cost_unmeasured": True})
    )
    df = RunGroupTable("g", [build_run_group_row(dto)]).to_dataframe()
    assert list(df["run_id"]) == ["run_a"]
    assert pd.isna(df["cost"].iloc[0])
