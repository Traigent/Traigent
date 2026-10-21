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


def _row(run: str, cfg: str, measures: dict | None, **extra) -> dict:
    """Browse row, field set of experiment_group_service._serialize_browse_row."""
    return {
        "configuration_run_id": cfg,
        "experiment_run_id": run,
        "experiment_id": "exp_1",
        "run_label": None,
        "trial_number": 1,
        "run_name": extra.pop("run_name", f"name-{run}"),
        "status": "completed",
        "created_at": "2026-10-01T00:00:00Z",
        "updated_at": "2026-10-01T00:00:00Z",
        "started_at": None,
        "completed_at": None,
        "configuration": {"model": "m", "temperature": 0.1},
        "parameters": {},
        "measures": measures,
        "summary_stats": None,
        "error_state": {
            "has_error": False,
            "error_code": None,
            "error_message_available": False,
        },
        "provenance": {},
        "comparison_state": None,
        **extra,
    }


def _page(items: list[dict], next_cursor: str | None = None) -> dict:
    """Cursor-mode envelope (list_group_configuration_runs_cursor)."""
    return {
        "success": True,
        "message": "ok",
        "data": {
            "items": items,
            "column_manifest": {},
            "cursor": {"next_cursor": next_cursor, "has_more": next_cursor is not None},
        },
    }


def _legacy_page(items: list[dict], has_next: bool) -> dict:
    """Legacy page-mode envelope (response_handler.paginated_response)."""
    return {
        "success": True,
        "message": "ok",
        "data": {
            "items": items,
            "pagination": {
                "page": 1,
                "per_page": 20,
                "total": 99,
                "total_pages": 5,
                "has_next": has_next,
                "has_prev": False,
            },
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
async def test_multi_run_group_follows_cursor_and_merges_run_ids() -> None:
    ns, http = _namespace(
        [
            _page(
                [
                    _row("run_a", "c1", {"accuracy": 0.9, "cost": 0.02}),
                    _row("run_b", "c2", {"accuracy": 0.8, "cost": 0.01}),
                ],
                next_cursor="cur_2",
            ),
            _page([_row("run_c", "c3", {"accuracy": 0.7, "cost": 0.5})]),
        ]
    )
    table = await ns.table("grp_1", "proj_abc", page_size=2)
    assert [r.run_id for r in table] == ["run_a", "run_b", "run_c"]
    assert table.rows[0].run_name == "name-run_a"
    assert table.rows[0].config == {"model": "m", "temperature": 0.1}
    assert table.rows[0].metrics == {"accuracy": 0.9}
    assert table.rows[0].cost == 0.02
    assert table.truncated is False
    first, second = (c.kwargs["params"] for c in http.get.await_args_list)
    assert first == {"limit": "2"}
    assert second == {"cursor": "cur_2", "limit": "2"}


@pytest.mark.asyncio
async def test_table_does_not_use_legacy_page_mode() -> None:
    """Legacy rows carry no measures; the table must request cursor mode."""
    ns, http = _namespace([_page([_row("run_a", "c1", {"cost": 0.02})])])
    await ns.table("grp_1", "proj_abc")
    params = http.get.await_args.kwargs["params"]
    assert "cursor" in params or "limit" in params
    assert "page" not in params and "per_page" not in params


@pytest.mark.asyncio
async def test_unmeasured_and_unpriced_cost_is_none_not_zero() -> None:
    ns, _ = _namespace(
        [
            _page(
                [
                    _row(
                        "run_a",
                        "c1",
                        {"accuracy": 0.9, "cost": 0.0, "cost_unmeasured": 1.0},
                    ),
                    _row("run_b", "c2", {"accuracy": 0.8}),
                    _row(
                        "run_c",
                        "c3",
                        {"accuracy": 0.7, "cost": 0.0, "cost_unpriced": 1.0},
                    ),
                    _row("run_d", "c4", None),
                ]
            )
        ]
    )
    table = await ns.table("grp_1", "proj_abc")
    assert [r.cost for r in table] == [None, None, None, None]
    assert [r.cost_unmeasured for r in table] == [True, False, True, False]
    assert all(r.cost_floor is None for r in table)
    assert table.rows[3].metrics == {}


@pytest.mark.asyncio
async def test_empty_group() -> None:
    ns, _ = _namespace([_page([])])
    table = await ns.table("grp_1", "proj_abc")
    assert len(table) == 0
    assert table.truncated is False


@pytest.mark.asyncio
async def test_max_rows_truncates_when_more_remain() -> None:
    ns, http = _namespace(
        [_page([_row("r1", "c1", {}), _row("r2", "c2", {})], next_cursor="cur_2")]
    )
    table = await ns.table("grp_1", "proj_abc", page_size=2, max_rows=2)
    assert len(table) == 2
    assert table.truncated is True
    assert http.get.await_count == 1


@pytest.mark.asyncio
async def test_max_rows_exact_fit_is_not_truncated() -> None:
    ns, _ = _namespace([_page([_row("r1", "c1", {}), _row("r2", "c2", {})])])
    table = await ns.table("grp_1", "proj_abc", page_size=2, max_rows=2)
    assert len(table) == 2
    assert table.truncated is False


@pytest.mark.asyncio
async def test_legacy_page_dto_reads_nested_pagination_and_sends_per_page() -> None:
    ns, http = _namespace([_legacy_page([_row("r1", "c1", None)], has_next=True)])
    reader = await ns._new_read_client()
    page = await reader.list_experiment_group_configuration_runs(
        "grp_1", "proj_abc", page=1, page_size=100
    )
    assert page.has_next is True
    assert page.total == 99
    assert page.page_size == 20
    assert http.get.await_args.kwargs["params"] == {"page": "1", "per_page": "100"}


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
async def test_transport_error_is_wrapped_value_free() -> None:
    import httpx

    from traigent.cloud.analytics_client import AnalyticsClientError

    boom = httpx.ConnectError(
        "could not reach https://secret-host.example/path?tok=zzz"
    )
    ns, _ = _namespace([boom])
    with pytest.raises(AnalyticsClientError) as info:
        await ns.table("grp_1", "proj_abc")
    assert "secret-host" not in str(info.value)
    assert "zzz" not in str(info.value)
    assert info.value.__cause__ is None


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
