"""Unit tests for the auto-pagination ``iter_*`` helpers (#1442).

Coverage goals
--------------
* 3-page fixture: all items are yielded exactly once and in order.
* Single-page fixture: terminates after one request.
* Empty first-page fixture: yields nothing, makes exactly one request.
* Infinite-loop guard: a backend that always returns has_next=True but
  never advances the page number terminates cleanly (no duplicates).
* Existing single-page ``list_*`` signatures are unchanged (smoke).
* The generic ``iter_pages`` helper is also exercised directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs

import pytest

from traigent.utils.pagination import iter_pages

# ---------------------------------------------------------------------------
# Helpers — minimal fake DTOs and path-parsing utilities
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FakePagination:
    page: int
    per_page: int
    total: int
    total_pages: int
    has_next: bool
    has_prev: bool


@dataclass(frozen=True)
class _FakeItem:
    id: str


@dataclass(frozen=True)
class _FakeListResponse:
    items: list[_FakeItem]
    pagination: _FakePagination


def _make_response(
    page: int,
    per_page: int,
    items: list[str],
    total: int,
    total_pages: int,
    has_next: bool,
) -> _FakeListResponse:
    return _FakeListResponse(
        items=[_FakeItem(id=i) for i in items],
        pagination=_FakePagination(
            page=page,
            per_page=per_page,
            total=total,
            total_pages=total_pages,
            has_next=has_next,
            has_prev=page > 1,
        ),
    )


def _page_from_path(path: str) -> int:
    """Extract the ``page`` query parameter from a URL path string.

    Handles both ``/api/v1beta/resource?page=2&per_page=2`` and bare
    ``?page=2&per_page=2`` forms.  Returns 1 if the parameter is absent.
    """
    # Strip a leading scheme+host so urlparse doesn't confuse bare paths.
    qs = path.split("?", 1)[1] if "?" in path else ""
    params = parse_qs(qs)
    return int(params.get("page", ["1"])[0])


# ---------------------------------------------------------------------------
# iter_pages — core helper
# ---------------------------------------------------------------------------


class TestIterPages:
    """Tests for the generic :func:`traigent.utils.pagination.iter_pages`."""

    def test_three_pages_all_items_yielded(self):
        """All items across 3 pages are returned in order."""
        responses = {
            1: _make_response(1, 2, ["a", "b"], 6, 3, has_next=True),
            2: _make_response(2, 2, ["c", "d"], 6, 3, has_next=True),
            3: _make_response(3, 2, ["e", "f"], 6, 3, has_next=False),
        }
        call_log: list[int] = []

        def list_fn(*, page: int, per_page: int, **_filters: Any):
            call_log.append(page)
            return responses[page]

        result = list(iter_pages(list_fn, per_page=2))
        assert [item.id for item in result] == ["a", "b", "c", "d", "e", "f"]
        assert call_log == [1, 2, 3], "exactly 3 requests — one per page"

    def test_single_page_terminates_after_one_request(self):
        """A single-page result stops immediately."""
        call_log: list[int] = []

        def list_fn(*, page: int, per_page: int, **_filters: Any):
            call_log.append(page)
            return _make_response(1, 20, ["x", "y"], 2, 1, has_next=False)

        result = list(iter_pages(list_fn, per_page=20))
        assert [item.id for item in result] == ["x", "y"]
        assert call_log == [1]

    def test_empty_first_page_yields_nothing(self):
        """An empty first page produces an empty iterator with one request."""
        call_log: list[int] = []

        def list_fn(*, page: int, per_page: int, **_filters: Any):
            call_log.append(page)
            return _make_response(1, 100, [], 0, 0, has_next=False)

        result = list(iter_pages(list_fn, per_page=100))
        assert result == []
        assert call_log == [1]

    def test_infinite_loop_guard_aborts_on_stale_page(self):
        """If the server always returns has_next=True with the same page number,
        the guard detects the stall and stops without emitting duplicates."""
        call_count = 0

        def list_fn(*, page: int, per_page: int, **_filters: Any):
            nonlocal call_count
            call_count += 1
            # Always returns page=1, has_next=True regardless of the requested page.
            return _make_response(1, 10, ["z"], 0, 99, has_next=True)

        result = list(iter_pages(list_fn, per_page=10))

        # First call yields the item; the second call sees pagination.page=1
        # (already seen) and the guard fires before yielding — so exactly 1
        # item is emitted and exactly 2 requests are made.
        assert call_count == 2, (
            f"guard should fire on the second call; got {call_count} calls"
        )
        assert [item.id for item in result] == ["z"], (
            "no duplicate items should be emitted"
        )

    def test_filters_forwarded_on_every_page(self):
        """Keyword filters are passed through verbatim on every page request."""
        received_args: list[dict[str, Any]] = []

        def list_fn(*, page: int, per_page: int, **filters: Any):
            received_args.append({"page": page, "per_page": per_page, **filters})
            has_next = page < 2
            return _make_response(page, per_page, ["i"], 2, 2, has_next=has_next)

        list(iter_pages(list_fn, per_page=50, search="hello", status="active"))

        assert len(received_args) == 2
        for args in received_args:
            assert args["search"] == "hello"
            assert args["status"] == "active"
            assert args["per_page"] == 50

    def test_positional_args_forwarded_on_every_page(self):
        """Positional arguments (e.g. queue_id) are forwarded on every call."""
        received_positional: list[tuple] = []

        def list_fn(queue_id, *, page: int, per_page: int, **_filters: Any):
            received_positional.append((queue_id,))
            has_next = page < 2
            return _make_response(page, per_page, ["j"], 2, 2, has_next=has_next)

        list(iter_pages(list_fn, "queue-42", per_page=10))

        assert len(received_positional) == 2
        for pos in received_positional:
            assert pos == ("queue-42",)


# ---------------------------------------------------------------------------
# EvaluationClient — iter_* smoke tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def _online(monkeypatch):
    """These tests use a mock request sender; opt out of offline mode."""
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "false")


def _eval_list_item(idx: int) -> dict[str, Any]:
    return {
        "id": f"ev_{idx}",
        "name": f"Evaluator {idx}",
        "description": "",
        "measure_id": "m1",
        "primary_measure_id": "m1",
        "target_type": "observability_trace",
        "judge_config": {
            "instructions": "Judge",
            "model_id": "gpt-4.1-mini",
            "context_type": "none",
            "parameters": {},
        },
        "sampling_rate": 1.0,
        "target_filters": {},
        "is_active": True,
        "measure": None,
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
    }


def _eval_page_response(page: int, total_pages: int) -> dict[str, Any]:
    idx = (page - 1) * 2 + 1
    return {
        "data": {
            "items": [_eval_list_item(idx), _eval_list_item(idx + 1)],
            "pagination": {
                "page": page,
                "per_page": 2,
                "total": total_pages * 2,
                "total_pages": total_pages,
                "has_next": page < total_pages,
                "has_prev": page > 1,
            },
        }
    }


class TestEvaluationClientIter:
    @pytest.fixture(autouse=True)
    def _use_online(self, _online):
        pass

    def test_iter_evaluators_three_pages(self):
        """iter_evaluators yields all items across 3 pages."""
        from traigent.evaluation import EvaluationClient

        pages_requested: list[int] = []

        def sender(method: str, path: str, payload: Any):
            p = _page_from_path(path)
            pages_requested.append(p)
            return _eval_page_response(p, 3)

        client = EvaluationClient(request_sender=sender)
        result = list(client.iter_evaluators(per_page=2))
        assert len(result) == 6
        assert result[0].id == "ev_1"
        assert result[5].id == "ev_6"
        assert pages_requested == [1, 2, 3]

    def test_list_evaluators_still_returns_single_page(self):
        """Existing list_evaluators is unchanged — returns a single page."""
        from traigent.evaluation import EvaluationClient

        def sender(method: str, path: str, payload: Any):
            return _eval_page_response(1, 3)

        client = EvaluationClient(request_sender=sender)
        resp = client.list_evaluators()
        # Returns a single page object, not an iterator
        assert hasattr(resp, "pagination")
        assert hasattr(resp, "items")
        assert len(resp.items) == 2  # only first page

    def test_iter_evaluators_empty_result(self):
        """iter_evaluators on an empty collection yields nothing."""
        from traigent.evaluation import EvaluationClient

        def sender(method: str, path: str, payload: Any):
            return {
                "data": {
                    "items": [],
                    "pagination": {
                        "page": 1,
                        "per_page": 100,
                        "total": 0,
                        "total_pages": 0,
                        "has_next": False,
                        "has_prev": False,
                    },
                }
            }

        client = EvaluationClient(request_sender=sender)
        result = list(client.iter_evaluators())
        assert result == []

    def test_iter_evaluator_runs_two_pages(self):
        """iter_evaluator_runs yields all items across 2 pages."""
        from traigent.evaluation import EvaluationClient

        def _run_item(idx: int) -> dict[str, Any]:
            return {
                "id": f"run_{idx}",
                "evaluator_id": "ev_1",
                "target_type": "observability_trace",
                "target_id": f"trace_{idx}",
                "status": "completed",
                "score": None,
                "raw_output": None,
                "error": None,
                "started_at": None,
                "completed_at": None,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
            }

        pages_seen: list[int] = []

        def sender(method: str, path: str, payload: Any):
            p = _page_from_path(path)
            pages_seen.append(p)
            idx = (p - 1) * 2 + 1
            return {
                "data": {
                    "items": [_run_item(idx), _run_item(idx + 1)],
                    "pagination": {
                        "page": p,
                        "per_page": 2,
                        "total": 4,
                        "total_pages": 2,
                        "has_next": p < 2,
                        "has_prev": p > 1,
                    },
                }
            }

        client = EvaluationClient(request_sender=sender)
        result = list(client.iter_evaluator_runs(per_page=2))
        assert len(result) == 4
        assert pages_seen == [1, 2]

    def test_iter_annotation_queue_items_positional_queue_id(self):
        """iter_annotation_queue_items forwards queue_id on every request."""
        from traigent.evaluation import EvaluationClient

        seen_paths: list[str] = []

        def _item(idx: int) -> dict[str, Any]:
            return {
                "id": f"item_{idx}",
                "queue_id": "q1",
                "target_type": "observability_trace",
                "target_id": f"trace_{idx}",
                "status": "pending",
                "assigned_user_id": None,
                "scores": [],
                "note": None,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
            }

        def sender(method: str, path: str, payload: Any):
            seen_paths.append(path)
            p = _page_from_path(path)
            idx = (p - 1) * 2 + 1
            return {
                "data": {
                    "items": [_item(idx), _item(idx + 1)],
                    "pagination": {
                        "page": p,
                        "per_page": 2,
                        "total": 4,
                        "total_pages": 2,
                        "has_next": p < 2,
                        "has_prev": p > 1,
                    },
                }
            }

        client = EvaluationClient(request_sender=sender)
        result = list(client.iter_annotation_queue_items("q1", per_page=2))
        assert len(result) == 4
        # Both paths should reference the queue ID
        for p in seen_paths:
            assert "annotation-queues/q1/items" in p


# ---------------------------------------------------------------------------
# ObservabilityClient — iter_traces and iter_sessions smoke
# ---------------------------------------------------------------------------


def _trace_item(idx: int) -> dict[str, Any]:
    return {
        "id": f"trace_{idx}",
        "name": f"Trace {idx}",
        "start_time": "2026-01-01T00:00:00+00:00",
        "end_time": "2026-01-01T00:01:00+00:00",
        "status": "ok",
        "environment": "production",
        "tags": [],
        "bookmarked": False,
        "user_id": None,
        "session_id": None,
        "release": None,
        "model": None,
        "input": {},
        "output": {},
        "metadata": {},
        "latency_ms": 1000,
        "total_tokens": 100,
        "prompt_tokens": 80,
        "completion_tokens": 20,
        "cost_usd": None,
    }


def _trace_page_response(page: int, total_pages: int) -> dict[str, Any]:
    idx = (page - 1) * 2 + 1
    return {
        "data": {
            "items": [_trace_item(idx), _trace_item(idx + 1)],
            "pagination": {
                "page": page,
                "per_page": 2,
                "total": total_pages * 2,
                "total_pages": total_pages,
                "has_next": page < total_pages,
                "has_prev": page > 1,
            },
        }
    }


class TestObservabilityClientIter:
    def test_iter_traces_two_pages(self, monkeypatch):
        """iter_traces yields all items across 2 pages."""
        monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "false")
        from traigent.observability.client import ObservabilityClient

        def sender(traces: list) -> dict | None:
            return None  # ingest path — unused

        client = ObservabilityClient(sender=sender, request_sender=None)

        pages_seen: list[int] = []

        def _request_json_mock(method: str, path: str, payload=None):
            p = _page_from_path(path)
            pages_seen.append(p)
            return _trace_page_response(p, 2)

        client._request_json = _request_json_mock  # type: ignore[method-assign]

        result = list(client.iter_traces(per_page=2))
        assert len(result) == 4
        assert pages_seen == [1, 2]

    def test_list_traces_unchanged(self, monkeypatch):
        """list_traces still returns a single-page response object."""
        monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "false")
        from traigent.observability.client import ObservabilityClient

        client = ObservabilityClient()

        def _request_json_mock(method: str, path: str, payload=None):
            return _trace_page_response(1, 5)

        client._request_json = _request_json_mock  # type: ignore[method-assign]

        resp = client.list_traces()
        assert hasattr(resp, "pagination")
        assert resp.pagination.has_next is True
        assert len(resp.items) == 2  # only first page returned

    def test_iter_sessions_two_pages(self, monkeypatch):
        """iter_sessions yields all items across 2 pages."""
        monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "false")
        from traigent.observability.client import ObservabilityClient

        def _session_item(idx: int) -> dict[str, Any]:
            return {
                "id": f"session_{idx}",
                "environment": "production",
                "user_id": None,
                "tags": [],
                "release": None,
                "trace_count": 1,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
            }

        def _session_page(page: int, total_pages: int) -> dict[str, Any]:
            idx = (page - 1) * 2 + 1
            return {
                "data": {
                    "items": [_session_item(idx), _session_item(idx + 1)],
                    "pagination": {
                        "page": page,
                        "per_page": 2,
                        "total": total_pages * 2,
                        "total_pages": total_pages,
                        "has_next": page < total_pages,
                        "has_prev": page > 1,
                    },
                }
            }

        client = ObservabilityClient()
        pages_seen: list[int] = []

        def _request_json_mock(method: str, path: str, payload=None):
            p = _page_from_path(path)
            pages_seen.append(p)
            return _session_page(p, 2)

        client._request_json = _request_json_mock  # type: ignore[method-assign]

        result = list(client.iter_sessions(per_page=2))
        assert len(result) == 4
        assert pages_seen == [1, 2]


# ---------------------------------------------------------------------------
# PromptManagementClient — iter_prompts smoke
# ---------------------------------------------------------------------------


def _prompt_item(idx: int) -> dict[str, Any]:
    return {
        "name": f"prompt_{idx}",
        "description": None,
        "prompt_type": "text",
        "labels": [],
        "version_count": 1,
        "latest_version": 1,
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
    }


def _prompt_page_response(page: int, total_pages: int) -> dict[str, Any]:
    idx = (page - 1) * 2 + 1
    return {
        "data": {
            "items": [_prompt_item(idx), _prompt_item(idx + 1)],
            "pagination": {
                "page": page,
                "per_page": 2,
                "total": total_pages * 2,
                "total_pages": total_pages,
                "has_next": page < total_pages,
                "has_prev": page > 1,
            },
        }
    }


class TestPromptClientIter:
    def test_iter_prompts_two_pages(self, monkeypatch):
        """iter_prompts yields all items across 2 pages."""
        monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "false")
        from traigent.prompts.client import PromptManagementClient

        pages_seen: list[int] = []

        def sender(method: str, path: str, payload=None):
            p = _page_from_path(path)
            pages_seen.append(p)
            return _prompt_page_response(p, 2)

        client = PromptManagementClient(request_sender=sender)
        result = list(client.iter_prompts(per_page=2))
        assert len(result) == 4
        assert pages_seen == [1, 2]

    def test_list_prompts_unchanged(self, monkeypatch):
        """list_prompts still returns a single-page response object."""
        monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "false")
        from traigent.prompts.client import PromptManagementClient

        def sender(method: str, path: str, payload=None):
            return _prompt_page_response(1, 3)

        client = PromptManagementClient(request_sender=sender)
        resp = client.list_prompts()
        assert hasattr(resp, "pagination")
        assert resp.pagination.has_next is True


# ---------------------------------------------------------------------------
# ProjectManagementClient — iter_projects smoke
# ---------------------------------------------------------------------------


def _project_item(idx: int) -> dict[str, Any]:
    return {
        "id": f"proj_{idx}",
        "name": f"Project {idx}",
        "slug": f"project-{idx}",
        "description": None,
        "status": "active",
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
    }


def _project_page_response(page: int, total_pages: int) -> dict[str, Any]:
    idx = (page - 1) * 2 + 1
    return {
        "data": {
            "items": [_project_item(idx), _project_item(idx + 1)],
            "pagination": {
                "page": page,
                "per_page": 2,
                "total": total_pages * 2,
                "total_pages": total_pages,
                "has_next": page < total_pages,
                "has_prev": page > 1,
            },
        }
    }


class TestProjectClientIter:
    def test_iter_projects_two_pages(self, monkeypatch):
        """iter_projects yields all items across 2 pages."""
        monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "false")
        from traigent.projects.client import ProjectManagementClient

        pages_seen: list[int] = []

        def sender(method: str, path: str, payload=None):
            p = _page_from_path(path)
            pages_seen.append(p)
            return _project_page_response(p, 2)

        client = ProjectManagementClient(request_sender=sender)
        result = list(client.iter_projects(per_page=2))
        assert len(result) == 4
        assert pages_seen == [1, 2]

    def test_list_projects_unchanged(self, monkeypatch):
        """list_projects still returns a single-page response object."""
        monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "false")
        from traigent.projects.client import ProjectManagementClient

        def sender(method: str, path: str, payload=None):
            return _project_page_response(1, 5)

        client = ProjectManagementClient(request_sender=sender)
        resp = client.list_projects()
        assert hasattr(resp, "pagination")
        assert resp.pagination.has_next is True
