"""Example-set transport: versioned URLs and the canonical envelope (#2368).

The backend serves example sets only under ``/api/v1`` and returns the
``/examples`` listing in the paginated ``{"data": {"items", "pagination"}}``
envelope. A reader that looked up a top-level ``examples`` key with a ``[]``
default turned that 200 into an empty dataset with no error.
"""

from __future__ import annotations

from typing import Any

import pytest

from traigent.cloud.dataset_converter import DatasetConverter, ExampleSetMetadata


class _FakeResponse:
    def __init__(self, status: int, body: Any) -> None:
        self.status = status
        self._body = body

    async def json(self) -> Any:
        return self._body

    async def text(self) -> str:
        return str(self._body)

    async def __aenter__(self) -> _FakeResponse:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None


class _FakeSession:
    """Records requests and replays queued responses in order."""

    def __init__(self, responses: list[_FakeResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, str, dict[str, Any] | None]] = []

    def _next(self, method: str, url: str, params: Any = None) -> _FakeResponse:
        self.calls.append((method, url, params))
        return self._responses.pop(0)

    def get(self, url: str, params: Any = None, **_: Any) -> _FakeResponse:
        return self._next("GET", url, params)

    def post(self, url: str, **_: Any) -> _FakeResponse:
        return self._next("POST", url)


BASE = "https://backend.example"


@pytest.fixture(autouse=True)
def _egress_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    # Transport is a fake session; lift the offline guard so the request
    # builders run.
    for name in ("TRAIGENT_OFFLINE", "TRAIGENT_OFFLINE_MODE"):
        monkeypatch.delenv(name, raising=False)


def _converter(responses: list[_FakeResponse]) -> tuple[DatasetConverter, _FakeSession]:
    converter = DatasetConverter(BASE)
    session = _FakeSession(responses)
    converter._session = session  # type: ignore[assignment]
    return converter, session


def _envelope(items: list[dict[str, Any]], has_next: bool) -> dict[str, Any]:
    return {
        "success": True,
        "data": {"items": items, "pagination": {"page": 1, "has_next": has_next}},
    }


@pytest.mark.asyncio
async def test_canonical_envelope_yields_examples_across_pages() -> None:
    converter, session = _converter(
        [
            _FakeResponse(200, _envelope([{"input": "a"}], has_next=True)),
            _FakeResponse(200, _envelope([{"input": "b"}], has_next=False)),
        ]
    )

    examples = await converter._fetch_backend_examples("set-1", no_egress=False)

    assert examples == [{"input": "a"}, {"input": "b"}]
    assert [call[1] for call in session.calls] == [
        f"{BASE}/api/v1/example-sets/set-1/examples"
    ] * 2
    assert [call[2]["page"] for call in session.calls] == [1, 2]


@pytest.mark.asyncio
async def test_legacy_examples_key_still_read() -> None:
    converter, _ = _converter([_FakeResponse(200, {"examples": [{"input": "a"}]})])

    assert await converter._fetch_backend_examples("set-1", no_egress=False) == [
        {"input": "a"}
    ]


@pytest.mark.asyncio
async def test_unrecognised_200_raises_instead_of_empty_dataset() -> None:
    converter, _ = _converter([_FakeResponse(200, {"success": True, "data": {}})])

    with pytest.raises(ValueError, match="Unrecognised example-set examples"):
        await converter._fetch_backend_examples("set-1", no_egress=False)


@pytest.mark.asyncio
async def test_all_example_set_calls_use_versioned_prefix() -> None:
    converter, session = _converter(
        [
            _FakeResponse(201, {"example_set_id": "set-1"}),
            _FakeResponse(200, {"stats": {}}),
            _FakeResponse(200, {"id": "set-1"}),
        ]
    )
    metadata = ExampleSetMetadata(
        example_set_id="",
        name="n",
        type="input-output",
        description="d",
        total_examples=0,
        created_from="test",
        privacy_mode=False,
    )

    await converter._create_backend_example_set(metadata, "agent-1", no_egress=False)
    await converter._upload_examples_to_backend("set-1", [], no_egress=False)
    await converter._fetch_backend_example_set("set-1", no_egress=False)

    assert [call[1] for call in session.calls] == [
        f"{BASE}/api/v1/example-sets",
        f"{BASE}/api/v1/example-sets/set-1/upload",
        f"{BASE}/api/v1/example-sets/set-1",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pagination", [None, {}, {"has_next": "false"}, {"has_next": 1}]
)
async def test_invalid_canonical_pagination_fails_loudly(pagination):
    converter, _ = _converter(
        [
            _FakeResponse(
                200, {"data": {"items": [{"input": "a"}], "pagination": pagination}}
            )
        ]
    )
    with pytest.raises(ValueError, match="pagination"):
        await converter._fetch_backend_examples("set-1", no_egress=False)


@pytest.mark.asyncio
async def test_empty_nonterminal_page_fails_instead_of_truncating():
    converter, _ = _converter([_FakeResponse(200, _envelope([], True))])
    with pytest.raises(ValueError, match="empty.*page"):
        await converter._fetch_backend_examples("set-1", no_egress=False)


@pytest.mark.asyncio
async def test_empty_terminal_page_is_valid():
    converter, _ = _converter([_FakeResponse(200, _envelope([], False))])
    assert await converter._fetch_backend_examples("set-1", no_egress=False) == []
