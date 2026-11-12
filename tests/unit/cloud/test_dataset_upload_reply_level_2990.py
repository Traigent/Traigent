"""Upload reply counts are read from ``stats`` and rejected rows never read as success (#2990)."""

from __future__ import annotations

import pytest

from traigent.cloud.dataset_converter import DatasetConverter
from traigent.evaluators.base import Dataset, EvaluationExample


def _dataset(n: int = 6) -> Dataset:
    return Dataset(
        name="ds",
        examples=[
            EvaluationExample(input_data={"text": f"t{i}"}, expected_output="ok")
            for i in range(n)
        ],
    )


async def _run(monkeypatch, reply):
    converter = DatasetConverter(backend_base_url="https://example.invalid")

    async def _create(*_a, **_k):
        return "set-1"

    async def _upload(*_a, **_k):
        return reply

    monkeypatch.setattr(converter, "_create_backend_example_set", _create)
    monkeypatch.setattr(converter, "_upload_examples_to_backend", _upload)
    return await converter.upload_sdk_dataset_to_backend(_dataset())


@pytest.mark.asyncio
async def test_nested_invalid_rows_are_an_error_not_success(monkeypatch):
    reply = {
        "message": "Examples uploaded successfully",
        "stats": {"added": 3, "invalid": 2, "skipped": 1},
        "row_errors": [
            {"row": 5, "reason": "duplicate input"},
            {"row": 6, "reason": "missing input"},
        ],
    }
    result = await _run(monkeypatch, reply)

    assert result.success is False
    assert result.converted_count == 3
    assert result.skipped_count == 1
    assert result.error_count == 2
    assert any("rejected 2" in e for e in result.errors)
    assert any("row 6" in e and "missing input" in e for e in result.errors)


@pytest.mark.asyncio
async def test_clean_nested_reply_is_success(monkeypatch):
    result = await _run(
        monkeypatch, {"stats": {"added": 6, "invalid": 0, "skipped": 0}}
    )

    assert result.success is True
    assert result.converted_count == 6
    assert result.errors == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reply",
    [
        {},
        {"message": "ok"},
        {"stats": {}},
        {"stats": {"added": 3}},
        {"stats": {"added": 3, "skipped": 0, "invalid": None}},
        {"stats": {"added": 3, "skipped": 0, "invalid": "2"}},
        {"stats": {"added": True, "skipped": 0, "invalid": 0}},
    ],
)
async def test_unreadable_reply_is_never_success(monkeypatch, reply):
    result = await _run(monkeypatch, reply)

    assert result.success is False
    assert result.errors


@pytest.mark.asyncio
async def test_skipped_duplicates_stay_success_but_are_visible(monkeypatch, caplog):
    caplog.set_level("WARNING", logger="traigent.cloud.dataset_converter")
    result = await _run(
        monkeypatch, {"stats": {"added": 5, "invalid": 0, "skipped": 1}}
    )

    assert result.success is True
    assert result.skipped_count == 1
    assert result.metadata["skipped_duplicates"] == 1
    warnings = [r for r in caplog.records if "skipped 1 example" in r.getMessage()]
    assert len(warnings) == 1
