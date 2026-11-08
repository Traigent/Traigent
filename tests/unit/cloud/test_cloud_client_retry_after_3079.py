"""Issue #3079: the status-probe Retry-After parser must read real aiohttp
headers (a CIMultiDictProxy, not a dict)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from types import SimpleNamespace

import pytest
from multidict import CIMultiDict, CIMultiDictProxy

from traigent.cloud.client import _get_retry_delay


def _response(**headers: str) -> SimpleNamespace:
    return SimpleNamespace(headers=CIMultiDictProxy(CIMultiDict(headers)))


def test_reads_retry_after_from_aiohttp_headers() -> None:
    assert _get_retry_delay(_response(**{"Retry-After": "1"})) == 1.0


def test_header_name_is_case_insensitive() -> None:
    assert _get_retry_delay(_response(**{"retry-after": "0.5"})) == 0.5


def test_http_date_form() -> None:
    when = format_datetime(datetime.now(UTC) + timedelta(seconds=30), usegmt=True)
    assert 25.0 <= _get_retry_delay(_response(**{"Retry-After": when})) <= 31.0


@pytest.mark.parametrize("value", ["soon", "-3", "nan", ""])
def test_unusable_values_mean_no_wait(value: str) -> None:
    assert _get_retry_delay(_response(**{"Retry-After": value})) == 0.0


def test_missing_header_or_headers() -> None:
    assert _get_retry_delay(_response()) == 0.0
    assert _get_retry_delay(SimpleNamespace()) == 0.0
