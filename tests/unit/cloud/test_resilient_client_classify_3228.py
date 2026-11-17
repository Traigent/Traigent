"""Regression tests for #3228: classify by status/type before loose substrings."""

import pytest

from traigent.cloud.resilient_client import ErrorType, ResilientClient


class _HttpErr(Exception):
    def __init__(self, msg, status):
        super().__init__(msg)
        self.status = status


@pytest.fixture
def client():
    return ResilientClient(max_retries=3)


@pytest.mark.parametrize(
    "msg,expected",
    [
        ("HTTP 403 Forbidden: password reset required", ErrorType.AUTH),
        ("HTTP 401 Unauthorized: ssl cert pinned", ErrorType.AUTH),
        ("HTTP 400 Bad Request: invalid connection string", ErrorType.CLIENT),
        ("HTTP 503 Service: network upstream", ErrorType.SERVER),
        ("connection reset by peer", ErrorType.NETWORK),
        ("something odd", ErrorType.UNKNOWN),
    ],
)
def test_untyped_message_classification(client, msg, expected):
    assert client.classify_error(ValueError(msg)) is expected


def test_untyped_auth_not_retried(client):
    err = ValueError("HTTP 403 Forbidden: password reset required")
    assert client.should_retry(err, 0) is False


def test_structured_status_beats_message(client):
    err = _HttpErr("connection reset", 401)
    assert client.classify_error(err) is ErrorType.AUTH
    assert client.should_retry(err, 0) is False


def test_type_name_network(client):
    assert client.classify_error(ConnectionResetError("x")) is ErrorType.NETWORK
    assert client.classify_error(TimeoutError("x")) is ErrorType.NETWORK


def test_port_number_not_status(client):
    assert client.classify_error(ValueError("failed port 4010 refused")) is (
        ErrorType.NETWORK
    )


def test_bare_number_in_message_is_not_a_status(client):
    err = ValueError("connection reset after reading 400 bytes")
    assert client.classify_error(err) is ErrorType.NETWORK
    assert client.should_retry(err, 0) is True


def test_unlisted_structured_4xx_is_client(client):
    err = _HttpErr("invalid connection string", 422)
    assert client.classify_error(err) is ErrorType.CLIENT
    assert client.should_retry(err, 0) is False


@pytest.mark.parametrize(
    "msg",
    ["400 bytes read before connection reset", "error code 1400: connection reset"],
)
def test_numbers_without_status_context_stay_network(client, msg):
    assert client.classify_error(ValueError(msg)) is ErrorType.NETWORK
