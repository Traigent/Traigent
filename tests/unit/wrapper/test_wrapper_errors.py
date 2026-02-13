"""Tests for traigent.wrapper.errors module."""

from traigent.wrapper.errors import (
    BadRequestError,
    HybridAPIError,
    RateLimitError,
    RequestTimeoutError,
    ServiceUnavailableError,
    UnauthorizedError,
)


class TestHybridAPIError:
    def test_base_error_attributes(self) -> None:
        err = HybridAPIError(
            "test error",
            status_code=418,
            error_code="TEAPOT",
            details={"key": "val"},
            headers={"X-Custom": "hdr"},
        )
        assert str(err) == "test error"
        assert err.status_code == 418
        assert err.error_code == "TEAPOT"
        assert err.details == {"key": "val"}
        assert err.headers == {"X-Custom": "hdr"}

    def test_base_error_defaults(self) -> None:
        err = HybridAPIError("msg", status_code=500, error_code="ERR")
        assert err.details is None
        assert err.headers == {}


class TestBadRequestError:
    def test_defaults(self) -> None:
        err = BadRequestError("bad input")
        assert err.status_code == 400
        assert err.error_code == "INVALID_REQUEST"
        assert err.details is None

    def test_custom_code(self) -> None:
        err = BadRequestError("x", error_code="VALIDATION", details={"field": "a"})
        assert err.error_code == "VALIDATION"
        assert err.details == {"field": "a"}


class TestUnauthorizedError:
    def test_defaults(self) -> None:
        err = UnauthorizedError()
        assert err.status_code == 401
        assert err.error_code == "UNAUTHORIZED"
        assert str(err) == "Authentication required"


class TestRequestTimeoutError:
    def test_defaults(self) -> None:
        err = RequestTimeoutError("timed out")
        assert err.status_code == 408
        assert err.error_code == "REQUEST_TIMEOUT"


class TestRateLimitError:
    def test_defaults(self) -> None:
        err = RateLimitError()
        assert err.status_code == 429
        assert err.error_code == "RATE_LIMITED"
        assert err.headers == {}

    def test_retry_after(self) -> None:
        err = RateLimitError(retry_after=30)
        assert err.headers["Retry-After"] == "30"


class TestServiceUnavailableError:
    def test_defaults(self) -> None:
        err = ServiceUnavailableError()
        assert err.status_code == 503
        assert err.error_code == "SERVICE_UNAVAILABLE"
        assert err.headers == {}

    def test_retry_after(self) -> None:
        err = ServiceUnavailableError(retry_after=60)
        assert err.headers["Retry-After"] == "60"
