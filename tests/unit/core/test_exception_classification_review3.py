"""Review round 3 for #2842/#2843: status hints and wrapped provider failures.

No real credentials: every string below is a placeholder.
"""

from __future__ import annotations

import pytest

from traigent.core.exception_handler import (
    VendorErrorCategory,
    classify_systematic_provider_failure,
    classify_vendor_error,
    is_provider_api_key_failure,
)


class _SdkError(Exception):
    __module__ = "openai"

    def __init__(self, status: int, text: str) -> None:
        super().__init__(text)
        self.status_code = status


class _HttpLibError(Exception):
    __module__ = "requests.exceptions"

    def __init__(self, status: int, text: str) -> None:
        super().__init__(text)
        self.status_code = status


def test_429_with_exhausted_quota_text_stays_fatal() -> None:
    exc = _SdkError(429, "You exceeded your current quota, please check your plan")
    assert classify_vendor_error(exc) is VendorErrorCategory.QUOTA_EXHAUSTED
    assert classify_systematic_provider_failure(exc) is not None


def test_plain_429_is_still_a_rate_limit() -> None:
    assert (
        classify_vendor_error(_SdkError(429, "Rate limit reached, retry in 2s"))
        is VendorErrorCategory.RATE_LIMIT
    )
    assert (
        classify_vendor_error(_SdkError(429, "Quota exceeded for requests per minute"))
        is VendorErrorCategory.RATE_LIMIT
    )


@pytest.mark.parametrize(
    "text",
    [
        "anthropic.AuthenticationError: invalid x-api-key",
        "litellm.AuthenticationError: bad key",
        "authentication_error: invalid x-api-key",
        "UnauthorizedError: nope",
        "QuotaExceededError: out of quota",
    ],
)
def test_exception_type_names_in_text_still_classify(text: str) -> None:
    assert classify_vendor_error(RuntimeError(text)) is not None


def test_identifier_glued_words_still_do_not_classify() -> None:
    assert classify_vendor_error(RuntimeError("no such column: quota_used")) is None
    assert (
        classify_vendor_error(RuntimeError("table authentication_log missing")) is None
    )


def test_raw_http_status_error_against_provider_endpoint_is_key_failure() -> None:
    exc = _HttpLibError(
        401, "401 Client Error: Unauthorized for url https://api.openai.com/v1/chat"
    )
    assert is_provider_api_key_failure(exc) is True


def test_user_service_401_is_not_key_failure() -> None:
    exc = _HttpLibError(401, "401 Unauthorized for url https://internal.example/tool")
    assert is_provider_api_key_failure(exc) is False


def test_wrapped_explicit_key_verdict_is_key_failure() -> None:
    exc = RuntimeError("Error: Incorrect API key provided: sk-placeholder")
    assert is_provider_api_key_failure(exc) is True


@pytest.mark.parametrize(
    "text",
    [
        "weather tool returned: Invalid API key",
        "Unexpected label 'invalid_api_key' in row 4",
    ],
)
def test_user_tool_key_wording_is_not_a_provider_verdict(text: str) -> None:
    assert is_provider_api_key_failure(RuntimeError(text)) is False


def test_provider_400_echoing_prompt_text_is_not_key_failure() -> None:
    exc = _SdkError(400, "content filtered: prompt mentions authentication and api key")
    assert is_provider_api_key_failure(exc) is False


def test_local_error_marker_does_not_hide_provider_status_text() -> None:
    text = "Error code: 401 - invalid_api_key; fallback raised TypeError: x"
    assert (
        classify_systematic_provider_failure(text) is VendorErrorCategory.AUTHENTICATION
    )


def test_botocore_style_dict_response_status_is_read() -> None:
    from traigent.core.exception_handler import http_status_of

    class ClientError(Exception):
        def __init__(self) -> None:
            super().__init__("An error occurred (UnrecognizedClientException)")
            self.response = {
                "Error": {"Code": "UnrecognizedClientException"},
                "ResponseMetadata": {"HTTPStatusCode": 403},
            }

    assert http_status_of(ClientError()) == 403


def test_429_credit_balance_with_retry_hint_stays_fatal() -> None:
    exc = _SdkError(
        429, "Your credit balance is too low. Please retry after topping up"
    )
    assert classify_vendor_error(exc) is VendorErrorCategory.INSUFFICIENT_FUNDS
    exc = _SdkError(429, "insufficient_quota: retry later")
    assert classify_vendor_error(exc) is VendorErrorCategory.QUOTA_EXHAUSTED


def test_aws_access_denied_is_not_a_credential_failure() -> None:
    from traigent.core.exception_handler import http_status_of

    class ClientError(Exception):
        def __init__(self, code: str, status: int) -> None:
            super().__init__(code)
            self.response = {
                "Error": {"Code": code},
                "ResponseMetadata": {"HTTPStatusCode": status},
            }

    assert http_status_of(ClientError("AccessDenied", 403)) is None
    assert http_status_of(ClientError("UnrecognizedClientException", 403)) == 403
