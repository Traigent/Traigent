"""Mock transport guard coverage (#2811) and visibility of a block (#2812).

All tests are offline: sockets are patched to fail the test if reached, and the
vendor clients are pointed at provider hosts that the guard must stop before
any connection is attempted.
"""

from __future__ import annotations

import logging
import socket
from collections.abc import Iterator

import pytest

from traigent import testing as traigent_testing
from traigent.config.provider_support import PROVIDER_SPECS
from traigent.utils import mock_transport_guard as guard
from traigent.utils.mock_transport_guard import (
    PROVIDER_HOSTS,
    MockModeTransportBlockedError,
    is_known_llm_provider_host,
)

FAKE_AWS_KEY_ID = "AKIAMOCKMOCKMOCKMOCK"  # pragma: allowlist secret
FAKE_AWS_SECRET = "mock-secret"  # pragma: allowlist secret


def _fake_aws_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", FAKE_AWS_KEY_ID)
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", FAKE_AWS_SECRET)


REPRO_HOSTS = (
    "inference-api.nousresearch.com",
    "us-central1-aiplatform.googleapis.com",
    "bedrock.us-east-1.amazonaws.com",
    "api.perplexity.ai",
    "x.cognitiveservices.azure.com",
)


@pytest.fixture(autouse=True)
def _mock_on(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv("TRAIGENT_MOCK_LLM", raising=False)
    monkeypatch.setenv("ENVIRONMENT", "development")
    traigent_testing._reset_for_tests()
    guard._reset_blocked_calls_for_tests()

    def _no_socket(*args: object, **kwargs: object) -> None:
        raise AssertionError("a socket connect was attempted under mock mode")

    monkeypatch.setattr(socket.socket, "connect", _no_socket)
    monkeypatch.setattr(socket.socket, "connect_ex", _no_socket)
    traigent_testing.enable_mock_mode_for_quickstart()
    yield
    traigent_testing._reset_for_tests()
    guard._reset_blocked_calls_for_tests()


@pytest.mark.parametrize("host", REPRO_HOSTS)
def test_missing_host_families_are_known_2811(host: str) -> None:
    assert is_known_llm_provider_host(host)


@pytest.mark.parametrize(
    "host",
    [
        "example.com",
        "bedrock.us-east-1.amazonaws.com.evil.example",
        "notaiplatform.com",
    ],
)
def test_unrelated_hosts_are_not_blocked_2811(host: str) -> None:
    assert not is_known_llm_provider_host(host)


def test_every_registered_provider_has_a_covered_host_2811() -> None:
    names = {spec.name for spec in PROVIDER_SPECS}
    assert names <= set(PROVIDER_HOSTS), (
        "provider(s) in PROVIDER_SPECS with no mock-guard host: "
        f"{sorted(names - set(PROVIDER_HOSTS))}"
    )
    for name in names:
        hosts, patterns = PROVIDER_HOSTS[name]
        assert hosts or patterns, f"{name} has no covered host"


def test_boto3_bedrock_runtime_is_blocked_before_any_socket_2811(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    boto3 = pytest.importorskip("boto3")
    _fake_aws_credentials(monkeypatch)
    client = boto3.client("bedrock-runtime", region_name="us-east-1")
    with pytest.raises(MockModeTransportBlockedError, match="bedrock-runtime"):
        client.converse(
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            messages=[{"role": "user", "content": [{"text": "hi"}]}],
        )


def test_boto3_non_provider_service_is_not_blocked_2811(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-LLM AWS host reaches the send boundary (and the socket stub)."""
    boto3 = pytest.importorskip("boto3")
    _fake_aws_credentials(monkeypatch)
    client = boto3.client("sts", region_name="us-east-1")
    with pytest.raises(Exception) as excinfo:  # noqa: PT011 - any non-guard failure
        client.get_caller_identity()
    assert not isinstance(excinfo.value, MockModeTransportBlockedError)


def test_grpc_channel_to_provider_host_is_blocked_2811() -> None:
    grpc = pytest.importorskip("grpc")
    with pytest.raises(MockModeTransportBlockedError, match="generativelanguage"):
        grpc.secure_channel(
            "generativelanguage.googleapis.com:443", grpc.ssl_channel_credentials()
        )
    with pytest.raises(MockModeTransportBlockedError):
        grpc.insecure_channel("dns:///us-central1-aiplatform.googleapis.com:443")


def test_grpc_channel_to_local_host_is_allowed_2811() -> None:
    grpc = pytest.importorskip("grpc")
    channel = grpc.insecure_channel("localhost:50051")
    channel.close()


def test_grpc_guard_inert_when_mock_off_2811() -> None:
    grpc = pytest.importorskip("grpc")
    traigent_testing.disable_mock_mode()
    channel = grpc.insecure_channel("generativelanguage.googleapis.com:443")
    channel.close()


def test_block_logs_one_warning_per_host_naming_mock_mode_2812(
    caplog: pytest.LogCaptureFixture,
) -> None:
    httpx = pytest.importorskip("httpx")
    with caplog.at_level(logging.WARNING):
        for _ in range(3):
            with pytest.raises(MockModeTransportBlockedError):
                httpx.post("https://api.openai.com/v1/x")
        with pytest.raises(MockModeTransportBlockedError):
            httpx.post("https://api.groq.com/v1/x")
    blocked = [r for r in caplog.records if "mock mode blocked" in r.getMessage()]
    assert len(blocked) == 2
    text = " ".join(r.getMessage() for r in blocked)
    assert "api.openai.com" in text and "api.groq.com" in text
    assert "embeddings" in text.lower() and "not mocked" in text
    assert guard.mock_blocked_calls() == {"api.openai.com": 3, "api.groq.com": 1}


def test_vendor_wrapped_connection_error_is_still_explained_2812(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """openai wraps the guard error into 'Connection error.'; the log says why."""
    openai = pytest.importorskip("openai")
    client = openai.OpenAI(api_key="sk-mock", max_retries=0)  # pragma: allowlist secret
    with caplog.at_level(logging.WARNING):
        with pytest.raises(openai.APIConnectionError) as excinfo:
            client.chat.completions.create(
                model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}]
            )
    assert str(excinfo.value) == "Connection error."
    assert any(
        "mock mode blocked" in r.getMessage() and "api.openai.com" in r.getMessage()
        for r in caplog.records
    )


@pytest.mark.asyncio
async def test_all_trials_failed_summary_names_mock_mode_2812() -> None:
    """A run whose only calls were blocked says so, not just 'all trials failed'."""
    httpx = pytest.importorskip("httpx")
    import traigent

    @traigent.optimize(
        configuration_space={"x": [1]},
        objectives=["accuracy"],
        eval_dataset=[{"input": {"q": "a"}, "output": "a"}],
        offline=True,
        max_trials=1,
    )
    def fn(q: str) -> str:
        httpx.post("https://api.openai.com/v1/chat/completions")
        return "a"

    result = await fn.optimize(max_trials=1, algorithm="grid")

    assert "MOCK_TRANSPORT_BLOCKED" in result.warning_codes
    assert "ALL_TRIALS_FAILED" in result.warning_codes
    text = " ".join(result.warnings)
    assert "Mock mode blocked" in text and "api.openai.com" in text
    assert "not a network outage" in text


def test_banner_names_exact_mocked_classes_and_guarded_transports_2813(
    caplog: pytest.LogCaptureFixture,
) -> None:
    traigent_testing._reset_for_tests()
    with caplog.at_level(logging.WARNING):
        traigent_testing.log_raw_client_scope_caveat()
    banner = " ".join(r.getMessage() for r in caplog.records)
    for covered in ("ChatOpenAI", "ChatAnthropic", "Bedrock", "LiteLLM"):
        assert covered in banner
    assert "Embeddings" in banner and "not mocked" in banner
    assert "ChatGoogleGenerativeAI" in banner
    for transport in ("httpx", "aiohttp", "requests", "botocore", "gRPC"):
        assert transport in banner
    assert "LiteLLM and LangChain calls are mocked" not in banner


@pytest.mark.parametrize(
    ("target", "host"),
    [
        ("generativelanguage.googleapis.com:443", "generativelanguage.googleapis.com"),
        ("dns:///api.openai.com:443", "api.openai.com"),
        ("dns:api.openai.com:443", "api.openai.com"),
        ("dns://8.8.8.8/api.openai.com:443", "api.openai.com"),
        ("passthrough:///api.openai.com:443", "api.openai.com"),
        ("xds:///api.openai.com", "api.openai.com"),
        ("user@api.openai.com:443", "api.openai.com"),
        ("[::1]:443", "::1"),
        ("ipv4:1.2.3.4:443", "1.2.3.4"),
        ("ipv6:[::1]:443", "::1"),
        ("https://api.openai.com/v1", "api.openai.com"),
        ("https://u:p@api.openai.com:443/x", "api.openai.com"),
        ("localhost:50051", "localhost"),
    ],
)
def test_grpc_target_host_parsing_2811(target: str, host: str) -> None:
    assert guard._grpc_target_host(target) == host


# --- red-team round: host table, anchoring, gRPC targets, inertness ---------


@pytest.mark.parametrize(
    "host",
    [
        # botocore dual-stack and non-"amazonaws.com" partition suffixes.
        "bedrock-runtime.us-east-1.api.aws",
        "bedrock-runtime-fips.us-gov-west-1.api.aws",
        "bedrock-runtime.cn-north-1.api.amazonwebservices.com.cn",
        "bedrock-runtime.eusc-de-east-1.amazonaws.eu",
        "bedrock-runtime.eusc-de-east-1.api.amazonwebservices.eu",
        "vpce-0abc-xyz.bedrock-runtime.us-east-1.vpce.amazonaws.com",
    ],
)
def test_bedrock_dualstack_and_partition_hosts_are_known_2811(host: str) -> None:
    assert is_known_llm_provider_host(host)


@pytest.mark.parametrize(
    "host",
    [
        # Virtual-hosted S3 bucket whose name merely starts with "bedrock".
        "bedrock-logs.s3.amazonaws.com",
        "bedrock-artifacts.s3.amazonaws.com.cn",
        "bedrock-x.sts.amazonaws.com",
    ],
)
def test_bedrock_prefixed_non_llm_hosts_are_not_blocked_2811(host: str) -> None:
    assert not is_known_llm_provider_host(host)


@pytest.mark.parametrize(
    "target",
    ["api.openai.com:https", "dns:///api.openai.com:https", "api.openai.com:grpc"],
)
def test_grpc_named_port_does_not_bypass_the_guard_2811(target: str) -> None:
    grpc = pytest.importorskip("grpc")
    with pytest.raises(MockModeTransportBlockedError):
        grpc.insecure_channel(target)


def test_block_message_names_the_canonical_host_2812() -> None:
    from types import SimpleNamespace

    with pytest.raises(MockModeTransportBlockedError) as excinfo:
        guard._raise_if_blocked(SimpleNamespace(host="API.OpenAI.com."))
    assert "(api.openai.com)" in str(excinfo.value)


def test_guard_is_inert_when_mock_off_even_if_host_extraction_fails_2811() -> None:
    def boom(self: object, request: object) -> object:
        raise AttributeError("no url on this request type")

    wrapped = guard._wrap_sync(lambda self, request: "sent", boom)
    traigent_testing.disable_mock_mode()
    assert wrapped(object(), object()) == "sent"


def test_guard_fails_closed_when_host_extraction_fails_under_mock_2811() -> None:
    def boom(self: object, request: object) -> object:
        raise AttributeError("no url on this request type")

    wrapped = guard._wrap_sync(lambda self, request: "sent", boom)
    with pytest.raises(MockModeTransportBlockedError, match="could not determine"):
        wrapped(object(), object())
