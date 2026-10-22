"""Validation tests for DatasetConverter safety checks."""

import os
import socket
import subprocess
import sys
import textwrap
import threading
from unittest.mock import Mock, patch

import pytest

from traigent.cloud.dataset_converter import DatasetConverter, is_sensitive_metadata_key
from traigent.cloud.url_security import CloudUrlUnreachableError
from traigent.config.backend_config import DEFAULT_CLOUD_URL

_PRIVATE_BACKEND = "https://bank.internal.example"

# SDK #2033: opt into the connected/backend code paths (see pyproject markers).
pytestmark = pytest.mark.backend_online


class TestDatasetConverterValidation:
    """Validate DatasetConverter input handling."""

    def test_backend_base_url_validation_rejects_invalid_urls(self):
        """Invalid backend URLs should raise ValueError."""
        with pytest.raises(ValueError):
            DatasetConverter(backend_base_url="ftp://example.com")

        with pytest.raises(ValueError):
            DatasetConverter(backend_base_url="  ")

        with pytest.raises(ValueError):
            DatasetConverter(
                backend_base_url="https://api.example.com/base?query=1#fragment"
            )

        with pytest.raises(ValueError):
            DatasetConverter(
                backend_base_url="https://user:pass@example.com"  # pragma: allowlist secret
            )

        with pytest.raises(ValueError):
            DatasetConverter(backend_base_url="https://api.example.com/../admin")

        with pytest.raises(ValueError):
            DatasetConverter(backend_base_url="https://api.example.com/\nadmin")

    def test_backend_base_url_normalization(self):
        """Valid backend URLs should be normalized."""
        converter = DatasetConverter(backend_base_url="https://api.example.com/base/")
        assert converter.backend_base_url == "https://api.example.com/base"

    def test_backend_base_url_defaults_to_cloud(self):
        """Default constructor should use the cloud backend, not localhost."""
        converter = DatasetConverter()
        assert converter.backend_base_url == DEFAULT_CLOUD_URL

    def test_validate_example_set_id_allows_uuid(self):
        """Example set identifiers should accept UUIDs and trim whitespace."""
        converter = DatasetConverter()
        uuid_value = "123e4567-e89b-12d3-a456-426614174000"
        assert converter._validate_example_set_id(f" {uuid_value} ") == uuid_value

    def test_validate_example_set_id_rejects_path_characters(self):
        """Path traversal characters must be rejected."""
        converter = DatasetConverter()
        with pytest.raises(ValueError):
            converter._validate_example_set_id("../evil")

        with pytest.raises(ValueError):
            converter._validate_example_set_id("bad/id")

        with pytest.raises(ValueError):
            converter._validate_example_set_id("unsafe\nid")

    @pytest.mark.asyncio
    async def test_backend_example_set_to_sdk_dataset_rejects_invalid_id(self):
        """Public API should reject unsafe identifiers before network calls."""
        converter = DatasetConverter()
        with pytest.raises(ValueError):
            await converter.backend_example_set_to_sdk_dataset("??invalid??")


class TestDatasetConverterSecurePath:
    """Tests for secure path handling in DatasetConverter."""

    def test_backend_examples_to_jsonl_with_output_path(self, tmp_path):
        """Test that JSONL output uses secure path validation."""
        converter = DatasetConverter()

        # Create backend examples (dict format)
        examples = [
            {"input": '{"question": "What is 2+2?"}', "output": "4"},
            {
                "input": '{"question": "What is the capital of France?"}',
                "output": "Paris",
            },
        ]

        # Write to output path
        output_file = tmp_path / "output.jsonl"
        result = converter.backend_examples_to_jsonl(examples, output_path=output_file)

        # Verify file was written
        assert output_file.exists()
        assert len(result) > 0

        # Verify content
        lines = output_file.read_text().strip().split("\n")
        assert len(lines) == 2

    def test_backend_examples_to_jsonl_with_absolute_output_path(self, tmp_path):
        """Test JSONL output with absolute path."""
        converter = DatasetConverter()

        examples = [
            {"input": '{"text": "Hello"}', "output": "Hi"},
        ]

        # Use absolute path
        output_file = tmp_path / "absolute_output.jsonl"
        abs_path = output_file.resolve()

        result = converter.backend_examples_to_jsonl(examples, output_path=abs_path)

        assert abs_path.exists()
        assert len(result) > 0

    def test_backend_examples_to_jsonl_without_output_path(self):
        """Test JSONL generation without writing to file."""
        converter = DatasetConverter()

        examples = [
            {"input": '{"prompt": "Test"}', "output": "Response"},
        ]

        # No output path - should return content without writing
        result = converter.backend_examples_to_jsonl(examples, output_path=None)

        assert len(result) > 0
        assert "Test" in result
        assert "Response" in result


class TestSensitiveMetadataKeyUnion:
    """Regression coverage for issue #1649: dataset_converter's redaction
    keyword list used to be a hand-rolled regex, independent from the
    observability sanitizers' lists. It now delegates to the canonical
    CREDENTIAL union set in `traigent.security.redaction`."""

    @pytest.mark.parametrize(
        "key",
        [
            # Still covered by the module's old regex.
            "password",
            "api_key",
            "authorization",
            "credit_card",
            "secret",
            "token",
            # Formerly missing here: only present in observability.decorators'
            # old fragment list.
            "credential",
            "private_key",
            "apikey",
        ],
    )
    def test_credential_union_keyword_is_sensitive(self, key: str) -> None:
        assert is_sensitive_metadata_key(key) is True

    @pytest.mark.parametrize("key", ["model_name", "prompt_version", "output_format"])
    def test_non_credential_key_is_not_sensitive(self, key: str) -> None:
        """Content-marker fragments (prompt/response/output/...) must NOT be
        applied here: this module never redacted them pre-unification, and
        keys like `prompt_version` are legitimate backend tags. The invariant
        from review: no path gains content-marker redaction it didn't have."""
        assert is_sensitive_metadata_key(key) is False


# --- Shared always-blocked checks (metadata / link-local / plaintext) ----------


@pytest.mark.parametrize(
    "url",
    [
        "http://169.254.169.254",
        "https://169.254.169.254",
        "https://[fd00:ec2::254]",
        "https://metadata.google.internal",
    ],
)
def test_dataset_converter_refuses_metadata_destinations(url: str) -> None:
    with pytest.raises(ValueError, match="metadata service|link-local"):
        DatasetConverter(backend_base_url=url)


@pytest.mark.asyncio
async def test_dataset_converter_refuses_name_resolving_to_metadata() -> None:
    from unittest.mock import patch

    # Construction does no DNS (imports must not resolve hosts). The metadata answer
    # is refused at the first credential-bearing call, before any request is sent.
    converter = DatasetConverter(backend_base_url="https://api.example.com")
    converter._session = Mock()
    with patch(
        "socket.getaddrinfo",
        return_value=[(2, 1, 6, "", ("169.254.169.254", 0))],
    ):
        with pytest.raises(ValueError, match="metadata service|link-local"):
            await converter._fetch_backend_example_set("example-set-1")
    converter._session.get.assert_not_called()


def test_dataset_converter_refuses_plaintext_http_in_production() -> None:
    from unittest.mock import patch

    with patch.dict("os.environ", {}, clear=True):
        with patch(
            "socket.getaddrinfo",
            return_value=[(2, 1, 6, "", ("93.184.216.34", 0))],
        ):
            with pytest.raises(ValueError, match="must use https in production"):
                DatasetConverter(backend_base_url="http://api.example.com")


def test_dataset_converter_private_host_policy_unchanged() -> None:
    # Only metadata/link-local and plaintext-in-production changed; private
    # ranges stay as they were for this converter.
    from unittest.mock import patch

    with patch.dict("os.environ", {}, clear=True):
        converter = DatasetConverter(backend_base_url="https://10.20.30.40:8443")
    assert converter.backend_base_url == "https://10.20.30.40:8443"


def _production_private_backend_env() -> dict[str, str]:
    """Environment of a bank deployment: production, private backend opted in."""
    # Minimal environment: inherited TRAIGENT_* flags (for example TRAIGENT_MOCK_LLM)
    # would trip the production guards in the child process.
    env = {
        key: os.environ[key] for key in ("PATH", "HOME", "LANG") if key in os.environ
    }
    env.update(
        {
            "ENVIRONMENT": "production",
            "TRAIGENT_ALLOW_PRIVATE_BACKEND": "1",
            "TRAIGENT_BACKEND_URL": _PRIVATE_BACKEND,
        }
    )
    return env


def test_import_does_not_resolve_backend_host_when_dns_fails() -> None:
    # A bank configures a private backend and its resolver cannot answer yet. Import
    # must still succeed: construction does no DNS; the first egress resolves.
    code = textwrap.dedent(
        """
        import socket

        def _no_dns(*args, **kwargs):
            raise socket.gaierror("no dns in this test")

        socket.getaddrinfo = _no_dns
        import traigent
        import traigent.cloud.backend_client  # noqa: F401
        print("IMPORT_OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=_production_private_backend_env(),
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "IMPORT_OK" in result.stdout


@pytest.mark.asyncio
async def test_first_egress_refuses_when_backend_host_does_not_resolve() -> None:
    session = Mock()
    with (
        patch.dict("os.environ", _production_private_backend_env(), clear=True),
        patch("socket.getaddrinfo", side_effect=socket.gaierror),
    ):
        converter = DatasetConverter(backend_base_url=_PRIVATE_BACKEND)
        converter._session = session
        with pytest.raises(CloudUrlUnreachableError, match="could not be resolved"):
            await converter._fetch_backend_example_set("example-set-1")
    session.get.assert_not_called()


class _FakeResponse:
    def __init__(self, payload: object) -> None:
        self.status = 200
        self._payload = payload

    async def json(self) -> object:
        return self._payload

    async def text(self) -> str:
        return ""


class _FakeRequest:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response

    async def __aenter__(self) -> _FakeResponse:
        return self._response

    async def __aexit__(self, *exc: object) -> bool:
        return False


def _examples_page(has_next: bool) -> dict[str, object]:
    return {"data": {"items": [{"id": "e"}], "pagination": {"has_next": has_next}}}


@pytest.mark.asyncio
async def test_every_paginated_request_is_checked_after_the_resolver_changes() -> None:
    # Page 1 resolves public; by page 2 the name answers the metadata address. The
    # second request must be refused before it is sent.
    pages = [_examples_page(True), _examples_page(False)]
    answers: list[str] = []

    def resolver(host: str, port: object, *args: object, **kwargs: object) -> list:
        ip = "93.184.216.34" if not answers else "169.254.169.254"
        answers.append(ip)
        return [(2, 1, 6, "", (ip, 0))]

    session = Mock()
    session.get = Mock(
        side_effect=lambda url, params=None: _FakeRequest(_FakeResponse(pages.pop(0)))
    )
    with (
        patch.dict("os.environ", {}, clear=True),
        patch("socket.getaddrinfo", side_effect=resolver),
    ):
        converter = DatasetConverter(backend_base_url="https://api.example.com")
        converter._session = session
        with pytest.raises(ValueError, match="metadata service|link-local"):
            await converter._fetch_backend_examples("example-set-1")
    assert len(answers) == 2
    assert session.get.call_count == 1


@pytest.mark.asyncio
async def test_egress_dns_lookup_runs_off_the_event_loop() -> None:
    # The blocking lookup must run in a worker thread, not on the loop thread.
    lookup_threads: list[int] = []

    def resolver(host: str, port: object, *args: object, **kwargs: object) -> list:
        lookup_threads.append(threading.get_ident())
        return [(2, 1, 6, "", ("93.184.216.34", 0))]

    session = Mock()
    session.get = Mock(
        side_effect=lambda url, params=None: _FakeRequest(_FakeResponse({}))
    )
    with (
        patch.dict("os.environ", {}, clear=True),
        patch("socket.getaddrinfo", side_effect=resolver),
    ):
        converter = DatasetConverter(backend_base_url="https://api.example.com")
        converter._session = session
        loop_thread = threading.get_ident()
        await converter._fetch_backend_example_set("example-set-1")
    assert lookup_threads, "no DNS lookup was made"
    assert all(thread != loop_thread for thread in lookup_threads)
    assert session.get.call_count == 1
