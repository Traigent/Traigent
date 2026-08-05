"""#1774 + #1777 — a broad handler must not launder a programming error.

One class, two instances. In each, a `try` whose handler is broad returns a benign
sentinel — FREE tier, or a synthetic "success" — so an `AttributeError` / missing
import symbol reads as normal degradation and disables a whole feature permanently,
undetectably, because the degraded result is a legitimate runtime outcome.

  #1774  core/license.py awaits `client.get_license_features()`, a method that has
         never existed in git history. Every paid tier validated through the cloud
         path silently resolved to FREE.
  #1777  cloud/production_mcp_client.py imports `StdioClientTransport` from
         `mcp.client.stdio`, a symbol that package does not export, so
         `MCP_AVAILABLE` was False even with `mcp` installed — and `call_tool` read
         `.result` / `.last_exception` off an object that has neither.
"""

from __future__ import annotations

import pytest


class TestMcpAvailabilityIsHonest:
    def test_mcp_available_reflects_the_package_not_a_nonexistent_symbol(self):
        """`mcp` is installed in this environment, so the flag must say so.

        Before the fix this asserted False, because the import block reached for a
        symbol `mcp.client.stdio` has never exported.
        """
        pytest.importorskip("mcp")
        import traigent.cloud.production_mcp_client as mod

        assert mod.MCP_AVAILABLE is True, (
            "MCP_AVAILABLE is False while `mcp` imports fine — the flag is keyed on "
            "something other than package availability (#1777)"
        )

    def test_the_nonexistent_symbol_is_no_longer_imported(self):
        """Pin the actual cause, not just its symptom."""
        import inspect

        import traigent.cloud.production_mcp_client as mod

        source = inspect.getsource(mod)
        import_block = source.split("logger = get_logger", 1)[0]

        assert "from mcp.client.stdio import StdioClientTransport" not in import_block

    def test_the_real_package_still_does_not_export_it(self):
        """Guards the premise. If `mcp` ever adds the symbol, revisit the port."""
        stdio = pytest.importorskip("mcp.client.stdio")

        assert not hasattr(stdio, "StdioClientTransport")
        assert hasattr(stdio, "stdio_client"), "the real API this needs porting to"

    def test_the_unported_transport_fails_loudly_rather_than_silently(self):
        """Honest about scope: the transport port is NOT done here.

        What must not happen is a NameError, or a quiet return to "unavailable" now
        that MCP_AVAILABLE is True. It raises NotImplementedError naming the work.
        """
        import inspect

        import traigent.cloud.production_mcp_client as mod

        connect_source = inspect.getsource(mod.ProductionMCPClient.connect)

        assert "NotImplementedError" in connect_source
        assert "stdio_client" in connect_source, "the message must name the real API"


class TestRetryContractIsUsedCorrectly:
    def test_execute_async_returns_the_unwrapped_value(self):
        """The premise behind the `call_tool` fix, asserted rather than assumed."""
        import inspect

        from traigent.utils.retry import RetryHandler

        source = inspect.getsource(RetryHandler.execute_async)

        # It unwraps (`return result.result`) and raises on failure — so a caller
        # reading `.success` / `.last_exception` off the return value is reading
        # attributes of the payload, not of a RetryResult.
        assert "return result.result" in source
        assert "raise result.error" in source

    def test_call_tool_no_longer_reads_retryresult_attributes_off_the_payload(self):
        import inspect

        import traigent.cloud.production_mcp_client as mod

        source = inspect.getsource(mod.ProductionMCPClient.call_tool)

        assert "result.last_exception" not in source, (
            "MCPResponse has no .last_exception; this raised AttributeError the "
            "moment the import bug was fixed (#1777)"
        )

    def test_a_programming_error_is_not_absorbed_into_synthetic_fallback(self):
        """The harm: fallback returns success=True with a fake `fallback_*` id.

        Laundering a defect through it reports a nonexistent backend resource as a
        successful creation.
        """
        import inspect

        import traigent.cloud.production_mcp_client as mod

        source = inspect.getsource(mod.ProductionMCPClient.call_tool)

        assert "except (AttributeError, TypeError, NameError, ImportError)" in source


class TestCloudLicenseValidationIsLoud:
    @pytest.mark.asyncio
    async def test_a_missing_client_method_logs_at_error_not_warning(self, monkeypatch):
        """#1774: a paid user silently becoming FREE must at least be visible.

        The tier outcome is unchanged (None -> FREE, which fails safe and never
        over-grants). What changes is that it is no longer indistinguishable from
        "the cloud was briefly unreachable".
        """
        from traigent.core.license import LicenseValidator

        class _ClientWithoutTheMethod:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_exc):
                return False

        import traigent.cloud.client as client_mod

        monkeypatch.setattr(
            client_mod, "TraigentCloudClient", _ClientWithoutTheMethod, raising=False
        )

        # Asserting on the logger call rather than caplog: this package uses a
        # custom logging facade whose records do not propagate to caplog, so a
        # caplog-based assertion here passes or fails for reasons unrelated to the
        # behaviour under test.
        import traigent.core.license as lic

        errors: list[str] = []
        monkeypatch.setattr(
            lic.logger,
            "error",
            lambda msg, *args, **kw: errors.append(
                str(msg) % args if args else str(msg)
            ),
        )

        validator = LicenseValidator()
        # Value is the literal word "placeholder" so the push-time secret scanner
        # recognises it as one. Any value works: the branch under test is reached
        # before the credential is ever read.
        validator._api_key = "placeholder"
        # The test environment enables offline mode, which short-circuits before the
        # cloud path is reached. Pinned off so the branch under test actually runs --
        # without this the test passes trivially by never entering the code it guards.
        validator._offline_mode = False

        result = await validator._validate_cloud_license()

        assert result is None, "must still fail safe rather than over-grant"
        assert any("get_license_features" in message for message in errors), (
            f"the dead path must be reported at ERROR naming the missing method; "
            f"got: {errors}"
        )

    def test_programming_errors_have_their_own_handler(self):
        """Pin the narrowing, so the broad handler cannot reclaim them."""
        import inspect

        from traigent.core import license as lic

        source = inspect.getsource(lic.LicenseValidator._validate_cloud_license)

        assert "except (AttributeError, TypeError, NameError)" in source
        # ...and it must come BEFORE the catch-all, or it never runs.
        assert source.index("except (AttributeError") < source.index("except Exception")


def test_no_silent_downgrade_paths_remain_unlabelled():
    """Both fixes must say WHY, not just behave better.

    A future reader seeing `return None` in a licence path needs to know it is a
    fallback rather than a verdict — that distinction is the whole defect class.
    """
    import inspect

    from traigent.core import license as lic

    source = inspect.getsource(lic.LicenseValidator._validate_cloud_license)

    assert "PROGRAMMING error" in source
    assert "fallback, not a verdict" in source


def test_unused_import_of_the_stub_class_is_gone():
    """The stub only existed to satisfy the bad import; it must not linger as a trap."""
    import traigent.cloud.production_mcp_client as mod

    assert not hasattr(mod, "StdioClientTransport") or not mod.MCP_AVAILABLE, (
        "a live StdioClientTransport stub alongside MCP_AVAILABLE=True would let "
        "connect() build a no-op transport again"
    )
