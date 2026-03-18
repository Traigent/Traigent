"""Pure-unit tests for JSEvaluator static helpers and config fields.

These tests do NOT spawn Node.js processes and can run directly.
They cover _lease_remaining, _consume_lease, and the new config fields.
"""

from unittest.mock import MagicMock

from traigent.evaluators.js_evaluator import JSEvaluator, JSEvaluatorConfig

# =============================================================================
# JSEvaluatorConfig new fields tests
# =============================================================================


class TestJSEvaluatorConfigNewFields:
    """Tests for the new JS runtime config fields."""

    def test_new_defaults(self):
        config = JSEvaluatorConfig(js_module="./test.js")
        assert config.js_use_npx is True
        assert config.js_runner_path is None
        assert config.js_node_executable == "node"

    def test_custom_new_fields(self):
        config = JSEvaluatorConfig(
            js_module="./test.js",
            js_use_npx=False,
            js_runner_path="/usr/local/bin/runner",
            js_node_executable="/usr/bin/node18",
        )
        assert config.js_use_npx is False
        assert config.js_runner_path == "/usr/local/bin/runner"
        assert config.js_node_executable == "/usr/bin/node18"


# =============================================================================
# _lease_remaining / _consume_lease static method tests
# =============================================================================


class TestLeaseHelpers:
    """Tests for _lease_remaining and _consume_lease static methods."""

    def test_lease_remaining_with_method(self):
        """Test remaining() as a callable method."""
        lease = MagicMock()
        lease.remaining = MagicMock(return_value=5)
        result = JSEvaluator._lease_remaining(lease)
        assert result == 5.0
        lease.remaining.assert_called_once()

    def test_lease_remaining_with_property_value(self):
        """Test remaining as a plain int attribute (property-style)."""

        class FakeLease:
            remaining = 7

        result = JSEvaluator._lease_remaining(FakeLease())
        assert result == 7.0

    def test_lease_remaining_with_real_property(self):
        """Test remaining as an actual @property descriptor."""

        class FakeLease:
            @property
            def remaining(self):
                return 9

        result = JSEvaluator._lease_remaining(FakeLease())
        assert result == 9.0

    def test_lease_remaining_missing_attribute(self):
        """Test missing remaining attribute returns 0.0."""

        class FakeLease:
            pass

        result = JSEvaluator._lease_remaining(FakeLease())
        assert result == 0.0

    def test_consume_lease_with_try_take(self):
        """Test _consume_lease prefers try_take when available."""

        class FakeLease:
            def try_take(self, count):
                self.taken = count

        lease = FakeLease()
        JSEvaluator._consume_lease(lease, 3)
        assert lease.taken == 3

    def test_consume_lease_falls_back_to_consume(self):
        """Test _consume_lease falls back to consume when try_take missing."""
        lease = MagicMock(spec=[])
        lease.consume = MagicMock()
        JSEvaluator._consume_lease(lease, 2)
        lease.consume.assert_called_once_with(2)

    def test_consume_lease_zero_count_returns_early(self):
        """Test _consume_lease with count <= 0 does nothing."""

        class FakeLease:
            def try_take(self, count):
                raise AssertionError("should not be called")

        JSEvaluator._consume_lease(FakeLease(), 0)
        JSEvaluator._consume_lease(FakeLease(), -1)

    def test_consume_lease_no_methods_is_noop(self):
        """Test _consume_lease with neither try_take nor consume."""

        class FakeLease:
            pass

        JSEvaluator._consume_lease(FakeLease(), 5)  # should not raise
