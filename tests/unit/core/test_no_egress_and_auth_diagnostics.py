"""#1776 + #1775 — a documented knob must reach the layer it names, and a 403 is not a bad key.

Both are contract/diagnosability defects rather than leaks. Neither over-grants; both
send the user somewhere useless.

  #1776  `no_egress=True` reached the TRANSPORT gate but not the EXECUTION policy, so
         an air-gapped caller who left `algorithm` at its "auto" default got
         CloudEgressBlockedError instead of a local run -- and the error text told
         them to clear the very flag they had deliberately set.
  #1775  `traigent auth whoami` / `keys` mapped 401 and 403 to one string. #1754
         (PR #1762) split them on the session path; the CLI kept the collapse, so an
         insufficient-scope 403 or a Cloudflare edge block both read as "bad key" and
         steered the user to rotate a perfectly valid credential.
"""

from __future__ import annotations

import pytest


class TestNoEgressReachesExecutionPolicy:
    def test_no_egress_is_folded_into_the_offline_predicate(self):
        """One predicate must drive BOTH the transport gate and the execution intent.

        Asserted at the call site because the alternative -- constructing a real
        TraigentClient -- pulls in cloud initialisation this test has no business
        exercising, and the defect is precisely that the argument was not passed on.
        """
        import inspect

        from traigent.traigent_client import TraigentClient

        source = inspect.getsource(TraigentClient.__init__)
        call = source.split("_resolve_execution_policy(", 1)[1].split(")", 1)[0]

        assert "no_egress" in call, (
            "no_egress is still dropped before the execution policy; the transport "
            "gate and the execution intent will disagree (#1776)"
        )

    def test_the_transport_flag_still_reflects_either_input(self):
        """The pre-existing behaviour this must not disturb."""
        import inspect

        from traigent.traigent_client import TraigentClient

        source = inspect.getsource(TraigentClient.__init__)

        assert "self.no_egress = bool(offline or no_egress)" in source

    @pytest.mark.parametrize(
        "offline,no_egress,expect_offline",
        [
            (False, False, False),
            (True, False, True),
            (False, True, True),  # the #1776 case
            (True, True, True),
        ],
    )
    def test_the_folded_predicate_truth_table(self, offline, no_egress, expect_offline):
        """`offline or no_egress` -- stated as a table so a future edit cannot quietly
        turn it into `and`, which would re-open the defect for exactly one input."""
        assert bool(offline or no_egress) is expect_offline


class TestAuthStatusDiagnostics:
    def test_401_and_403_no_longer_share_a_message(self):
        from traigent.cli.auth_commands import _ERROR_STATUS_MAP

        unauthenticated, _ = _ERROR_STATUS_MAP[401]
        unauthorized, _ = _ERROR_STATUS_MAP[403]

        assert unauthenticated != unauthorized, (
            "401 and 403 map to one string again -- a scope problem reads as a bad "
            "key and the user is told to rotate a valid credential (#1775)"
        )

    def test_403_is_categorised_as_authorization_not_authentication(self):
        """The category drives the remediation hint, so it has to be right."""
        from traigent.cli.auth_commands import _ERROR_STATUS_MAP

        assert _ERROR_STATUS_MAP[401][1] == "authentication"
        assert _ERROR_STATUS_MAP[403][1] == "authorization"

    def test_403_names_the_scope_rather_than_the_key(self):
        from traigent.cli.auth_commands import _ERROR_STATUS_MAP

        message, _ = _ERROR_STATUS_MAP[403]

        assert "scope" in message.lower()
        assert "invalid" not in message.lower(), (
            "the key is valid; do not say otherwise"
        )

    @pytest.mark.parametrize(
        "body",
        [
            "error code: 1010",
            "Cloudflare Ray ID: cf-ray-abc",
            "<html>Attention Required! | Cloudflare</html>",
            "request blocked by WAF",
            "edge_blocked",
        ],
    )
    def test_an_edge_block_is_recognised(self, body):
        from traigent.cli.auth_commands import _looks_edge_blocked

        assert _looks_edge_blocked(body) is True

    @pytest.mark.parametrize(
        "body",
        [
            "",
            "insufficient scope: experiment.write",
            '{"error": "forbidden", "detail": "missing permission"}',
        ],
    )
    def test_a_genuine_traigent_403_is_not_mistaken_for_an_edge_block(self, body):
        """The false-red direction. Calling a real scope problem an edge block would
        send the user to inspect their network instead of their key's permissions."""
        from traigent.cli.auth_commands import _looks_edge_blocked

        assert _looks_edge_blocked(body) is False

    def test_the_edge_vocabulary_matches_the_session_path(self):
        """Two different answers for one HTTP response is the defect being fixed.

        The session classifier already recognises these signals; if the two lists
        drift, `whoami` and a session-creation failure will disagree about the same
        backend response.
        """
        import inspect

        from traigent.cli.auth_commands import _EDGE_BLOCK_SIGNALS
        from traigent.core import backend_session_manager as bsm

        session_source = inspect.getsource(bsm)

        missing = [s for s in _EDGE_BLOCK_SIGNALS if f'"{s}"' not in session_source]
        assert not missing, (
            f"CLI edge signals absent from the session classifier: {missing} -- the "
            f"two surfaces would classify the same 403 differently"
        )
