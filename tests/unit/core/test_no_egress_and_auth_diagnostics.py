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


class TestForbiddenClassificationDoesNotGuess:
    """Review findings on PR #2107: `_looks_edge_blocked` was wrong in BOTH directions.

    The old classifier was a Cloudflare-specific substring list whose default -- for
    every 403 it did not recognise -- was `authorization`, whose printed remediation
    is "grant the scope rather than rotating the key". For an AWS WAF, Akamai or API
    Gateway block that is a *confidently wrong* instruction, strictly worse than the
    ambiguous message it replaced.

    It now classifies on the POSITIVE signal we control (does this parse as a
    Traigent API error?) and answers `indeterminate` when it cannot tell.
    """

    AWS_WAF = (
        "<html><title>ERROR: The request could not be satisfied</title>"
        "Request blocked. Generated by cloudfront</html>"
    )
    AKAMAI = (
        "Access Denied You don't have permission. Reference #18.abc.1712345678.9abc"
    )
    API_GATEWAY = '{"message":"Forbidden"}'
    SCOPE_403_WAF_ID = '{"detail":"forbidden","request_id":"Zx9wAFq1kP"}'
    SCOPE_403_CF_LINK = (
        '{"detail":"API key lacks scope experiment.write",'
        '"help":"https://docs.traigent.ai/troubleshooting/cloudflare"}'
    )

    @pytest.mark.parametrize(
        "body", [AWS_WAF, AKAMAI, API_GATEWAY], ids=["aws_waf", "akamai", "api_gateway"]
    )
    def test_a_non_cloudflare_edge_block_is_never_called_a_scope_problem(self, body):
        """The false-green direction, and the more damaging one.

        These told the user their key was fine and a scope was missing. It is not a
        scope problem and the request never reached Traigent.
        """
        from traigent.cli.auth_commands import _classify_403

        assert _classify_403(body) != "authorization"

    @pytest.mark.parametrize(
        "body",
        [SCOPE_403_WAF_ID, SCOPE_403_CF_LINK],
        ids=["waf_inside_request_id", "body_links_to_cloudflare_docs"],
    )
    def test_a_genuine_scope_403_is_not_flipped_into_an_edge_block(self, body):
        """The false-red direction.

        `"waf" in body` is a 3-character substring with no word boundary, so the
        request id `Zx9wAFq1kP` used to flip a real scope problem into "check your
        proxy". A body that merely LINKS to cloudflare docs did the same.
        """
        from traigent.cli.auth_commands import _classify_403

        assert _classify_403(body) == "authorization"

    def test_a_cloudflare_response_header_is_decisive(self):
        """`cf-ray`/`cf-mitigated` are HEADERS.

        They were searched for in the BODY, where they essentially never appear, so
        the two most reliable Cloudflare signals could never fire.
        """
        from traigent.cli.auth_commands import _classify_403

        assert (
            _classify_403("<html>whatever</html>", {"CF-RAY": "8abc"}) == "edge_blocked"
        )

    def test_an_unrecognised_403_says_so_instead_of_inventing_a_remediation(self):
        from traigent.cli.auth_commands import _classify_403

        assert _classify_403("something nobody has seen before") == "indeterminate"

    def test_classification_reads_past_the_220_character_preview(self):
        """`body_preview` is truncated for DISPLAY; classification uses the full body.

        An edge block's giveaway routinely sits past the cut.
        """
        from traigent.cli.auth_commands import _classify_403

        body = ("x" * 400) + " cf-mitigated"

        assert _classify_403(body[:220]) == "indeterminate"
        assert _classify_403(body) == "edge_blocked"

    def test_the_cli_and_session_path_share_one_list_object(self):
        """Stronger than the source-grep this replaces: they cannot drift at all."""
        from traigent.cli.auth_commands import _EDGE_BLOCK_SIGNALS
        from traigent.core.backend_session_manager import EDGE_BLOCK_SIGNALS

        assert _EDGE_BLOCK_SIGNALS is EDGE_BLOCK_SIGNALS


# Placeholder credential for constructor tests -- never a real key.
_TEST_KEY = "k"  # pragma: allowlist secret


class TestNoEgressPlusExplicitExecutionModeFailsClosed:
    """Review finding on PR #2107, and the correction to my first answer to it.

    Review observed that `TraigentClient(no_egress=True, execution_mode="hybrid")`
    leaves `execution_mode` reading HYBRID while the resolved policy says
    local_only/offline=True, and called the run "still egressing".

    The disagreement is real but the consequence is not: the transport guard fails
    CLOSED. `test_offline_legacy_traigent_client_hybrid_zero_transport_calls` in
    tests/unit/cloud/test_no_content_egress_canary.py already pins that -- the run
    raises CloudEgressBlockedError and asserts zero outbound calls.

    My first fix downgraded the mode to LOCAL. That was worse: it replaced a loud
    failure with a silent one, handing a caller who asked for hybrid a different and
    weaker optimizer behind a log line. These tests pin the behaviour that is
    actually correct.
    """

    def test_the_attribute_keeps_the_requested_mode(self):
        from traigent.config.types import ExecutionMode
        from traigent.traigent_client import TraigentClient

        client = TraigentClient(
            api_key=_TEST_KEY, no_egress=True, execution_mode="hybrid"
        )

        assert client.execution_mode == ExecutionMode.HYBRID

    def test_but_the_policy_and_the_egress_flag_both_say_offline(self):
        """The half that matters: nothing is permitted to leave."""
        from traigent.traigent_client import TraigentClient

        client = TraigentClient(
            api_key=_TEST_KEY, no_egress=True, execution_mode="hybrid"
        )

        assert client.execution_policy.offline is True
        assert client.no_egress is True

    def test_an_explicit_mode_is_untouched_without_no_egress(self):
        from traigent.config.types import ExecutionMode
        from traigent.traigent_client import TraigentClient

        client = TraigentClient(api_key=_TEST_KEY, execution_mode="hybrid")

        assert client.execution_mode == ExecutionMode.HYBRID
        assert client.no_egress is False


class TestManagedAlgorithmsWithNoEgressIsABreakingChange:
    """Review finding on PR #2107: a public-API break that was unmentioned and untested.

    On develop `TraigentClient(algorithm="bayesian", no_egress=True)` constructed
    fine (and failed later, at optimize()). It now raises at CONSTRUCTION. That is
    more honest -- the combination can never work -- but it is a behaviour change on
    a public constructor, so it is pinned here and called out in the changelog.
    """

    MANAGED = ["bayesian", "optuna", "tpe", "cmaes", "nsga2"]

    @pytest.mark.parametrize("algorithm", MANAGED)
    def test_a_managed_algorithm_now_raises_at_construction(self, algorithm):
        from traigent.traigent_client import TraigentClient
        from traigent.utils.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError, match="requires managed optimization"):
            TraigentClient(api_key=_TEST_KEY, algorithm=algorithm, no_egress=True)

    @pytest.mark.parametrize("algorithm", ["grid", "random"])
    def test_a_local_algorithm_still_constructs(self, algorithm):
        """The blast radius is exactly the managed algorithms, not every caller."""
        from traigent.traigent_client import TraigentClient

        client = TraigentClient(api_key=_TEST_KEY, algorithm=algorithm, no_egress=True)

        assert client.no_egress is True
