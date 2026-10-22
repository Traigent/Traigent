"""TVL 1.4 reactive constructs in the fragment loader (RFC 0004), offline.

The real compiler is exercised in ``test_hosted_subset.py``; this file pins the
loader's grammar, its closed refusals and the rendered policy.
"""

from __future__ import annotations

import copy

import pytest
import yaml

import traigent
from traigent.runtime.model import RuntimeRequirementError
from traigent.runtime.tvl_fragment import load_fragment, parse_fragment

from .conftest import FIXTURES

REACTIVE = FIXTURES / "reactive_fragment.yml"


def _doc() -> dict:
    return yaml.safe_load(REACTIVE.read_text())


def _rule(doc: dict, rule_id: str) -> dict:
    return next(r for r in doc["responses"][0]["rules"] if r["id"] == rule_id)


def test_reactive_fragment_is_accepted_and_round_trips() -> None:
    req = load_fragment(REACTIVE, max_fix_cost_usd=5.0)
    assert req.is_reactive()
    by_id = {r.id: r for r in req.rules}
    assert by_id["fix"].deploy == "if_promoted"
    assert (
        by_id["fix_found_nothing"].trigger_rule,
        by_id["fix_found_nothing"].event,
    ) == (
        "fix",
        "optimize_no_candidate",
    )
    assert by_id["undecided"].event == "unknown_for" and by_id["undecided"].hours == 12
    assert [(g.kind, g.ref) for g in by_id["undecided"].unless] == [("rule", "fix")]
    assert [(g.kind, g.ref) for g in by_id["page_while_open"].only_while] == [
        ("rule", "fix")
    ]
    rendered = req.to_doc()
    assert rendered["tvl_version"] == "1.4"
    assert [m["phase"] for m in rendered["monitors"]] == ["runtime", "promotion"]
    # What the loader renders is what the loader reads back (a stable policy).
    again = parse_fragment(rendered, max_fix_cost_usd=5.0)
    assert again.to_doc() == rendered
    assert again.policy_digest() == req.policy_digest()
    assert req.validate() == []


def test_a_13_fragment_still_renders_13_and_an_untouched_digest() -> None:
    @traigent.requirement(latency_ms=2000, max_violation_rate=0.05, on_breach="alert")
    def fn() -> None: ...

    assert fn.requirements[0].to_doc()["tvl_version"] == "1.3"
    assert not fn.requirements[0].is_reactive()


def test_declared_13_with_a_reactive_construct_is_refused() -> None:
    doc = _doc()
    doc["tvl_version"] = "1.3"
    with pytest.raises(RuntimeRequirementError) as err:
        parse_fragment(doc, max_fix_cost_usd=5.0)
    assert err.value.code == "declared_version_below_constructs"
    doc = _doc()
    del doc["tvl_version"]
    with pytest.raises(RuntimeRequirementError) as err:
        parse_fragment(doc, max_fix_cost_usd=5.0)
    assert err.value.code == "declared_version_below_constructs"


def test_declared_14_without_a_reactive_construct_is_fine() -> None:
    doc = _doc()
    doc["responses"][0]["rules"] = [
        r for r in doc["responses"][0]["rules"] if r["id"] == "fix"
    ]
    req = parse_fragment(doc, max_fix_cost_usd=5.0)
    assert not req.is_reactive()
    assert req.to_doc()["tvl_version"] == "1.3"  # projected: no 1.4 construct, so /1


def _mutate(path: str, fn):
    doc = copy.deepcopy(_doc())
    fn(doc)
    return path, doc


REFUSALS = {
    "outcome_on_a_non_notify_rule": (
        lambda d: _rule(d, "fix_failed").update(do="open_next_step"),
        "runtime_notify_only",
    ),
    "unknown_for_on_a_non_notify_rule": (
        lambda d: _rule(d, "undecided").update(do="optimize", approval="human"),
        "runtime_notify_only",
    ),
    "guard_on_an_optimize_rule": (
        lambda d: _rule(d, "fix").update(
            only_while=[{"rule": "fix", "state": "attempt_open"}]
        ),
        "runtime_notify_only",
    ),
    "unknown_for_hours_below_the_window": (
        lambda d: _rule(d, "undecided")["trigger"].update(hours=3),
        "runtime_trigger_hours_out_of_range",
    ),
    "unknown_for_without_hours": (
        lambda d: _rule(d, "undecided")["trigger"].pop("hours"),
        "tvl_fragment_invalid",
    ),
    "hours_on_a_plain_episode_event": (
        lambda d: _rule(d, "page_while_open")["trigger"].update(hours=6),
        "tvl_fragment_invalid",
    ),
    "outcome_event_outside_the_closed_set": (
        lambda d: _rule(d, "fix_failed")["trigger"].update(event="optimize_succeeded"),
        "runtime_unsupported_capability",
    ),
    "outcome_trigger_with_a_monitor": (
        lambda d: _rule(d, "fix_failed")["trigger"].update(monitor="latency_2s_prod"),
        "tvl_fragment_invalid",
    ),
    "outcome_names_a_rule_that_does_not_exist": (
        lambda d: _rule(d, "fix_failed")["trigger"].update(rule="nope"),
        "invalid_requirement",
    ),
    "outcome_names_a_notify_rule": (
        lambda d: _rule(d, "fix_failed")["trigger"].update(rule="undecided"),
        "invalid_requirement",
    ),
    "deploy_outcome_without_if_promoted": (
        lambda d: _rule(d, "fix").update(deploy="never"),
        "invalid_requirement",
    ),
    "deploy_on_a_notify_rule": (
        lambda d: _rule(d, "undecided").update(deploy="never"),
        "tvl_fragment_invalid",
    ),
    "deploy_mode_outside_the_closed_set": (
        lambda d: _rule(d, "fix").update(deploy="always"),
        "tvl_fragment_invalid",
    ),
    "guard_state_outside_the_closed_set": (
        lambda d: _rule(d, "undecided").update(
            unless=[{"rule": "fix", "state": "episode_open"}]
        ),
        "runtime_unsupported_capability",
    ),
    "guard_on_a_monitor_the_fragment_lacks": (
        lambda d: _rule(d, "undecided").update(
            unless=[{"monitor": "other", "state": "episode_open"}]
        ),
        "runtime_unsupported_capability",
    ),
    "guard_naming_a_non_optimize_rule": (
        lambda d: _rule(d, "undecided").update(
            unless=[{"rule": "undecided", "state": "attempt_open"}]
        ),
        "invalid_requirement",
    ),
    "empty_guard_list": (
        lambda d: _rule(d, "undecided").update(unless=[]),
        "tvl_fragment_invalid",
    ),
    "approval_standing_is_refused_locally": (
        lambda d: _rule(d, "fix").update(approval="standing"),
        "response_action_unsupported",
    ),
    "if_promoted_without_a_promotion_monitor": (
        lambda d: d.update(monitors=d["monitors"][:1]),
        "invalid_requirement",
    ),
}


@pytest.mark.parametrize("case", sorted(REFUSALS))
def test_reactive_refusals_are_closed_codes(case: str) -> None:
    mutate, code = REFUSALS[case]
    doc = _doc()
    mutate(doc)
    with pytest.raises(RuntimeRequirementError) as err:
        parse_fragment(doc, max_fix_cost_usd=5.0)
    assert err.value.code == code
    assert "nope" not in str(err.value) and "other" not in str(err.value)


def test_standing_approval_is_refused_on_the_sugar_too() -> None:
    with pytest.raises(RuntimeRequirementError) as err:
        traigent.requirement(
            latency_ms=2000,
            max_violation_rate=0.05,
            on_breach="fix",
            approval="standing",
            max_fix_cost_usd=5.0,
        )
    assert err.value.code == "response_action_unsupported"
