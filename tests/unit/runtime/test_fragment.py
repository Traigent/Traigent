"""TVL fragment loader: accepted shape, closed refusals, no echo of user values."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import traigent
from traigent.runtime.model import RuntimeRequirementError
from traigent.runtime.tvl_fragment import load_fragment, parse_fragment

from .conftest import FIXTURES

FRAGMENT = FIXTURES / "latency.tvl.yml"


def _doc() -> dict:
    return yaml.safe_load(FRAGMENT.read_text())


def test_design_fragment_is_accepted() -> None:
    req = load_fragment(FRAGMENT, max_fix_cost_usd=5.0)
    assert (req.name, req.latency_threshold_ms, req.max_violation_rate) == (
        "latency_2s",
        2000,
        0.05,
    )
    assert req.monitor_name == "latency_2s_prod"
    assert [r.action for r in req.rules] == [
        "notify",
        "open_next_step",
        "optimize",
        "notify",
    ]
    assert req.window_hours == 6 and req.min_samples == 200  # backend_v1 defaults
    assert req.source == "tvl"


def test_sugar_and_fragment_build_the_same_tvl_object() -> None:
    sugar = traigent.requirement(
        latency_ms=2000, max_violation_rate=0.05, on_breach="alert", name="latency_2s"
    )

    @sugar
    def fn() -> None: ...

    frag = {
        "tvl_version": "1.3",
        "requirements": [_doc()["requirements"][0]],
        "monitors": [
            {**_doc()["monitors"][0], "name": "latency_2s_monitor"},
        ],
        "responses": fn.requirements[0].to_doc()["responses"],
    }
    parsed = parse_fragment(frag)
    assert parsed.to_doc() == fn.requirements[0].to_doc()
    assert parsed.policy_digest() == fn.requirements[0].policy_digest()


def test_decorator_accepts_tvl_fragment_with_execution_metadata() -> None:
    @traigent.requirement(tvl=str(FRAGMENT), max_fix_cost_usd=2.5, agent_id="a1")
    def fn() -> None: ...

    req = fn.requirements[0]
    assert (req.max_fix_cost_usd, req.agent_id, req.source) == (2.5, "a1", "tvl")


def _mutations():
    def second_monitor(d):
        d["monitors"].append(dict(d["monitors"][0], name="other"))

    def rollback(d):
        d["responses"][0]["rules"][0] = {
            "id": "rb",
            "trigger": {"monitor": "latency_2s_prod", "event": "episode_opened"},
            "do": "rollback",
        }

    def standing(d):
        d["responses"][0]["rules"][2]["approval"] = "standing"

    def pack(d):
        d["pack"] = {"module": "p"}

    def burn(d):
        d["requirements"][0]["kind"] = "burn"

    def cost(d):
        d["requirements"][0]["violation"]["metric"] = "cost_usd"

    def bad_event(d):
        d["responses"][0]["rules"][0]["trigger"]["event"] = "stale"

    def second_response(d):
        d["responses"].append(dict(d["responses"][0], name="b"))

    def non_backend_profile(d):
        d["monitors"][0]["profile"] = "local"

    return {
        "second_monitor": (
            second_monitor,
            "runtime_unsupported_capability",
            "monitors[1]",
        ),
        "rollback": (
            rollback,
            "response_action_unsupported",
            "responses[0].rules[0].do",
        ),
        "standing": (
            standing,
            "response_action_unsupported",
            "responses[0].rules[2].approval",
        ),
        "pack": (pack, "runtime_unsupported_capability", "<root>.<key>"),
        "burn": (burn, "runtime_unsupported_capability", "requirements[0].kind"),
        "cost_usd": (
            cost,
            "runtime_unsupported_capability",
            "requirements[0].violation.metric",
        ),
        "bad_event": (
            bad_event,
            "runtime_unsupported_capability",
            "responses[0].rules[0].trigger.event",
        ),
        "second_response": (
            second_response,
            "runtime_unsupported_capability",
            "responses[1]",
        ),
        "profile": (
            non_backend_profile,
            "runtime_unsupported_capability",
            "monitors[0].profile",
        ),
    }


@pytest.mark.parametrize("case", sorted(_mutations()))
def test_unsupported_constructs_refused_with_code_and_path(case: str) -> None:
    mutate, code, path = _mutations()[case]
    doc = _doc()
    mutate(doc)
    with pytest.raises(RuntimeRequirementError) as err:
        parse_fragment(doc)
    assert (err.value.code, err.value.path) == (code, path)


def test_errors_never_echo_user_values() -> None:
    secret = "SECRET_VALUE_f3a9"
    doc = _doc()
    doc["requirements"][0]["violation"]["metric"] = secret
    doc[secret] = 1
    for mutate in (
        lambda d: d["requirements"][0].__setitem__("name", secret + " bad name"),
        lambda d: d["monitors"][0]["evidence"].__setitem__("source", secret),
        lambda d: d["responses"][0]["rules"][0].__setitem__("do", secret),
    ):
        d = _doc()
        mutate(d)
        with pytest.raises(RuntimeRequirementError) as err:
            parse_fragment(d)
        assert secret not in str(err.value) and secret not in repr(err.value.path)
    with pytest.raises(RuntimeRequirementError) as err:
        parse_fragment(doc)
    assert secret not in str(err.value)


def test_unreadable_or_oversized_file_is_closed_error(tmp_path: Path) -> None:
    with pytest.raises(RuntimeRequirementError) as err:
        load_fragment(tmp_path / "missing.yml")
    assert err.value.code == "tvl_fragment_invalid"
    big = tmp_path / "big.yml"
    big.write_text("x: " + "a" * 70_000)
    with pytest.raises(RuntimeRequirementError):
        load_fragment(big)
    bad = tmp_path / "bad.yml"
    bad.write_text("a: [unterminated")
    with pytest.raises(RuntimeRequirementError):
        load_fragment(bad)
