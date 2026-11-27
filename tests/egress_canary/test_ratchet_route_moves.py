"""Exercise the ratchet's route-move rules with synthetic leak identities."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from . import pins, test_egress_ratchet
from .canaries import Leak
from .pins import Identity, ratchet_problems
from .scenario import ScenarioResult

OLD = "POST /r/old"
NEW = "POST /r/new"
OTHER = "POST /r/other"
THIRD = "POST /r/third"
IDS = {"S1", "S2"}


@pytest.fixture(autouse=True)
def routes(monkeypatch: pytest.MonkeyPatch) -> None:
    for route in (OLD, NEW, OTHER, THIRD):
        monkeypatch.setitem(pins.ROUTE_INVENTORY_IDS, route, frozenset({"S1"}))


@pytest.fixture
def declared_move(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pins, "ROUTE_MOVES", {OLD: NEW}, raising=False)


def at(
    route: str, location: str = "body.a", field_path: str = "$.x", kind: str = "kind_a"
) -> Identity:
    return (route, location, field_path, kind)


def entry(
    identity: Identity,
    moved_to: str | None = None,
    reason: str = "reviewed",
    reference: str = "INV:S1",
) -> dict[str, str]:
    route, location, field_path, kind = identity
    row = {
        "route": route,
        "location": location,
        "field_path": field_path,
        "kind": kind,
        "reason": reason,
        "reference": reference,
    }
    if moved_to is not None:
        row["moved_to"] = moved_to
    return row


def problems(
    live: set[Identity],
    anchor: set[Identity],
    register: set[Identity],
    dispositions: list[dict[str, str]] | None = None,
) -> list[str]:
    return ratchet_problems(live, anchor, register, dispositions or [], ids=IDS)


def rejects_addition(probs: list[str], identity: Identity) -> bool:
    return any(p.startswith("new leak ") and p.endswith(repr(identity)) for p in probs)


@pytest.mark.usefixtures("declared_move")
def test_pure_route_move_passes():
    old = at(OLD)
    assert problems({at(NEW)}, {old}, {at(NEW)}, [entry(old, moved_to=NEW)]) == []


def test_shared_inventory_section_without_a_declaration_is_rejected(monkeypatch):
    monkeypatch.setattr(pins, "ROUTE_MOVES", {}, raising=False)
    source = "POST /api/v1/example-sets/{id}/upload"
    destination = "POST /api/v1/example-sets"
    old = at(source, location="body.raw", field_path="", kind="example_id")
    new = at(destination, *old[1:])
    assert pins.ROUTE_INVENTORY_IDS[source] == pins.ROUTE_INVENTORY_IDS[destination]
    row = entry(old, moved_to=destination, reference="INV:E1")
    probs = ratchet_problems({new}, {old}, {new}, [row], ids={"E1"})
    assert rejects_addition(probs, new)


@pytest.mark.usefixtures("declared_move")
def test_moved_to_must_match_the_declared_destination():
    old, third = at(OLD), at(THIRD)
    probs = problems({third}, {old}, {third}, [entry(old, moved_to=THIRD)])
    assert rejects_addition(probs, third)


def test_route_move_table_has_unique_destinations():
    assert len(pins.ROUTE_MOVES) == len(set(pins.ROUTE_MOVES.values()))


@pytest.mark.parametrize(
    ("declare", "changes", "expected"),
    [
        (True, {}, True),
        (False, {}, False),
        (True, {"location": "body.b"}, False),
        (True, {"field_path": "$.z"}, False),
        (True, {"kind": "kind_b"}, False),
    ],
    ids=["declared", "undeclared", "location", "field_path", "kind"],
)
def test_register_route_rekey_against_baseline(monkeypatch, declare, changes, expected):
    old, new = at(OLD), at(NEW, **changes)
    monkeypatch.setattr(
        pins, "ROUTE_MOVES", {OLD: NEW} if declare else {}, raising=False
    )
    monkeypatch.setattr(test_egress_ratchet, "BASELINE", {old})
    monkeypatch.setattr(test_egress_ratchet, "REGISTER", {new})
    monkeypatch.setattr(
        test_egress_ratchet, "load_dispositions", lambda: [entry(old, moved_to=NEW)]
    )
    if expected:
        test_egress_ratchet.test_manifest_and_dispositions_are_consistent()
    else:
        with pytest.raises(AssertionError, match="register may only shrink"):
            test_egress_ratchet.test_manifest_and_dispositions_are_consistent()


def check_manifest(
    monkeypatch: pytest.MonkeyPatch,
    baseline: set[Identity],
    register: set[Identity],
    dispositions: list[dict[str, str]],
) -> None:
    monkeypatch.setattr(test_egress_ratchet, "BASELINE", baseline)
    monkeypatch.setattr(test_egress_ratchet, "REGISTER", register)
    monkeypatch.setattr(test_egress_ratchet, "load_dispositions", lambda: dispositions)
    test_egress_ratchet.test_manifest_and_dispositions_are_consistent()


@pytest.mark.usefixtures("declared_move")
def test_move_one_leak_and_remove_another_on_the_same_route(monkeypatch):
    old, new = at(OLD), at(NEW)
    fixed = at(OLD, field_path="$.y")
    dispositions = [entry(old, moved_to=NEW), entry(fixed)]

    assert problems({new}, {old, fixed}, {new}, dispositions) == []
    check_manifest(monkeypatch, {old, fixed}, {new}, dispositions)


@pytest.mark.usefixtures("declared_move")
def test_plain_removal_on_the_wrong_route_is_rejected(monkeypatch):
    old, new = at(OLD), at(NEW)
    fixed = at(OLD, field_path="$.y")
    wrong_route = at(NEW, field_path="$.y")
    dispositions = [entry(old, moved_to=NEW), entry(wrong_route)]

    with pytest.raises(
        AssertionError, match="every identity removed from the register"
    ):
        check_manifest(monkeypatch, {old, fixed}, {new}, dispositions)


@pytest.mark.parametrize(
    ("declared", "changes"),
    [
        (True, {}),
        (False, {}),
        (True, {"location": "body.key"}),
        (True, {"field_path": "$.other"}),
        (True, {"kind": "function_name"}),
    ],
    ids=["declared", "undeclared", "location", "field_path", "kind"],
)
def test_legacy_baseline_checks_the_route_move(monkeypatch, declared, changes):
    agent = Leak(NEW, "body.value", "agent_name", "raw", "$.name")
    other_leaks = [
        Leak(OTHER, "body.value", "evaluation_set", "raw", "$.eval"),
        Leak(OTHER, "body.value", "evaluator_id", "raw", "$.evaluator"),
    ]
    old = (OLD, *agent.identity[1:])
    monkeypatch.setattr(pins, "ROUTE_MOVES", {OLD: NEW} if declared else {})
    monkeypatch.setattr(
        test_egress_ratchet, "BASELINE", {old, *(leak.identity for leak in other_leaks)}
    )
    monkeypatch.setattr(
        test_egress_ratchet, "load_dispositions", lambda: [entry(old, moved_to=NEW)]
    )
    scenario = ScenarioResult(
        requests=[],
        guard_violations=[],
        unhandled_routes=[],
        optimization_error=None,
        result_source=None,
        leaks=[replace(agent, **changes), *other_leaks],
    )

    if declared and not changes:
        test_egress_ratchet.test_legacy_default_run_leaks_stay_inside_the_baseline(
            scenario
        )
    else:
        with pytest.raises(AssertionError, match="legacy shape grew leaks"):
            test_egress_ratchet.test_legacy_default_run_leaks_stay_inside_the_baseline(
                scenario
            )


@pytest.mark.parametrize("destination", [NEW, THIRD], ids=["direct", "transitive"])
@pytest.mark.usefixtures("declared_move")
def test_legacy_baseline_carries_an_identity_with_a_plain_removal(
    monkeypatch, destination
):
    agent = Leak(OLD, "body.value", "agent_name", "raw", "$.name")
    evaluator = Leak(OLD, "body.value", "evaluator_id", "raw", "$.evaluator_id")
    evaluation_set = Leak(OTHER, "body.value", "evaluation_set", "raw", "$.eval")
    dispositions = [entry(agent.identity, moved_to=NEW), entry(evaluator.identity)]
    if destination == THIRD:
        monkeypatch.setitem(pins.ROUTE_MOVES, NEW, THIRD)
        dispositions.append(entry(replace(agent, route=NEW).identity, moved_to=THIRD))
    baseline = {agent.identity, evaluator.identity, evaluation_set.identity}
    register = {replace(agent, route=destination).identity, evaluation_set.identity}
    check_manifest(monkeypatch, baseline, register, dispositions)

    scenario = ScenarioResult(
        requests=[],
        guard_violations=[],
        unhandled_routes=[],
        optimization_error=None,
        result_source=None,
        leaks=[
            replace(agent, route=destination),
            replace(evaluator, route=destination),
            evaluation_set,
        ],
    )
    test_egress_ratchet.test_legacy_default_run_leaks_stay_inside_the_baseline(scenario)


@pytest.mark.usefixtures("declared_move")
def test_later_removal_from_the_moved_route_passes(monkeypatch):
    old, new = at(OLD), at(NEW)
    history = [entry(old, moved_to=NEW)]
    check_manifest(monkeypatch, {old}, {new}, history)

    dispositions = [*history, entry(new)]
    assert problems(set(), {new}, set(), dispositions) == []
    check_manifest(monkeypatch, {old}, set(), dispositions)


@pytest.mark.usefixtures("declared_move")
def test_later_declared_move_from_the_moved_route_passes(monkeypatch):
    old, new, third = at(OLD), at(NEW), at(THIRD)
    history = [entry(old, moved_to=NEW)]
    check_manifest(monkeypatch, {old}, {new}, history)

    monkeypatch.setitem(pins.ROUTE_MOVES, NEW, THIRD)
    dispositions = [*history, entry(new, moved_to=THIRD)]
    assert problems({third}, {new}, {third}, dispositions) == []
    check_manifest(monkeypatch, {old}, {third}, dispositions)


@pytest.mark.usefixtures("declared_move")
def test_move_chain_stops_at_each_identitys_last_disposition(monkeypatch):
    old, new, third = at(OLD), at(NEW), at(THIRD)
    other_old = at(OLD, field_path="$.y")
    other_new = at(NEW, field_path="$.y")
    monkeypatch.setitem(pins.ROUTE_MOVES, NEW, THIRD)
    dispositions = [
        entry(old, moved_to=NEW),
        entry(new, moved_to=THIRD),
        entry(other_old, moved_to=NEW),
        entry(other_new),
    ]

    check_manifest(monkeypatch, {old, other_old}, {third}, dispositions)


def test_route_move_cycle_raises(monkeypatch):
    monkeypatch.setattr(pins, "ROUTE_MOVES", {OLD: NEW, NEW: OLD})
    with pytest.raises(ValueError, match="cycle in ROUTE_MOVES"):
        pins.effective_baseline({at(OLD)}, [])
    with pytest.raises(ValueError, match="cycle in ROUTE_MOVES"):
        pins.route_level_baseline({at(OTHER)})


def test_empty_route_moves_preserve_the_checked_in_baseline(monkeypatch):
    dispositions = [row for row in pins.load_dispositions() if "moved_to" not in row]
    monkeypatch.setattr(pins, "ROUTE_MOVES", {})
    assert pins.effective_baseline(pins.BASELINE, dispositions) == pins.BASELINE
    assert pins.route_level_baseline(pins.BASELINE) == pins.BASELINE


@pytest.mark.usefixtures("declared_move")
def test_stale_plain_disposition_is_rejected(monkeypatch):
    old, new = at(OLD), at(NEW)
    dispositions = [entry(old, moved_to=NEW), entry(at(OTHER))]
    with pytest.raises(AssertionError, match=r"stale: .*POST /r/other"):
        check_manifest(monkeypatch, {old}, {new}, dispositions)


@pytest.mark.usefixtures("declared_move")
def test_manifest_still_requires_the_move_disposition(monkeypatch):
    with pytest.raises(
        AssertionError, match=r"register may only shrink: .*POST /r/new"
    ):
        check_manifest(monkeypatch, {at(OLD)}, {at(NEW)}, [])


@pytest.mark.usefixtures("declared_move")
def test_move_history_cannot_start_outside_the_baseline(monkeypatch):
    monkeypatch.setitem(pins.ROUTE_MOVES, OTHER, OLD)
    dispositions = [entry(at(OTHER), moved_to=OLD), entry(at(OLD), moved_to=NEW)]
    with pytest.raises(AssertionError, match=r"stale: .*POST /r/other"):
        check_manifest(monkeypatch, {at(OLD)}, {at(NEW)}, dispositions)


def test_plain_removal_still_passes():
    old = at(OLD)
    assert problems(set(), {old}, set(), [entry(old)]) == []


def test_unexplained_addition_still_fails():
    added = at(NEW)
    probs = problems({added}, set(), {added})
    assert rejects_addition(probs, added)


@pytest.mark.parametrize(
    "changed",
    [{"location": "body.b"}, {"field_path": "$.z"}, {"kind": "kind_b"}],
    ids=["location", "field_path", "kind"],
)
@pytest.mark.usefixtures("declared_move")
def test_move_that_changes_location_field_path_or_kind_is_rejected(changed):
    old = at(OLD)
    moved = at(NEW, **changed)
    probs = problems({moved}, {old}, {moved}, [entry(old, moved_to=NEW)])
    assert rejects_addition(probs, moved)


@pytest.mark.usefixtures("declared_move")
def test_copy_where_the_old_identity_stays_live_is_rejected():
    # The unrelated removal of ``gone`` keeps the total count equal, so only the copy
    # (old still live, new added) can make this fail.
    old, gone, new = at(OLD), at(OTHER, field_path="$.gone"), at(NEW)
    probs = problems(
        {old, new},
        {old, gone},
        {new},
        [entry(old, moved_to=NEW), entry(gone)],
    )
    assert rejects_addition(probs, new)


def test_two_old_identities_folding_into_one_new_identity_is_rejected(monkeypatch):
    monkeypatch.setattr(pins, "ROUTE_MOVES", {OLD: NEW, OTHER: NEW}, raising=False)
    probs = problems(
        {at(NEW)},
        {at(OLD), at(OTHER)},
        {at(NEW)},
        [entry(at(OLD), moved_to=NEW), entry(at(OTHER), moved_to=NEW)],
    )
    assert rejects_addition(probs, at(NEW))


@pytest.mark.usefixtures("declared_move")
def test_one_old_identity_cannot_be_claimed_by_two_added_identities():
    old, gone = at(OLD), at(OTHER, field_path="$.gone")
    probs = problems(
        {at(NEW), at(THIRD)},
        {old, gone},
        {at(NEW), at(THIRD)},
        [entry(old, moved_to=NEW), entry(old, moved_to=THIRD), entry(gone)],
    )
    assert rejects_addition(probs, at(NEW))
    assert rejects_addition(probs, at(THIRD))


@pytest.mark.usefixtures("declared_move")
def test_one_disposition_cannot_cover_two_additions_on_the_target_route():
    old, gone = at(OLD), at(OTHER, field_path="$.gone")
    new, extra = at(NEW), at(NEW, field_path="$.extra")
    probs = problems(
        {new, extra}, {old, gone}, {new, extra}, [entry(old, moved_to=NEW), entry(gone)]
    )
    assert rejects_addition(probs, extra)


def test_moved_to_must_name_the_added_route(monkeypatch):
    monkeypatch.setattr(pins, "ROUTE_MOVES", {OLD: THIRD}, raising=False)
    old = at(OLD)
    probs = problems({at(NEW)}, {old}, {at(NEW)}, [entry(old, moved_to=THIRD)])
    assert rejects_addition(probs, at(NEW))


@pytest.mark.usefixtures("declared_move")
def test_added_route_without_inventory_mapping_is_rejected(monkeypatch):
    monkeypatch.delitem(pins.ROUTE_INVENTORY_IDS, NEW)
    old = at(OLD)
    probs = problems({at(NEW)}, {old}, {at(NEW)}, [entry(old, moved_to=NEW)])
    assert rejects_addition(probs, at(NEW))


@pytest.mark.usefixtures("declared_move")
def test_added_route_mapped_to_a_different_inventory_id_is_rejected(monkeypatch):
    monkeypatch.setitem(pins.ROUTE_INVENTORY_IDS, NEW, frozenset({"S2"}))
    old = at(OLD)
    probs = problems({at(NEW)}, {old}, {at(NEW)}, [entry(old, moved_to=NEW)])
    assert rejects_addition(probs, at(NEW))


@pytest.mark.usefixtures("declared_move")
def test_valid_move_plus_extra_identity_is_rejected():
    old, extra = at(OLD), at(OTHER, field_path="$.extra")
    probs = problems(
        {at(NEW), extra},
        {old},
        {at(NEW), extra},
        [entry(old, moved_to=NEW)],
    )
    assert rejects_addition(probs, at(NEW))
    assert rejects_addition(probs, extra)


@pytest.mark.usefixtures("declared_move")
def test_per_kind_growth_hidden_by_a_stable_total_is_rejected():
    # Total stays at two; kind_a goes from one identity to two.
    old, other = at(OLD, kind="kind_a"), at(OTHER, kind="kind_b")
    moved, extra = at(NEW, kind="kind_a"), at(THIRD, kind="kind_a")
    probs = problems(
        {moved, extra},
        {old, other},
        {moved, extra},
        [entry(old, moved_to=NEW), entry(other)],
    )
    assert rejects_addition(probs, moved)
    assert rejects_addition(probs, extra)


def test_unknown_moved_to_with_no_replacement_is_rejected():
    # Red before the implementation: today's code accepts this as a plain removal.
    old = at(OLD)
    assert problems(set(), {old}, set(), [entry(old, moved_to="POST /r/unknown")])


@pytest.mark.usefixtures("declared_move")
def test_move_without_matching_addition_is_not_a_plain_removal():
    old = at(OLD)
    assert problems(set(), {old}, set(), [entry(old, moved_to=NEW)])


@pytest.mark.usefixtures("declared_move")
def test_already_anchored_move_can_stay_in_disposition_history():
    new = at(NEW)
    assert problems({new}, {new}, {new}, [entry(at(OLD), moved_to=NEW)]) == []


def test_unknown_destination_fails_even_without_an_anchor_delta():
    new = at(NEW)
    assert problems({new}, {new}, {new}, [entry(at(OLD), moved_to="POST /r/unknown")])


@pytest.mark.usefixtures("declared_move")
def test_register_must_gain_the_new_identity():
    old = at(OLD)
    probs = problems({at(NEW)}, {old}, set(), [entry(old, moved_to=NEW)])
    assert rejects_addition(probs, at(NEW))


@pytest.mark.usefixtures("declared_move")
def test_register_must_drop_the_old_identity():
    old = at(OLD)
    probs = problems({at(NEW)}, {old}, {at(NEW), old}, [entry(old, moved_to=NEW)])
    assert rejects_addition(probs, at(NEW))


@pytest.mark.parametrize("reference", ["INV:NOPE", "INV:S2"])
@pytest.mark.usefixtures("declared_move")
def test_move_with_invalid_reference_is_rejected(reference):
    # INV:NOPE is unknown; INV:S2 is a known id that is not a section of this route.
    old = at(OLD)
    row = entry(old, moved_to=NEW, reference=reference)
    probs = problems({at(NEW)}, {old}, {at(NEW)}, [row])
    assert rejects_addition(probs, at(NEW))


@pytest.mark.usefixtures("declared_move")
def test_move_with_blank_reason_is_rejected():
    old = at(OLD)
    row = entry(old, moved_to=NEW, reason="   ")
    probs = problems({at(NEW)}, {old}, {at(NEW)}, [row])
    assert rejects_addition(probs, at(NEW))


@pytest.mark.usefixtures("declared_move")
def test_moves_require_inventory_references_but_removals_accept_file_references(
    tmp_path: Path,
) -> None:
    (tmp_path / "payload.py").write_text("payload = {}\n", encoding="utf-8")
    old, new = (at(route, field_path="$.payload") for route in (OLD, NEW))
    reference = "payload.py:1"

    removal = entry(old, reference=reference)
    assert ratchet_problems(set(), {old}, set(), [removal], tmp_path, IDS) == []

    move = entry(old, moved_to=NEW, reference=reference)
    probs = ratchet_problems({new}, {old}, {new}, [move], tmp_path, IDS)
    assert rejects_addition(probs, new)
