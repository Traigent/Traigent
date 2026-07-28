"""Issue #2031: the OptimizationResult round-trip manifest must stay complete.

Both persisted result formats used to rebuild the dataclass with a hand-written
constructor call naming a subset of its fields, so every field added afterwards
was dropped on load with every test still green. The fix replaces both call
sites with a single manifest that partitions **all** the dataclass's fields into
"restored" and "deliberately reset".

The tests here are the guard on that manifest rather than on either loader: they
read ``dataclasses.fields(OptimizationResult)`` directly, so a 28th field fails
:func:`test_manifest_covers_every_optimization_result_field` on its first run
until whoever adds it records a decision. The per-loader value assertions live in
``tests/unit/core/test_config_state_manager_reload_2031.py`` and
``tests/unit/utils/test_persistence.py``.

Cost safety: these are pure serializer unit tests. No ``OptimizedFunction``, no
``optimize()``, no evaluator or optimizer is constructed, so there is no LLM
call, no network, and no spend.
"""
# Traceability: CONC-Layer-Data CONC-Quality-Reliability FUNC-STORAGE REQ-STOR-007

from __future__ import annotations

import copy
import dataclasses
import logging
import socket
from datetime import UTC, datetime

import pytest

from traigent.api.types import OptimizationResult, OptimizationStatus, TrialResult
from traigent.utils import optimization_result_persistence as manifest
from traigent.utils.optimization_result_persistence import (
    RESULT_RESET,
    RESULT_RESTORE,
    RESULT_SCHEMA_VERSION,
    SCHEMA_VERSION_KEY,
    _SENTINELS,
    decode_result,
    encode_result_fields,
)

_LOGGER_NAME = "traigent.utils.optimization_result_persistence"


def _sentinel_result() -> OptimizationResult:
    """A result whose every restorable field holds a non-default value."""
    return OptimizationResult(**copy.deepcopy(_SENTINELS))


def _encoded(result: OptimizationResult | None = None) -> dict:
    return encode_result_fields(result if result is not None else _sentinel_result())


# --- The manifest itself ----------------------------------------------------


def test_manifest_covers_every_optimization_result_field() -> None:
    """RESTORE + RESET must be an exact *partition* of the dataclass.

    This is the acceptance criterion for #2031: the defect was not any one
    missing field, it was that adding a field silently opted it out of
    persistence. Enumerating ``dataclasses.fields`` means the manifest cannot go
    stale — a 28th field fails here until its author decides, in the manifest,
    whether it survives a round trip.

    Both halves of "partition" are asserted, and neither is redundant:

    * **coverage** — the union must be exactly the declared field names. A 28th
      field (or a stale name left behind by a rename) fails this.
    * **disjointness** — the two sets must not overlap. A union-only assertion
      passes when a field is listed in *both* sets, which is not a decision but
      a contradiction: RESTORE says "must survive a round trip", RESET says
      "must not", and the decoder honours whichever it happens to consult.
    """
    declared = {field.name for field in dataclasses.fields(OptimizationResult)}

    assert RESULT_RESTORE | RESULT_RESET == declared, (
        "OptimizationResult fields not classified in "
        "traigent/utils/optimization_result_persistence.py: "
        f"unclassified={sorted(declared - (RESULT_RESTORE | RESULT_RESET))}, "
        f"classified but not declared on the dataclass="
        f"{sorted((RESULT_RESTORE | RESULT_RESET) - declared)}"
    )
    assert RESULT_RESTORE & RESULT_RESET == frozenset(), (
        "A field cannot be both restored and reset; the manifest contradicts "
        f"itself for: {sorted(RESULT_RESTORE & RESULT_RESET)}"
    )
    assert len(declared) == 27, (
        f"OptimizationResult now declares {len(declared)} fields, not 27. If you "
        f"added one, updating this number is the LAST step, not the first: a "
        f"restorable field also needs a _SENTINELS entry, a bumped "
        f"RESULT_SCHEMA_VERSION, a FIELD_INTRODUCED_IN entry at that bumped "
        f"version, and a frozen golden artifact for the version you just left "
        f"behind (see _GOLDEN_ARTIFACTS below). Change this number alone and "
        f"every result already on disk stops loading."
    )
    # Belt and braces on the partition: with the two sets disjoint, their sizes
    # must add up to the declared count. This fails on a duplicate even if the
    # two assertions above were ever weakened independently.
    assert len(RESULT_RESTORE) + len(RESULT_RESET) == len(declared)


def test_reset_set_is_exactly_the_two_documented_non_durable_fields() -> None:
    """Non-restoration is a contract, not an omission — so name the members.

    ``sync_session_id`` is pinned by the #2020 suites; ``_experiment_stats`` is
    an ``init=False`` memo cache recomputed from ``trials``. Anything else
    landing in RESET is a fidelity regression and must be argued for here.
    """
    assert RESULT_RESET == frozenset({"sync_session_id", "_experiment_stats"})


def test_sentinel_table_covers_every_restored_field() -> None:
    """Every restorable field needs a reference value to be tested against.

    Without this, a field could be added to RESTORE and still never be asserted
    on by the round-trip suites — the manifest would claim coverage it does not
    have.
    """
    assert set(_SENTINELS) == RESULT_RESTORE


def test_sentinel_values_are_all_non_default() -> None:
    """A sentinel equal to the field's default proves nothing on reload."""
    defaults = {
        field.name: (
            field.default_factory()
            if field.default_factory is not dataclasses.MISSING
            else field.default
        )
        for field in dataclasses.fields(OptimizationResult)
    }
    equal_to_default = [
        name
        for name, value in _SENTINELS.items()
        if defaults[name] is not dataclasses.MISSING and value == defaults[name]
    ]
    assert equal_to_default == []


def test_manifest_drives_behaviour_and_the_round_trip_pins_can_fail(
    monkeypatch,
) -> None:
    """Mutation probe: shrink the manifest and the fidelity assertions must fail.

    Layers 1-3 above are enumerations; this proves they are not vacuous. With
    ``total_cost`` removed from RESTORE the encoder stops writing it and the
    decoder stops requiring it, so the field comes back at its default — exactly
    the pre-#2031 defect, reproduced on demand.
    """
    shrunk = RESULT_RESTORE - {"total_cost"}
    monkeypatch.setattr(manifest, "RESULT_RESTORE", shrunk)

    encoded = _encoded()
    assert "total_cost" not in encoded

    restored = decode_result(encoded, trials=[])
    assert restored.total_cost is None
    assert _SENTINELS["total_cost"] == 0.0234  # … which is NOT the sentinel


# --- Versioned artifacts ----------------------------------------------------


def test_encode_decode_round_trips_every_restored_field() -> None:
    """Decoder-level fidelity, independent of either on-disk format."""
    encoded = _encoded()
    restored = decode_result(encoded, trials=copy.deepcopy(_SENTINELS["trials"]))

    for name in sorted(RESULT_RESTORE - {"trials"}):
        assert getattr(restored, name) == _SENTINELS[name], f"{name} did not survive"
    assert [trial.trial_id for trial in restored.trials] == ["trial-sentinel-0"]


def test_preset_selection_survives_a_rationale_longer_than_the_280_char_clamp() -> None:
    """The encoder must use ``asdict``, not ``to_dict`` (captain decision 3).

    ``PresetSelection.to_dict`` routes through ``to_metadata``, which truncates
    ``selection_rationale`` to 280 characters and drops it entirely when falsy.
    A persistence encoder that loses data the reader can see is not a round trip.
    """
    original = _SENTINELS["preset_selection"]
    assert len(original.selection_rationale) > 280

    restored = decode_result(_encoded(), trials=[])

    assert restored.preset_selection is not None
    assert restored.preset_selection.selection_rationale == original.selection_rationale
    assert len(restored.preset_selection.selection_rationale) == len(
        original.selection_rationale
    )


def test_reset_fields_are_dropped_even_when_the_artifact_supplies_them() -> None:
    """A hand-edited or forked-SDK artifact cannot resurrect a RESET field.

    ``sync_session_id`` is a live handle into *this* machine's session store
    (#2020) and ``_experiment_stats`` is a memo cache; neither is restorable
    data, whatever the file says.
    """
    encoded = _encoded()
    encoded["sync_session_id"] = "20260727_005247_744918_answer_b1960eba"
    encoded["_experiment_stats"] = {"total_duration": 1.0}

    restored = decode_result(encoded, trials=[])

    assert restored.sync_session_id is None
    assert restored._experiment_stats is None


def test_versioned_artifact_missing_a_declared_field_is_corruption() -> None:
    """No blanket ``.get(field, default)`` — that is the defect, re-armed.

    A versioned artifact promises every restorable field, so an absent key means
    the file is damaged, not that it is old. Silently defaulting here would
    recreate #2031 for the next field somebody adds.
    """
    encoded = _encoded()
    del encoded["stop_reason"]

    with pytest.raises(ValueError, match="stop_reason"):
        decode_result(encoded, trials=[], artifact_name="damaged")


def test_unsupported_schema_version_fails_loudly() -> None:
    """A newer artifact must not be silently mis-decoded by an older SDK.

    The message must name *both* versions and must not call the artifact
    corrupt: it is intact, this reader is simply older than it.
    """
    encoded = _encoded()
    encoded[SCHEMA_VERSION_KEY] = 999

    with pytest.raises(ValueError) as excinfo:
        decode_result(encoded, trials=[], artifact_name="from-the-future")

    message = str(excinfo.value)
    assert "999" in message
    assert f"version {RESULT_SCHEMA_VERSION}" in message
    assert "not corrupt" in message


def test_a_schema_version_that_is_not_a_version_is_rejected() -> None:
    """``"1"``/``1.5``/``0``/``null`` are not versions this format ever wrote."""
    for bogus in ("1", 1.5, 0, -1, True, None):
        encoded = _encoded()
        encoded[SCHEMA_VERSION_KEY] = bogus

        with pytest.raises(ValueError, match=SCHEMA_VERSION_KEY):
            decode_result(encoded, trials=[], artifact_name="bogus-version")


def test_an_explicit_null_version_is_corruption_not_an_unversioned_artifact() -> None:
    """``"_schema_version": null`` is a declared version, and an invalid one.

    Reading the key with ``.get()`` and testing ``is not None`` makes the two
    indistinguishable, so a damaged *versioned* artifact routes down the legacy
    path — where every missing field is expected and quietly defaulted. One
    hand-edited value would then restore the blanket ``.get(name, default)``
    that #2031 exists to remove: here ``stop_reason`` is missing too, and must
    not come back as ``None`` under the legacy reading.
    """
    payload = _encoded()
    payload[SCHEMA_VERSION_KEY] = None
    del payload["stop_reason"]

    with pytest.raises(ValueError) as excinfo:
        decode_result(
            payload,
            trials=[],
            legacy_format="config_state",
            artifact_name="null-version.json",
        )

    message = str(excinfo.value)
    assert SCHEMA_VERSION_KEY in message
    assert "None" in message


# --- Schema evolution -------------------------------------------------------
#
# The strict "a versioned artifact promises every restorable field" rule is only
# safe if the format can also grow. Without an introduction table, adding a
# restorable field would reject every artifact already on disk — a fresh
# save -> load test would go green while all real history became unreadable, and
# bumping the version would not help either, because there would be no reader
# for the older one. These tests are the guard on that evolution path; they use
# an existing field re-declared as "introduced in version 2" so the mechanism is
# exercised end to end rather than described.


#: The fields the version loop actually reads out of an artifact. ``trials`` is
#: decoded by each format's own decoder and is never a key in the payload.
_VERSIONED_FIELDS = RESULT_RESTORE - {manifest._CALLER_SUPPLIED}

#: The version a hypothetical next release would write.
_NEXT_VERSION = RESULT_SCHEMA_VERSION + 1


def _as_next_version_world(monkeypatch, *introduced_now: str) -> None:
    """Pretend the next release added ``introduced_now`` and bumped the version.

    The table handed to the module is a **complete** one — every other field at
    its real introduction version — because the real table is exact-coverage and
    the decoder refuses to guess for a field it does not find there. Deriving
    both the version and the table from the module keeps these tests working
    after a real bump instead of turning into three more hardcoded ``2``s to
    mechanically edit.
    """
    table = dict(manifest.FIELD_INTRODUCED_IN)
    table.update(dict.fromkeys(introduced_now, _NEXT_VERSION))
    monkeypatch.setattr(manifest, "RESULT_SCHEMA_VERSION", _NEXT_VERSION)
    monkeypatch.setattr(manifest, "FIELD_INTRODUCED_IN", table)


def test_field_introduction_table_covers_every_restored_field_exactly() -> None:
    """Exact coverage, not a subset: an omitted entry must never mean "v1".

    A subset assertion passes vacuously on an empty table, and the single
    mistake the table exists to catch — adding a restorable field and forgetting
    its entry — is exactly the one an implicit "absent means version 1" default
    swallows. It does not merely go unnoticed: the decoder then believes every
    artifact ever written already carried the key, so it reports all of them as
    damaged. Requiring the entry turns that into this test failing.
    """
    missing = sorted(_VERSIONED_FIELDS - set(manifest.FIELD_INTRODUCED_IN))
    assert not missing, (
        f"FIELD_INTRODUCED_IN has no entry for {missing}. Record the "
        f"RESULT_SCHEMA_VERSION that introduced each — for a field you are "
        f"adding now that means bumping RESULT_SCHEMA_VERSION (currently "
        f"{RESULT_SCHEMA_VERSION}) and using the bumped value, never 1. Entering "
        f"1 claims every artifact already on disk carried the key, and makes "
        f"every one of them fail to load."
    )
    extra = sorted(set(manifest.FIELD_INTRODUCED_IN) - _VERSIONED_FIELDS)
    assert not extra, (
        f"FIELD_INTRODUCED_IN names {extra}, which the version loop never reads "
        f"(not restored, or caller-supplied like 'trials')"
    )

    for name, version in manifest.FIELD_INTRODUCED_IN.items():
        assert 1 <= version <= RESULT_SCHEMA_VERSION, (
            f"FIELD_INTRODUCED_IN['{name}'] = {version} is not a version "
            f"between 1 and RESULT_SCHEMA_VERSION ({RESULT_SCHEMA_VERSION})"
        )


def test_a_field_with_no_introduction_entry_is_an_sdk_defect_not_corruption(
    monkeypatch,
) -> None:
    """The runtime half of exact coverage: no silent "absent means version 1".

    If the table is ever incomplete anyway, the decoder must say so as a defect
    in *this build* — pointing at the file that needs the entry — rather than
    accusing the artifact of being damaged.
    """
    table = {
        name: version
        for name, version in manifest.FIELD_INTRODUCED_IN.items()
        if name != "stop_reason"
    }
    monkeypatch.setattr(manifest, "FIELD_INTRODUCED_IN", table)

    encoded = _encoded()
    del encoded["stop_reason"]

    with pytest.raises(RuntimeError) as excinfo:
        decode_result(encoded, trials=[], artifact_name="fine.json")

    message = str(excinfo.value)
    assert "stop_reason" in message
    assert "FIELD_INTRODUCED_IN" in message
    assert "optimization_result_persistence.py" in message


def test_an_older_artifact_still_loads_after_a_new_field_is_added(
    monkeypatch, caplog
) -> None:
    """The regression this whole section exists for.

    An artifact written today must still load tomorrow, when someone has added a
    restorable field and bumped the version. The new field comes back as its
    **declared dataclass default** — not a fabricated value, but exactly what a
    record written before the field existed genuinely held — and every field
    today's writer did record survives untouched.
    """
    old_artifact = _encoded()  # written by today's build
    assert old_artifact[SCHEMA_VERSION_KEY] == RESULT_SCHEMA_VERSION
    del old_artifact["best_config_margin"]  # … a field this version lacked

    _as_next_version_world(monkeypatch, "best_config_margin")

    with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
        restored = decode_result(
            old_artifact, trials=[], artifact_name="written-last-release.json"
        )

    assert restored.best_config_margin is None  # the dataclass default
    for name in sorted(RESULT_RESTORE - {"trials", "best_config_margin"}):
        assert getattr(restored, name) == _SENTINELS[name], f"{name} did not survive"

    # The log says what happened, and says it accurately.
    assert "best_config_margin" in caplog.text
    assert "written-last-release.json" in caplog.text
    assert "corrupt" not in caplog.text.lower()


def test_an_older_artifact_missing_a_field_its_own_version_declared_is_corruption(
    monkeypatch,
) -> None:
    """Being older is not a licence to default *everything*.

    ``stop_reason`` existed at version 1, so an older artifact without it is
    damaged — the introduction table must not degrade into the blanket
    ``.get(name, default)`` that #2031 exists to remove.
    """
    old_artifact = _encoded()
    del old_artifact["stop_reason"]

    _as_next_version_world(monkeypatch, "best_config_margin")

    with pytest.raises(ValueError) as excinfo:
        decode_result(old_artifact, trials=[], artifact_name="damaged-old")

    message = str(excinfo.value)
    assert "stop_reason" in message
    assert f"schema version {RESULT_SCHEMA_VERSION}" in message
    # The field predates this reader's own build, so damage is the leading
    # explanation and the introduction table is the secondary one.
    assert message.index("damaged or was truncated") < message.index(
        "FIELD_INTRODUCED_IN"
    )


def test_a_current_version_artifact_must_carry_the_newly_added_field(
    monkeypatch,
) -> None:
    """The new field is only optional for artifacts that predate it."""
    new_artifact = _encoded()
    _as_next_version_world(monkeypatch, "best_config_margin")
    new_artifact[SCHEMA_VERSION_KEY] = _NEXT_VERSION
    del new_artifact["best_config_margin"]

    with pytest.raises(ValueError, match="best_config_margin"):
        decode_result(new_artifact, trials=[], artifact_name="damaged-current")


def test_a_missing_field_this_build_just_added_is_not_called_damage(
    monkeypatch,
) -> None:
    """The false-corruption verdict, pinned.

    Somebody adds a restorable field, records it in FIELD_INTRODUCED_IN at
    version 1 instead of a bumped version, and every artifact ever written now
    lacks a key its declared version supposedly included. The reader cannot tell
    that from real damage — but it must not *assert* damage, because here the
    file is intact and the fix is one line in this SDK. The message has to name
    the reader's own table as the leading explanation and say what to change.
    """
    # `best_config_margin` stands in for the newly added field: it is declared as
    # having existed since the version the artifact itself claims.
    monkeypatch.setattr(
        manifest,
        "FIELD_INTRODUCED_IN",
        {**manifest.FIELD_INTRODUCED_IN, "best_config_margin": RESULT_SCHEMA_VERSION},
    )
    artifact = _encoded()
    del artifact["best_config_margin"]

    with pytest.raises(ValueError) as excinfo:
        decode_result(artifact, trials=[], artifact_name="history.json")

    message = str(excinfo.value)
    assert "best_config_margin" in message
    assert "RESULT_SCHEMA_VERSION" in message
    assert "FIELD_INTRODUCED_IN['best_config_margin']" in message
    # Damage is offered as the alternative, never as the verdict.
    assert message.index("FIELD_INTRODUCED_IN") < message.index(
        "damaged or was truncated"
    )
    assert "The artifact is damaged" not in message


def test_the_error_for_a_newer_artifact_blames_the_reader_not_the_artifact() -> None:
    """An artifact from a *newer* build is refused, and told the truth about.

    It may encode fields and semantics this reader would silently drop, so it
    cannot be decoded — but it is intact, and the message must say so rather
    than sending its owner looking for file damage.
    """
    encoded = _encoded()
    encoded[SCHEMA_VERSION_KEY] = _NEXT_VERSION

    with pytest.raises(ValueError) as excinfo:
        decode_result(encoded, trials=[], artifact_name="newer.json")

    message = str(excinfo.value)
    assert f"schema version {_NEXT_VERSION}" in message
    assert f"version {RESULT_SCHEMA_VERSION}" in message
    assert "not corrupt" in message


def test_unversioned_artifact_without_a_named_legacy_format_is_rejected() -> None:
    """Legacy migrations are named, not guessed."""
    encoded = _encoded()
    del encoded[SCHEMA_VERSION_KEY]

    with pytest.raises(ValueError, match=SCHEMA_VERSION_KEY):
        decode_result(encoded, trials=[])


def test_unrecognized_status_member_decodes_to_unknown(caplog) -> None:
    """#1302 AC3: never assert success or failure on a status we do not know."""
    encoded = _encoded()
    encoded["status"] = "quantum_superposition"

    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        restored = decode_result(encoded, trials=[])

    assert restored.status is OptimizationStatus.UNKNOWN
    assert "quantum_superposition" in caplog.text


def test_schema_version_is_stamped_into_the_encoded_payload() -> None:
    """The payload is self-describing, so it can be handed straight back."""
    assert _encoded()[SCHEMA_VERSION_KEY] == RESULT_SCHEMA_VERSION


# --- Frozen artifacts from every released schema version --------------------
#
# Everything above this line is written in terms of the manifest, so it moves
# when the manifest moves. That is what makes the evolution tests readable, and
# it is also why none of them can catch the one mistake that actually bricks
# users' history: adding a restorable field and not bumping the version. The
# monkeypatched tests build their "old" artifact from today's encoder, so a
# forgotten bump changes both sides at once and they stay green.
#
# These artifacts are committed literals. They do not move. Each one is a real
# payload as the named schema version wrote it, and the contract is simply that
# every version this build claims to read still loads. A forgotten bump makes
# the golden v1 payload short of the new field, at a version that (with no
# bump) is said to have included it — and this suite goes red instead of a
# user's reload.
#
# When you bump RESULT_SCHEMA_VERSION, freeze the payload the *outgoing* version
# wrote by adding an entry here; the count test below will ask you for it.

_GOLDEN_V1: dict = {
    "_schema_version": 1,
    "algorithm": "grid_search",
    "best_config": {"model": "cheap", "temperature": 0.0},
    "best_config_margin": {"verdict": "clear_winner", "delta": 0.08},
    "best_score": 0.87,
    "cloud_url": "https://portal.traigent.ai/experiments/exp-golden-v1",
    "convergence_info": {"converged": True, "iterations": 4},
    "duration": 31.5,
    "experiment_id": "exp-golden-v1",
    "experiment_run_id": "run-golden-v1",
    "metadata": {
        "function_name": "answer_question",
        "configuration_space": {"model": ["cheap", "smart"]},
        "session_summary": {"winning_trial_ids": ["golden-trial-0"]},
    },
    "metrics": {"accuracy": 0.87, "cost": 0.0119},
    "objectives": ["accuracy"],
    "optimization_id": "opt-golden-v1",
    "preset_selection": {
        "preset_name": "balanced",
        "params": {"temperature": 0.0},
        "selection_grade": "advisory",
        "selection_rationale": "balanced led on accuracy at equal cost.",
        "status": "selected",
        "selected_config": {"model": "cheap"},
        "selected_configs": [{"model": "cheap"}],
        "selected_trial_indices": [0],
    },
    "reason_code": None,
    "run_label": "answer_question_20260211_091500_9c4d1e",
    "source": "local",
    "status": "completed",
    "stop_reason": "max_trials_reached",
    "timestamp": "2026-02-11T09:15:00+00:00",
    "total_cost": 0.0119,
    "total_tokens": 8210,
    "warning_codes": ["PRICING_INCOMPLETE"],
    "warnings": ["Model 'mystery-1' priced at $0 — spend is under-reported."],
}

#: One frozen payload per schema version this build claims to read.
_GOLDEN_ARTIFACTS: dict[int, dict] = {1: _GOLDEN_V1}


def test_a_golden_artifact_is_frozen_for_every_readable_schema_version() -> None:
    """Bumping the version is not the last step — freeze what the old one wrote.

    Without a frozen payload for it, a version this build promises to read has
    nothing pinning that promise: the evolution tests all synthesise their "old"
    artifact from today's encoder, so they move with the code they are meant to
    catch.
    """
    expected = set(range(1, RESULT_SCHEMA_VERSION + 1))
    assert set(_GOLDEN_ARTIFACTS) == expected, (
        f"_GOLDEN_ARTIFACTS covers {sorted(_GOLDEN_ARTIFACTS)} but this build "
        f"reads schema versions {sorted(expected)}. Freeze a literal payload as "
        f"the missing version's writer produced it — copy it out of a real "
        f"metadata.json / results.json written by that release, do not generate "
        f"it from encode_result_fields()."
    )


@pytest.mark.parametrize("version", sorted(_GOLDEN_ARTIFACTS))
def test_a_frozen_artifact_from_every_released_version_still_loads(version) -> None:
    """The one test a forgotten version bump cannot slip past.

    An author adds a 28th restorable field, gives it a sentinel, updates the
    field count — and forgets to bump RESULT_SCHEMA_VERSION. Every test written
    against the manifest still passes, because they all encode with the same
    build they decode with. This one does not: the frozen payload predates the
    new field and cannot carry it, so decoding it at the version it declares —
    which the un-bumped build says already included the field — raises. The
    author sees that here, instead of every user seeing it on their next reload.
    """
    restored = decode_result(
        copy.deepcopy(_GOLDEN_ARTIFACTS[version]),
        trials=[],
        artifact_name=f"golden-v{version}.json",
    )

    assert isinstance(restored, OptimizationResult)


def test_the_frozen_v1_artifact_restores_the_values_it_recorded() -> None:
    """… and loading it is not enough: it must come back as what it said.

    Pinned against literals rather than against ``_SENTINELS``, so a decoder
    change that happens to move both cannot pass unnoticed.
    """
    restored = decode_result(
        copy.deepcopy(_GOLDEN_V1), trials=[], artifact_name="golden-v1.json"
    )

    assert restored.optimization_id == "opt-golden-v1"
    assert restored.status is OptimizationStatus.COMPLETED
    assert restored.source == "local"  # never the "backend" default (#1265)
    assert restored.total_cost == 0.0119
    assert restored.total_tokens == 8210
    assert restored.stop_reason == "max_trials_reached"
    assert restored.warnings == [
        "Model 'mystery-1' priced at $0 — spend is under-reported."
    ]
    assert restored.warning_codes == ["PRICING_INCOMPLETE"]
    assert restored.best_config == {"model": "cheap", "temperature": 0.0}
    assert restored.best_score == 0.87
    assert restored.best_config_margin == {"verdict": "clear_winner", "delta": 0.08}
    assert restored.timestamp == datetime(2026, 2, 11, 9, 15, tzinfo=UTC)
    assert restored.metadata["session_summary"] == {
        "winning_trial_ids": ["golden-trial-0"]
    }
    assert restored.preset_selection is not None
    assert restored.preset_selection.preset_name == "balanced"
    assert restored.cloud_url == "https://portal.traigent.ai/experiments/exp-golden-v1"
    assert restored.experiment_run_id == "run-golden-v1"
    # #2020: not restorable from any artifact, whatever its version.
    assert restored.sync_session_id is None


def test_the_frozen_v1_artifact_is_a_complete_v1_payload() -> None:
    """The baseline is only a baseline if it is not itself missing a key.

    A golden artifact that silently lost a field would stop failing on the very
    mistake it exists to catch, so its key set is asserted against the fields
    version 1 introduced — never against today's ``RESULT_RESTORE``, which is
    what a new field changes.
    """
    v1_fields = {
        name for name, version in manifest.FIELD_INTRODUCED_IN.items() if version == 1
    }

    assert set(_GOLDEN_V1) - {SCHEMA_VERSION_KEY} == v1_fields, (
        "the frozen v1 payload no longer matches the fields FIELD_INTRODUCED_IN "
        "says version 1 had. If you just added a field, its entry belongs at a "
        "bumped RESULT_SCHEMA_VERSION, not at 1 — the v1 payload is frozen and "
        "must not grow."
    )


# --- The nested envelope ----------------------------------------------------


def test_an_envelope_and_payload_that_disagree_on_the_version_are_rejected() -> None:
    """Two stamps, one truth. See ``verify_envelope_version``.

    ``PersistenceManager`` writes the version on both the envelope and the
    nested payload, but only the payload's copy is decoded. Left unchecked, the
    envelope's copy is decorative: a file whose envelope claims a version this
    build cannot read would still decode at whatever version its payload claims,
    silently dropping everything the higher version encodes — the exact outcome
    the future-version refusal exists to prevent.
    """
    with pytest.raises(ValueError) as excinfo:
        manifest.verify_envelope_version(
            {SCHEMA_VERSION_KEY: _NEXT_VERSION},
            {SCHEMA_VERSION_KEY: RESULT_SCHEMA_VERSION},
            artifact_name="two-faced",
        )

    message = str(excinfo.value)
    assert str(_NEXT_VERSION) in message
    assert str(RESULT_SCHEMA_VERSION) in message
    assert "two-faced" in message


def test_a_version_stamped_on_only_one_half_of_the_artifact_is_rejected() -> None:
    """Both stamps are written together, so one alone means the file was edited."""
    with pytest.raises(ValueError, match="result_fields"):
        manifest.verify_envelope_version(
            {SCHEMA_VERSION_KEY: RESULT_SCHEMA_VERSION}, {"best_score": 0.5}
        )

    with pytest.raises(ValueError, match="envelope"):
        manifest.verify_envelope_version(
            {"best_score": 0.5}, {SCHEMA_VERSION_KEY: RESULT_SCHEMA_VERSION}
        )


def test_matching_versions_and_a_wholly_unversioned_artifact_both_pass() -> None:
    """The two shapes a loader-written file can legitimately have."""
    manifest.verify_envelope_version(
        {SCHEMA_VERSION_KEY: RESULT_SCHEMA_VERSION},
        {SCHEMA_VERSION_KEY: RESULT_SCHEMA_VERSION},
    )
    # A pre-#2031 artifact: no envelope stamp, and no nested payload at all.
    manifest.verify_envelope_version({"best_score": 0.5}, None)


# --- Legacy artifacts -------------------------------------------------------


def _legacy_config_state_payload() -> dict:
    """An artifact as ``ConfigStateManager`` wrote it before #2031."""
    payload = _encoded()
    del payload[SCHEMA_VERSION_KEY]
    for name in ("total_cost", "total_tokens", "warnings", "warning_codes", "source"):
        del payload[name]
    return payload


def test_legacy_artifact_defaults_missing_fields_and_says_so(caplog) -> None:
    """Reduced fidelity must be visible, and visible only in the log."""
    payload = _legacy_config_state_payload()

    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        restored = decode_result(
            payload,
            trials=[],
            legacy_format="config_state",
            artifact_name="old.json",
        )

    assert restored.total_cost is None
    assert restored.total_tokens is None
    assert restored.warnings == []
    assert restored.warning_codes == []
    for name in ("total_cost", "total_tokens", "warnings", "warning_codes"):
        assert name in caplog.text
    assert "old.json" in caplog.text


def test_legacy_source_is_unknown_never_the_backend_default() -> None:
    """Captain decision 1 / the #1265 regression, pinned.

    ``source`` defaults to ``"backend"`` on the dataclass. An artifact that never
    recorded provenance may well be a local run, so claiming backend provenance
    for it is a lie the loader must not tell — "unknown" is the honest answer.
    """
    restored = decode_result(
        _legacy_config_state_payload(),
        trials=[],
        legacy_format="config_state",
        artifact_name="old.json",
    )

    assert restored.source == "unknown"


def test_legacy_restore_never_mutates_the_user_facing_warning_channels() -> None:
    """Finding A-c: loader-internal concerns do not belong in user payloads.

    ``warnings`` / ``warning_codes`` / ``metadata`` are round-tripped verbatim
    (and pinned as such by #2026). Injecting a "this artifact was lossy" entry
    into them would corrupt that contract and would surface a storage detail as
    a money-correctness warning.
    """
    payload = _legacy_config_state_payload()
    payload["metadata"] = {"function_name": "answer_question"}

    restored = decode_result(
        payload, trials=[], legacy_format="config_state", artifact_name="old.json"
    )

    assert restored.warnings == []
    assert restored.warning_codes == []
    assert restored.metadata == {"function_name": "answer_question"}


def test_defaulted_collections_are_fresh_instances_per_load() -> None:
    """Two loads must never share one mutable default."""
    payload = _legacy_config_state_payload()

    first = decode_result(payload, trials=[], legacy_format="config_state")
    first.warnings.append("mutated by the caller")
    first.warning_codes.append("MUTATED")
    second = decode_result(payload, trials=[], legacy_format="config_state")

    assert second.warnings == []
    assert second.warning_codes == []


# --- Offline determinism ----------------------------------------------------


def test_cloud_references_restore_verbatim_without_touching_the_network(
    monkeypatch,
) -> None:
    """Restoration asserts provenance, not reachability.

    ``cloud_url`` / ``experiment_id`` / ``experiment_run_id`` are a record of
    where this run was tracked. Decoding must never try to validate them, so a
    reload works offline and on a machine that cannot reach the portal at all.
    """

    def _no_sockets(*args, **kwargs):  # pragma: no cover - only fires on regression
        raise AssertionError("decode_result must not open a socket")

    monkeypatch.setattr(socket, "socket", _no_sockets)
    monkeypatch.setattr(socket, "create_connection", _no_sockets)

    encoded = _encoded()
    encoded["cloud_url"] = "https://unreachable.invalid/experiments/exp-2031"
    encoded["experiment_run_id"] = "run-that-no-longer-exists"

    restored = decode_result(encoded, trials=[])

    assert restored.cloud_url == "https://unreachable.invalid/experiments/exp-2031"
    assert restored.experiment_run_id == "run-that-no-longer-exists"
    assert restored.experiment_id == _SENTINELS["experiment_id"]


# --- Encoding shapes --------------------------------------------------------


def test_encoded_payload_is_json_safe_for_the_typed_fields() -> None:
    """status / timestamp / preset_selection are the three non-trivial types."""
    encoded = _encoded()

    assert encoded["status"] == "cancelled"
    assert (
        encoded["timestamp"]
        == datetime(2026, 3, 15, 14, 30, 22, tzinfo=UTC).isoformat()
    )
    assert isinstance(encoded["preset_selection"], dict)
    assert encoded["preset_selection"]["preset_name"] == "balanced"


def test_encoder_never_writes_a_reset_field() -> None:
    """The writer side of the #2020 guarantee."""
    result = _sentinel_result()
    result.sync_session_id = "20260727_005247_744918_answer_b1960eba"
    assert result.experiment_stats is not None  # populate the memo cache

    encoded = encode_result_fields(result)

    assert "sync_session_id" not in encoded
    assert "_experiment_stats" not in encoded


def test_trials_are_supplied_by_the_caller_not_the_manifest() -> None:
    """Each format keeps its own trial decoder; the manifest never encodes them."""
    assert "trials" in RESULT_RESTORE
    assert "trials" not in _encoded()

    trials = [
        TrialResult(
            trial_id="caller-supplied",
            config={},
            metrics={},
            status=_SENTINELS["trials"][0].status,
            duration=0.0,
            timestamp=datetime(2026, 3, 15, tzinfo=UTC),
        )
    ]
    restored = decode_result(_encoded(), trials=trials)

    assert [trial.trial_id for trial in restored.trials] == ["caller-supplied"]


# --- Duck-typed callers -----------------------------------------------------


class _PartialResult:
    """A result-like object carrying only the attributes ``save_result`` read.

    ``PersistenceManager.save_result`` has always accepted any object shaped
    like a result — its curated metadata block reads ~10 attributes — and
    ``tests/unit/cli/test_main_cli_security.py`` exercises the CLI through
    exactly such a stub. Required fields (``optimization_id``, ``timestamp``)
    are present because they have no declared default; everything else is
    deliberately absent.
    """

    def __init__(self) -> None:
        self.status = OptimizationStatus.COMPLETED
        self.trials: list[TrialResult] = []
        self.best_config = {"param": 1}
        self.best_score = 0.75
        self.metadata = {"function_name": "fake_optimized_function"}
        self.algorithm = "grid"
        self.objectives = ["accuracy"]
        self.preset_selection = None
        self.duration = 0.01
        self.convergence_info: dict = {}
        self.optimization_id = "opt-partial"
        self.timestamp = datetime(2026, 3, 15, 14, 30, 22, tzinfo=UTC)


def test_encodes_a_duck_typed_partial_result_using_declared_defaults() -> None:
    """A partial result must encode, not raise ``AttributeError``.

    Reading all 24 encoded fields by plain attribute access silently narrowed
    ``save_result`` to ``OptimizationResult`` alone and broke two CLI
    path-traversal tests that have nothing to do with persistence. The fallback
    is the field's own declared dataclass default, so it can only ever fire for
    a non-dataclass caller — a real ``OptimizationResult`` always has all 27
    attributes — and it invents no value the dataclass would not have produced.
    """
    encoded = encode_result_fields(_PartialResult())

    # Present attributes are encoded from the object, not defaulted.
    assert encoded["best_score"] == 0.75
    assert encoded["algorithm"] == "grid"
    assert encoded["status"] == "completed"

    # Absent ones fall back to the value the dataclass itself declares.
    defaults = {
        field.name: field.default
        for field in dataclasses.fields(OptimizationResult)
        if field.default is not dataclasses.MISSING
    }
    for name in ("total_cost", "stop_reason", "experiment_id", "best_config_margin"):
        assert encoded[name] is defaults[name] is None
    assert encoded["source"] == defaults["source"] == "backend"

    # default_factory fields get a fresh instance, never a shared one.
    assert encoded["warnings"] == []
    assert encoded["warnings"] is not encode_result_fields(_PartialResult())["warnings"]


def test_missing_required_field_raises_a_named_error_not_attribute_error() -> None:
    """A field with no declared default has nothing honest to fall back to."""
    partial = _PartialResult()
    del partial.optimization_id

    with pytest.raises(TypeError, match=r"_PartialResult.*'optimization_id'"):
        encode_result_fields(partial)
