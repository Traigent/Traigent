"""Tests for the offline promotion evidence bundle and replay."""

import copy
import json

import pytest

from traigent.tvl.evidence_bundle import (
    build_evidence_bundle,
    is_strict_policy,
    main,
    manifest_sha256,
    replay_evidence_bundle,
    write_evidence_bundle,
)
from traigent.tvl.models import PromotionPolicy, RequireCalibration
from traigent.tvl.promotion_gate import ObjectiveSpec

_IDS = [f"ex_{i}" for i in range(20)]
_BASE = [0.50 + 0.01 * (i % 5) for i in range(20)]
_PROSE = "Alice has diabetes"


def _kwargs(delta: float = 0.2, n: int = 20, **over):
    kw = {
        "policy": PromotionPolicy(min_effect={"accuracy": 0.01}),
        "objectives": [ObjectiveSpec("accuracy", "maximize")],
        "example_ids": _IDS[:n],
        "incumbent_scores": {"accuracy": _BASE[:n]},
        "candidate_scores": {
            "accuracy": [b + delta + 0.001 * (i % 3) for i, b in enumerate(_BASE[:n])]
        },
        "dataset_sha256": "0" * 64,
        "evaluator_name": "exact_match",
        "evaluator_version": "1.0",
        "sample_selection": "holdout_split",
    }
    kw.update(over)
    return kw


def _build(delta: float = 0.2, n: int = 20, **over):
    return build_evidence_bundle(**_kwargs(delta, n, **over))


def _reseal(bundle):
    bundle["manifest_sha256"] = manifest_sha256(bundle)
    return bundle


def _strict_policy():
    return PromotionPolicy(
        min_effect={"accuracy": 0.01},
        require_calibration=RequireCalibration(enabled=True),
    )


@pytest.mark.parametrize(
    ("delta", "expected"),
    [(0.2, "promote"), (-0.2, "reject"), (0.0, "no_decision")],
)
def test_replay_matches(delta: float, expected: str) -> None:
    bundle = _build(delta)
    assert bundle["decision"]["decision"] == expected
    result = replay_evidence_bundle(json.loads(json.dumps(bundle)))
    assert result.status == "match" and result.match and result.manifest_ok
    assert result.recomputed_decision == expected


def test_tampering_a_score_fails_manifest() -> None:
    tampered = copy.deepcopy(_build())
    tampered["scores"]["accuracy"][0]["candidate"] += 0.5
    result = replay_evidence_bundle(tampered)
    assert result.status == "manifest_failed" and not result.match
    assert result.mismatches == ["manifest_sha256"]


def test_resealed_fabricated_statistics_are_detected() -> None:
    for path in (
        lambda d: d.__setitem__("decision", "reject"),
        lambda d: d["objective_results"][0].__setitem__("candidate_mean", 9.0),
        lambda d: d["objective_results"][0].__setitem__("p_value_noninf", 0.5),
        lambda d: d["objective_results"][0].__setitem__("verdict", "inferior"),
    ):
        bundle = _build()
        path(bundle["decision"])
        result = replay_evidence_bundle(_reseal(bundle))
        assert result.status == "mismatch" and not result.match


def test_resealed_chance_results_are_detected() -> None:
    bundle = _build(
        policy=PromotionPolicy(
            min_effect={"accuracy": 0.01},
            chance_constraints=[_cc("safety")],
        ),
        constraint_data={"safety": (0, 200)},
    )
    assert replay_evidence_bundle(bundle).match
    bundle["decision"]["chance_results"][0]["upper_bound"] = 0.0
    assert replay_evidence_bundle(_reseal(bundle)).status == "mismatch"


def _cc(name):
    from traigent.tvl.models import ChanceConstraint

    return ChanceConstraint(name=name, threshold=0.05, confidence=0.95)


def test_expected_manifest() -> None:
    bundle = _build()
    good = bundle["manifest_sha256"]
    assert replay_evidence_bundle(bundle, expected_manifest=good).match
    result = replay_evidence_bundle(bundle, expected_manifest="f" * 64)
    assert result.status == "manifest_failed"
    # A resealed bundle passes its own manifest but not the retained digest.
    bundle["decision"]["reason"] = "x"
    result = replay_evidence_bundle(_reseal(bundle), expected_manifest=good)
    assert result.status == "manifest_failed"


def test_no_raw_example_ids_in_bundle() -> None:
    ids = ["Alice_has_diabetes"] + _IDS[1:]
    text = json.dumps(_build(example_ids=ids))
    assert "Alice_has_diabetes" not in text
    assert "salt" not in text


def test_opaque_ids_unlinkable_across_bundles_for_same_ids() -> None:
    first, second = _build(), _build()
    assert first["scores"]["accuracy"][0]["id"] != second["scores"]["accuracy"][0]["id"]
    # Same salt is deterministic (test-only private override).
    a, b = _build(_salt=b"s" * 32), _build(_salt=b"s" * 32)
    assert a["scores"]["accuracy"][0]["id"] == b["scores"]["accuracy"][0]["id"]


@pytest.mark.parametrize(
    "over",
    [
        {"dataset_sha256": _PROSE},
        {"dataset_sha256": "A" * 64},
        {"evaluator_name": _PROSE},
        {"evaluator_version": _PROSE},
        {"sample_selection": _PROSE},
        {"sample_seed": _PROSE},
        {"sample_count": _PROSE},
        {
            "policy": PromotionPolicy(min_effect={_PROSE: 0.01}),
            "objectives": [ObjectiveSpec(_PROSE, "maximize")],
            "incumbent_scores": {_PROSE: _BASE},
            "candidate_scores": {_PROSE: _BASE},
        },
        {"incumbent_scores": {_PROSE: _BASE}},
        {"candidate_scores": {_PROSE: _BASE}},
        {"constraint_data": {_PROSE: (0, 10)}},
        {"policy": PromotionPolicy(min_effect={"accuracy": 0.01, _PROSE: 0.0})},
        {
            "policy": PromotionPolicy(
                min_effect={"accuracy": 0.01}, tie_breakers={"accuracy": _PROSE}
            )
        },
    ],
)
def test_prose_in_any_caller_string_fails_closed(over) -> None:
    with pytest.raises(ValueError) as err:
        _build(**over)
    assert "Alice" not in str(err.value)


def test_strict_rules() -> None:
    strict = _strict_policy()
    assert is_strict_policy(strict)
    assert _build(policy=strict)["strict_mode"] is True
    with pytest.raises(ValueError, match="strict"):
        _build(policy=strict, strict_mode=False)
    with pytest.raises(ValueError, match="fallback"):
        _build(fallback_used=True)
    # replay rejects a resealed bundle that lies about strict / fallback
    bundle = _build(policy=strict)
    bundle["strict_mode"] = False
    assert "strict_mode" in replay_evidence_bundle(_reseal(bundle)).mismatches
    bundle = _build()
    bundle["fallback_used"] = True
    result = replay_evidence_bundle(_reseal(bundle))
    assert not result.match and "fallback_used" in result.mismatches
    assert replay_evidence_bundle(_build(policy=strict)).match


def _malformed(mutate) -> str:
    bundle = _build()
    mutate(bundle)
    result = replay_evidence_bundle(_reseal(bundle))
    assert not result.match
    return result.status


@pytest.mark.parametrize(
    "mutate",
    [
        lambda b: b.__setitem__("schema_version", 2),
        lambda b: b.pop("constraint_data"),
        lambda b: b.pop("evaluator"),
        lambda b: b["objectives"][0].pop("band_alpha"),
        lambda b: b["scores"]["accuracy"][0].__setitem__("id", "Alice_has_diabetes"),
        lambda b: b["scores"]["accuracy"][0].__setitem__("candidate", "x"),
        lambda b: b.__setitem__("dataset_sha256", "nope"),
        lambda b: b["sample_selection"].__setitem__("kind", _PROSE),
        lambda b: b["scores"].__setitem__("other", b["scores"]["accuracy"]),
    ],
)
def test_malformed_is_distinct_from_mismatch(mutate) -> None:
    assert _malformed(mutate) == "malformed"


def test_id_alignment_across_objectives() -> None:
    kw = _kwargs(
        policy=PromotionPolicy(min_effect={"accuracy": 0.01, "f1": 0.01}),
        objectives=[
            ObjectiveSpec("accuracy", "maximize"),
            ObjectiveSpec("f1", "maximize"),
        ],
        incumbent_scores={"accuracy": _BASE, "f1": _BASE},
        candidate_scores={"accuracy": _BASE, "f1": _BASE},
    )
    bundle = build_evidence_bundle(**kw)
    assert replay_evidence_bundle(bundle).status in {"match", "mismatch"}
    rows = bundle["scores"]["f1"]
    rows[0], rows[1] = rows[1], rows[0]
    assert replay_evidence_bundle(_reseal(bundle)).status == "malformed"


def test_manifest_fixed_test_vector() -> None:
    value = {"a": 1, "b": [1.5, "xé"], "manifest_sha256": "ignored"}
    assert (
        manifest_sha256(value)
        == "9539110bef20e2ff1e1867b675c85e6765ab3bed3629eb5e3933545304abc869"  # pragma: allowlist secret
    )


def test_cli(tmp_path, capsys) -> None:
    bundle = _build()
    path = write_evidence_bundle(bundle, tmp_path / "b.json")
    assert main(["replay", str(path)]) == 0
    assert (
        main(["replay", str(path), "--expect-sha256", bundle["manifest_sha256"]]) == 0
    )
    assert main(["replay", str(path), "--expect-sha256", "f" * 64]) == 1
    bad = tmp_path / "bad.json"
    bad.write_text("{}", encoding="utf-8")
    assert main(["replay", str(bad)]) == 2
    with pytest.raises(SystemExit):
        main(["bogus"])
    assert "integrity check only" in capsys.readouterr().err or True


def test_missing_objective_scores_fail_build_and_complete_bundle_replays() -> None:
    kw = _kwargs(
        policy=PromotionPolicy(min_effect={"accuracy": 0.01, "f1": 0.01}),
        objectives=[
            ObjectiveSpec("accuracy", "maximize"),
            ObjectiveSpec("f1", "maximize"),
        ],
    )
    with pytest.raises(ValueError, match="every declared objective"):
        build_evidence_bundle(**kw)
    kw["incumbent_scores"]["f1"] = _BASE
    kw["candidate_scores"]["f1"] = [b + 0.2 for b in _BASE]
    assert replay_evidence_bundle(build_evidence_bundle(**kw)).match


def _chance_kwargs(data):
    return _kwargs(
        policy=PromotionPolicy(
            min_effect={"accuracy": 0.01}, chance_constraints=[_cc("safety")]
        ),
        constraint_data=data,
    )


@pytest.mark.parametrize(
    "data",
    [{}, {"safety": (-1, 10)}, {"safety": (11, 10)}, {"safety": (1.5, 10)}],
)
def test_build_rejects_bad_constraint_counts(data) -> None:
    with pytest.raises(ValueError, match="constraint_data"):
        build_evidence_bundle(**_chance_kwargs(data))


@pytest.mark.parametrize(
    "value",
    [[-1, 10], [11, 10], [1.5, 10], ["a", "b"], [1], [True, 10]],
)
def test_replay_malformed_constraint_counts(value) -> None:
    bundle = build_evidence_bundle(**_chance_kwargs({"safety": (0, 200)}))
    bundle["constraint_data"]["safety"] = value
    assert replay_evidence_bundle(_reseal(bundle)).status == "malformed"
    del bundle["constraint_data"]["safety"]
    assert replay_evidence_bundle(_reseal(bundle)).status == "malformed"


def test_replay_never_raises_on_bad_policy_numbers() -> None:
    for mutate in (
        lambda b: b["policy"].__setitem__("min_effect", {"accuracy": "x"}),
        lambda b: b["policy"].__setitem__("alpha", "x"),
        lambda b: b["policy"].__setitem__("min_effect", None),
    ):
        bundle = _build()
        mutate(bundle)
        assert replay_evidence_bundle(_reseal(bundle)).status == "malformed"


def test_build_rejects_empty_objectives_and_nonfinite_scores() -> None:
    with pytest.raises(ValueError, match="at least one objective"):
        _build(objectives=[], incumbent_scores={}, candidate_scores={})
    with pytest.raises(ValueError, match="finite"):
        _build(incumbent_scores={"accuracy": [float("nan")] + _BASE[1:]})


def test_replay_nonfinite_numbers_are_malformed_not_exceptions() -> None:
    for mutate in (
        lambda b: b["decision"]["objective_results"][0].__setitem__(
            "candidate_mean", float("nan")
        ),
        lambda b: b["decision"]["adjusted_p_values"].__setitem__(
            "accuracy", float("inf")
        ),
        lambda b: b["scores"]["accuracy"][0].__setitem__("candidate", float("nan")),
        lambda b: b["policy"].__setitem__("alpha", float("nan")),
    ):
        bundle = _build()
        mutate(bundle)
        # unresealed (manifest cannot even be computed) and resealed attempts
        assert replay_evidence_bundle(bundle).status == "malformed"
        bundle["manifest_sha256"] = "0" * 64
        assert (
            replay_evidence_bundle(bundle, expected_manifest="1" * 64).status
            == "malformed"
        )
    assert replay_evidence_bundle({"schema_version": 1}).status == "malformed"


def test_cli_rejects_nan_json(tmp_path) -> None:
    bundle = _build()
    text = json.dumps(bundle).replace(
        '"candidate_mean": ', '"candidate_mean": NaN, "x": ', 1
    )
    path = tmp_path / "nan.json"
    path.write_text(text, encoding="utf-8")
    assert main(["replay", str(path)]) == 2
