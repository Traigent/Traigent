"""Tests for the R3 selection receipt (``traigent.core.selection_receipt``).

The receipt is projected from the SDK's own ``SelectionResult`` and must
validate against the pinned TraigentSchema ``session_aggregation_schema.json``
``selection`` contract (Draft 7). Fail-on-old marker: the module does not exist
before R3 stage 3.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import os
from datetime import UTC, datetime
from typing import Any

import pytest
from jsonschema import Draft7Validator

from traigent.api.types import TrialResult, TrialStatus
from traigent.core.result_selection import SelectionResult, select_best_configuration
from traigent.core.selection_receipt import (
    MAX_ELIGIBLE_TRIALS,
    build_selection_receipt,
    eligible_trial_ids_digest,
    sanitize_selection_receipt,
)

# Known answer from TraigentSchema df06e6dcb tests/test_session_aggregation_selection.py:
# sha256 over the UTF-8 bytes of '["trial_a","trial_b","trial_c"]'.
_KAT_IDS = ["trial_a", "trial_b", "trial_c"]
_KAT_DIGEST_HEX = "4a31e7194d0ef1862bfa5db184d1486776997a10810f1a72c67ab24aef08e005"

_SENTINEL = "SENTINEL_PROMPT_CONTENT_9f2c"


def _selection_validator() -> Draft7Validator:
    """Validator for the ``selection`` property of the INSTALLED pinned schema."""
    import traigent_schema

    path = os.path.join(
        os.path.dirname(traigent_schema.__file__),
        "schemas",
        "optimization",
        "session_aggregation_schema.json",
    )
    with open(path, encoding="utf-8") as handle:
        schema = json.load(handle)
    assert "selection" in schema["properties"], (
        "installed traigent-schema predates the R3 selection receipt; "
        "pip install -r scripts/ci/schema-pin.txt"
    )
    wrapper = {
        "$schema": schema["$schema"],
        "definitions": schema["definitions"],
        **schema["properties"]["selection"],
    }
    Draft7Validator.check_schema(wrapper)
    return Draft7Validator(wrapper)


def _assert_valid(receipt: dict[str, Any]) -> None:
    errors = sorted(_selection_validator().iter_errors(receipt), key=str)
    assert errors == [], [e.message for e in errors]


def _trial(
    trial_id: str,
    config: dict[str, Any],
    per_example: list[float] | None,
    *,
    aggregate: float | None = None,
) -> TrialResult:
    metadata: dict[str, Any] = {"successful_examples": 1}
    if per_example is not None:
        metadata = {
            "successful_examples": max(len(per_example), 1),
            "example_results": [
                {"example_id": f"e{i}", "metrics": {"accuracy": value}}
                for i, value in enumerate(per_example)
            ],
        }
        aggregate = sum(per_example) / len(per_example)
    return TrialResult(
        trial_id=trial_id,
        config=config,
        metrics={"accuracy": aggregate},
        status=TrialStatus.COMPLETED,
        duration=1.0,
        timestamp=datetime.now(UTC),
        metadata=metadata,
    )


def _select(trials: list[TrialResult], *, aggregate: bool = False) -> SelectionResult:
    return select_best_configuration(
        trials,
        "accuracy",
        config_space_keys=["model", "system_prompt"],
        aggregate_configs=aggregate,
        comparability_mode="legacy",
        objective_order=["accuracy"],
        objective_orientations={"accuracy": "maximize"},
    )


def _clear_selection() -> SelectionResult:
    return _select(
        [
            _trial("trial_w", {"model": "A"}, [1.0] * 40),
            _trial("trial_r", {"model": "B"}, [1.0] * 10 + [0.0] * 30),
        ]
    )


def _tie_selection() -> SelectionResult:
    return _select(
        [
            _trial("trial_w", {"model": "A"}, [1.0] * 20 + [0.0] * 20),
            _trial("trial_r", {"model": "B"}, [1.0] * 18 + [0.0] * 22),
        ]
    )


def _na_selection() -> SelectionResult:
    return _select(
        [
            _trial("trial_w", {"model": "A"}, None, aggregate=0.9),
            _trial("trial_r", {"model": "B"}, None, aggregate=0.8),
            _trial("trial_z", {"model": "C"}, None, aggregate=0.7),
        ]
    )


def _repeated_config_selection() -> SelectionResult:
    return _select(
        [
            _trial("trial_w2", {"model": "A"}, [1.0] * 30 + [0.0] * 10),
            _trial("trial_w1", {"model": "A"}, [1.0] * 32 + [0.0] * 8),
            _trial("trial_r1", {"model": "B"}, [1.0] * 10 + [0.0] * 30),
        ],
        aggregate=True,
    )


def _single_config_selection() -> SelectionResult:
    return _select([_trial("trial_only", {"model": "A"}, [1.0] * 5)])


SCENARIOS = {
    "clear": (_clear_selection, "clear"),
    "statistical_tie": (_tie_selection, "statistical_tie"),
    "na": (_na_selection, "na"),
    "repeated_config": (_repeated_config_selection, None),
    "no_runner_up": (_single_config_selection, None),
}


class TestDigest:
    def test_known_answer_matches_schema_fixture(self):
        assert eligible_trial_ids_digest(_KAT_IDS) == "sha256:" + _KAT_DIGEST_HEX

    def test_digest_is_order_independent_over_canonical_preimage(self):
        assert eligible_trial_ids_digest(
            ["trial_c", "trial_a", "trial_b"]
        ) == eligible_trial_ids_digest(_KAT_IDS)


class TestBuildFromRealSelection:
    @pytest.mark.parametrize("name", sorted(SCENARIOS))
    def test_receipt_validates_against_pinned_schema(self, name):
        factory, expected_verdict = SCENARIOS[name]
        selection = factory()
        receipt = build_selection_receipt(selection)
        assert receipt is not None
        _assert_valid(receipt)
        if expected_verdict is not None:
            assert receipt["margin"]["verdict"] == expected_verdict

    @pytest.mark.parametrize("name", sorted(SCENARIOS))
    def test_ids_are_exactly_the_ranked_set_sorted_unique(self, name):
        selection = SCENARIOS[name][0]()
        receipt = build_selection_receipt(selection)
        assert receipt is not None
        ranked = selection.ranking_eligible_trial_ids
        assert receipt["eligible_trial_ids"] == sorted(set(ranked))
        assert len(receipt["eligible_trial_ids"]) == len(ranked)
        assert receipt["eligible_trial_count"] == len(receipt["eligible_trial_ids"])
        assert receipt["eligible_trial_ids_digest"] == eligible_trial_ids_digest(
            receipt["eligible_trial_ids"]
        )
        assert receipt["winner_trial_id"] == selection.best_trial_id
        assert receipt["disposition"] == "accepted"
        assert "attestation" not in receipt
        assert receipt["selection_reason"] == selection.reason_code

    def test_margin_is_projected_not_recomputed(self):
        selection = _clear_selection()
        receipt = build_selection_receipt(selection)
        source = selection.best_config_margin
        margin = receipt["margin"]
        for key in (
            "winner_trial_id",
            "runner_up_trial_id",
            "delta",
            "p_value",
            "verdict",
            "test",
            "n_shared_examples",
            "effective_alpha",
            "n_configs",
        ):
            assert margin[key] == source[key]
        assert margin["ci95"] == list(source["ci95"])
        assert margin["winner_trial_id"] == receipt["winner_trial_id"]

    def test_repeated_config_counts_trials_not_configs(self):
        receipt = build_selection_receipt(_repeated_config_selection())
        assert receipt["eligible_trial_count"] == 3
        assert receipt["margin"]["n_configs"] == 2

    def test_no_runner_up_sends_null_margin(self):
        receipt = build_selection_receipt(_single_config_selection())
        assert "margin" in receipt and receipt["margin"] is None

    def test_unsorted_eligible_ids_are_sent_sorted(self):
        selection = _repeated_config_selection()
        assert selection.ranking_eligible_trial_ids != sorted(
            selection.ranking_eligible_trial_ids
        )
        receipt = build_selection_receipt(selection)
        assert receipt["eligible_trial_ids"] == ["trial_r1", "trial_w1", "trial_w2"]


def _synthetic(**overrides: Any) -> SelectionResult:
    base = SelectionResult(
        best_config={"model": "A"},
        best_score=0.9,
        session_summary=None,
        best_trial_id="trial_b",
        ranking_eligible_trial_ids=["trial_c", "trial_b", "trial_a"],
        best_config_margin={
            "runner_up": {"model": "B"},
            "runner_up_trial_id": "trial_a",
            "winner_trial_id": "trial_b",
            "primary_objective": "accuracy",
            "alpha": 0.05,
            "effective_alpha": 0.025,
            "n_configs": 3,
            "delta": 0.1,
            "ci95": (0.01, 0.2),
            "p_value": 0.01,
            "verdict": "clear",
            "test": "paired_t",
            "n_shared_examples": 12,
        },
    )
    return dataclasses.replace(base, **overrides)


class TestNoReceipt:
    def test_no_winner_returns_none(self):
        selection = select_best_configuration(
            [],
            "accuracy",
            config_space_keys=["model"],
            aggregate_configs=False,
        )
        assert selection.best_trial_id is None
        assert build_selection_receipt(selection) is None

    def test_no_eligible_set_returns_none(self):
        assert (
            build_selection_receipt(_synthetic(ranking_eligible_trial_ids=None)) is None
        )
        assert (
            build_selection_receipt(_synthetic(ranking_eligible_trial_ids=[])) is None
        )

    def test_winner_outside_eligible_set_returns_none_with_content_free_log(
        self, caplog
    ):
        with caplog.at_level(logging.WARNING, logger="traigent"):
            receipt = build_selection_receipt(_synthetic(best_trial_id="trial_zzz"))
        assert receipt is None
        messages = " ".join(record.getMessage() for record in caplog.records)
        assert "not in the ranking-eligible trial set" in messages
        assert "trial_zzz" not in messages and "trial_a" not in messages

    def test_non_wire_valid_id_returns_none(self):
        selection = _synthetic(
            ranking_eligible_trial_ids=["trial_b", f"trial {_SENTINEL}"]
        )
        assert build_selection_receipt(selection) is None

    def test_over_max_eligible_returns_none(self):
        ids = [f"t{i:05d}" for i in range(MAX_ELIGIBLE_TRIALS + 1)]
        selection = _synthetic(best_trial_id=ids[0], ranking_eligible_trial_ids=ids)
        assert build_selection_receipt(selection) is None

    def test_at_max_eligible_is_valid(self):
        ids = [f"t{i:05d}" for i in range(MAX_ELIGIBLE_TRIALS)]
        receipt = build_selection_receipt(
            _synthetic(
                best_trial_id=ids[0],
                ranking_eligible_trial_ids=ids,
                best_config_margin=None,
            )
        )
        assert receipt is not None and receipt["eligible_trial_count"] == 10000
        _assert_valid(receipt)

    def test_never_raises(self):
        class Exploding:
            @property
            def best_trial_id(self):
                raise RuntimeError(_SENTINEL)

        assert build_selection_receipt(Exploding()) is None
        assert build_selection_receipt(None) is None
        assert build_selection_receipt(object()) is None


class TestNonFiniteAndInvalidMargin:
    @pytest.mark.parametrize(
        "patch",
        [
            {"delta": math.nan},
            {"delta": math.inf},
            {"p_value": math.nan},
            {"ci95": (0.0, math.inf)},
            {"ci95": (-math.inf, 0.1)},
            {"effective_alpha": math.nan},
            {"verdict": "sure"},
            {"test": "paired t test"},
            {"p_value": 1.5},
            {"n_configs": 1},
            {"n_shared_examples": 0},
            {"runner_up_trial_id": None},
        ],
    )
    def test_invalid_margin_is_omitted_not_nulled(self, patch):
        margin = {**_synthetic().best_config_margin, **patch}
        receipt = build_selection_receipt(_synthetic(best_config_margin=margin))
        assert receipt is not None
        assert "margin" not in receipt
        _assert_valid(receipt)
        assert "NaN" not in json.dumps(receipt)
        assert "Infinity" not in json.dumps(receipt)

    def test_na_with_non_finite_delta_is_omitted(self):
        margin = {
            **_synthetic().best_config_margin,
            "verdict": "na",
            "ci95": None,
            "p_value": None,
            "n_shared_examples": 0,
            "delta": math.nan,
        }
        receipt = build_selection_receipt(_synthetic(best_config_margin=margin))
        assert "margin" not in receipt

    def test_na_with_null_delta_is_kept(self):
        margin = {
            **_synthetic().best_config_margin,
            "verdict": "na",
            "ci95": None,
            "p_value": None,
            "n_shared_examples": 0,
            "delta": None,
        }
        receipt = build_selection_receipt(_synthetic(best_config_margin=margin))
        assert receipt["margin"]["delta"] is None
        _assert_valid(receipt)


class TestContentCanary:
    def test_sentinel_config_values_prompts_and_reasons_never_appear(self):
        trials = [
            _trial(
                "trial_w",
                {"model": "A", "system_prompt": f"You are {_SENTINEL} winner"},
                [1.0] * 40,
            ),
            _trial(
                "trial_r",
                {"model": "B", "system_prompt": f"You are {_SENTINEL} runner"},
                [1.0] * 10 + [0.0] * 30,
            ),
        ]
        for trial in trials:
            trial.metadata["example_results"][0]["input"] = _SENTINEL
        selection = _select(trials)
        # The SDK margin payload itself carries the runner-up config dict.
        assert _SENTINEL in json.dumps(selection.best_config_margin, default=str)
        selection.best_config_margin["reason"] = f"free text {_SENTINEL}"
        receipt = build_selection_receipt(selection)
        assert receipt is not None
        assert _SENTINEL not in json.dumps(receipt)
        _assert_valid(receipt)

    def test_free_text_reason_code_becomes_null(self):
        receipt = build_selection_receipt(
            _synthetic(reason_code=f"because {_SENTINEL}")
        )
        assert receipt["selection_reason"] is None
        assert _SENTINEL not in json.dumps(receipt)


class TestSanitizer:
    def _receipt(self) -> dict[str, Any]:
        receipt = build_selection_receipt(_synthetic())
        assert receipt is not None
        return receipt

    def test_idempotent_on_builder_output(self):
        receipt = self._receipt()
        assert sanitize_selection_receipt(receipt) == receipt
        null_margin = build_selection_receipt(_synthetic(best_config_margin=None))
        assert sanitize_selection_receipt(null_margin) == null_margin

    def test_strips_unknown_free_text_and_attestation_keys(self):
        receipt = self._receipt()
        receipt["attestation"] = "client_attested_server_bound"
        receipt["note"] = _SENTINEL
        receipt["margin"] = {
            **receipt["margin"],
            "runner_up": {"system_prompt": _SENTINEL},
            "reason": _SENTINEL,
        }
        out = sanitize_selection_receipt(receipt)
        assert out is not None
        assert "attestation" not in out and "note" not in out
        assert set(out["margin"]) == set(self._receipt()["margin"])
        assert _SENTINEL not in json.dumps(out)
        _assert_valid(out)

    def test_absent_margin_stays_absent(self):
        receipt = self._receipt()
        del receipt["margin"]
        assert "margin" not in sanitize_selection_receipt(receipt)

    def test_free_text_reason_nulled(self):
        receipt = {**self._receipt(), "selection_reason": f"x {_SENTINEL}"}
        assert sanitize_selection_receipt(receipt)["selection_reason"] is None

    @pytest.mark.parametrize(
        "mutate",
        [
            lambda r: r.update(disposition="rejected_inconsistent"),
            lambda r: r.update(
                eligible_trial_ids=list(reversed(r["eligible_trial_ids"]))
            ),
            lambda r: r.update(eligible_trial_count=r["eligible_trial_count"] + 1),
            lambda r: r.update(eligible_trial_ids_digest="sha256:" + "0" * 64),
            lambda r: r.update(winner_trial_id="trial_zzz"),
            lambda r: r.update(
                eligible_trial_ids=r["eligible_trial_ids"] + ["trial_a"]
            ),
            lambda r: r.update(eligible_trial_count=True),
        ],
    )
    def test_inconsistent_receipt_is_dropped(self, mutate):
        receipt = self._receipt()
        mutate(receipt)
        assert sanitize_selection_receipt(receipt) is None

    @pytest.mark.parametrize("raw", [None, "accepted", [], {"disposition": "accepted"}])
    def test_malformed_input_is_dropped(self, raw):
        assert sanitize_selection_receipt(raw) is None

    def test_rejected_form_is_never_sent(self):
        assert (
            sanitize_selection_receipt(
                {"disposition": "rejected_inconsistent", "reason": "invalid_receipt"}
            )
            is None
        )
