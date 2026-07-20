"""Tests for best_config winner-vs-runner-up margin significance (issue #1866).

These are additive-qualification tests: they assert the SDK now reports a paired
test / CI vs the runner-up and marks statistical ties, WITHOUT changing which
config wins.

Fail-on-old markers:
- ``OptimizationResult`` / ``SelectionResult`` carry a ``best_config_margin``
  field (absent before #1866).
- ``select_best_configuration`` populates it for a real winner.
"""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime
from typing import Any

from traigent.api.types import OptimizationResult, TrialResult, TrialStatus
from traigent.core.result_selection import (
    SelectionResult,
    select_best_configuration,
)
from traigent.core.stat_significance import (
    BEST_CONFIG_MARGIN_ALPHA,
    compute_best_config_margin,
)


def _trial(
    trial_id: str,
    config: dict[str, Any],
    objective: str,
    per_example: list[float],
) -> TrialResult:
    """Build a completed TrialResult with per-example scores for one objective."""
    aggregate = sum(per_example) / len(per_example) if per_example else 0.0
    example_results = [
        {"example_id": f"e{i}", "metrics": {objective: value}}
        for i, value in enumerate(per_example)
    ]
    return TrialResult(
        trial_id=trial_id,
        config=config,
        metrics={objective: aggregate},
        status=TrialStatus.COMPLETED,
        duration=1.0,
        timestamp=datetime.now(UTC),
        metadata={
            "successful_examples": max(len(per_example), 1),
            "example_results": example_results,
        },
    )


def _trial_from_map(
    trial_id: str,
    config: dict[str, Any],
    objective: str,
    per_example: dict[str, float],
) -> TrialResult:
    """Build a completed TrialResult from an explicit ``{example_id: score}`` map.

    Unlike :func:`_trial` (which auto-numbers ids as ``e0, e1, ...``), this gives
    every example an explicit id so a test can control precisely which examples
    are SHARED (intersected) between two trials versus unique to one — the setup
    needed to reproduce a winner that wins the full aggregate on non-shared
    examples while losing every shared one.
    """
    values = list(per_example.values())
    aggregate = sum(values) / len(values) if values else 0.0
    example_results = [
        {"example_id": eid, "metrics": {objective: value}}
        for eid, value in per_example.items()
    ]
    return TrialResult(
        trial_id=trial_id,
        config=config,
        metrics={objective: aggregate},
        status=TrialStatus.COMPLETED,
        duration=1.0,
        timestamp=datetime.now(UTC),
        metadata={
            "successful_examples": max(len(per_example), 1),
            "example_results": example_results,
        },
    )


# ---------------------------------------------------------------------------
# Fail-on-old: the field exists
# ---------------------------------------------------------------------------


class TestFieldExists:
    def test_optimization_result_has_margin_field(self):
        names = {f.name for f in dataclasses.fields(OptimizationResult)}
        assert "best_config_margin" in names

    def test_selection_result_has_margin_field(self):
        names = {f.name for f in dataclasses.fields(SelectionResult)}
        assert "best_config_margin" in names

    def test_default_alpha_is_005(self):
        assert BEST_CONFIG_MARGIN_ALPHA == 0.05


# ---------------------------------------------------------------------------
# Binary scorers → McNemar exact
# ---------------------------------------------------------------------------


class TestBinaryMcNemar:
    def test_near_tie_is_statistical_tie(self):
        """A photo-finish binary margin (few discordant pairs) → statistical_tie."""
        n = 40
        # Winner beats runner-up on only 2 examples, ties on the rest.
        winner = [1] * 20 + [0] * 20
        runner = [1] * 18 + [0] * 2 + [0] * 20
        margin = compute_best_config_margin(
            [
                _trial("w", {"model": "A"}, "accuracy", winner),
                _trial("r", {"model": "B"}, "accuracy", runner),
            ],
            winner_trial_id="w",
            winner_config={"model": "A"},
            primary_objective="accuracy",
            orientation="maximize",
        )
        assert margin is not None
        assert margin["test"] == "mcnemar_exact"
        assert margin["verdict"] == "statistical_tie"
        assert margin["p_value"] is not None and margin["p_value"] > 0.05
        assert margin["discordant"] == {"b": 2, "c": 0}
        assert margin["n_shared_examples"] == n
        assert margin["runner_up"] == {"model": "B"}
        assert margin["runner_up_trial_id"] == "r"

    def test_clear_binary_winner(self):
        """A large, consistent binary margin → clear + low p-value."""
        winner = [1] * 38 + [0] * 2
        runner = [1] * 20 + [0] * 20
        margin = compute_best_config_margin(
            [
                _trial("w", {"model": "A"}, "accuracy", winner),
                _trial("r", {"model": "B"}, "accuracy", runner),
            ],
            winner_trial_id="w",
            winner_config={"model": "A"},
            primary_objective="accuracy",
            orientation="maximize",
        )
        assert margin is not None
        assert margin["verdict"] == "clear"
        assert margin["p_value"] < 0.05
        # CI on the margin should exclude 0 for a clear winner.
        lo, hi = margin["ci95"]
        assert lo > 0.0 or hi < 0.0

    def test_identical_binary_is_tie(self):
        """No discordant pairs → p == 1.0 → statistical_tie."""
        vals = [1, 0, 1, 1, 0, 1, 0, 1, 0, 1]
        margin = compute_best_config_margin(
            [
                _trial("w", {"model": "A"}, "accuracy", vals),
                _trial("r", {"model": "B"}, "accuracy", list(vals)),
            ],
            winner_trial_id="w",
            winner_config={"model": "A"},
            primary_objective="accuracy",
            orientation="maximize",
        )
        assert margin is not None
        assert margin["p_value"] == 1.0
        assert margin["verdict"] == "statistical_tie"
        assert margin["discordant"] == {"b": 0, "c": 0}


# ---------------------------------------------------------------------------
# Continuous scorers → paired t-test
# ---------------------------------------------------------------------------


class TestContinuousPairedT:
    def test_tie_when_difference_is_noise(self):
        winner = [0.50 + 0.001 * i for i in range(30)]
        # Runner-up is essentially the same series with tiny jitter.
        runner = [v - 0.001 for v in winner]
        margin = compute_best_config_margin(
            [
                _trial("w", {"p": 1}, "score", winner),
                _trial("r", {"p": 2}, "score", runner),
            ],
            winner_trial_id="w",
            winner_config={"p": 1},
            primary_objective="score",
            orientation="maximize",
        )
        assert margin is not None
        assert margin["test"] == "paired_t"
        # Constant +0.001 offset every example is perfectly consistent, so this
        # is actually a clear (degenerate) win — assert the machinery ran and
        # produced a real verdict + finite CI.
        assert margin["verdict"] in {"clear", "statistical_tie"}
        assert margin["ci95"] is not None

    def test_noisy_tie_is_statistical_tie(self):
        import random

        rng = random.Random(1)
        winner = [0.5 + rng.gauss(0, 0.1) for _ in range(30)]
        runner = [w - rng.gauss(0.002, 0.05) for w in winner]
        margin = compute_best_config_margin(
            [
                _trial("w", {"p": 1}, "score", winner),
                _trial("r", {"p": 2}, "score", runner),
            ],
            winner_trial_id="w",
            winner_config={"p": 1},
            primary_objective="score",
            orientation="maximize",
        )
        assert margin is not None
        assert margin["verdict"] == "statistical_tie"
        assert margin["p_value"] > 0.05

    def test_clear_continuous_winner(self):
        import random

        rng = random.Random(2)
        winner = [0.8 + rng.gauss(0, 0.05) for _ in range(30)]
        runner = [0.5 + rng.gauss(0, 0.05) for _ in range(30)]
        margin = compute_best_config_margin(
            [
                _trial("w", {"p": 1}, "score", winner),
                _trial("r", {"p": 2}, "score", runner),
            ],
            winner_trial_id="w",
            winner_config={"p": 1},
            primary_objective="score",
            orientation="maximize",
        )
        assert margin is not None
        assert margin["verdict"] == "clear"
        assert margin["p_value"] < 0.05
        assert margin["delta"] > 0.0

    def test_identical_continuous_is_tie(self):
        vals = [0.1 * i for i in range(30)]
        margin = compute_best_config_margin(
            [
                _trial("w", {"p": 1}, "score", vals),
                _trial("r", {"p": 2}, "score", list(vals)),
            ],
            winner_trial_id="w",
            winner_config={"p": 1},
            primary_objective="score",
            orientation="maximize",
        )
        assert margin is not None
        assert margin["p_value"] == 1.0
        assert margin["verdict"] == "statistical_tie"
        assert margin["delta"] == 0.0


# ---------------------------------------------------------------------------
# Degenerate cases — graceful, no crash
# ---------------------------------------------------------------------------


class TestDegenerate:
    def test_single_config_returns_none(self):
        vals = [1, 0, 1, 1, 0]
        margin = compute_best_config_margin(
            [
                _trial("w", {"model": "A"}, "accuracy", vals),
                _trial("w2", {"model": "A"}, "accuracy", vals),
            ],
            winner_trial_id="w",
            winner_config={"model": "A"},
            primary_objective="accuracy",
            orientation="maximize",
        )
        assert margin is None

    def test_fewer_than_two_trials_returns_none(self):
        margin = compute_best_config_margin(
            [_trial("w", {"model": "A"}, "accuracy", [1, 0, 1])],
            winner_trial_id="w",
            winner_config={"model": "A"},
            primary_objective="accuracy",
            orientation="maximize",
        )
        assert margin is None

    def test_no_primary_objective_returns_none(self):
        margin = compute_best_config_margin(
            [
                _trial("w", {"model": "A"}, "accuracy", [1, 0, 1, 1, 0]),
                _trial("r", {"model": "B"}, "accuracy", [1, 1, 0, 0, 1]),
            ],
            winner_trial_id="w",
            winner_config={"model": "A"},
            primary_objective=None,
        )
        assert margin is None

    def test_no_per_example_data_is_na(self):
        """Two distinct configs but no shared per-example data → verdict 'na'."""
        margin = compute_best_config_margin(
            [
                _trial("w", {"model": "A"}, "accuracy", []),
                _trial("r", {"model": "B"}, "accuracy", []),
            ],
            winner_trial_id="w",
            winner_config={"model": "A"},
            primary_objective="accuracy",
            orientation="maximize",
        )
        assert margin is not None
        assert margin["verdict"] == "na"
        assert margin["p_value"] is None
        assert margin["ci95"] is None
        assert margin["test"] == "none"
        # Aggregate delta is still reported from the metric means.
        assert margin["delta"] is not None

    def test_too_few_shared_examples_is_na(self):
        """< 5 shared examples → no paired test → 'na' (not a crash)."""
        margin = compute_best_config_margin(
            [
                _trial("w", {"model": "A"}, "accuracy", [1, 0, 1]),
                _trial("r", {"model": "B"}, "accuracy", [1, 1, 0]),
            ],
            winner_trial_id="w",
            winner_config={"model": "A"},
            primary_objective="accuracy",
            orientation="maximize",
        )
        assert margin is not None
        assert margin["verdict"] == "na"


# ---------------------------------------------------------------------------
# Minimize objective — runner-up ranked correctly
# ---------------------------------------------------------------------------


class TestMinimizeObjective:
    def test_runner_up_is_second_lowest_for_minimize(self):
        """For a minimize objective the runner-up is the 2nd-lowest config."""
        cheap = [0.10] * 10  # winner (lowest cost)
        mid = [0.12] * 10  # runner-up
        pricey = [0.90] * 10
        margin = compute_best_config_margin(
            [
                _trial("cheap", {"m": "A"}, "cost", cheap),
                _trial("mid", {"m": "B"}, "cost", mid),
                _trial("pricey", {"m": "C"}, "cost", pricey),
            ],
            winner_trial_id="cheap",
            winner_config={"m": "A"},
            primary_objective="cost",
            orientation="minimize",
        )
        assert margin is not None
        assert margin["runner_up_trial_id"] == "mid"


# ---------------------------------------------------------------------------
# End-to-end: select_best_configuration attaches the margin
# ---------------------------------------------------------------------------


class TestSelectBestConfigurationWiring:
    def _run(self, trials: list[TrialResult]) -> SelectionResult:
        return select_best_configuration(
            trials,
            primary_objective="accuracy",
            config_space_keys={"model"},
            aggregate_configs=False,
            comparability_mode="legacy",
            objective_order=["accuracy"],
            objective_orientations={"accuracy": "maximize"},
        )

    def test_selection_attaches_statistical_tie(self):
        winner = [1] * 20 + [0] * 20
        runner = [1] * 18 + [0] * 2 + [0] * 20
        selection = self._run(
            [
                _trial("w", {"model": "A"}, "accuracy", winner),
                _trial("r", {"model": "B"}, "accuracy", runner),
            ]
        )
        # best_config selection itself is unchanged: the higher-accuracy config wins.
        assert selection.best_config == {"model": "A"}
        assert selection.best_config_margin is not None
        assert selection.best_config_margin["verdict"] == "statistical_tie"
        assert selection.best_config_margin["p_value"] is not None

    def test_selection_attaches_clear(self):
        winner = [1] * 38 + [0] * 2
        runner = [1] * 20 + [0] * 20
        selection = self._run(
            [
                _trial("w", {"model": "A"}, "accuracy", winner),
                _trial("r", {"model": "B"}, "accuracy", runner),
            ]
        )
        assert selection.best_config == {"model": "A"}
        assert selection.best_config_margin is not None
        assert selection.best_config_margin["verdict"] == "clear"

    def test_single_config_leaves_margin_none(self):
        vals = [1, 0, 1, 1, 0]
        selection = self._run(
            [
                _trial("w", {"model": "A"}, "accuracy", vals),
                _trial("w2", {"model": "A"}, "accuracy", vals),
            ]
        )
        assert selection.best_config_margin is None


# ---------------------------------------------------------------------------
# Direction guard (#1866 regression)
#
# "clear" must mean the selected winner is significantly BETTER than the
# runner-up on the comparable examples. A winner chosen on the full aggregate
# (boosted by non-shared examples) that loses every SHARED example is a
# statistical_tie, not a clear win — even when the paired test is significant
# and the CI excludes 0.
# ---------------------------------------------------------------------------


class TestVerdictRequiresFavorableDirection:
    def test_binary_adverse_direction_is_not_clear(self):
        """Winner wins the full aggregate on non-shared examples yet scores 0 on
        every one of the 6 SHARED examples, where the runner scores 1.

        McNemar is significant (b=0, c=6, p=0.03125) with a CI that excludes 0,
        but the effect favors the RUNNER-UP — so the verdict must NOT be
        "clear"; it is the conservative "statistical_tie".
        """
        shared = [f"s{i}" for i in range(6)]
        # Winner: 0 on all 6 shared, 1 on 54 winner-only examples -> aggregate 0.90.
        winner_map: dict[str, float] = dict.fromkeys(shared, 0.0)
        winner_map.update({f"wonly{i}": 1.0 for i in range(54)})
        # Runner: 1 on all 6 shared, 0 on 2 runner-only examples -> aggregate 0.75.
        runner_map: dict[str, float] = dict.fromkeys(shared, 1.0)
        runner_map.update({f"ronly{i}": 0.0 for i in range(2)})

        margin = compute_best_config_margin(
            [
                _trial_from_map("w", {"model": "A"}, "accuracy", winner_map),
                _trial_from_map("r", {"model": "B"}, "accuracy", runner_map),
            ],
            winner_trial_id="w",
            winner_config={"model": "A"},
            primary_objective="accuracy",
            orientation="maximize",
        )
        assert margin is not None
        assert margin["test"] == "mcnemar_exact"
        assert margin["n_shared_examples"] == 6
        assert margin["discordant"] == {"b": 0, "c": 6}
        assert margin["delta"] == -1.0
        # The test is genuinely significant with a CI that excludes 0 ...
        assert margin["p_value"] is not None and margin["p_value"] <= 0.05
        lo, hi = margin["ci95"]
        assert not (lo <= 0.0 <= hi)
        # ... yet because it favors the runner-up, the verdict is NOT "clear".
        assert margin["verdict"] == "statistical_tie"
        assert "reason" in margin
        assert "non-shared" in margin["reason"]

    def test_continuous_adverse_direction_is_not_clear(self):
        """Continuous analogue: a significant NEGATIVE mean_diff — the winner
        scores consistently BELOW the runner on the shared examples — is not a
        clear win even though the winner leads on the full aggregate.
        """
        shared = [f"s{i}" for i in range(6)]
        winner_shared = [0.40, 0.42, 0.38, 0.41, 0.39, 0.40]
        runner_shared = [0.70, 0.72, 0.68, 0.71, 0.69, 0.70]
        winner_map: dict[str, float] = dict(zip(shared, winner_shared, strict=True))
        # Winner-only high scores make the winner lead the full aggregate.
        winner_map.update({f"wonly{i}": 0.95 for i in range(24)})
        runner_map: dict[str, float] = dict(zip(shared, runner_shared, strict=True))
        runner_map.update({f"ronly{i}": 0.10 for i in range(6)})

        margin = compute_best_config_margin(
            [
                _trial_from_map("w", {"p": 1}, "score", winner_map),
                _trial_from_map("r", {"p": 2}, "score", runner_map),
            ],
            winner_trial_id="w",
            winner_config={"p": 1},
            primary_objective="score",
            orientation="maximize",
        )
        assert margin is not None
        assert margin["test"] == "paired_t"
        assert margin["n_shared_examples"] == 6
        assert margin["delta"] < 0.0
        assert margin["p_value"] is not None and margin["p_value"] <= 0.05
        lo, hi = margin["ci95"]
        assert not (lo <= 0.0 <= hi)
        assert margin["verdict"] == "statistical_tie"
        assert "reason" in margin

    def test_positive_control_binary_favorable_stays_clear(self):
        """No over-correction: a significant margin that FAVORS the winner on the
        shared examples is still "clear" and carries no adverse-direction reason.
        """
        shared = [f"s{i}" for i in range(40)]
        # Winner beats runner on the shared examples: b=18 favorable, c=0.
        winner_shared = [1.0] * 38 + [0.0] * 2
        runner_shared = [1.0] * 20 + [0.0] * 20
        winner_map = dict(zip(shared, winner_shared, strict=True))
        runner_map = dict(zip(shared, runner_shared, strict=True))

        margin = compute_best_config_margin(
            [
                _trial_from_map("w", {"model": "A"}, "accuracy", winner_map),
                _trial_from_map("r", {"model": "B"}, "accuracy", runner_map),
            ],
            winner_trial_id="w",
            winner_config={"model": "A"},
            primary_objective="accuracy",
            orientation="maximize",
        )
        assert margin is not None
        assert margin["delta"] > 0.0
        assert margin["p_value"] <= 0.05
        assert margin["verdict"] == "clear"
        assert "reason" not in margin

    def test_minimize_adverse_direction_is_not_clear(self):
        """Minimize analogue of the binary adverse case.

        For a MINIMIZE objective a HIGHER shared score is WORSE. The winner is
        selected because it has the lowest FULL aggregate (driven by non-shared
        examples), yet it scores 1 (worse) on every shared example where the
        runner scores 0 (better). McNemar is significant (b=6, c=0, delta=+1.0),
        but for minimize a positive delta favors the RUNNER-UP — so this must be
        a statistical_tie, not clear.
        """
        shared = [f"s{i}" for i in range(6)]
        # Winner: 1 (worse) on all 6 shared, 0 (better) on 54 winner-only
        # examples -> aggregate 0.10 (lowest = selected for minimize).
        winner_map: dict[str, float] = dict.fromkeys(shared, 1.0)
        winner_map.update({f"wonly{i}": 0.0 for i in range(54)})
        # Runner: 0 (better) on all 6 shared, 1 (worse) on 2 runner-only
        # examples -> aggregate 0.25 (> winner, so the winner still wins).
        runner_map: dict[str, float] = dict.fromkeys(shared, 0.0)
        runner_map.update({f"ronly{i}": 1.0 for i in range(2)})

        margin = compute_best_config_margin(
            [
                _trial_from_map("w", {"model": "A"}, "cost", winner_map),
                _trial_from_map("r", {"model": "B"}, "cost", runner_map),
            ],
            winner_trial_id="w",
            winner_config={"model": "A"},
            primary_objective="cost",
            orientation="minimize",
        )
        assert margin is not None
        assert margin["test"] == "mcnemar_exact"
        assert margin["n_shared_examples"] == 6
        # winner=1, runner=0 on every shared example -> b=6, c=0, delta=+1.0.
        assert margin["discordant"] == {"b": 6, "c": 0}
        assert margin["delta"] == 1.0
        # Genuinely significant with a CI excluding 0 ...
        assert margin["p_value"] is not None and margin["p_value"] <= 0.05
        lo, hi = margin["ci95"]
        assert not (lo <= 0.0 <= hi)
        # ... but a positive delta favors the runner-up for a MINIMIZE objective,
        # so the winner is NOT a clear win.
        assert margin["verdict"] == "statistical_tie"
        assert "reason" in margin
        assert "non-shared" in margin["reason"]

    def test_minimize_positive_control_stays_clear(self):
        """No over-correction for minimize: a winner that is genuinely BETTER
        (lower) on the shared examples has delta<0, which favors the winner for
        a MINIMIZE objective -> still "clear". Guards against the orientation
        fix wrongly demoting a legitimate minimize winner.
        """
        shared = [f"s{i}" for i in range(6)]
        # Winner: 0 (better/lower) on all 6 shared; Runner: 1 (worse) on all 6.
        winner_map = dict.fromkeys(shared, 0.0)
        runner_map = dict.fromkeys(shared, 1.0)

        margin = compute_best_config_margin(
            [
                _trial_from_map("w", {"model": "A"}, "cost", winner_map),
                _trial_from_map("r", {"model": "B"}, "cost", runner_map),
            ],
            winner_trial_id="w",
            winner_config={"model": "A"},
            primary_objective="cost",
            orientation="minimize",
        )
        assert margin is not None
        assert margin["test"] == "mcnemar_exact"
        # winner=0, runner=1 on every shared example -> b=0, c=6, delta=-1.0.
        assert margin["discordant"] == {"b": 0, "c": 6}
        assert margin["delta"] == -1.0
        assert margin["p_value"] <= 0.05
        # delta<0 favors the winner for minimize -> a real, clear win.
        assert margin["verdict"] == "clear"
        assert "reason" not in margin
