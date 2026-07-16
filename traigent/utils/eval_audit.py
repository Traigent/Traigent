"""Eval-dataset defect detectors over the per-example x per-config matrix.

# Traceability: CONC-Layer-Data CONC-Quality-Observability FUNC-ANALYTICS FUNC-STORAGE REQ-STOR-007 SYNC-StorageLogging

Every optimization run already produces the per-example x per-config outcome
matrix (persisted by #1838; see ``traigent.utils.outcome_matrix``). That
by-product doubles as an eval-dataset *auditor*: deterministic detectors over the
outcome/token cells surface examples whose recorded gold label is likely wrong,
ambiguous, or otherwise defective — with no extra LLM calls and no network.

This module implements that computation as a **pure function of the matrix**
(plus, for D7, the per-cell ``predicted`` answer the matrix now carries). It is
the consumer half of #1838.

On top of #1880's binary detectors it also ships #1881's **continuous defect
score**: a per-item heuristic suspicion score plus a dataset-relative percentile,
so a reviewer gets a ranked worklist over ALL scored items rather than an
unordered flag set. The score is a logistic function over three per-item
telemetry features (``mean_wrong`` / ``never_correct`` / ``instability``). The
shipped ``DEFECT_SCORE_*`` coefficients are **illustrative defaults, hand-chosen
for sensible ordering — NOT a calibrated model**; the trustworthy default output
is the dataset-relative ``defect_percentile`` (rank), while the absolute
``defect_score`` becomes a calibrated probability only after you refit the
coefficients on your own labeled subset (see :func:`compute_defect_scores` and
the ``DEFECT_SCORE_*`` constants). The binary flags then become expressible as
``defect_score >= threshold``. This layer reuses the same per-cell primitives
(:func:`cell_is_correct`) and stays a pure, deterministic function of the matrix.

Detectors
---------
**D1 never-correct.** An example that is *wrong in every configuration it ran in*
(``success`` false, or the per-cell scoring signal below the success threshold).
Correlated failure across independent configs is strong evidence the item itself
is defective rather than merely hard. Works for single-model grids (multiple
configs of one model still give multiple columns). The audit's >=2-config gate
guarantees >=2 columns *exist*, but not that every *row* ran in all of them — a
sparse row (an example run in only one trial) has a single determinable cell.
Per-row eligibility is therefore enforced regardless of sparse rows: an example
is eligible for D1 only when it has a determinable outcome in >=2 columns
(``len(outcomes) >= 2`` in the code below), so sparse rows are excluded, never
crash, and are never flagged.

**D3 token-leak.** An example scored *correct in every configuration* yet whose
token usage is a strong upper outlier among the always-correct population — a
difficulty-matched cohort, since they were all solved. Rule: among always-correct
items that carry ``tokens.total``, take each item's mean total-token usage across
its correct cells. Two paths pick the outlier fence:

* **Normal (``IQR > 0``):** flag values above the Tukey upper fence
  ``Q3 + k * IQR`` (``k = 1.5`` by default).
* **Degenerate (``IQR == 0``):** when ``Q1 == Q3`` the cohort's tokens are
  clustered on a single value and the Tukey fence collapses onto it, so a genuine
  outlier (e.g. ``[100, 100, 100, 100, 1000]``) would be silently dropped.
  Fall back to a median-multiple rule — flag values above
  ``median * D3_DEGENERATE_MEDIAN_MULTIPLE`` (2.5 by default).

Skipped entirely when fewer than ``_MIN_D3_POPULATION`` (4) always-correct items
carry token data, since a robust fence is not defined below that. Works for
single-model grids.

**D7 cross-family consensus-on-wrong.** Models from >=2 *unrelated families*
(openai/anthropic/google/...) that are all wrong on an example but agree on the
*same* non-gold answer. Cross-family agreement on a specific wrong answer is the
"the gold is wrong and here is the fix" signal: the agreed answer is exposed as
``suggested_answer``. Requires the config space to span >=2 families; single-
family runs get D1 + D3 only. Family is inferred from the config's model name by
the prefix heuristic in ``_FAMILY_PREFIXES`` / :func:`model_family`.

Lift (base-rate framing)
------------------------
The summary reports, per detector, a ``lift`` computable straight from the matrix
so the user knows how concentrated the flags are versus chance. Each is
documented on its detector below; briefly:

* **D1 lift** = observed never-correct rate / the rate expected if per-config
  errors were *independent* at the observed base error rate
  (``p_wrong ** column_count``). Lift >> 1 means item failures are correlated
  across configs — a systematic-defect signal, not random noise.
* **D3 lift** = the mean token-burn *multiple* of flagged items versus the median
  always-correct item (``flagged_tokens / median_tokens``). This is the same
  "defect items burn N x tokens" framing the validation study reports.
* **D7 lift** = observed cross-family-consensus-on-wrong rate / ``p_wrong ** 2``
  (the chance two configs are *both* wrong on an item under independence). The
  denominator ignores the extra "agree on the same answer across families"
  constraint, so it is a conservative null — real enrichment is larger.

Everything here is deterministic and side-effect free.
"""

from __future__ import annotations

import bisect
import json
import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from traigent.api.types import (
        DefectSignal,
        DetectorStat,
        EvalAudit,
        EvalAuditFlag,
        ItemDefectScore,
    )

#: Default per-cell success threshold: a cell's scoring signal at or above this
#: counts as correct. Benchmark signals here are typically 0/1, so 0.5 is a clean
#: split; callers can override.
DEFAULT_SUCCESS_THRESHOLD = 0.5

#: Tukey upper-fence multiplier for the D3 token-leak outlier rule
#: (flag ``value > Q3 + k * IQR``).
DEFAULT_TOKEN_IQR_K = 1.5

#: D3 needs at least this many always-correct items carrying token data before a
#: robust IQR (and therefore an outlier rule) is meaningful.
_MIN_D3_POPULATION = 4

#: D3 fallback for a *degenerate* always-correct cohort whose IQR is 0 (Q1 == Q3,
#: i.e. the tokens are clustered on a single value). The Tukey fence collapses
#: onto that cluster value there and would silently drop a genuine outlier, so
#: instead flag items whose mean tokens exceed ``median * this multiple``.
D3_DEGENERATE_MEDIAN_MULTIPLE = 2.5

#: Model-name -> family prefix heuristic. Keys are lowercase substrings/prefixes
#: matched against the (provider-stripped, lowercased) model name; the first
#: matching family wins in iteration order. Documented and extend-only: unknown
#: models fall through to family ``"unknown"`` and never anchor D7 consensus.
_FAMILY_PREFIXES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("openai", ("gpt", "o1", "o3", "o4", "chatgpt", "davinci", "text-embedding")),
    ("anthropic", ("claude",)),
    ("google", ("gemini", "palm", "bison", "gemma")),
    ("meta", ("llama", "meta-llama", "codellama")),
    ("mistral", ("mistral", "mixtral", "codestral", "ministral")),
    ("cohere", ("command", "cohere")),
    ("deepseek", ("deepseek",)),
    ("qwen", ("qwen",)),
    ("xai", ("grok",)),
)

#: Family assigned when no prefix matches. Never anchors D7 consensus (D7 needs
#: >=2 *known, distinct* families).
UNKNOWN_FAMILY = "unknown"


# ---------------------------------------------------------------------------
# Continuous defect score (#1881) — default logistic coefficients
# ---------------------------------------------------------------------------
# The score is ``sigmoid(b0 + Σ wi·xi)`` over three per-item telemetry features
# computed purely from the outcome matrix:
#   * ``mean_wrong``    = 1 − mean correctness across the item's configs — the
#                          graded generalization of "never-correct". Dominant
#                          signal: an item wrong in 5/6 configs is nearly as
#                          suspect as one wrong in 6/6, and per the #1881 fit this
#                          coefficient absorbs most of ``never_correct``'s weight.
#   * ``never_correct`` = 1.0 when the item is wrong in EVERY config it ran, else
#                          0.0 — a smaller residual bump on top of ``mean_wrong``.
#   * ``instability``   = config-disagreement rate = ``min(n_correct, n_wrong) /
#                          n_outcomes`` in [0, 0.5]; 0 when configs are unanimous,
#                          0.5 when evenly split (models can't agree → ambiguous).
#
# These are ILLUSTRATIVE DEFAULT coefficients — ROUND, hand-chosen values picked
# only so the score gives a sensible *ordering* (never-correct > mixed >
# always-correct) out of the box. They are NOT the AUC-0.83 fitted coefficients:
# the #1881 validation study (in-sample AUC ~0.83) is evidence the FEATURE
# APPROACH works — that these three telemetry features separate defective items —
# and is NOT the provenance of these particular constants. Because they are not
# fitted, the ABSOLUTE ``defect_score`` is not a calibrated probability; only its
# RANK is trustworthy by default, so the default output to rely on is the
# dataset-relative ``defect_percentile``, not the raw score.
#
# To get an absolutely-calibrated probability, refit a logistic regression on your
# own labeled subset over the same three features and pass the resulting ``b0`` +
# weights via the ``intercept`` / ``weights`` parameters of
# :func:`compute_defect_scores` (and :func:`compute_eval_audit`) — no code change
# required.
#
# Ordering sanity of the defaults (illustrative scores, NOT calibrated
# probabilities):
#   * always-correct, stable item (mean_wrong=0, never_correct=0, instability=0)
#     → z = -4.0 → score ~0.018 (low, as it should be);
#   * never-correct item (mean_wrong=1, never_correct=1, instability=0)
#     → z = +3.5 → score ~0.971 (high);
#   * evenly-split item (mean_wrong=0.5, never_correct=0, instability=0.5)
#     → z = -0.5 → score ~0.378 (the graded middle).

#: Illustrative logistic intercept ``b0`` (hand-chosen default, NOT fitted).
#: Negative so the baseline (always-correct, stable) item scores near zero —
#: defects are the rare positive class.
DEFECT_SCORE_INTERCEPT = -4.0

#: Illustrative logistic feature weights ``wi`` (hand-chosen defaults, NOT the
#: AUC-0.83 fitted values). ``mean_wrong`` dominates by design (see above).
DEFECT_SCORE_WEIGHTS: dict[str, float] = {
    "mean_wrong": 6.0,
    "never_correct": 1.5,
    "instability": 1.0,
}

#: The ordered feature vector the score consumes. Fixing the order keeps the
#: logit sum and the ``contributing_signals`` output deterministic.
DEFECT_SCORE_FEATURES: tuple[str, ...] = ("mean_wrong", "never_correct", "instability")

#: Tiny ABSOLUTE floor used only to drop true-zero-value features (a feature that
#: simply didn't fire) from ``contributing_signals``. It is deliberately far below
#: any plausible ``weight * value`` so it never doubles as a suspicion threshold —
#: signal *selection* is scale-aware (see :data:`DEFECT_SIGNAL_RELATIVE_KEEP`).
DEFECT_SIGNAL_ZERO_EPS = 1e-9

#: Scale-aware keep fraction: a contribution is reported as a signal when its
#: magnitude is at least this fraction of the item's LARGEST contribution
#: magnitude. Being relative (not absolute), it survives any refit weight scale —
#: a max-suspicion item always yields at least its top signal, even under tiny
#: weights — where a fixed absolute cutoff would silently blank the explanation.
DEFECT_SIGNAL_RELATIVE_KEEP = 0.05


# ---------------------------------------------------------------------------
# Model-family inference
# ---------------------------------------------------------------------------
def model_family(model: Any) -> str:
    """Infer a model's provider family from its name.

    Strips a ``provider/model`` prefix, lowercases, and matches against
    :data:`_FAMILY_PREFIXES`. Returns :data:`UNKNOWN_FAMILY` for empty or
    unrecognized names. Pure and deterministic.
    """
    if not isinstance(model, str) or not model.strip():
        return UNKNOWN_FAMILY
    name = model.strip().lower()
    # Prefer the model half of an explicit "provider/model" identifier, but also
    # let a recognized provider prefix decide (e.g. "openai/…", "anthropic/…").
    provider = name.split("/", 1)[0] if "/" in name else ""
    tail = name.split("/", 1)[1] if "/" in name else name
    for family, prefixes in _FAMILY_PREFIXES:
        if provider == family:
            return family
        for prefix in prefixes:
            if tail.startswith(prefix) or name.startswith(prefix):
                return family
    return UNKNOWN_FAMILY


def _config_model(config: Any) -> Any:
    """Best-effort extraction of the model name from a trial config."""
    if isinstance(config, dict):
        for key in ("model", "model_name", "llm", "llm_model"):
            value = config.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return None


# ---------------------------------------------------------------------------
# Per-cell primitives (reusable by #1881)
# ---------------------------------------------------------------------------
def cell_is_correct(cell: dict[str, Any], threshold: float) -> bool | None:
    """Whether one outcome cell counts as correct.

    Returns ``None`` when correctness is indeterminable (no signal at all). A
    cell that errored or is explicitly ``success == False`` is wrong (``False``);
    otherwise the scoring signal (``score`` then ``accuracy``) decides against
    ``threshold``, falling back to the ``success`` flag.
    """
    if cell.get("error") is not None:
        return False
    if cell.get("success") is False:
        return False
    score = cell.get("score")
    if isinstance(score, (int, float)) and not isinstance(score, bool):
        return float(score) >= threshold
    accuracy = cell.get("accuracy")
    if isinstance(accuracy, (int, float)) and not isinstance(accuracy, bool):
        return float(accuracy) >= threshold
    success = cell.get("success")
    if isinstance(success, bool):
        return success
    return None


def _cell_total_tokens(cell: dict[str, Any]) -> int | None:
    """Total token count for a cell, or ``None`` when unavailable."""
    tokens = cell.get("tokens")
    if not isinstance(tokens, dict):
        return None
    total = tokens.get("total")
    if isinstance(total, bool) or not isinstance(total, (int, float)):
        return None
    return int(total)


def _answer_key(value: Any) -> str | None:
    """Normalize a predicted answer into a comparable key.

    Strings are trimmed and lowercased; other values are canonicalized via
    ``json.dumps`` (sorted keys, ``default=str``). ``None`` (no captured answer)
    yields ``None`` so a missing answer never anchors consensus.
    """
    if value is None:
        return None
    if isinstance(value, str):
        key = value.strip().lower()
        return key or None
    try:
        return json.dumps(value, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return str(value)


def _quantile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolation quantile (type-7, numpy default) over sorted data."""
    if not sorted_values:
        raise ValueError("quantile of empty sequence")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    frac = position - lower
    return sorted_values[lower] * (1 - frac) + sorted_values[upper] * frac


# ---------------------------------------------------------------------------
# Continuous defect score (#1881)
# ---------------------------------------------------------------------------
def _logistic(z: float) -> float:
    """Numerically-stable logistic ``1 / (1 + exp(-z))`` in [0, 1]."""
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _item_defect_features(outcomes: list[bool]) -> dict[str, float]:
    """Per-item telemetry features from an item's determinable outcomes.

    Requires ``len(outcomes) >= 2`` (the caller enforces it): with a single
    determinable cell ``mean_wrong`` and ``instability`` are ill-defined, so such
    items are excluded from scoring rather than scored on one column.

    Returns ``mean_wrong`` (1 − mean correctness), ``never_correct`` (1.0 iff wrong
    everywhere), and ``instability`` (minority-outcome fraction in [0, 0.5]).
    """
    n = len(outcomes)
    n_correct = sum(1 for o in outcomes if o)
    n_wrong = n - n_correct
    return {
        "mean_wrong": n_wrong / n,
        "never_correct": 1.0 if n_correct == 0 else 0.0,
        "instability": min(n_correct, n_wrong) / n,
    }


def compute_defect_scores(
    matrix: dict[str, Any] | None,
    *,
    success_threshold: float = DEFAULT_SUCCESS_THRESHOLD,
    intercept: float = DEFECT_SCORE_INTERCEPT,
    weights: dict[str, float] | None = None,
) -> list[ItemDefectScore]:
    """Continuous per-item defect scores over an outcome matrix (issue #1881).

    Pure, deterministic function of the matrix — no LLM calls, no network. For
    each item that ran (with a determinable outcome) in ``>=2`` configs, computes
    a heuristic suspicion score ``sigmoid(intercept + Σ weight·feature)`` in
    ``[0, 1]`` over the telemetry features from :func:`_item_defect_features`,
    plus its dataset-relative percentile and the per-feature contributions that
    drove it.

    With the DEFAULT (illustrative, hand-chosen) coefficients the absolute
    ``defect_score`` is NOT a calibrated probability — only its RANK is
    meaningful, so rely on ``defect_percentile``. Pass ``intercept`` / ``weights``
    refit on your own labeled subset to obtain an absolutely-calibrated score.

    Items run in ``<2`` configs are excluded (their features are undefined). An
    all-correct item scores low; a never-correct item scores high.

    Args:
        matrix: An outcome-matrix dict (see
            :func:`traigent.utils.outcome_matrix.build_outcome_matrix`), or ``None``.
        success_threshold: Per-cell correctness cutoff (default 0.5).
        intercept: Logistic intercept ``b0`` (default :data:`DEFECT_SCORE_INTERCEPT`).
        weights: Logistic feature weights (default :data:`DEFECT_SCORE_WEIGHTS`).
            Pass a refit dict to swap in customer-fitted coefficients.

    Returns:
        One :class:`~traigent.api.types.ItemDefectScore` per scored item, sorted
        by ``defect_score`` descending (ties broken by ``example_id`` for a
        deterministic order). Empty when the matrix has no scorable items.

    Percentile convention:
        ``defect_percentile`` = fraction of scored items whose score is ``<=`` this
        item's, i.e. ``count(score_j <= score_i) / N``. The most suspicious item is
        ``1.0``; tied items share an identical percentile (all tied members are
        counted by ``<=``); a single-item run yields ``1.0``. Percentile is
        therefore monotonic non-decreasing in ``defect_score``.
    """
    from traigent.api.types import ItemDefectScore

    coeffs = DEFECT_SCORE_WEIGHTS if weights is None else weights
    if not matrix:
        return []
    examples = matrix.get("examples") or []
    if not examples:
        return []

    raw: list[tuple[str, float, dict[str, float]]] = []  # (id, score, features)
    for example in examples:
        example_id = str(example.get("example_id"))
        cells = example.get("cells") or {}
        outcomes = [
            correct
            for cell in cells.values()
            if (correct := cell_is_correct(cell, success_threshold)) is not None
        ]
        if len(outcomes) < 2:
            continue  # features undefined on a single column -> excluded
        features = _item_defect_features(outcomes)
        z = intercept + sum(
            coeffs.get(name, 0.0) * features[name] for name in DEFECT_SCORE_FEATURES
        )
        raw.append((example_id, _logistic(z), features))

    if not raw:
        return []

    # Percentile: fraction of scored items with score <= this item's (ties share
    # a percentile; a single item -> 1.0). ``bisect_right`` over the sorted scores
    # counts exactly the ``<=`` members.
    sorted_scores = sorted(score for _, score, _ in raw)
    total = len(sorted_scores)

    scored: list[ItemDefectScore] = []
    for example_id, score, features in raw:
        percentile = bisect.bisect_right(sorted_scores, score) / total
        scored.append(
            ItemDefectScore(
                example_id=example_id,
                defect_score=score,
                defect_percentile=percentile,
                features=features,
                contributing_signals=_build_signals(features, coeffs),
            )
        )

    # Ranked worklist: most suspicious first, deterministic tie-break by id.
    scored.sort(key=lambda s: (-s.defect_score, s.example_id))
    return scored


def _build_signals(
    features: dict[str, float], weights: dict[str, float]
) -> list[DefectSignal]:
    """Per-feature ``weight * value`` contributions that drove the item's score.

    Selection is **scale-aware and sign-honest** so it stays correct under an
    arbitrary refit:

    * A contribution is kept when its magnitude is at least
      :data:`DEFECT_SIGNAL_RELATIVE_KEEP` of the item's LARGEST contribution
      magnitude (a tiny absolute floor, :data:`DEFECT_SIGNAL_ZERO_EPS`, still
      drops true-zero features). Because the cutoff is relative, a max-suspicion
      item always yields at least its top signal even under tiny refit weights —
      a fixed absolute cutoff would silently blank the explanation there.
    * Signals are ranked by contribution **magnitude** (strongest driver first),
      and each signal's ``contribution`` carries its SIGN: a positive value raised
      suspicion, a negative one (possible only under a negative refit weight)
      LOWERED it. An all-correct item has an empty list — nothing fired.

    With the default (all-positive) coefficients every contribution is
    non-negative, so magnitude-ranking equals descending-value ranking and the
    default output is unchanged.
    """
    from traigent.api.types import DefectSignal

    contributions = [
        (name, features.get(name, 0.0), weights.get(name, 0.0))
        for name in DEFECT_SCORE_FEATURES
    ]
    max_magnitude = max(
        (abs(weight * value) for _, value, weight in contributions),
        default=0.0,
    )
    if max_magnitude <= DEFECT_SIGNAL_ZERO_EPS:
        return []  # nothing fired (all features true-zero) -> no explanation
    keep_floor = max(
        DEFECT_SIGNAL_ZERO_EPS, DEFECT_SIGNAL_RELATIVE_KEEP * max_magnitude
    )

    signals: list[DefectSignal] = []
    for name, value, weight in contributions:
        contribution = weight * value
        if abs(contribution) >= keep_floor:
            signals.append(
                DefectSignal(
                    feature=name,
                    value=value,
                    weight=weight,
                    contribution=contribution,
                )
            )
    # Rank by magnitude so the strongest driver is first regardless of sign; the
    # contribution's sign then tells the reviewer the direction (raise vs lower).
    signals.sort(key=lambda s: abs(s.contribution), reverse=True)
    return signals


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------
def _detector_stat(
    name: str,
    *,
    active: bool,
    flagged_count: int,
    eligible_count: int,
    lift: float | None,
    note: str | None,
) -> DetectorStat:
    from traigent.api.types import DetectorStat

    flag_rate = flagged_count / eligible_count if eligible_count else 0.0
    return DetectorStat(
        name=name,
        active=active,
        flagged_count=flagged_count,
        eligible_count=eligible_count,
        flag_rate=flag_rate,
        lift=lift,
        note=note,
    )


def compute_eval_audit(
    matrix: dict[str, Any] | None,
    *,
    success_threshold: float = DEFAULT_SUCCESS_THRESHOLD,
    token_iqr_k: float = DEFAULT_TOKEN_IQR_K,
    defect_score_intercept: float = DEFECT_SCORE_INTERCEPT,
    defect_score_weights: dict[str, float] | None = None,
) -> EvalAudit | None:
    """Run the eval-defect detectors + continuous scorer over an outcome matrix.

    Pure function of the matrix produced by
    :func:`traigent.utils.outcome_matrix.build_outcome_matrix` (D7 additionally
    reads the per-cell ``predicted`` answer that matrix now carries). Makes no
    LLM calls and no network requests. Produces both #1880's binary ``flagged``
    detectors and #1881's continuous ``scored`` worklist (see
    :func:`compute_defect_scores`).

    Args:
        matrix: An outcome-matrix dict, or ``None``.
        success_threshold: Per-cell correctness cutoff (default 0.5).
        token_iqr_k: Tukey fence multiplier for D3 (default 1.5).
        defect_score_intercept: Logistic intercept for the continuous defect
            score (default :data:`DEFECT_SCORE_INTERCEPT`).
        defect_score_weights: Logistic feature weights for the continuous defect
            score (default :data:`DEFECT_SCORE_WEIGHTS`); pass a refit dict to
            swap in customer-fitted coefficients.

    Returns:
        An :class:`~traigent.api.types.EvalAudit`, or ``None`` when the audit is
        not applicable — no matrix, no examples, or fewer than two configurations
        (columns). The <2-config case is the "not requested / nothing to audit"
        signal: never-correct and consensus are meaningless on a single column.
    """
    from traigent.api.types import EvalAudit, EvalAuditFlag, EvalAuditSummary

    if not matrix:
        return None
    examples = matrix.get("examples") or []
    trials = matrix.get("trials") or []
    if not examples or len(trials) < 2:
        return None

    # trial_id -> family, and the set of distinct KNOWN families present.
    trial_family: dict[str, str] = {}
    known_families: set[str] = set()
    for column in trials:
        trial_id = column.get("trial_id")
        family = model_family(_config_model(column.get("config")))
        if trial_id is not None:
            trial_family[str(trial_id)] = family
        if family != UNKNOWN_FAMILY:
            known_families.add(family)
    family_count = len(known_families)
    d7_active = family_count >= 2

    example_count = len(examples)

    # Base error rate across all determinable cells (drives D1/D7 lift nulls).
    wrong_cells = 0
    total_cells = 0
    for example in examples:
        for cell in (example.get("cells") or {}).values():
            correct = cell_is_correct(cell, success_threshold)
            if correct is None:
                continue
            total_cells += 1
            if not correct:
                wrong_cells += 1
    p_wrong = wrong_cells / total_cells if total_cells else 0.0

    # --- D1 never-correct + D7 consensus-on-wrong (single pass over rows) -----
    d1_flags: dict[str, EvalAuditFlag] = {}
    d7_flags: dict[str, EvalAuditFlag] = {}
    d1_eligible = 0
    always_correct: list[tuple[str, float]] = []  # (example_id, mean tokens)

    for example in examples:
        example_id = str(example.get("example_id"))
        cells = example.get("cells") or {}
        outcomes: list[bool] = []
        correct_token_totals: list[int] = []
        # answer_key -> set of families that produced that WRONG answer
        wrong_answer_families: dict[str, set[str]] = {}
        wrong_answer_value: dict[str, Any] = {}

        for trial_id, cell in cells.items():
            correct = cell_is_correct(cell, success_threshold)
            if correct is None:
                continue
            outcomes.append(correct)
            if correct:
                tokens = _cell_total_tokens(cell)
                if tokens is not None:
                    correct_token_totals.append(tokens)
            else:
                # Collect for D7: which families produced which wrong answer.
                family = trial_family.get(str(trial_id), UNKNOWN_FAMILY)
                if family == UNKNOWN_FAMILY:
                    continue
                key = _answer_key(cell.get("predicted"))
                if key is None:
                    continue
                wrong_answer_families.setdefault(key, set()).add(family)
                wrong_answer_value.setdefault(key, cell.get("predicted"))

        # D1: eligible when the example ran (with a determinable outcome) in >=2
        # columns; flagged when wrong in all of them.
        if len(outcomes) >= 2:
            d1_eligible += 1
            if not any(outcomes):
                d1_flags[example_id] = EvalAuditFlag(
                    example_id=example_id, detectors=["never-correct"]
                )

        # D3 population: always-correct items carrying token data.
        if outcomes and all(outcomes) and correct_token_totals:
            mean_tokens = sum(correct_token_totals) / len(correct_token_totals)
            always_correct.append((example_id, mean_tokens))

        # D7: a wrong answer shared by >=2 distinct known families.
        if d7_active:
            for key, families in wrong_answer_families.items():
                if len(families) >= 2:
                    d7_flags[example_id] = EvalAuditFlag(
                        example_id=example_id,
                        detectors=["cross-family-consensus-on-wrong"],
                        suggested_answer=wrong_answer_value.get(key),
                    )
                    break

    # --- D3 token-leak (outlier among the always-correct cohort) -------------
    d3_flags: dict[str, EvalAuditFlag] = {}
    d3_eligible = len(always_correct)
    d3_note: str | None = None
    d3_lift: float | None = None
    if d3_eligible < _MIN_D3_POPULATION:
        d3_note = (
            f"skipped: only {d3_eligible} always-correct item(s) carry token "
            f"data (need >= {_MIN_D3_POPULATION} for a robust IQR)"
        )
    else:
        values = sorted(v for _, v in always_correct)
        q1 = _quantile(values, 0.25)
        q3 = _quantile(values, 0.75)
        iqr = q3 - q1
        median = _quantile(values, 0.5)
        if iqr > 0:
            # Normal path: Tukey upper fence over a cohort with real spread.
            fence = q3 + token_iqr_k * iqr
        else:
            # Degenerate cohort (Q1 == Q3 -> IQR == 0): tokens are clustered on a
            # single value, so the Tukey fence would sit exactly on that value and
            # a genuine outlier (e.g. [100,100,100,100,1000]) would be dropped
            # even though it is a clear token leak. Fall back to a median-multiple
            # rule so a clear outlier over a zero-spread cohort is still caught.
            fence = median * D3_DEGENERATE_MEDIAN_MULTIPLE
        burn_multiples: list[float] = []
        for example_id, mean_tokens in always_correct:
            if mean_tokens > fence:
                d3_flags[example_id] = EvalAuditFlag(
                    example_id=example_id, detectors=["token-leak"]
                )
                if median > 0:
                    burn_multiples.append(mean_tokens / median)
        if burn_multiples:
            d3_lift = sum(burn_multiples) / len(burn_multiples)

    # --- Lift for D1 / D7 (enrichment over an independence null) -------------
    column_count = len(trials)
    d1_lift = _enrichment_lift(len(d1_flags), example_count, p_wrong, column_count)
    d7_lift = (
        _enrichment_lift(len(d7_flags), example_count, p_wrong, 2)
        if d7_active
        else None
    )

    # --- Merge flags across detectors ----------------------------------------
    merged: dict[str, EvalAuditFlag] = {}
    for source in (d1_flags, d3_flags, d7_flags):
        for example_id, flag in source.items():
            if example_id not in merged:
                merged[example_id] = EvalAuditFlag(
                    example_id=example_id, detectors=[], suggested_answer=None
                )
            existing = merged[example_id]
            existing.detectors.extend(flag.detectors)
            if flag.suggested_answer is not None and existing.suggested_answer is None:
                existing.suggested_answer = flag.suggested_answer

    flagged = [merged[example_id] for example_id in sorted(merged)]

    summary = EvalAuditSummary(
        example_count=example_count,
        config_count=column_count,
        family_count=family_count,
        total_flagged=len(flagged),
        detectors={
            "never-correct": _detector_stat(
                "never-correct",
                active=True,
                flagged_count=len(d1_flags),
                eligible_count=d1_eligible,
                lift=d1_lift,
                note=None,
            ),
            "token-leak": _detector_stat(
                "token-leak",
                active=d3_eligible >= _MIN_D3_POPULATION,
                flagged_count=len(d3_flags),
                eligible_count=d3_eligible,
                lift=d3_lift,
                note=d3_note,
            ),
            "cross-family-consensus-on-wrong": _detector_stat(
                "cross-family-consensus-on-wrong",
                active=d7_active,
                flagged_count=len(d7_flags),
                eligible_count=example_count if d7_active else 0,
                lift=d7_lift,
                note=(
                    None
                    if d7_active
                    else f"inactive: needs >= 2 model families, found {family_count}"
                ),
            ),
        },
    )
    scored = compute_defect_scores(
        matrix,
        success_threshold=success_threshold,
        intercept=defect_score_intercept,
        weights=defect_score_weights,
    )
    return EvalAudit(flagged=flagged, summary=summary, scored=scored)


def _enrichment_lift(
    flagged: int, population: int, p_wrong: float, exponent: int
) -> float | None:
    """Observed flag rate divided by an independence-null expectation.

    ``null = p_wrong ** exponent`` is the rate expected if per-config errors were
    independent at the base error rate. Returns ``None`` when the null is 0 (no
    wrong cells) or the population is empty; ``0.0`` when nothing is flagged.
    """
    if population == 0 or p_wrong <= 0.0:
        return None
    if flagged == 0:
        return 0.0
    null = p_wrong**exponent
    if null <= 0.0:
        return None
    return (flagged / population) / null
