"""Eval-dataset defect detectors over the per-example x per-config matrix.

# Traceability: CONC-Layer-Data CONC-Quality-Observability FUNC-ANALYTICS FUNC-STORAGE REQ-STOR-007 SYNC-StorageLogging

Every optimization run already produces the per-example x per-config outcome
matrix (persisted by #1838; see ``traigent.utils.outcome_matrix``). That
by-product doubles as an eval-dataset *auditor*: deterministic detectors over the
outcome/token cells surface examples whose recorded gold label is likely wrong,
ambiguous, or otherwise defective — with no extra LLM calls and no network.

This module implements that computation as a **pure function of the matrix**
(plus, for D7, the per-cell ``predicted`` answer the matrix now carries). It is
the consumer half of #1838 and the base layer for #1881 (a continuous defect
score stacks on the same per-item features computed here).

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

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from traigent.api.types import DetectorStat, EvalAudit, EvalAuditFlag

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
) -> EvalAudit | None:
    """Run the eval-defect detectors over an outcome matrix.

    Pure function of the matrix produced by
    :func:`traigent.utils.outcome_matrix.build_outcome_matrix` (D7 additionally
    reads the per-cell ``predicted`` answer that matrix now carries). Makes no
    LLM calls and no network requests.

    Args:
        matrix: An outcome-matrix dict, or ``None``.
        success_threshold: Per-cell correctness cutoff (default 0.5).
        token_iqr_k: Tukey fence multiplier for D3 (default 1.5).

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
    return EvalAudit(flagged=flagged, summary=summary)


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
