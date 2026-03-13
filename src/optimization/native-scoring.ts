import { ValidationError } from '../core/errors.js';
import type {
  NormalizedObjectiveDefinition,
  OptimizationTrialRecord,
  PromotionDecision,
  TvlPromotionPolicy,
} from './types.js';
import { getCompletedTrials } from './native-constraints.js';
import {
  buildPromotionDecision,
  compareTrialsWithStatisticalPromotion,
} from './native-promotion.js';

export function getObjectiveMetric(
  trial: OptimizationTrialRecord,
  objective: NormalizedObjectiveDefinition,
): number {
  const value = trial.metrics[objective.metric];
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new ValidationError(
      `Trial "${trial.trialId}" is missing numeric metric "${objective.metric}".`,
    );
  }
  return value;
}

function objectiveScoreValue(
  value: number,
  objective: NormalizedObjectiveDefinition,
): number {
  if (objective.direction === 'band') {
    const band = objective.band;
    if (!band) {
      throw new ValidationError(
        `Objective "${objective.metric}" is missing band metadata.`,
      );
    }
    if (value < band.low) {
      return -(band.low - value);
    }
    if (value > band.high) {
      return -(value - band.high);
    }
    return 0;
  }

  return objective.direction === 'minimize' ? -value : value;
}

function compareObjectiveWithTolerance(
  candidateValue: number,
  incumbentValue: number,
  objective: NormalizedObjectiveDefinition,
  epsilon: number,
): -1 | 0 | 1 {
  if (objective.direction === 'band') {
    const band = objective.band;
    if (!band) {
      throw new ValidationError(
        `Objective "${objective.metric}" is missing band metadata.`,
      );
    }
    const candidateDeviation =
      candidateValue < band.low
        ? band.low - candidateValue
        : candidateValue > band.high
          ? candidateValue - band.high
          : 0;
    const incumbentDeviation =
      incumbentValue < band.low
        ? band.low - incumbentValue
        : incumbentValue > band.high
          ? incumbentValue - band.high
          : 0;

    if (candidateDeviation + epsilon < incumbentDeviation) {
      return 1;
    }
    if (candidateDeviation > incumbentDeviation + epsilon) {
      return -1;
    }
    return 0;
  }

  if (objective.direction === 'maximize') {
    if (candidateValue > incumbentValue + epsilon) {
      return 1;
    }
    if (candidateValue + epsilon < incumbentValue) {
      return -1;
    }
    return 0;
  }

  if (candidateValue + epsilon < incumbentValue) {
    return 1;
  }
  if (candidateValue > incumbentValue + epsilon) {
    return -1;
  }
  return 0;
}

function compareTieBreakerMetric(
  candidate: OptimizationTrialRecord,
  incumbent: OptimizationTrialRecord,
  metric: string,
  direction: 'maximize' | 'minimize',
): -1 | 0 | 1 {
  const candidateValue = candidate.metrics[metric];
  const incumbentValue = incumbent.metrics[metric];
  if (
    typeof candidateValue !== 'number' ||
    !Number.isFinite(candidateValue) ||
    typeof incumbentValue !== 'number' ||
    !Number.isFinite(incumbentValue)
  ) {
    return 0;
  }

  if (direction === 'maximize') {
    if (candidateValue > incumbentValue) {
      return 1;
    }
    if (candidateValue < incumbentValue) {
      return -1;
    }
    return 0;
  }

  if (candidateValue < incumbentValue) {
    return 1;
  }
  if (candidateValue > incumbentValue) {
    return -1;
  }
  return 0;
}

function compareTrialsWithPromotionPolicy(
  candidate: OptimizationTrialRecord,
  incumbent: OptimizationTrialRecord,
  objectives: readonly NormalizedObjectiveDefinition[],
  policy: TvlPromotionPolicy,
): -1 | 0 | 1 {
  const statisticalComparison = compareTrialsWithStatisticalPromotion(
    candidate,
    incumbent,
    objectives,
    policy,
  );
  if (statisticalComparison !== undefined) {
    return statisticalComparison;
  }

  let candidateBetter = false;
  let incumbentBetter = false;

  for (const objective of objectives) {
    const epsilon = policy.minEffect?.[objective.metric] ?? 0;
    const comparison = compareObjectiveWithTolerance(
      getObjectiveMetric(candidate, objective),
      getObjectiveMetric(incumbent, objective),
      objective,
      epsilon,
    );
    if (comparison > 0) {
      candidateBetter = true;
    } else if (comparison < 0) {
      incumbentBetter = true;
    }
  }

  if (candidateBetter && !incumbentBetter) {
    return 1;
  }
  if (incumbentBetter && !candidateBetter) {
    return -1;
  }

  if (policy.tieBreakers) {
    for (const [metric, direction] of Object.entries(policy.tieBreakers)) {
      const comparison = compareTieBreakerMetric(
        candidate,
        incumbent,
        metric,
        direction,
      );
      if (comparison !== 0) {
        return comparison;
      }
    }
  }

  // Preserve incumbent order on full ties so "best" remains stable across runs.
  return 0;
}

export function selectBestTrialWithPromotionDecision(
  trials: OptimizationTrialRecord[],
  objectives: readonly NormalizedObjectiveDefinition[],
  promotionPolicy?: TvlPromotionPolicy,
): { bestTrial: OptimizationTrialRecord | null; promotionDecision?: PromotionDecision } {
  const completedTrials = getCompletedTrials(trials);
  if (completedTrials.length === 0) {
    return { bestTrial: null };
  }

  if (!promotionPolicy) {
    return {
      bestTrial: selectBestTrial(trials, objectives, undefined),
    };
  }

  let bestTrial = completedTrials[0]!;
  let lastDecision: PromotionDecision | undefined = buildPromotionDecision(
    bestTrial,
    undefined,
    objectives,
    promotionPolicy,
  );

  for (const candidate of completedTrials.slice(1)) {
    const decision = buildPromotionDecision(
      candidate,
      bestTrial,
      objectives,
      promotionPolicy,
    );
    if (decision) {
      candidate.promotionDecision = decision;
    }
    if (
      compareTrialsWithPromotionPolicy(
        candidate,
        bestTrial,
        objectives,
        promotionPolicy,
      ) > 0
    ) {
      bestTrial = candidate;
      lastDecision = decision;
    }
  }

  return {
    bestTrial,
    promotionDecision: lastDecision,
  };
}

export function computeSearchScore(
  metrics: OptimizationTrialRecord['metrics'],
  objectives: readonly NormalizedObjectiveDefinition[],
): number {
  let weightedTotal = 0;
  let totalWeight = 0;

  for (const objective of objectives) {
    const value = metrics[objective.metric];
    if (typeof value !== 'number' || !Number.isFinite(value)) {
      throw new ValidationError(
        `Trial metrics are missing numeric objective "${objective.metric}".`,
      );
    }
    const signedValue = objectiveScoreValue(value, objective);
    weightedTotal += signedValue * objective.weight;
    totalWeight += objective.weight;
  }

  return totalWeight === 0 ? 0 : weightedTotal / totalWeight;
}

export function selectBestTrial(
  trials: OptimizationTrialRecord[],
  objectives: readonly NormalizedObjectiveDefinition[],
  promotionPolicy?: TvlPromotionPolicy,
): OptimizationTrialRecord | null {
  const completedTrials = getCompletedTrials(trials);
  if (completedTrials.length === 0) return null;

  if (promotionPolicy) {
    return selectBestTrialWithPromotionDecision(
      trials,
      objectives,
      promotionPolicy,
    ).bestTrial;
  }

  const ranges = objectives.map((objective) => {
    const values = completedTrials.map((trial) =>
      objectiveScoreValue(getObjectiveMetric(trial, objective), objective),
    );
    return {
      objective,
      min: values.reduce(
        (currentMinimum, value) => Math.min(currentMinimum, value),
        Number.POSITIVE_INFINITY,
      ),
      max: values.reduce(
        (currentMaximum, value) => Math.max(currentMaximum, value),
        Number.NEGATIVE_INFINITY,
      ),
    };
  });

  let bestTrial: OptimizationTrialRecord | null = null;
  let bestScore = Number.NEGATIVE_INFINITY;

  for (const trial of completedTrials) {
    let weightedScore = 0;
    let totalWeight = 0;

    for (const range of ranges) {
      const value = objectiveScoreValue(
        getObjectiveMetric(trial, range.objective),
        range.objective,
      );
      let normalized = 1;

      if (range.max !== range.min) {
        normalized = (value - range.min) / (range.max - range.min);
      }

      weightedScore += normalized * range.objective.weight;
      totalWeight += range.objective.weight;
    }

    const score = totalWeight === 0 ? 0 : weightedScore / totalWeight;
    if (score > bestScore) {
      bestScore = score;
      bestTrial = trial;
    }
  }

  return bestTrial;
}

export function hasPlateau(
  trials: OptimizationTrialRecord[],
  objectives: readonly NormalizedObjectiveDefinition[],
  plateau:
    | {
        window: number;
        minImprovement: number;
      }
    | undefined,
): boolean {
  const completedTrials = getCompletedTrials(trials);
  if (!plateau || completedTrials.length <= plateau.window) {
    return false;
  }

  const bestHistory: number[] = [];
  let best = Number.NEGATIVE_INFINITY;

  for (const trial of completedTrials) {
    best = Math.max(best, computeSearchScore(trial.metrics, objectives));
    bestHistory.push(best);
  }

  const current = bestHistory.at(-1)!;
  const previous = bestHistory.at(-(plateau.window + 1))!;
  return current - previous < plateau.minImprovement;
}
