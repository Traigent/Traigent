import { ValidationError } from '../core/errors.js';
import type {
  NormalizedObjectiveDefinition,
  OptimizationTrialRecord,
  PromotionChanceConstraintResult,
  PromotionDecision,
  PromotionObjectiveResult,
  TvlPromotionPolicy,
} from './types.js';

interface ChanceConstraintCounts {
  successes: number;
  trials: number;
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function asFiniteNumberArray(value: unknown): number[] | undefined {
  if (!Array.isArray(value)) {
    return undefined;
  }
  const numericValues = value.filter(
    (entry): entry is number => typeof entry === 'number' && Number.isFinite(entry),
  );
  return numericValues.length === value.length ? numericValues : undefined;
}

function isBinarySampleSeries(values: readonly number[]): boolean {
  return values.every((value) => value === 0 || value === 1);
}

export function getTrialMetricSamples(
  trial: OptimizationTrialRecord,
  metricName: string,
): number[] | undefined {
  const metadata = isPlainObject(trial.metadata) ? trial.metadata : undefined;
  const metricSamples = metadata?.['metricSamples'];
  if (!isPlainObject(metricSamples)) {
    return undefined;
  }

  return asFiniteNumberArray(metricSamples[metricName]);
}

function getExplicitChanceConstraintCounts(
  trial: OptimizationTrialRecord,
  metricName: string,
): ChanceConstraintCounts | undefined {
  const metadata = isPlainObject(trial.metadata) ? trial.metadata : undefined;
  const rawCounts = metadata?.['chanceConstraintCounts'];
  if (!isPlainObject(rawCounts)) {
    return undefined;
  }

  const entry = rawCounts[metricName];
  if (!isPlainObject(entry)) {
    return undefined;
  }

  const successes = entry['successes'];
  const trials = entry['trials'];
  if (
    typeof successes !== 'number' ||
    !Number.isInteger(successes) ||
    successes < 0 ||
    typeof trials !== 'number' ||
    !Number.isInteger(trials) ||
    trials < 0 ||
    successes > trials
  ) {
    throw new ValidationError(
      `Trial "${trial.trialId}" has invalid chanceConstraintCounts for "${metricName}".`,
    );
  }

  return { successes, trials };
}

function deriveChanceConstraintCountsFromSamples(
  trial: OptimizationTrialRecord,
  metricName: string,
): ChanceConstraintCounts | undefined {
  const samples = getTrialMetricSamples(trial, metricName);
  if (!samples || samples.length === 0) {
    return undefined;
  }
  if (!isBinarySampleSeries(samples)) {
    return undefined;
  }

  const successes = samples.reduce(
    (sum: number, value: number) => sum + value,
    0,
  );
  return { successes, trials: samples.length };
}

function getChanceConstraintCounts(
  trial: OptimizationTrialRecord,
  metricName: string,
): ChanceConstraintCounts | undefined {
  return (
    getExplicitChanceConstraintCounts(trial, metricName) ??
    deriveChanceConstraintCountsFromSamples(trial, metricName)
  );
}

function inverseNormalCdf(p: number): number {
  if (p <= 0) {
    return Number.NEGATIVE_INFINITY;
  }
  if (p >= 1) {
    return Number.POSITIVE_INFINITY;
  }

  if (p > 0.5) {
    return -inverseNormalCdf(1 - p);
  }

  const t = Math.sqrt(-2 * Math.log(p));
  const c0 = 2.515517;
  const c1 = 0.802853;
  const c2 = 0.010328;
  const d1 = 1.432788;
  const d2 = 0.189269;
  const d3 = 0.001308;

  return -(
    t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t)
  );
}

function logGamma(value: number): number {
  const coefficients = [
    676.5203681218851,
    -1259.1392167224028,
    771.3234287776531,
    -176.6150291621406,
    12.507343278686905,
    -0.13857109526572012,
    9.984369578019572e-6,
    1.5056327351493116e-7,
  ];

  if (value < 0.5) {
    return (
      Math.log(Math.PI) -
      Math.log(Math.sin(Math.PI * value)) -
      logGamma(1 - value)
    );
  }

  let x = 0.9999999999998099;
  const shifted = value - 1;
  for (let index = 0; index < coefficients.length; index += 1) {
    x += coefficients[index]! / (shifted + index + 1);
  }

  const t = shifted + coefficients.length - 0.5;
  return (
    0.5 * Math.log(2 * Math.PI) +
    (shifted + 0.5) * Math.log(t) -
    t +
    Math.log(x)
  );
}

function logBeta(alpha: number, beta: number): number {
  return logGamma(alpha) + logGamma(beta) - logGamma(alpha + beta);
}

function computeBetaCoefficient(x: number, alpha: number, beta: number): number {
  try {
    const logBt =
      alpha * Math.log(x) + beta * Math.log(1 - x) - logBeta(alpha, beta);
    return Math.exp(logBt);
  } catch {
    return 0;
  }
}

function cfStep(
  aa: number,
  c: number,
  d: number,
  fpMin: number,
): { c: number; d: number; delta: number } {
  let nextD = 1 + aa * d;
  if (Math.abs(nextD) < fpMin) {
    nextD = fpMin;
  }

  let nextC = 1 + aa / c;
  if (Math.abs(nextC) < fpMin) {
    nextC = fpMin;
  }

  nextD = 1 / nextD;
  return {
    c: nextC,
    d: nextD,
    delta: nextD * nextC,
  };
}

function regularizedBeta(x: number, alpha: number, beta: number): number {
  if (x <= 0) {
    return 0;
  }
  if (x >= 1) {
    return 1;
  }

  if (x > (alpha + 1) / (alpha + beta + 2)) {
    return 1 - regularizedBeta(1 - x, beta, alpha);
  }

  const bt = computeBetaCoefficient(x, alpha, beta);
  if (bt === 0) {
    return 0;
  }

  const qab = alpha + beta;
  const qap = alpha + 1;
  const qam = alpha - 1;
  const fpMin = Number.MIN_VALUE / Number.EPSILON;

  let c = 1;
  let d = 1 - (qab * x) / qap;
  if (Math.abs(d) < fpMin) {
    d = fpMin;
  }
  d = 1 / d;
  let fraction = d;

  for (let m = 1; m <= 200; m += 1) {
    const m2 = 2 * m;
    let aa = (m * (beta - m) * x) / ((qam + m2) * (alpha + m2));
    ({ c, d } = cfStep(aa, c, d, fpMin));
    fraction *= d * c;

    aa = (-((alpha + m) * (qab + m) * x)) / ((alpha + m2) * (qap + m2));
    const step = cfStep(aa, c, d, fpMin);
    c = step.c;
    d = step.d;
    fraction *= step.delta;

    if (Math.abs(step.delta - 1) < 1e-10) {
      break;
    }
  }
  return (bt * fraction) / alpha;
}

function betaPdf(x: number, alpha: number, beta: number): number {
  if (x <= 0 || x >= 1) {
    return 0;
  }
  try {
    const logValue =
      (alpha - 1) * Math.log(x) +
      (beta - 1) * Math.log(1 - x) -
      logBeta(alpha, beta);
    return Math.exp(logValue);
  } catch {
    return 0;
  }
}

function betaQuantileApprox(p: number, alpha: number, beta: number): number {
  if (p <= 0) {
    return 0;
  }
  if (p >= 1) {
    return 1;
  }

  const mean = alpha / (alpha + beta);
  const variance =
    (alpha * beta) /
    ((alpha + beta) ** 2 * (alpha + beta + 1));
  const standardDeviation = Math.sqrt(variance);
  const z = inverseNormalCdf(p);
  let x = Math.max(0.001, Math.min(0.999, mean + z * standardDeviation));

  for (let iteration = 0; iteration < 10; iteration += 1) {
    const fx = regularizedBeta(x, alpha, beta) - p;
    if (Math.abs(fx) < 1e-10) {
      break;
    }
    const fpx = betaPdf(x, alpha, beta);
    if (fpx < 1e-15) {
      break;
    }

    const step = fx / fpx;
    x = Math.max(0.001, Math.min(0.999, x - 0.5 * step));
  }

  return x;
}

export function clopperPearsonLowerBound(
  successes: number,
  trials: number,
  confidence: number,
): number {
  if (trials <= 0) {
    throw new ValidationError(`trials must be positive, got ${trials}`);
  }
  if (successes < 0 || successes > trials) {
    throw new ValidationError(
      `successes must be in [0, ${trials}], got ${successes}`,
    );
  }
  if (confidence <= 0 || confidence >= 1) {
    throw new ValidationError(
      `confidence must be in (0, 1), got ${confidence}`,
    );
  }

  if (successes === 0) {
    return 0;
  }

  const alpha = 1 - confidence;
  if (trials < 30) {
    return betaQuantileApprox(alpha / 2, successes, trials - successes + 1);
  }

  // Use Wilson as a stable large-sample approximation; keep the public API name
  // aligned with the more conservative small-sample branch above.
  const z = inverseNormalCdf(1 - alpha / 2);
  const pHat = successes / trials;
  const denominator = 1 + (z * z) / trials;
  const center = (pHat + (z * z) / (2 * trials)) / denominator;
  const margin =
    (z *
      Math.sqrt(
        (pHat * (1 - pHat)) / trials + (z * z) / (4 * trials * trials),
      )) /
    denominator;

  return Math.max(0, center - margin);
}

export function evaluatePromotionChanceConstraints(
  trial: OptimizationTrialRecord,
  policy: TvlPromotionPolicy | undefined,
): PromotionChanceConstraintResult[] {
  if (!policy?.chanceConstraints || policy.chanceConstraints.length === 0) {
    return [];
  }

  return policy.chanceConstraints.map((constraint) => {
    const counts = getChanceConstraintCounts(trial, constraint.name);
    if (!counts || counts.trials === 0) {
      return {
        name: constraint.name,
        satisfied: false,
        observedRate: 0,
        lowerBound: 0,
        threshold: constraint.threshold,
        confidence: constraint.confidence,
      };
    }

    const observedRate = counts.successes / counts.trials;
    const lowerBound = clopperPearsonLowerBound(
      counts.successes,
      counts.trials,
      constraint.confidence,
    );
    return {
      name: constraint.name,
      satisfied: lowerBound >= constraint.threshold,
      observedRate,
      lowerBound,
      threshold: constraint.threshold,
      confidence: constraint.confidence,
    };
  });
}

export function getPromotionRejectionReason(
  trial: OptimizationTrialRecord,
  policy: TvlPromotionPolicy | undefined,
): string | undefined {
  const results = evaluatePromotionChanceConstraints(trial, policy);
  if (results.length === 0) {
    return undefined;
  }

  const failed = results.filter((result) => !result.satisfied);
  if (failed.length === 0) {
    return undefined;
  }

  const failedNames = failed.map((result) => result.name).join(', ');
  return `Promotion policy chance constraints rejected the trial: ${failedNames}.`;
}

interface PairedComparisonResult {
  pValue: number;
  effectSize: number;
}

interface StatisticalObjectiveResult {
  name: string;
  direction: NormalizedObjectiveDefinition['direction'];
  candidateBetter: boolean;
  pValue: number;
  effectSize: number;
  epsilon: number;
  candidateMean: number;
  incumbentMean: number;
}

function mean(values: readonly number[]): number {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function sampleVariance(values: readonly number[], meanValue: number): number {
  if (values.length <= 1) {
    return 0;
  }
  return (
    values.reduce((sum, value) => sum + (value - meanValue) ** 2, 0) /
    (values.length - 1)
  );
}

export function benjaminiHochbergAdjust(pValues: readonly number[]): number[] {
  if (pValues.length === 0) {
    return [];
  }
  if (pValues.length === 1) {
    return [Math.min(pValues[0]!, 1)];
  }

  const indexed = pValues.map((value, index) => [value, index] as const);
  indexed.sort((left, right) => left[0] - right[0]);

  const adjusted = new Array<number>(pValues.length).fill(0);
  let previous = indexed.at(-1)![0];
  adjusted[indexed.at(-1)![1]] = Math.min(previous, 1);

  for (let rank = pValues.length - 1; rank > 0; rank -= 1) {
    const [pValue, originalIndex] = indexed[rank - 1]!;
    let current = (pValue * pValues.length) / rank;
    current = Math.min(current, previous, 1);
    adjusted[originalIndex] = current;
    previous = current;
  }

  return adjusted;
}

function tCdfApprox(t: number, degreesOfFreedom: number): number {
  if (degreesOfFreedom <= 0) {
    return 0.5;
  }
  if (t === Number.POSITIVE_INFINITY) {
    return 1;
  }
  if (t === Number.NEGATIVE_INFINITY) {
    return 0;
  }
  if (degreesOfFreedom > 100) {
    return normalCdf(t);
  }
  if (Math.abs(t) < 1e-15) {
    return 0.5;
  }

  const x = degreesOfFreedom / (degreesOfFreedom + t * t);
  const betaValue = regularizedBeta(x, degreesOfFreedom / 2, 0.5);
  return t > 0 ? 1 - 0.5 * betaValue : 0.5 * betaValue;
}

function erfApprox(x: number): number {
  const sign = x < 0 ? -1 : 1;
  const absoluteX = Math.abs(x);
  const a1 = 0.254829592;
  const a2 = -0.284496736;
  const a3 = 1.421413741;
  const a4 = -1.453152027;
  const a5 = 1.061405429;
  const p = 0.3275911;

  const t = 1 / (1 + p * absoluteX);
  const y =
    1 -
    (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) *
      t *
      Math.exp(-absoluteX * absoluteX));
  return sign * y;
}

function normalCdf(x: number): number {
  return 0.5 * (1 + erfApprox(x / Math.sqrt(2)));
}

function pairedComparisonTest(
  candidateSamples: readonly number[],
  incumbentSamples: readonly number[],
  epsilon: number,
  direction: 'greater' | 'less',
): PairedComparisonResult {
  if (
    candidateSamples.length === 0 ||
    candidateSamples.length !== incumbentSamples.length
  ) {
    throw new ValidationError(
      'Paired promotion comparison requires equal-length non-empty samples.',
    );
  }

  const differences = candidateSamples.map(
    (value, index) => value - incumbentSamples[index]!,
  );
  const differenceMean = mean(differences);
  const differenceVariance = sampleVariance(differences, differenceMean);
  const standardError =
    differences.length > 1 && differenceVariance > 0
      ? Math.sqrt(differenceVariance / differences.length)
      : 0;
  const degreesOfFreedom = Math.max(1, differences.length - 1);

  if (standardError < 1e-15) {
    const reject =
      direction === 'greater'
        ? differenceMean > epsilon
        : differenceMean < -epsilon;
    return {
      pValue: reject ? 0 : 1,
      effectSize: differenceMean,
    };
  }

  const tStatistic =
    direction === 'greater'
      ? (differenceMean - epsilon) / standardError
      : (differenceMean + epsilon) / standardError;
  const pValue =
    direction === 'greater'
      ? 1 - tCdfApprox(tStatistic, degreesOfFreedom)
      : tCdfApprox(tStatistic, degreesOfFreedom);

  return {
    pValue,
    effectSize: differenceMean,
  };
}

interface TostResult {
  isEquivalent: boolean;
  pLower: number;
  pUpper: number;
  sampleMean: number;
}

function tostEquivalenceTest(
  samples: readonly number[],
  band: { low: number; high: number },
  alpha: number,
): TostResult {
  if (samples.length === 0) {
    throw new ValidationError('Cannot perform TOST on empty samples.');
  }

  const sampleMean = mean(samples);
  if (samples.length === 1) {
    return {
      isEquivalent: sampleMean > band.low && sampleMean < band.high,
      pLower: sampleMean > band.low ? 0 : 1,
      pUpper: sampleMean < band.high ? 0 : 1,
      sampleMean,
    };
  }

  const sampleStd = Math.sqrt(sampleVariance(samples, sampleMean));
  const standardError = sampleStd / Math.sqrt(samples.length);
  if (standardError < 1e-15) {
    const isEquivalent = sampleMean > band.low && sampleMean < band.high;
    return {
      isEquivalent,
      pLower: sampleMean > band.low ? 0 : 1,
      pUpper: sampleMean < band.high ? 0 : 1,
      sampleMean,
    };
  }

  const degreesOfFreedom = samples.length - 1;
  const tLower = (sampleMean - band.low) / standardError;
  const tUpper = (sampleMean - band.high) / standardError;
  const pLower = 1 - tCdfApprox(tLower, degreesOfFreedom);
  const pUpper = tCdfApprox(tUpper, degreesOfFreedom);

  return {
    isEquivalent: pLower < alpha && pUpper < alpha,
    pLower,
    pUpper,
    sampleMean,
  };
}

function compareBandedWithTost(
  candidateSamples: readonly number[],
  incumbentSamples: readonly number[],
  band: { low: number; high: number },
  alpha: number,
  epsilon: number,
): PairedComparisonResult & { candidateBetter: boolean } {
  const candidateTost = tostEquivalenceTest(candidateSamples, band, alpha);
  const incumbentTost = tostEquivalenceTest(incumbentSamples, band, alpha);

  const center = (band.low + band.high) / 2;
  const candidateDistance = Math.abs(candidateTost.sampleMean - center);
  const incumbentDistance = Math.abs(incumbentTost.sampleMean - center);

  if (candidateTost.isEquivalent && !incumbentTost.isEquivalent) {
    return {
      candidateBetter: true,
      pValue: Math.max(candidateTost.pLower, candidateTost.pUpper),
      effectSize: incumbentDistance - candidateDistance,
    };
  }

  if (!candidateTost.isEquivalent && incumbentTost.isEquivalent) {
    return {
      candidateBetter: false,
      pValue: 1,
      effectSize: incumbentDistance - candidateDistance,
    };
  }

  const candidateBetter = candidateDistance < incumbentDistance - epsilon;
  // This fallback is a deterministic distance heuristic, not a formal
  // significance test. We preserve a pseudo-p-value only so the result can
  // flow through the same promotion-policy machinery as the standard path.
  const pValue = Math.max(
    0,
    Math.min(
      1,
      0.5 *
        (1 + (candidateDistance - incumbentDistance) / Math.max(incumbentDistance, 1e-10)),
    ),
  );

  return {
    candidateBetter,
    pValue,
    effectSize: incumbentDistance - candidateDistance,
  };
}

function compareObjectiveStatistically(
  candidate: OptimizationTrialRecord,
  incumbent: OptimizationTrialRecord,
  objective: NormalizedObjectiveDefinition,
  policy: TvlPromotionPolicy,
): StatisticalObjectiveResult | undefined {
  const candidateSamples = getTrialMetricSamples(candidate, objective.metric);
  const incumbentSamples = getTrialMetricSamples(incumbent, objective.metric);
  if (
    !candidateSamples ||
    !incumbentSamples ||
    candidateSamples.length === 0 ||
    candidateSamples.length !== incumbentSamples.length
  ) {
    return undefined;
  }

  const epsilon = policy.minEffect?.[objective.metric] ?? 0;
  if (objective.direction === 'band') {
    if (!objective.band) {
      throw new ValidationError(
        `Objective "${objective.metric}" is missing band metadata.`,
      );
    }
    const comparison = compareBandedWithTost(
      candidateSamples,
      incumbentSamples,
      objective.band,
      objective.band.alpha,
      epsilon,
    );
    return {
      name: objective.metric,
      direction: objective.direction,
      candidateBetter: comparison.candidateBetter,
      pValue: comparison.pValue,
      effectSize: comparison.effectSize,
      epsilon,
      candidateMean: mean(candidateSamples),
      incumbentMean: mean(incumbentSamples),
    };
  }

  const comparison = pairedComparisonTest(
    candidateSamples,
    incumbentSamples,
    epsilon,
    objective.direction === 'maximize' ? 'greater' : 'less',
  );
  return {
    name: objective.metric,
    direction: objective.direction,
    candidateBetter: comparison.pValue < (policy.alpha ?? 0.05),
    pValue: comparison.pValue,
    effectSize:
      objective.direction === 'maximize'
        ? comparison.effectSize
        : -comparison.effectSize,
    epsilon,
    candidateMean: mean(candidateSamples),
    incumbentMean: mean(incumbentSamples),
  };
}

function buildDeterministicObjectiveResults(
  candidate: OptimizationTrialRecord,
  incumbent: OptimizationTrialRecord,
  objectives: readonly NormalizedObjectiveDefinition[],
  policy: TvlPromotionPolicy,
): PromotionObjectiveResult[] {
  return objectives.map((objective) => {
    const candidateMean = Number(candidate.metrics[objective.metric]);
    const incumbentMean = Number(incumbent.metrics[objective.metric]);
    const epsilon = policy.minEffect?.[objective.metric] ?? 0;
    let effectSize = 0;
    let candidateBetter = false;

    if (objective.direction === 'band') {
      const band = objective.band;
      if (!band) {
        throw new ValidationError(
          `Objective "${objective.metric}" is missing band metadata.`,
        );
      }
      const center = (band.low + band.high) / 2;
      const candidateDistance = Math.abs(candidateMean - center);
      const incumbentDistance = Math.abs(incumbentMean - center);
      effectSize = incumbentDistance - candidateDistance;
      candidateBetter = candidateDistance < incumbentDistance - epsilon;
    } else if (objective.direction === 'maximize') {
      effectSize = candidateMean - incumbentMean;
      candidateBetter = candidateMean > incumbentMean + epsilon;
    } else {
      effectSize = incumbentMean - candidateMean;
      candidateBetter = candidateMean + epsilon < incumbentMean;
    }

    return {
      name: objective.metric,
      direction: objective.direction,
      candidateBetter,
      effectSize,
      epsilon,
      candidateMean,
      incumbentMean,
      method: 'deterministic',
    };
  });
}

function compareTieBreakerValues(
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

function evaluateObjectiveResults(
  objectiveResults: readonly PromotionObjectiveResult[],
  alpha: number,
): { anyBetter: boolean; anyWorse: boolean; dominanceSatisfied: boolean } {
  let anyBetter = false;
  let anyWorse = false;

  for (const result of objectiveResults) {
    const adjustedPValue = result.adjustedPValue ?? result.pValue;
    if (
      result.candidateBetter &&
      (adjustedPValue === undefined || adjustedPValue < alpha)
    ) {
      anyBetter = true;
    } else if (!result.candidateBetter && result.effectSize < -result.epsilon) {
      anyWorse = true;
    }
  }

  return {
    anyBetter,
    anyWorse,
    dominanceSatisfied: anyBetter && !anyWorse,
  };
}

function createNoDecision(
  reason: string,
  candidateTrialId?: string,
  incumbentTrialId?: string,
): PromotionDecision {
  return {
    decision: 'no_decision',
    reason,
    objectiveResults: [],
    chanceResults: [],
    adjustedPValues: {},
    dominanceSatisfied: false,
    method: 'none',
    candidateTrialId,
    incumbentTrialId,
  };
}

export function buildPromotionDecision(
  candidate: OptimizationTrialRecord,
  incumbent: OptimizationTrialRecord | undefined,
  objectives: readonly NormalizedObjectiveDefinition[],
  policy: TvlPromotionPolicy | undefined,
): PromotionDecision | undefined {
  if (!policy) {
    return undefined;
  }

  const chanceResults = evaluatePromotionChanceConstraints(candidate, policy);
  const failedChanceResults = chanceResults.filter((result) => !result.satisfied);
  if (failedChanceResults.length > 0) {
    return {
      decision: 'reject',
      reason: `Chance constraints not satisfied: ${failedChanceResults.map((result) => result.name).join(', ')}`,
      objectiveResults: [],
      chanceResults,
      adjustedPValues: {},
      dominanceSatisfied: false,
      method: 'chance-constraints',
      candidateTrialId: candidate.trialId,
      incumbentTrialId: incumbent?.trialId,
    };
  }

  if (!incumbent) {
    return createNoDecision(
      'No incumbent trial available for promotion comparison.',
      candidate.trialId,
      undefined,
    );
  }

  const statisticalResults = objectives
    .map((objective) =>
      compareObjectiveStatistically(candidate, incumbent, objective, policy),
    )
    .filter(
      (result): result is StatisticalObjectiveResult => result !== undefined,
    );

  const alpha = policy.alpha ?? 0.05;

  if (statisticalResults.length === objectives.length) {
    const adjustedPValuesArray =
      policy.adjust === 'BH' && statisticalResults.length > 1
        ? benjaminiHochbergAdjust(statisticalResults.map((result) => result.pValue))
        : statisticalResults.map((result) => result.pValue);

    const objectiveResults = statisticalResults.map((result, index) => ({
      name: result.name,
      direction: result.direction,
      candidateBetter: result.candidateBetter,
      pValue: result.pValue,
      adjustedPValue: adjustedPValuesArray[index],
      effectSize: result.effectSize,
      epsilon: result.epsilon,
      candidateMean: result.candidateMean,
      incumbentMean: result.incumbentMean,
      method: 'statistical',
    })) satisfies PromotionObjectiveResult[];

    const adjustedPValues = Object.fromEntries(
      objectiveResults.map((result) => [
        result.name,
        result.adjustedPValue ?? result.pValue ?? 1,
      ]),
    );
    const { anyWorse, dominanceSatisfied } = evaluateObjectiveResults(
      objectiveResults,
      alpha,
    );

    if (dominanceSatisfied || anyWorse) {
      return {
        decision: dominanceSatisfied ? 'promote' : 'reject',
        reason: dominanceSatisfied
          ? 'Candidate satisfies statistical promotion criteria.'
          : 'Candidate is dominated by the incumbent on one or more objectives.',
        objectiveResults,
        chanceResults,
        adjustedPValues,
        dominanceSatisfied,
        method: 'statistical',
        candidateTrialId: candidate.trialId,
        incumbentTrialId: incumbent.trialId,
      };
    }

    const deterministicObjectiveResults = buildDeterministicObjectiveResults(
      candidate,
      incumbent,
      objectives,
      policy,
    );
    const deterministicEvaluation = evaluateObjectiveResults(
      deterministicObjectiveResults,
      alpha,
    );
    if (deterministicEvaluation.dominanceSatisfied) {
      return {
        decision: 'promote',
        reason:
          'Statistical promotion produced no decision; deterministic fallback promoted the candidate.',
        objectiveResults: deterministicObjectiveResults,
        chanceResults,
        adjustedPValues,
        dominanceSatisfied: true,
        method: 'deterministic',
        candidateTrialId: candidate.trialId,
        incumbentTrialId: incumbent.trialId,
      };
    }
    if (deterministicEvaluation.anyWorse) {
      return {
        decision: 'reject',
        reason:
          'Statistical promotion produced no decision; deterministic fallback rejected the candidate.',
        objectiveResults: deterministicObjectiveResults,
        chanceResults,
        adjustedPValues,
        dominanceSatisfied: false,
        method: 'deterministic',
        candidateTrialId: candidate.trialId,
        incumbentTrialId: incumbent.trialId,
      };
    }

    if (policy.tieBreakers) {
      for (const [metric, direction] of Object.entries(policy.tieBreakers)) {
        const comparison = compareTieBreakerValues(
          candidate,
          incumbent,
          metric,
          direction,
        );
        if (comparison > 0) {
          return {
            decision: 'promote',
            reason: `Statistical promotion produced no decision; tie-breaker "${metric}" promoted the candidate.`,
            objectiveResults: deterministicObjectiveResults,
            chanceResults,
            adjustedPValues,
            dominanceSatisfied: false,
            method: 'deterministic',
            candidateTrialId: candidate.trialId,
            incumbentTrialId: incumbent.trialId,
          };
        }
        if (comparison < 0) {
          return {
            decision: 'reject',
            reason: `Statistical promotion produced no decision; tie-breaker "${metric}" kept the incumbent.`,
            objectiveResults: deterministicObjectiveResults,
            chanceResults,
            adjustedPValues,
            dominanceSatisfied: false,
            method: 'deterministic',
            candidateTrialId: candidate.trialId,
            incumbentTrialId: incumbent.trialId,
          };
        }
      }
    }

    return {
      decision: 'no_decision',
      reason: 'Statistical promotion produced no decision and deterministic fallback did not separate the trials.',
      objectiveResults,
      chanceResults,
      adjustedPValues,
      dominanceSatisfied: false,
      method: 'statistical',
      candidateTrialId: candidate.trialId,
      incumbentTrialId: incumbent.trialId,
    };
  }

  const objectiveResults = buildDeterministicObjectiveResults(
    candidate,
    incumbent,
    objectives,
    policy,
  );
  const { anyWorse, dominanceSatisfied } = evaluateObjectiveResults(
    objectiveResults,
    alpha,
  );

  if (!dominanceSatisfied && !anyWorse && policy.tieBreakers) {
    for (const [metric, direction] of Object.entries(policy.tieBreakers)) {
      const comparison = compareTieBreakerValues(
        candidate,
        incumbent,
        metric,
        direction,
      );
      if (comparison > 0) {
        return {
          decision: 'promote',
          reason: `Tie-breaker "${metric}" promoted the candidate after deterministic comparison.`,
          objectiveResults,
          chanceResults,
          adjustedPValues: {},
          dominanceSatisfied: false,
          method: 'deterministic',
          candidateTrialId: candidate.trialId,
          incumbentTrialId: incumbent.trialId,
        };
      }
      if (comparison < 0) {
        return {
          decision: 'reject',
          reason: `Tie-breaker "${metric}" kept the incumbent after deterministic comparison.`,
          objectiveResults,
          chanceResults,
          adjustedPValues: {},
          dominanceSatisfied: false,
          method: 'deterministic',
          candidateTrialId: candidate.trialId,
          incumbentTrialId: incumbent.trialId,
        };
      }
    }
  }

  return {
    decision: dominanceSatisfied
      ? 'promote'
      : anyWorse
        ? 'reject'
        : 'no_decision',
    reason: dominanceSatisfied
      ? 'Candidate satisfies deterministic promotion criteria.'
      : anyWorse
        ? 'Candidate is dominated by the incumbent on one or more objectives.'
        : 'Insufficient evidence for deterministic promotion.',
    objectiveResults,
    chanceResults,
    adjustedPValues: {},
    dominanceSatisfied,
    method: 'deterministic',
    candidateTrialId: candidate.trialId,
    incumbentTrialId: incumbent.trialId,
  };
}

export function compareTrialsWithStatisticalPromotion(
  candidate: OptimizationTrialRecord,
  incumbent: OptimizationTrialRecord,
  objectives: readonly NormalizedObjectiveDefinition[],
  policy: TvlPromotionPolicy,
): -1 | 0 | 1 | undefined {
  const decision = buildPromotionDecision(candidate, incumbent, objectives, policy);
  if (!decision || decision.method !== 'statistical') {
    return undefined;
  }

  if (decision.decision === 'promote') {
    return 1;
  }
  if (decision.decision === 'reject') {
    return -1;
  }
  return 0;
}
