import { PythonRandom } from './python-random.js';
import { getCompletedTrials } from './native-constraints.js';
import {
  applyDefaultConfig,
  buildFloatValues,
  buildIntValues,
  clamp,
  configKey,
  discreteCardinality,
  ensureLogBounds,
  getOrderedParameterEntries,
  roundToPrecision,
  sampleCandidateConfig,
} from './native-space.js';
import type { CandidateConfig } from './native-space.js';
import { computeSearchScore } from './native-scoring.js';
import type {
  FloatParamDefinition,
  IntParamDefinition,
  NormalizedOptimizationSpec,
  OptimizationTrialRecord,
  ParameterDefinition,
} from './types.js';

function vectorizeConfig(
  config: CandidateConfig,
  entries: [string, ParameterDefinition][]
): number[] {
  const vector: number[] = [];

  for (const [name, definition] of entries) {
    const rawValue = config[name];

    switch (definition.type) {
      case 'enum': {
        const index = definition.values.indexOf(rawValue as never);
        const denominator = Math.max(definition.values.length - 1, 1);
        vector.push(index <= 0 ? 0 : index / denominator);
        break;
      }
      case 'int':
      case 'float': {
        const value = typeof rawValue === 'number' ? rawValue : Number(rawValue ?? definition.min);
        if (definition.scale === 'log') {
          ensureLogBounds(name, definition);
          const minLog = Math.log10(definition.min);
          const maxLog = Math.log10(definition.max);
          const currentLog = Math.log10(clamp(value, definition.min, definition.max));
          vector.push(maxLog === minLog ? 0 : (currentLog - minLog) / (maxLog - minLog));
        } else {
          vector.push(
            definition.max === definition.min
              ? 0
              : (value - definition.min) / (definition.max - definition.min)
          );
        }
        break;
      }
    }
  }

  return vector;
}

function euclideanDistance(left: number[], right: number[]): number {
  let total = 0;
  for (let index = 0; index < left.length; index += 1) {
    const delta = (left[index] ?? 0) - (right[index] ?? 0);
    total += delta * delta;
  }
  return Math.sqrt(total);
}

function estimateBayesianAcquisition(
  candidate: number[],
  observedVectors: number[][],
  observedScores: number[]
): number {
  if (observedVectors.length === 0) {
    return 0;
  }

  const bandwidth = Math.max(0.08, Math.sqrt(candidate.length || 1) / 6);
  let weightedTotal = 0;
  let weightSum = 0;
  let weightedVariance = 0;
  let nearestDistance = Number.POSITIVE_INFINITY;

  for (let index = 0; index < observedVectors.length; index += 1) {
    const distance = euclideanDistance(candidate, observedVectors[index]!);
    const weight = Math.exp(-(distance * distance) / (2 * bandwidth * bandwidth));
    weightedTotal += observedScores[index]! * weight;
    weightSum += weight;
    nearestDistance = Math.min(nearestDistance, distance);
  }

  const mean = weightedTotal / weightSum;

  for (let index = 0; index < observedVectors.length; index += 1) {
    const distance = euclideanDistance(candidate, observedVectors[index]!);
    const weight = Math.exp(-(distance * distance) / (2 * bandwidth * bandwidth));
    const delta = observedScores[index]! - mean;
    weightedVariance += weight * delta * delta;
  }

  const variance = weightSum === 0 ? 0 : weightedVariance / weightSum;
  const exploration = Math.sqrt(Math.max(variance, 0)) + nearestDistance;
  return mean + 0.35 * exploration;
}

function sampleLogValue(
  name: string,
  definition: FloatParamDefinition | IntParamDefinition,
  random: PythonRandom
): number {
  ensureLogBounds(name, definition);
  const minLog = Math.log10(definition.min);
  const maxLog = Math.log10(definition.max);
  const exponent = random.uniform(minLog, maxLog);
  return 10 ** exponent;
}

function buildBayesianLocalCandidate(
  baseline: CandidateConfig,
  entries: [string, ParameterDefinition][],
  random: PythonRandom
): CandidateConfig {
  const candidate: CandidateConfig = {};

  for (const [name, definition] of entries) {
    const baselineValue = baseline[name];

    switch (definition.type) {
      case 'enum': {
        if (random.random() < 0.75 && baselineValue !== undefined) {
          candidate[name] = baselineValue;
        } else {
          candidate[name] = random.choice(definition.values);
        }
        break;
      }
      case 'int': {
        const center = typeof baselineValue === 'number' ? baselineValue : definition.min;
        if (definition.scale === 'log') {
          const sampled = sampleLogValue(name, definition, random);
          candidate[name] = clamp(
            Math.round((center + sampled) / 2),
            definition.min,
            definition.max
          );
          break;
        }
        const span = Math.max(1, Math.round((definition.max - definition.min) / 4));
        const lower = clamp(center - span, definition.min, definition.max);
        const upper = clamp(center + span, definition.min, definition.max);
        candidate[name] =
          definition.step !== undefined && definition.step !== 1
            ? random.choice(
                buildIntValues(name, {
                  ...definition,
                  min: lower,
                  max: upper,
                })
              )
            : random.randint(lower, upper);
        break;
      }
      case 'float': {
        const center = typeof baselineValue === 'number' ? baselineValue : definition.min;
        if (definition.scale === 'log') {
          const sampled = sampleLogValue(name, definition, random);
          const mixed = Math.sqrt(center * sampled);
          candidate[name] =
            definition.step !== undefined
              ? random.choice(
                  buildFloatValues(name, {
                    ...definition,
                    min: Math.min(center, sampled),
                    max: Math.max(center, sampled),
                  })
                )
              : roundToPrecision(clamp(mixed, definition.min, definition.max));
          break;
        }
        const span = Math.max((definition.max - definition.min) / 6, definition.step ?? 0.01);
        const lower = clamp(center - span, definition.min, definition.max);
        const upper = clamp(center + span, definition.min, definition.max);
        const sampled = random.uniform(lower, upper);
        if (definition.step === undefined) {
          candidate[name] = roundToPrecision(sampled);
        } else {
          const snapped =
            Math.round((sampled - definition.min) / definition.step) * definition.step +
            definition.min;
          candidate[name] = roundToPrecision(clamp(snapped, definition.min, definition.max));
        }
        break;
      }
    }
  }

  return candidate;
}

export function suggestBayesianConfig(
  spec: NormalizedOptimizationSpec,
  trials: OptimizationTrialRecord[],
  random: PythonRandom,
  maxTrials: number,
  evaluatePreTrialConstraints: (
    spec: NormalizedOptimizationSpec,
    config: CandidateConfig
  ) => boolean
): { config: CandidateConfig | null; exhaustive: boolean } {
  const entries = getOrderedParameterEntries(spec.configurationSpace);
  const seen = new Set(trials.map((trial) => configKey(trial.config)));
  const completedTrials = getCompletedTrials(trials);
  const cardinality = discreteCardinality(entries);

  if (cardinality !== null && seen.size >= cardinality) {
    return { config: null, exhaustive: true };
  }

  const initialRandomSamples = Math.min(maxTrials, Math.max(5, entries.length * 2));
  if (trials.length < initialRandomSamples) {
    for (let attempt = 0; attempt < 512; attempt += 1) {
      const candidate = applyDefaultConfig(spec, sampleCandidateConfig(entries, random));
      if (!seen.has(configKey(candidate)) && evaluatePreTrialConstraints(spec, candidate)) {
        return { config: candidate, exhaustive: false };
      }
    }
  }

  const observedVectors = completedTrials.map((trial) => vectorizeConfig(trial.config, entries));
  const observedScores = completedTrials.map((trial) =>
    computeSearchScore(trial.metrics, spec.objectives)
  );
  const sortedByScore = [...completedTrials].sort(
    (left, right) =>
      computeSearchScore(right.metrics, spec.objectives) -
      computeSearchScore(left.metrics, spec.objectives)
  );

  let bestCandidate: CandidateConfig | null = null;
  let bestAcquisition = Number.NEGATIVE_INFINITY;

  const candidateBudget = Math.max(256, entries.length * 256);
  for (let attempt = 0; attempt < candidateBudget; attempt += 1) {
    const baseline =
      attempt < sortedByScore.length * 32
        ? sortedByScore[Math.floor(attempt / 32)]?.config
        : undefined;
    const candidate =
      baseline === undefined
        ? applyDefaultConfig(spec, sampleCandidateConfig(entries, random))
        : applyDefaultConfig(spec, buildBayesianLocalCandidate(baseline, entries, random));

    if (seen.has(configKey(candidate)) || !evaluatePreTrialConstraints(spec, candidate)) {
      continue;
    }

    const acquisition = estimateBayesianAcquisition(
      vectorizeConfig(candidate, entries),
      observedVectors,
      observedScores
    );

    if (acquisition > bestAcquisition) {
      bestAcquisition = acquisition;
      bestCandidate = candidate;
    }
  }

  if (!bestCandidate) {
    for (let attempt = 0; attempt < 1024; attempt += 1) {
      const candidate = applyDefaultConfig(spec, sampleCandidateConfig(entries, random));
      if (!seen.has(configKey(candidate)) && evaluatePreTrialConstraints(spec, candidate)) {
        return { config: candidate, exhaustive: false };
      }
    }
    return { config: null, exhaustive: cardinality !== null };
  }

  return { config: bestCandidate, exhaustive: false };
}
