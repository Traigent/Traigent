import { randomUUID } from 'node:crypto';

import { TrialContext } from '../core/context.js';
import { TimeoutError, ValidationError } from '../core/errors.js';
import { MetricsSchema, type Metrics, type TrialConfig } from '../dtos/trial.js';
import { PythonRandom } from './python-random.js';
import type {
  FloatParamDefinition,
  IntParamDefinition,
  NativeOptimizeOptions,
  NativeTrialFunctionResult,
  NormalizedObjectiveDefinition,
  NormalizedOptimizationSpec,
  OptimizationResult,
  OptimizationTrialRecord,
  ParameterDefinition,
} from './types.js';

type NativeTrialFunction = (
  trialConfig: TrialConfig,
) => Promise<NativeTrialFunctionResult>;

type CandidateConfig = Record<string, unknown>;

type TrialOutcome =
  | {
      status: 'completed';
      record: OptimizationTrialRecord;
    }
  | {
      status: 'timeout' | 'error';
      errorMessage: string;
    };

function ensureFiniteNumber(
  value: unknown,
  message: string,
): asserts value is number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new ValidationError(message);
  }
}

function roundToPrecision(value: number): number {
  return Number.parseFloat(value.toPrecision(12));
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

function canonicalize(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map(canonicalize);
  }
  if (value && typeof value === 'object') {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>)
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([key, nested]) => [key, canonicalize(nested)]),
    );
  }
  return value;
}

function stableJson(value: unknown): string {
  return JSON.stringify(canonicalize(value));
}

function configKey(config: CandidateConfig): string {
  return stableJson(config);
}

function getOrderedParameterEntries(
  configurationSpace: NormalizedOptimizationSpec['configurationSpace'],
): [string, ParameterDefinition][] {
  const names = Object.keys(configurationSpace).sort((left, right) => {
    if (left === 'model' && right !== 'model') return 1;
    if (right === 'model' && left !== 'model') return -1;
    return left.localeCompare(right);
  });

  return names.map((name) => [name, configurationSpace[name]!]);
}

function ensureLogBounds(
  name: string,
  definition: FloatParamDefinition | IntParamDefinition,
): void {
  if (definition.min <= 0 || definition.max <= 0) {
    throw new ValidationError(
      `Log-scaled parameter "${name}" requires min and max to be greater than 0.`,
    );
  }
}

function buildLinearIntValues(definition: IntParamDefinition): number[] {
  const step = definition.step ?? 1;
  if (!Number.isInteger(step) || step <= 0) {
    throw new ValidationError(
      'Grid search requires int parameters to use a positive integer step.',
    );
  }

  const values: number[] = [];
  for (let value = definition.min; value <= definition.max; value += step) {
    values.push(value);
  }

  if (values.at(-1) !== definition.max) {
    values.push(definition.max);
  }

  return [...new Set(values)];
}

function buildLogIntValues(name: string, definition: IntParamDefinition): number[] {
  ensureLogBounds(name, definition);
  if (definition.step === undefined) {
    throw new ValidationError(
      'Grid search requires log-scaled int parameters to define a multiplicative step.',
    );
  }
  if (!Number.isFinite(definition.step) || definition.step <= 1) {
    throw new ValidationError(
      'Grid search requires log-scaled int parameters to use a multiplicative step greater than 1.',
    );
  }

  const values: number[] = [];
  let current = definition.min;
  while (current <= definition.max) {
    values.push(current);
    const next = Math.round(current * definition.step);
    if (next <= current) {
      throw new ValidationError(
        `Log-scaled int parameter "${name}" requires step to advance the range.`,
      );
    }
    current = next;
  }

  if (values.at(-1) !== definition.max) {
    values.push(definition.max);
  }

  return [...new Set(values)];
}

function buildIntValues(name: string, definition: IntParamDefinition): number[] {
  if (definition.scale === 'log') {
    return buildLogIntValues(name, definition);
  }
  return buildLinearIntValues(definition);
}

function buildLinearFloatValues(definition: FloatParamDefinition): number[] {
  if (definition.step === undefined) {
    throw new ValidationError(
      'Grid search requires float parameters to define step.',
    );
  }
  if (definition.step <= 0 || !Number.isFinite(definition.step)) {
    throw new ValidationError(
      'Grid search requires float parameters to use a positive finite step.',
    );
  }

  const values: number[] = [];
  const epsilon = definition.step / 1000;
  for (
    let value = definition.min;
    value <= definition.max + epsilon;
    value += definition.step
  ) {
    values.push(
      roundToPrecision(
        clamp(
          definition.min +
            Math.round((value - definition.min) / definition.step) *
              definition.step,
          definition.min,
          definition.max,
        ),
      ),
    );
  }

  if (values.at(-1) !== definition.max) {
    values.push(roundToPrecision(definition.max));
  }

  return [...new Set(values)];
}

function buildLogFloatValues(
  name: string,
  definition: FloatParamDefinition,
): number[] {
  ensureLogBounds(name, definition);
  if (definition.step === undefined) {
    throw new ValidationError(
      'Grid search requires log-scaled float parameters to define a multiplicative step.',
    );
  }
  if (!Number.isFinite(definition.step) || definition.step <= 1) {
    throw new ValidationError(
      'Grid search requires log-scaled float parameters to use a multiplicative step greater than 1.',
    );
  }

  const values: number[] = [];
  let current = definition.min;
  while (current <= definition.max) {
    values.push(roundToPrecision(current));
    const next = current * definition.step;
    if (next <= current) {
      throw new ValidationError(
        `Log-scaled float parameter "${name}" requires step to advance the range.`,
      );
    }
    current = next;
  }

  if (values.at(-1) !== definition.max) {
    values.push(roundToPrecision(definition.max));
  }

  return [...new Set(values)];
}

function buildFloatValues(
  name: string,
  definition: FloatParamDefinition,
): number[] {
  if (definition.scale === 'log') {
    return buildLogFloatValues(name, definition);
  }
  return buildLinearFloatValues(definition);
}

function buildDiscreteValues(
  name: string,
  definition: ParameterDefinition,
): unknown[] {
  switch (definition.type) {
    case 'enum':
      return [...definition.values];
    case 'int':
      return buildIntValues(name, definition);
    case 'float':
      return buildFloatValues(name, definition);
    default:
      throw new ValidationError(`Unsupported parameter type for "${name}".`);
  }
}

function isDiscreteDefinition(definition: ParameterDefinition): boolean {
  switch (definition.type) {
    case 'enum':
      return true;
    case 'int':
      return true;
    case 'float':
      return definition.step !== undefined;
    default:
      return false;
  }
}

function discreteCardinality(
  entries: [string, ParameterDefinition][],
): number | null {
  if (!entries.every(([, definition]) => isDiscreteDefinition(definition))) {
    return null;
  }

  return entries.reduce(
    (product, [name, definition]) => product * buildDiscreteValues(name, definition).length,
    1,
  );
}

function sampleLogValue(
  name: string,
  definition: FloatParamDefinition | IntParamDefinition,
  random: PythonRandom,
): number {
  ensureLogBounds(name, definition);
  const minLog = Math.log10(definition.min);
  const maxLog = Math.log10(definition.max);
  const exponent = random.uniform(minLog, maxLog);
  return 10 ** exponent;
}

function sampleParameter(
  name: string,
  definition: ParameterDefinition,
  random: PythonRandom,
): unknown {
  switch (definition.type) {
    case 'enum':
      return random.choice(definition.values);
    case 'int': {
      if (definition.scale === 'log') {
        if (definition.step !== undefined) {
          return random.choice(buildIntValues(name, definition));
        }
        return clamp(
          Math.round(sampleLogValue(name, definition, random)),
          definition.min,
          definition.max,
        );
      }

      if (definition.step !== undefined && definition.step !== 1) {
        return random.choice(buildIntValues(name, definition));
      }
      return random.randint(definition.min, definition.max);
    }
    case 'float': {
      if (definition.scale === 'log') {
        if (definition.step !== undefined) {
          return random.choice(buildFloatValues(name, definition));
        }
        return roundToPrecision(sampleLogValue(name, definition, random));
      }

      const sampled = random.uniform(definition.min, definition.max);
      if (definition.step === undefined) {
        return roundToPrecision(sampled);
      }

      const snapped =
        Math.round((sampled - definition.min) / definition.step) *
          definition.step +
        definition.min;
      return roundToPrecision(clamp(snapped, definition.min, definition.max));
    }
    default:
      return undefined;
  }
}

function cartesianProduct(
  entries: [string, unknown[]][],
): CandidateConfig[] {
  let product: CandidateConfig[] = [{}];

  for (const [name, values] of entries) {
    const next: CandidateConfig[] = [];
    for (const candidate of product) {
      for (const value of values) {
        next.push({ ...candidate, [name]: value });
      }
    }
    product = next;
  }

  return product;
}

function buildCandidatePlan(
  spec: NormalizedOptimizationSpec,
  options: NativeOptimizeOptions,
): { configs: CandidateConfig[]; exhaustive: boolean } {
  const entries = getOrderedParameterEntries(spec.configurationSpace);

  if (options.algorithm === 'grid') {
    const product = cartesianProduct(
      entries.map(
        ([name, definition]) =>
          [name, buildDiscreteValues(name, definition)] as [string, unknown[]],
      ),
    );

    return {
      configs: product.slice(0, options.maxTrials),
      exhaustive: product.length <= options.maxTrials,
    };
  }

  const random = new PythonRandom(options.randomSeed ?? 0);
  const configs: CandidateConfig[] = [];
  const seen = new Set<string>();
  const cardinality = discreteCardinality(entries);
  const uniqueOnly = cardinality !== null;

  while (configs.length < options.maxTrials) {
    const candidate = Object.fromEntries(
      entries.map(([name, definition]) => [
        name,
        sampleParameter(name, definition, random),
      ]),
    );

    if (!uniqueOnly) {
      configs.push(candidate);
      continue;
    }

    const key = configKey(candidate);
    if (seen.has(key)) {
      if (cardinality !== null && seen.size >= cardinality) {
        break;
      }
      continue;
    }

    seen.add(key);
    configs.push(candidate);

    if (cardinality !== null && seen.size >= cardinality) {
      break;
    }
  }

  return {
    configs,
    exhaustive: cardinality !== null && configs.length === cardinality,
  };
}

async function resolveEvaluationRows(
  spec: NormalizedOptimizationSpec,
): Promise<readonly unknown[]> {
  if (spec.evaluation?.data) return spec.evaluation.data;
  if (spec.evaluation?.loadData) return spec.evaluation.loadData();

  throw new ValidationError(
    'optimize() requires spec.evaluation.data or spec.evaluation.loadData.',
  );
}

function validateOptimizeOptions(
  options: NativeOptimizeOptions,
): NativeOptimizeOptions {
  if (!options || typeof options !== 'object') {
    throw new ValidationError('optimize() options are required.');
  }
  if (
    options.algorithm !== 'grid' &&
    options.algorithm !== 'random' &&
    options.algorithm !== 'bayesian'
  ) {
    throw new ValidationError(
      'optimize() only supports algorithm "grid", "random", or "bayesian".',
    );
  }
  if (!Number.isInteger(options.maxTrials) || options.maxTrials <= 0) {
    throw new ValidationError(
      'optimize() requires maxTrials to be a positive integer.',
    );
  }
  if (
    options.randomSeed !== undefined &&
    (!Number.isInteger(options.randomSeed) || options.randomSeed < 0)
  ) {
    throw new ValidationError(
      'optimize() randomSeed must be a non-negative integer when provided.',
    );
  }
  if (
    options.timeoutMs !== undefined &&
    (!Number.isInteger(options.timeoutMs) || options.timeoutMs <= 0)
  ) {
    throw new ValidationError(
      'optimize() timeoutMs must be a positive integer when provided.',
    );
  }

  return options;
}

function getObjectiveMetric(
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

function computeSearchScore(
  metrics: Metrics,
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

    const signedValue = objective.direction === 'minimize' ? -value : value;
    weightedTotal += signedValue * objective.weight;
    totalWeight += objective.weight;
  }

  return totalWeight === 0 ? 0 : weightedTotal / totalWeight;
}

function selectBestTrial(
  trials: OptimizationTrialRecord[],
  objectives: readonly NormalizedObjectiveDefinition[],
): OptimizationTrialRecord | null {
  if (trials.length === 0) return null;

  const ranges = objectives.map((objective) => {
    const values = trials.map((trial) => getObjectiveMetric(trial, objective));
    return {
      objective,
      min: Math.min(...values),
      max: Math.max(...values),
    };
  });

  let bestTrial: OptimizationTrialRecord | null = null;
  let bestScore = Number.NEGATIVE_INFINITY;

  for (const trial of trials) {
    let weightedScore = 0;
    let totalWeight = 0;

    for (const range of ranges) {
      const value = getObjectiveMetric(trial, range.objective);
      let normalized = 1;

      if (range.max !== range.min) {
        normalized = (value - range.min) / (range.max - range.min);
      }

      if (range.objective.direction === 'minimize') {
        normalized = 1 - normalized;
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

function createTrialConfig(
  config: CandidateConfig,
  trialNumber: number,
  totalRows: number,
  experimentRunId: string,
): TrialConfig {
  return {
    trial_id: `trial_${trialNumber}_${randomUUID()}`,
    trial_number: trialNumber,
    experiment_run_id: experimentRunId,
    config,
    dataset_subset: {
      indices: Array.from({ length: totalRows }, (_, index) => index),
      total: totalRows,
    },
  };
}

function normalizeDuration(
  result: NativeTrialFunctionResult,
  fallbackDuration: number,
): number {
  if (
    result.duration !== undefined &&
    Number.isFinite(result.duration) &&
    result.duration >= 0
  ) {
    return result.duration;
  }

  return fallbackDuration;
}

function toErrorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }
  return String(error);
}

async function executeTrial(
  trialFn: NativeTrialFunction,
  trialConfig: TrialConfig,
  timeoutMs: number | undefined,
): Promise<TrialOutcome> {
  const start = Date.now();
  let timeoutId: NodeJS.Timeout | undefined;

  try {
    const trialPromise = Promise.resolve(
      TrialContext.run(trialConfig, async () => trialFn(trialConfig)),
    );

    const rawResult = await Promise.race(
      [
        trialPromise,
        ...(timeoutMs === undefined
          ? []
          : [
              new Promise<never>((_, reject) => {
                timeoutId = setTimeout(() => {
                  reject(new TimeoutError('Trial timeout', timeoutMs));
                }, timeoutMs);
              }),
            ]),
      ],
    );

    if (!rawResult || typeof rawResult !== 'object') {
      throw new ValidationError(
        'optimize() trial function must resolve to an object containing metrics.',
      );
    }

    const metricsParse = MetricsSchema.safeParse(rawResult.metrics);
    if (!metricsParse.success) {
      throw new ValidationError(
        `optimize() trial metrics are invalid: ${metricsParse.error.message}`,
      );
    }

    return {
      status: 'completed',
      record: {
        trialId: trialConfig.trial_id,
        trialNumber: trialConfig.trial_number,
        config: { ...trialConfig.config },
        metrics: metricsParse.data,
        duration: normalizeDuration(rawResult, (Date.now() - start) / 1000),
        metadata: rawResult.metadata,
      },
    };
  } catch (error) {
    if (error instanceof ValidationError) {
      throw error;
    }
    if (error instanceof TimeoutError) {
      return {
        status: 'timeout',
        errorMessage: error.message,
      };
    }
    return {
      status: 'error',
      errorMessage: toErrorMessage(error),
    };
  } finally {
    if (timeoutId) {
      clearTimeout(timeoutId);
    }
  }
}

function finalizeResult(
  trials: OptimizationTrialRecord[],
  objectives: readonly NormalizedObjectiveDefinition[],
  stopReason: OptimizationResult['stopReason'],
  totalCostUsd: number,
  errorMessage?: string,
): OptimizationResult {
  const bestTrial = selectBestTrial(trials, objectives);

  return {
    bestConfig: bestTrial?.config ?? null,
    bestMetrics: bestTrial?.metrics ?? null,
    trials,
    stopReason,
    totalCostUsd,
    errorMessage,
  };
}

function vectorizeConfig(
  config: CandidateConfig,
  entries: [string, ParameterDefinition][],
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
        const value =
          typeof rawValue === 'number' ? rawValue : Number(rawValue ?? definition.min);
        if (definition.scale === 'log') {
          ensureLogBounds(name, definition);
          const minLog = Math.log10(definition.min);
          const maxLog = Math.log10(definition.max);
          const currentLog = Math.log10(clamp(value, definition.min, definition.max));
          vector.push(
            maxLog === minLog ? 0 : (currentLog - minLog) / (maxLog - minLog),
          );
        } else {
          vector.push(
            definition.max === definition.min
              ? 0
              : (value - definition.min) / (definition.max - definition.min),
          );
        }
        break;
      }
      default:
        break;
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
  observedScores: number[],
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

  const mean =
    weightSum === 0
      ? observedScores.reduce((sum, score) => sum + score, 0) /
        observedScores.length
      : weightedTotal / weightSum;

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

function buildBayesianLocalCandidate(
  baseline: CandidateConfig,
  entries: [string, ParameterDefinition][],
  random: PythonRandom,
): CandidateConfig {
  const candidate: CandidateConfig = {};

  for (const [name, definition] of entries) {
    const baselineValue = baseline[name];

    switch (definition.type) {
      case 'enum': {
        candidate[name] =
          random.random() < 0.75 && baselineValue !== undefined
            ? baselineValue
            : random.choice(definition.values);
        break;
      }
      case 'int': {
        const center =
          typeof baselineValue === 'number' ? baselineValue : definition.min;
        if (definition.scale === 'log') {
          const sampled = sampleLogValue(name, definition, random);
          candidate[name] = clamp(
            Math.round((center + sampled) / 2),
            definition.min,
            definition.max,
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
                }),
              )
            : random.randint(lower, upper);
        break;
      }
      case 'float': {
        const center =
          typeof baselineValue === 'number' ? baselineValue : definition.min;
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
                  }),
                )
              : roundToPrecision(clamp(mixed, definition.min, definition.max));
          break;
        }

        const span = Math.max(
          (definition.max - definition.min) / 6,
          definition.step ?? 0.01,
        );
        const lower = clamp(center - span, definition.min, definition.max);
        const upper = clamp(center + span, definition.min, definition.max);
        const sampled = random.uniform(lower, upper);
        if (definition.step === undefined) {
          candidate[name] = roundToPrecision(sampled);
        } else {
          const snapped =
            Math.round((sampled - definition.min) / definition.step) *
              definition.step +
            definition.min;
          candidate[name] = roundToPrecision(
            clamp(snapped, definition.min, definition.max),
          );
        }
        break;
      }
      default:
        break;
    }
  }

  return candidate;
}

function sampleCandidateConfig(
  entries: [string, ParameterDefinition][],
  random: PythonRandom,
): CandidateConfig {
  return Object.fromEntries(
    entries.map(([name, definition]) => [
      name,
      sampleParameter(name, definition, random),
    ]),
  );
}

function suggestBayesianConfig(
  spec: NormalizedOptimizationSpec,
  trials: OptimizationTrialRecord[],
  random: PythonRandom,
  maxTrials: number,
): { config: CandidateConfig | null; exhaustive: boolean } {
  const entries = getOrderedParameterEntries(spec.configurationSpace);
  const seen = new Set(trials.map((trial) => configKey(trial.config)));
  const cardinality = discreteCardinality(entries);

  if (cardinality !== null && seen.size >= cardinality) {
    return { config: null, exhaustive: true };
  }

  const initialRandomSamples = Math.min(maxTrials, Math.max(5, entries.length * 2));
  if (trials.length < initialRandomSamples) {
    for (let attempt = 0; attempt < 512; attempt += 1) {
      const candidate = sampleCandidateConfig(entries, random);
      if (!seen.has(configKey(candidate))) {
        return { config: candidate, exhaustive: false };
      }
    }
  }

  const observedVectors = trials.map((trial) => vectorizeConfig(trial.config, entries));
  const observedScores = trials.map((trial) =>
    computeSearchScore(trial.metrics, spec.objectives),
  );
  const sortedByScore = [...trials].sort(
    (left, right) =>
      computeSearchScore(right.metrics, spec.objectives) -
      computeSearchScore(left.metrics, spec.objectives),
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
        ? sampleCandidateConfig(entries, random)
        : buildBayesianLocalCandidate(baseline, entries, random);

    if (seen.has(configKey(candidate))) {
      continue;
    }

    const acquisition = estimateBayesianAcquisition(
      vectorizeConfig(candidate, entries),
      observedVectors,
      observedScores,
    );

    if (acquisition > bestAcquisition) {
      bestAcquisition = acquisition;
      bestCandidate = candidate;
    }
  }

  if (!bestCandidate) {
    for (let attempt = 0; attempt < 1024; attempt += 1) {
      const candidate = sampleCandidateConfig(entries, random);
      if (!seen.has(configKey(candidate))) {
        return { config: candidate, exhaustive: false };
      }
    }
    return { config: null, exhaustive: cardinality !== null };
  }

  return { config: bestCandidate, exhaustive: false };
}

function updateTotalCost(
  spec: NormalizedOptimizationSpec,
  trialRecord: OptimizationTrialRecord,
  totalCostUsd: number,
): number {
  if (spec.budget?.maxCostUsd !== undefined) {
    const costMetric = trialRecord.metrics['cost'];
    ensureFiniteNumber(
      costMetric,
      'budget.maxCostUsd requires every trial to return numeric metrics.cost.',
    );
    return totalCostUsd + costMetric;
  }

  if (
    typeof trialRecord.metrics['cost'] === 'number' &&
    Number.isFinite(trialRecord.metrics['cost'])
  ) {
    return totalCostUsd + trialRecord.metrics['cost'];
  }

  return totalCostUsd;
}

async function runSequentialPlan(
  trialFn: NativeTrialFunction,
  spec: NormalizedOptimizationSpec,
  options: NativeOptimizeOptions,
  configs: CandidateConfig[],
  exhaustive: boolean,
  evaluationRows: readonly unknown[],
): Promise<OptimizationResult> {
  const experimentRunId = `native_${randomUUID()}`;
  const trials: OptimizationTrialRecord[] = [];
  let totalCostUsd = 0;
  let stopReason: OptimizationResult['stopReason'] = exhaustive
    ? 'completed'
    : 'maxTrials';

  for (const [index, candidateConfig] of configs.entries()) {
    const trialConfig = createTrialConfig(
      candidateConfig,
      index + 1,
      evaluationRows.length,
      experimentRunId,
    );

    const outcome = await executeTrial(trialFn, trialConfig, options.timeoutMs);
    if (outcome.status !== 'completed') {
      return finalizeResult(
        trials,
        spec.objectives,
        outcome.status,
        totalCostUsd,
        outcome.errorMessage,
      );
    }

    for (const objective of spec.objectives) {
      getObjectiveMetric(outcome.record, objective);
    }

    totalCostUsd = updateTotalCost(spec, outcome.record, totalCostUsd);
    trials.push(outcome.record);

    if (
      spec.budget?.maxCostUsd !== undefined &&
      totalCostUsd >= spec.budget.maxCostUsd
    ) {
      stopReason = 'budget';
      break;
    }
  }

  return finalizeResult(trials, spec.objectives, stopReason, totalCostUsd);
}

async function runBayesianPlan(
  trialFn: NativeTrialFunction,
  spec: NormalizedOptimizationSpec,
  options: NativeOptimizeOptions,
  evaluationRows: readonly unknown[],
): Promise<OptimizationResult> {
  const experimentRunId = `native_${randomUUID()}`;
  const trials: OptimizationTrialRecord[] = [];
  const random = new PythonRandom(options.randomSeed ?? 0);
  let totalCostUsd = 0;
  let stopReason: OptimizationResult['stopReason'] = 'maxTrials';

  for (let index = 0; index < options.maxTrials; index += 1) {
    const suggestion = suggestBayesianConfig(spec, trials, random, options.maxTrials);
    if (!suggestion.config) {
      stopReason = suggestion.exhaustive ? 'completed' : 'maxTrials';
      break;
    }

    const trialConfig = createTrialConfig(
      suggestion.config,
      index + 1,
      evaluationRows.length,
      experimentRunId,
    );

    const outcome = await executeTrial(trialFn, trialConfig, options.timeoutMs);
    if (outcome.status !== 'completed') {
      return finalizeResult(
        trials,
        spec.objectives,
        outcome.status,
        totalCostUsd,
        outcome.errorMessage,
      );
    }

    for (const objective of spec.objectives) {
      getObjectiveMetric(outcome.record, objective);
    }

    totalCostUsd = updateTotalCost(spec, outcome.record, totalCostUsd);
    trials.push(outcome.record);

    if (
      spec.budget?.maxCostUsd !== undefined &&
      totalCostUsd >= spec.budget.maxCostUsd
    ) {
      stopReason = 'budget';
      break;
    }
  }

  return finalizeResult(trials, spec.objectives, stopReason, totalCostUsd);
}

export async function runNativeOptimization(
  trialFn: NativeTrialFunction,
  spec: NormalizedOptimizationSpec,
  rawOptions: NativeOptimizeOptions,
): Promise<OptimizationResult> {
  const options = validateOptimizeOptions(rawOptions);
  const evaluationRows = await resolveEvaluationRows(spec);

  if (!Array.isArray(evaluationRows) || evaluationRows.length === 0) {
    throw new ValidationError(
      'optimize() requires evaluation data to be a non-empty array.',
    );
  }

  if (options.algorithm === 'bayesian') {
    return runBayesianPlan(trialFn, spec, options, evaluationRows);
  }

  const { configs, exhaustive } = buildCandidatePlan(spec, options);
  return runSequentialPlan(
    trialFn,
    spec,
    options,
    configs,
    exhaustive,
    evaluationRows,
  );
}
