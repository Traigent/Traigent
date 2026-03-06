import { randomUUID } from 'node:crypto';

import { TrialContext } from '../core/context.js';
import { ValidationError } from '../core/errors.js';
import { MetricsSchema, type TrialConfig } from '../dtos/trial.js';
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

function createSeededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function ensureFiniteNumber(
  value: unknown,
  message: string,
): asserts value is number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new ValidationError(message);
  }
}

function roundFloat(value: number, step?: number): number {
  if (!step) return Number(value.toFixed(6));
  const scaled = Math.round(value / step) * step;
  return Number(scaled.toFixed(6));
}

function buildIntValues(definition: IntParamDefinition): number[] {
  if (definition.scale === 'log') {
    throw new ValidationError(
      'Native optimize() does not support log-scaled int parameters yet.',
    );
  }

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

function buildFloatValues(definition: FloatParamDefinition): number[] {
  if (definition.scale === 'log') {
    throw new ValidationError(
      'Native optimize() does not support log-scaled float parameters yet.',
    );
  }
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
    values.push(roundFloat(Math.min(value, definition.max), definition.step));
  }

  if (values.at(-1) !== definition.max) {
    values.push(roundFloat(definition.max, definition.step));
  }

  return [...new Set(values)];
}

function buildGridValues(
  name: string,
  definition: ParameterDefinition,
): unknown[] {
  switch (definition.type) {
    case 'enum':
      return [...definition.values];
    case 'int':
      return buildIntValues(definition);
    case 'float':
      return buildFloatValues(definition);
    default:
      throw new ValidationError(`Unsupported parameter type for ${name}.`);
  }
}

function sampleParameter(
  definition: ParameterDefinition,
  random: () => number,
): unknown {
  switch (definition.type) {
    case 'enum': {
      const index = Math.floor(random() * definition.values.length);
      return definition.values[index];
    }
    case 'int': {
      if (definition.scale === 'log') {
        throw new ValidationError(
          'Native optimize() does not support log-scaled int parameters yet.',
        );
      }
      const step = definition.step ?? 1;
      const values = buildIntValues({ ...definition, step });
      const index = Math.floor(random() * values.length);
      return values[index];
    }
    case 'float': {
      if (definition.scale === 'log') {
        throw new ValidationError(
          'Native optimize() does not support log-scaled float parameters yet.',
        );
      }
      const raw =
        definition.min + random() * (definition.max - definition.min);
      return roundFloat(raw, definition.step);
    }
    default:
      return undefined;
  }
}

function cartesianProduct(
  entries: [string, unknown[]][],
): Record<string, unknown>[] {
  let product: Record<string, unknown>[] = [{}];

  for (const [name, values] of entries) {
    const next: Record<string, unknown>[] = [];
    for (const candidate of product) {
      for (const value of values) {
        next.push({ ...candidate, [name]: value });
      }
    }
    product = next;
  }

  return product;
}

function buildCandidateConfigs(
  spec: NormalizedOptimizationSpec,
  options: NativeOptimizeOptions,
): { configs: Record<string, unknown>[]; exhaustive: boolean } {
  const random = createSeededRandom(options.randomSeed ?? 0xdecafbad);
  const entries = Object.entries(spec.configurationSpace);

  if (options.algorithm === 'random') {
    const configs = Array.from({ length: options.maxTrials }, () =>
      Object.fromEntries(
        entries.map(([name, definition]) => [
          name,
          sampleParameter(definition, random),
        ]),
      ),
    );

    return { configs, exhaustive: false };
  }

  const values = entries.map(([name, definition]) => [
    name,
    buildGridValues(name, definition),
  ] as [string, unknown[]]);
  const product = cartesianProduct(values);

  return {
    configs: product.slice(0, options.maxTrials),
    exhaustive: product.length <= options.maxTrials,
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
  if (options.algorithm !== 'grid' && options.algorithm !== 'random') {
    throw new ValidationError(
      'optimize() only supports algorithm "grid" or "random".',
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
  config: Record<string, unknown>,
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

  const { configs, exhaustive } = buildCandidateConfigs(spec, options);
  const experimentRunId = `native_${randomUUID()}`;
  const trials: OptimizationTrialRecord[] = [];
  let totalCostUsd = 0;
  let stopReason: OptimizationResult['stopReason'] =
    options.algorithm === 'grid' && exhaustive ? 'completed' : 'maxTrials';

  for (const [index, candidateConfig] of configs.entries()) {
    const trialNumber = index + 1;
    const trialConfig = createTrialConfig(
      candidateConfig,
      trialNumber,
      evaluationRows.length,
      experimentRunId,
    );
    const start = Date.now();

    const rawResult = await TrialContext.run(
      trialConfig,
      async () => trialFn(trialConfig),
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

    const duration = normalizeDuration(
      rawResult,
      (Date.now() - start) / 1000,
    );

    const trialRecord: OptimizationTrialRecord = {
      trialId: trialConfig.trial_id,
      trialNumber,
      config: { ...trialConfig.config },
      metrics: metricsParse.data,
      duration,
      metadata: rawResult.metadata,
    };

    for (const objective of spec.objectives) {
      getObjectiveMetric(trialRecord, objective);
    }

    if (spec.budget?.maxCostUsd !== undefined) {
      const costMetric = trialRecord.metrics.cost;
      ensureFiniteNumber(
        costMetric,
        'budget.maxCostUsd requires every trial to return numeric metrics.cost.',
      );
      totalCostUsd += costMetric;
    } else if (
      typeof trialRecord.metrics.cost === 'number' &&
      Number.isFinite(trialRecord.metrics.cost)
    ) {
      totalCostUsd += trialRecord.metrics.cost;
    }

    trials.push(trialRecord);

    if (
      spec.budget?.maxCostUsd !== undefined &&
      totalCostUsd >= spec.budget.maxCostUsd
    ) {
      stopReason = 'budget';
      break;
    }
  }

  const bestTrial = selectBestTrial(trials, spec.objectives);

  return {
    bestConfig: bestTrial?.config ?? null,
    bestMetrics: bestTrial?.metrics ?? null,
    trials,
    stopReason,
    totalCostUsd,
  };
}
