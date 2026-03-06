import { createHash, randomUUID } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';

import { TrialCancelledError, TrialContext } from '../core/context.js';
import {
  CancelledError,
  TimeoutError,
  ValidationError,
} from '../core/errors.js';
import { MetricsSchema, type Metrics, type TrialConfig } from '../dtos/trial.js';
import { PythonRandom, type SerializedPythonRandomState } from './python-random.js';
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
      status: 'timeout' | 'error' | 'cancelled';
      trialNumber: number;
      errorMessage: string;
    };

interface ValidatedOptimizeOptions extends NativeOptimizeOptions {
  trialConcurrency: number;
}

interface CandidatePlan {
  configs: CandidateConfig[];
  exhaustive: boolean;
}

interface CheckpointState {
  version: 1;
  algorithm: ValidatedOptimizeOptions['algorithm'];
  specHash: string;
  optionsHash: string;
  experimentRunId: string;
  exhaustive?: boolean;
  candidateConfigs?: CandidateConfig[];
  samplerState?: SerializedPythonRandomState;
  completedTrials: OptimizationTrialRecord[];
  totalCostUsd: number;
}

interface ResumeState {
  experimentRunId: string;
  exhaustive?: boolean;
  candidateConfigs?: CandidateConfig[];
  samplerState?: SerializedPythonRandomState;
  completedTrials: OptimizationTrialRecord[];
  totalCostUsd: number;
}

interface CheckpointContext {
  specHash: string;
  optionsHash: string;
  path: string;
}

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

function hashJson(value: unknown): string {
  return createHash('sha256').update(stableJson(value)).digest('hex');
}

function configKey(config: CandidateConfig): string {
  return stableJson(config);
}

function compareParameterNames(left: string, right: string): number {
  if (left === 'model' && right !== 'model') return 1;
  if (right === 'model' && left !== 'model') return -1;
  return left.localeCompare(right);
}

function getOrderedParameterEntries(
  configurationSpace: NormalizedOptimizationSpec['configurationSpace'],
): [string, ParameterDefinition][] {
  const names = Object.keys(configurationSpace);
  const inDegree = new Map(names.map((name) => [name, 0]));
  const dependents = new Map<string, string[]>();

  for (const name of names) {
    for (const dependency of Object.keys(configurationSpace[name]!.conditions ?? {})) {
      inDegree.set(name, (inDegree.get(name) ?? 0) + 1);
      const siblings = dependents.get(dependency) ?? [];
      siblings.push(name);
      dependents.set(dependency, siblings);
    }
  }

  const queue = names
    .filter((name) => (inDegree.get(name) ?? 0) === 0)
    .sort(compareParameterNames);
  const ordered: string[] = [];

  while (queue.length > 0) {
    const current = queue.shift()!;
    ordered.push(current);

    for (const dependent of dependents.get(current) ?? []) {
      const nextDegree = (inDegree.get(dependent) ?? 0) - 1;
      inDegree.set(dependent, nextDegree);
      if (nextDegree === 0) {
        queue.push(dependent);
        queue.sort(compareParameterNames);
      }
    }
  }

  if (ordered.length !== names.length) {
    throw new ValidationError(
      'Conditional parameters cannot form dependency cycles.',
    );
  }

  return ordered.map((name) => [name, configurationSpace[name]!]);
}

function isParameterActive(
  definition: ParameterDefinition,
  config: CandidateConfig,
): boolean {
  return Object.entries(definition.conditions ?? {}).every(
    ([name, expected]) => config[name] === expected,
  );
}

function getInactiveParameterValue(
  name: string,
  definition: ParameterDefinition,
): unknown {
  if (definition.default === undefined) {
    throw new ValidationError(
      `Conditional parameter "${name}" requires a default fallback value.`,
    );
  }
  return definition.default;
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
  const limit = definition.max;
  while (current <= limit) {
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

function isDiscreteSpace(entries: [string, ParameterDefinition][]): boolean {
  return entries.every(([, definition]) => isDiscreteDefinition(definition));
}

function discreteCardinality(
  entries: [string, ParameterDefinition][],
): number | null {
  const configs = buildDiscreteCandidateConfigs(entries);
  return configs?.length ?? null;
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

function buildDiscreteCandidateConfigs(
  entries: [string, ParameterDefinition][],
): CandidateConfig[] | null {
  if (!isDiscreteSpace(entries)) {
    return null;
  }

  let product: CandidateConfig[] = [{}];

  for (const [name, definition] of entries) {
    const next: CandidateConfig[] = [];
    for (const candidate of product) {
      if (!isParameterActive(definition, candidate)) {
        next.push({
          ...candidate,
          [name]: getInactiveParameterValue(name, definition),
        });
        continue;
      }

      for (const value of buildDiscreteValues(name, definition)) {
        next.push({ ...candidate, [name]: value });
      }
    }
    product = next;
  }

  return product;
}

function sampleConditionalCandidateConfig(
  entries: [string, ParameterDefinition][],
  random: PythonRandom,
): CandidateConfig {
  const candidate: CandidateConfig = {};

  for (const [name, definition] of entries) {
    candidate[name] = isParameterActive(definition, candidate)
      ? sampleParameter(name, definition, random)
      : getInactiveParameterValue(name, definition);
  }

  return candidate;
}

function buildGridCandidatePlan(
  entries: [string, ParameterDefinition][],
  options: ValidatedOptimizeOptions,
): CandidatePlan {
  const product = buildDiscreteCandidateConfigs(entries);
  if (!product) {
    for (const [name, definition] of entries) {
      buildDiscreteValues(name, definition);
    }
    throw new ValidationError(
      'Grid search requires every parameter to be discrete or stepped.',
    );
  }

  return {
    configs: product.slice(0, options.maxTrials),
    exhaustive: product.length <= options.maxTrials,
  };
}

function buildRandomCandidatePlan(
  entries: [string, ParameterDefinition][],
  options: ValidatedOptimizeOptions,
): CandidatePlan {
  const random = new PythonRandom(options.randomSeed ?? 0);
  const configs: CandidateConfig[] = [];
  const seen = new Set<string>();
  const cardinality = discreteCardinality(entries);
  const uniqueOnly = cardinality !== null;

  while (configs.length < options.maxTrials) {
    const candidate = sampleConditionalCandidateConfig(entries, random);

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

function buildCandidatePlan(
  spec: NormalizedOptimizationSpec,
  options: ValidatedOptimizeOptions,
): CandidatePlan {
  const entries = getOrderedParameterEntries(spec.configurationSpace);

  if (options.algorithm === 'grid') {
    return buildGridCandidatePlan(entries, options);
  }

  if (options.algorithm === 'random') {
    return buildRandomCandidatePlan(entries, options);
  }

  return { configs: [], exhaustive: false };
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
): ValidatedOptimizeOptions {
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

  const trialConcurrency = options.trialConcurrency ?? 1;
  if (!Number.isInteger(trialConcurrency) || trialConcurrency <= 0) {
    throw new ValidationError(
      'optimize() trialConcurrency must be a positive integer when provided.',
    );
  }
  if (options.algorithm === 'bayesian' && trialConcurrency > 1) {
    throw new ValidationError(
      'optimize() bayesian does not support trialConcurrency > 1 yet.',
    );
  }

  if (options.plateau !== undefined) {
    if (
      !options.plateau ||
      !Number.isInteger(options.plateau.window) ||
      options.plateau.window <= 0
    ) {
      throw new ValidationError(
        'optimize() plateau.window must be a positive integer.',
      );
    }
    if (
      typeof options.plateau.minImprovement !== 'number' ||
      !Number.isFinite(options.plateau.minImprovement) ||
      options.plateau.minImprovement < 0
    ) {
      throw new ValidationError(
        'optimize() plateau.minImprovement must be a finite number >= 0.',
      );
    }
  }

  if (options.checkpoint !== undefined) {
    if (
      typeof options.checkpoint.key !== 'string' ||
      options.checkpoint.key.trim().length === 0
    ) {
      throw new ValidationError('optimize() checkpoint.key must be non-empty.');
    }
    if (
      options.checkpoint.dir !== undefined &&
      options.checkpoint.dir.trim().length === 0
    ) {
      throw new ValidationError(
        'optimize() checkpoint.dir must be non-empty when provided.',
      );
    }
  }

  return {
    ...options,
    trialConcurrency,
  };
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

function hasPlateau(
  trials: OptimizationTrialRecord[],
  objectives: readonly NormalizedObjectiveDefinition[],
  plateau: NonNullable<ValidatedOptimizeOptions['plateau']> | undefined,
): boolean {
  if (!plateau || trials.length <= plateau.window) {
    return false;
  }

  const bestHistory: number[] = [];
  let best = Number.NEGATIVE_INFINITY;

  for (const trial of trials) {
    best = Math.max(best, computeSearchScore(trial.metrics, objectives));
    bestHistory.push(best);
  }

  const current = bestHistory.at(-1)!;
  const previous = bestHistory.at(-(plateau.window + 1))!;
  return current - previous < plateau.minImprovement;
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
  signal: AbortSignal | undefined,
): Promise<TrialOutcome> {
  const controller = new AbortController();
  const listeners: Array<() => void> = [];

  let timeoutId: NodeJS.Timeout | undefined;
  const start = Date.now();

  try {
    const trialPromise = Promise.resolve(
      TrialContext.run(
        trialConfig,
        async () => trialFn(trialConfig),
        controller.signal,
      ),
    );

    const timeoutPromise =
      timeoutMs === undefined
        ? undefined
        : new Promise<never>((_, reject) => {
            timeoutId = setTimeout(() => {
              controller.abort();
              reject(new TimeoutError('Trial timeout', timeoutMs));
            }, timeoutMs);
          });

    const cancelPromise =
      signal === undefined
        ? undefined
        : new Promise<never>((_, reject) => {
            const onAbort = () => {
              controller.abort();
              reject(new CancelledError('Optimization cancelled'));
            };

            if (signal.aborted) {
              onAbort();
              return;
            }

            signal.addEventListener('abort', onAbort, { once: true });
            listeners.push(() => signal.removeEventListener('abort', onAbort));
          });

    const rawResult = await Promise.race(
      [trialPromise, timeoutPromise, cancelPromise].filter(
        (value): value is Promise<NativeTrialFunctionResult> =>
          value !== undefined,
      ),
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

    const duration = normalizeDuration(rawResult, (Date.now() - start) / 1000);
    return {
      status: 'completed',
      record: {
        trialId: trialConfig.trial_id,
        trialNumber: trialConfig.trial_number,
        config: { ...trialConfig.config },
        metrics: metricsParse.data,
        duration,
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
        trialNumber: trialConfig.trial_number,
        errorMessage: error.message,
      };
    }
    if (
      error instanceof CancelledError ||
      error instanceof TrialCancelledError ||
      controller.signal.aborted
    ) {
      return {
        status: 'cancelled',
        trialNumber: trialConfig.trial_number,
        errorMessage: toErrorMessage(error),
      };
    }
    return {
      status: 'error',
      trialNumber: trialConfig.trial_number,
      errorMessage: toErrorMessage(error),
    };
  } finally {
    if (timeoutId) {
      clearTimeout(timeoutId);
    }
    for (const removeListener of listeners) {
      removeListener();
    }
  }
}

async function ensureCheckpointDirectory(path: string): Promise<void> {
  await mkdir(path, { recursive: true });
}

async function resolveCheckpointContext(
  spec: NormalizedOptimizationSpec,
  options: ValidatedOptimizeOptions,
): Promise<CheckpointContext | undefined> {
  if (!options.checkpoint) {
    return undefined;
  }

  const dir = options.checkpoint.dir ?? join(process.cwd(), '.traigent-checkpoints');
  await ensureCheckpointDirectory(dir);

  const specHash = hashJson(spec);
  const optionsHash = hashJson({
    algorithm: options.algorithm,
    maxTrials: options.maxTrials,
    randomSeed: options.randomSeed,
    timeoutMs: options.timeoutMs,
    trialConcurrency: options.trialConcurrency,
    plateau: options.plateau,
  });
  const filename = `${createHash('sha256')
    .update(options.checkpoint.key)
    .digest('hex')}.json`;

  return {
    specHash,
    optionsHash,
    path: join(dir, filename),
  };
}

async function loadCheckpointState(
  checkpoint: CheckpointContext | undefined,
  options: ValidatedOptimizeOptions,
): Promise<ResumeState | undefined> {
  if (!checkpoint || !options.checkpoint?.resume) {
    return undefined;
  }

  try {
    const raw = await readFile(checkpoint.path, 'utf8');
    const parsed = JSON.parse(raw) as CheckpointState;
    if (
      parsed.version !== 1 ||
      parsed.specHash !== checkpoint.specHash ||
      parsed.optionsHash !== checkpoint.optionsHash
    ) {
      throw new ValidationError(
        'optimize() checkpoint does not match the current spec/options.',
      );
    }

    return {
      experimentRunId: parsed.experimentRunId,
      exhaustive: parsed.exhaustive,
      candidateConfigs: parsed.candidateConfigs,
      samplerState: parsed.samplerState,
      completedTrials: parsed.completedTrials ?? [],
      totalCostUsd: parsed.totalCostUsd ?? 0,
    };
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') {
      return undefined;
    }
    throw error;
  }
}

async function saveCheckpointState(
  checkpoint: CheckpointContext | undefined,
  options: ValidatedOptimizeOptions,
  state: ResumeState,
): Promise<void> {
  if (!checkpoint || !options.checkpoint) {
    return;
  }

  const payload: CheckpointState = {
    version: 1,
    algorithm: options.algorithm,
    specHash: checkpoint.specHash,
    optionsHash: checkpoint.optionsHash,
    experimentRunId: state.experimentRunId,
    exhaustive: state.exhaustive,
    candidateConfigs: state.candidateConfigs,
    samplerState: state.samplerState,
    completedTrials: state.completedTrials,
    totalCostUsd: state.totalCostUsd,
  };

  await writeFile(checkpoint.path, `${JSON.stringify(payload, null, 2)}\n`, 'utf8');
}

function finalizeResult(
  trials: OptimizationTrialRecord[],
  objectives: readonly NormalizedObjectiveDefinition[],
  stopReason: OptimizationResult['stopReason'],
  totalCostUsd: number,
  errorMessage?: string,
): OptimizationResult {
  const orderedTrials = [...trials].sort(
    (left, right) => left.trialNumber - right.trialNumber,
  );
  const bestTrial = selectBestTrial(orderedTrials, objectives);

  return {
    bestConfig: bestTrial?.config ?? null,
    bestMetrics: bestTrial?.metrics ?? null,
    trials: orderedTrials,
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
    if (!isParameterActive(definition, candidate)) {
      candidate[name] = getInactiveParameterValue(name, definition);
      continue;
    }

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
  return sampleConditionalCandidateConfig(entries, random);
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

  const trialCost = trialRecord.metrics['cost'];
  if (typeof trialCost === 'number' && Number.isFinite(trialCost)) {
    return totalCostUsd + trialCost;
  }

  return totalCostUsd;
}

async function runCandidatePlan(
  trialFn: NativeTrialFunction,
  spec: NormalizedOptimizationSpec,
  options: ValidatedOptimizeOptions,
  plan: CandidatePlan,
  resume: ResumeState,
  evaluationRows: readonly unknown[],
  checkpoint: CheckpointContext | undefined,
): Promise<OptimizationResult> {
  const trials = [...resume.completedTrials];
  let totalCostUsd = resume.totalCostUsd;
  let stopReason: OptimizationResult['stopReason'] =
    plan.exhaustive ? 'completed' : 'maxTrials';
  let errorMessage: string | undefined;
  let committedTrialNumber = trials.length;

  const abortController = new AbortController();
  if (options.signal) {
    if (options.signal.aborted) {
      abortController.abort();
      return finalizeResult(
        trials,
        spec.objectives,
        'cancelled',
        totalCostUsd,
        'Optimization cancelled',
      );
    }
    options.signal.addEventListener('abort', () => abortController.abort(), {
      once: true,
    });
  }

  type ActiveTask = { trialNumber: number; promise: Promise<TrialOutcome> };
  const active = new Map<number, ActiveTask>();
  const buffered = new Map<number, TrialOutcome>();

  let nextIndex = committedTrialNumber;
  const maybeDispatch = (): void => {
    while (
      !abortController.signal.aborted &&
      active.size < options.trialConcurrency &&
      nextIndex < plan.configs.length
    ) {
      const trialNumber = nextIndex + 1;
      const trialConfig = createTrialConfig(
        plan.configs[nextIndex]!,
        trialNumber,
        evaluationRows.length,
        resume.experimentRunId,
      );
      const promise = executeTrial(
        trialFn,
        trialConfig,
        options.timeoutMs,
        abortController.signal,
      );
      active.set(trialNumber, { trialNumber, promise });
      nextIndex += 1;
    }
  };

  maybeDispatch();

  while (active.size > 0) {
    const { trialNumber, outcome } = await Promise.race(
      [...active.values()].map(async (task) => ({
        trialNumber: task.trialNumber,
        outcome: await task.promise,
      })),
    );
    active.delete(trialNumber);
    buffered.set(trialNumber, outcome);

    while (buffered.has(committedTrialNumber + 1)) {
      const current = buffered.get(committedTrialNumber + 1)!;
      buffered.delete(committedTrialNumber + 1);
      committedTrialNumber += 1;

      if (current.status !== 'completed') {
        stopReason = current.status;
        errorMessage = current.errorMessage;
        abortController.abort();
        break;
      }

      for (const objective of spec.objectives) {
        getObjectiveMetric(current.record, objective);
      }

      totalCostUsd = updateTotalCost(spec, current.record, totalCostUsd);
      trials.push(current.record);

      await saveCheckpointState(checkpoint, options, {
        ...resume,
        exhaustive: plan.exhaustive,
        candidateConfigs: plan.configs,
        completedTrials: [...trials].sort(
          (left, right) => left.trialNumber - right.trialNumber,
        ),
        totalCostUsd,
      });

      if (
        spec.budget?.maxCostUsd !== undefined &&
        totalCostUsd >= spec.budget.maxCostUsd
      ) {
        stopReason = 'budget';
        abortController.abort();
        break;
      }

      if (hasPlateau(trials, spec.objectives, options.plateau)) {
        stopReason = 'plateau';
        abortController.abort();
        break;
      }
    }

    if (abortController.signal.aborted) {
      if (
        options.signal?.aborted &&
        (stopReason === 'completed' || stopReason === 'maxTrials')
      ) {
        stopReason = 'cancelled';
        errorMessage ??= 'Optimization cancelled';
      }
      break;
    }

    maybeDispatch();
  }

  await Promise.allSettled([...active.values()].map((task) => task.promise));
  return finalizeResult(trials, spec.objectives, stopReason, totalCostUsd, errorMessage);
}

async function runBayesianPlan(
  trialFn: NativeTrialFunction,
  spec: NormalizedOptimizationSpec,
  options: ValidatedOptimizeOptions,
  resume: ResumeState,
  evaluationRows: readonly unknown[],
  checkpoint: CheckpointContext | undefined,
): Promise<OptimizationResult> {
  const trials = [...resume.completedTrials].sort(
    (left, right) => left.trialNumber - right.trialNumber,
  );
  let totalCostUsd = resume.totalCostUsd;
  let stopReason: OptimizationResult['stopReason'] = 'maxTrials';
  let errorMessage: string | undefined;
  const random = resume.samplerState
    ? new PythonRandom(resume.samplerState)
    : new PythonRandom(options.randomSeed ?? 0);

  for (let index = trials.length; index < options.maxTrials; index += 1) {
    if (options.signal?.aborted) {
      stopReason = 'cancelled';
      errorMessage = 'Optimization cancelled';
      break;
    }

    const suggestion = suggestBayesianConfig(spec, trials, random, options.maxTrials);
    if (!suggestion.config) {
      stopReason = 'completed';
      break;
    }

    const trialNumber = index + 1;
    const trialConfig = createTrialConfig(
      suggestion.config,
      trialNumber,
      evaluationRows.length,
      resume.experimentRunId,
    );

    const outcome = await executeTrial(
      trialFn,
      trialConfig,
      options.timeoutMs,
      options.signal,
    );

    if (outcome.status !== 'completed') {
      stopReason = outcome.status;
      errorMessage = outcome.errorMessage;
      break;
    }

    for (const objective of spec.objectives) {
      getObjectiveMetric(outcome.record, objective);
    }

    totalCostUsd = updateTotalCost(spec, outcome.record, totalCostUsd);
    trials.push(outcome.record);

    await saveCheckpointState(checkpoint, options, {
      ...resume,
      exhaustive: suggestion.exhaustive,
      samplerState: random.serialize(),
      completedTrials: [...trials],
      totalCostUsd,
    });

    if (
      spec.budget?.maxCostUsd !== undefined &&
      totalCostUsd >= spec.budget.maxCostUsd
    ) {
      stopReason = 'budget';
      break;
    }

    if (hasPlateau(trials, spec.objectives, options.plateau)) {
      stopReason = 'plateau';
      break;
    }
  }

  return finalizeResult(trials, spec.objectives, stopReason, totalCostUsd, errorMessage);
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

  const checkpoint = await resolveCheckpointContext(spec, options);
  const resume =
    (await loadCheckpointState(checkpoint, options)) ?? {
      experimentRunId: `native_${randomUUID()}`,
      completedTrials: [],
      totalCostUsd: 0,
    };

  if (options.algorithm === 'bayesian') {
    return runBayesianPlan(
      trialFn,
      spec,
      options,
      resume,
      evaluationRows,
      checkpoint,
    );
  }

  const plan = {
    ...buildCandidatePlan(spec, options),
    configs: resume.candidateConfigs ?? buildCandidatePlan(spec, options).configs,
    exhaustive:
      resume.exhaustive ?? buildCandidatePlan(spec, options).exhaustive,
  };

  return runCandidatePlan(
    trialFn,
    spec,
    options,
    plan,
    {
      ...resume,
      candidateConfigs: plan.configs,
      exhaustive: plan.exhaustive,
    },
    evaluationRows,
    checkpoint,
  );
}
