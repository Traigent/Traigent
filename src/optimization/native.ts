import { createHash, randomUUID } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';

import { TrialCancelledError, TrialContext } from '../core/context.js';
import { CancelledError, TimeoutError, ValidationError } from '../core/errors.js';
import { MetricsSchema, type TrialConfig } from '../dtos/trial.js';
import { PythonRandom, type SerializedPythonRandomState } from './python-random.js';
import { applyPostTrialGuards, evaluatePreTrialConstraints } from './native-constraints.js';
import {
  applyDefaultConfig,
  buildDiscreteValues,
  cartesianProduct,
  clamp,
  configKey,
  discreteCardinality,
  getOrderedParameterEntries,
  hashJson,
  sampleParameter,
} from './native-space.js';
import {
  assertTrialCostMetricAvailable,
  extractTrialCost,
  normalizeCostMetrics,
} from './native-cost.js';
import {
  aggregateRepetitionMetrics,
  collectMetricSamples,
  mergeMetricSamples,
} from './native-reps.js';
import {
  getObjectiveMetric,
  hasPlateau,
  selectBestTrial,
  selectBestTrialWithPromotionDecision,
} from './native-scoring.js';
import { suggestBayesianConfig } from './native-bayesian.js';
import type {
  NativeOptimizeOptions,
  NativeOptimizationReportingSummary,
  NativeTrialFunctionResult,
  NormalizedObjectiveDefinition,
  NormalizedOptimizationSpec,
  OptimizationResult,
  OptimizationTrialRecord,
  ParameterDefinition,
} from './types.js';
import type { CandidateConfig } from './native-space.js';

type NativeTrialFunction = (trialConfig: TrialConfig) => Promise<NativeTrialFunctionResult>;

type TrialOutcome =
  | {
      status: 'completed';
      record: OptimizationTrialRecord;
      actualCostUsd: number;
      evaluatedExamples: number;
    }
  | {
      status: 'rejected';
      record: OptimizationTrialRecord;
      actualCostUsd: number;
      evaluatedExamples: number;
    }
  | {
      status: 'timeout' | 'error' | 'cancelled';
      trialNumber: number;
      errorMessage: string;
      actualCostUsd: number;
      evaluatedExamples: number;
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
  trials?: OptimizationTrialRecord[];
  completedTrials?: OptimizationTrialRecord[];
  totalCostUsd: number;
  totalExamplesEvaluated?: number;
}

interface ResumeState {
  experimentRunId: string;
  exhaustive?: boolean;
  candidateConfigs?: CandidateConfig[];
  samplerState?: SerializedPythonRandomState;
  trials: OptimizationTrialRecord[];
  totalCostUsd: number;
  totalExamplesEvaluated: number;
}

interface CheckpointContext {
  specHash: string;
  optionsHash: string;
  path: string;
}

function asMetricSampleMap(value: unknown): Record<string, readonly number[]> | undefined {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) {
    return undefined;
  }

  const sampleMap: Record<string, readonly number[]> = {};
  for (const [metricName, samples] of Object.entries(value as Record<string, unknown>)) {
    if (!Array.isArray(samples)) {
      return undefined;
    }
    const numericSamples = samples.filter(
      (entry): entry is number => typeof entry === 'number' && Number.isFinite(entry)
    );
    if (numericSamples.length !== samples.length) {
      return undefined;
    }
    sampleMap[metricName] = numericSamples;
  }

  return sampleMap;
}

function mergeChanceConstraintCounts(
  repetitionRecords: readonly OptimizationTrialRecord[]
): Record<string, { successes: number; trials: number }> | undefined {
  const merged: Record<string, { successes: number; trials: number }> = {};

  for (const record of repetitionRecords) {
    const metadata =
      record.metadata !== null &&
      typeof record.metadata === 'object' &&
      !Array.isArray(record.metadata)
        ? (record.metadata as Record<string, unknown>)
        : undefined;
    const rawCounts = metadata?.['chanceConstraintCounts'];
    if (
      rawCounts === null ||
      rawCounts === undefined ||
      typeof rawCounts !== 'object' ||
      Array.isArray(rawCounts)
    ) {
      continue;
    }

    for (const [metricName, entry] of Object.entries(rawCounts as Record<string, unknown>)) {
      if (
        entry === null ||
        typeof entry !== 'object' ||
        Array.isArray(entry) ||
        typeof (entry as Record<string, unknown>)['successes'] !== 'number' ||
        typeof (entry as Record<string, unknown>)['trials'] !== 'number'
      ) {
        continue;
      }

      const existing = merged[metricName] ?? { successes: 0, trials: 0 };
      const counts = entry as Record<string, number>;
      existing.successes += counts['successes']!;
      existing.trials += counts['trials']!;
      merged[metricName] = existing;
    }
  }

  return Object.keys(merged).length === 0 ? undefined : merged;
}

function buildGridCandidatePlan(
  spec: NormalizedOptimizationSpec,
  entries: [string, ParameterDefinition][],
  options: ValidatedOptimizeOptions
): CandidatePlan {
  const values = entries.map(
    ([name, definition]) => [name, buildDiscreteValues(name, definition)] as [string, unknown[]]
  );
  const product = cartesianProduct(values)
    .map((candidate) => applyDefaultConfig(spec, candidate))
    .filter((candidate) => evaluatePreTrialConstraints(spec, candidate, toErrorMessage));

  return {
    configs: product.slice(0, options.maxTrials),
    exhaustive: product.length <= options.maxTrials,
  };
}

function buildRandomCandidatePlan(
  spec: NormalizedOptimizationSpec,
  entries: [string, ParameterDefinition][],
  options: ValidatedOptimizeOptions
): CandidatePlan {
  const random = new PythonRandom(options.randomSeed ?? 0);
  const configs: CandidateConfig[] = [];
  const seen = new Set<string>();
  const cardinality = discreteCardinality(entries);
  const uniqueOnly = cardinality !== null;

  let attempts = 0;
  const maxAttempts = Math.max(options.maxTrials * 200, 1_000);

  while (configs.length < options.maxTrials && attempts < maxAttempts) {
    attempts += 1;
    const candidate = applyDefaultConfig(
      spec,
      Object.fromEntries(
        entries.map(([name, definition]) => [name, sampleParameter(name, definition, random)])
      )
    );

    if (!evaluatePreTrialConstraints(spec, candidate, toErrorMessage)) {
      if (cardinality !== null && seen.size >= cardinality) {
        break;
      }
      continue;
    }

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
  options: ValidatedOptimizeOptions
): CandidatePlan {
  const entries = getOrderedParameterEntries(spec.configurationSpace);

  if (options.algorithm === 'grid') {
    return buildGridCandidatePlan(spec, entries, options);
  }

  if (options.algorithm === 'random') {
    return buildRandomCandidatePlan(spec, entries, options);
  }

  return { configs: [], exhaustive: false };
}

async function resolveEvaluationRows(
  spec: NormalizedOptimizationSpec
): Promise<readonly unknown[]> {
  if (spec.evaluation?.data) return spec.evaluation.data;
  if (spec.evaluation?.loadData) return spec.evaluation.loadData();

  throw new ValidationError(
    'optimize() requires spec.evaluation.data or spec.evaluation.loadData.'
  );
}

function validateOptimizeOptions(options: NativeOptimizeOptions): ValidatedOptimizeOptions {
  if (!options || typeof options !== 'object') {
    throw new ValidationError('optimize() options are required.');
  }
  if (options.mode !== undefined && options.mode !== 'native') {
    throw new ValidationError('optimize() native mode only accepts mode: "native".');
  }
  if (
    options.algorithm !== 'grid' &&
    options.algorithm !== 'random' &&
    options.algorithm !== 'bayesian'
  ) {
    throw new ValidationError(
      'optimize() only supports algorithm "grid", "random", or "bayesian".'
    );
  }
  if (!Number.isInteger(options.maxTrials) || options.maxTrials <= 0) {
    throw new ValidationError('optimize() requires maxTrials to be a positive integer.');
  }
  if (
    options.randomSeed !== undefined &&
    (!Number.isInteger(options.randomSeed) || options.randomSeed < 0)
  ) {
    throw new ValidationError(
      'optimize() randomSeed must be a non-negative integer when provided.'
    );
  }
  if (
    options.timeoutMs !== undefined &&
    (!Number.isInteger(options.timeoutMs) || options.timeoutMs <= 0)
  ) {
    throw new ValidationError('optimize() timeoutMs must be a positive integer when provided.');
  }

  const trialConcurrency = options.trialConcurrency ?? 1;
  if (!Number.isInteger(trialConcurrency) || trialConcurrency <= 0) {
    throw new ValidationError(
      'optimize() trialConcurrency must be a positive integer when provided.'
    );
  }
  if (options.algorithm === 'bayesian' && trialConcurrency > 1) {
    throw new ValidationError('optimize() bayesian does not support trialConcurrency > 1 yet.');
  }

  if (options.plateau !== undefined) {
    if (
      !options.plateau ||
      !Number.isInteger(options.plateau.window) ||
      options.plateau.window <= 0
    ) {
      throw new ValidationError('optimize() plateau.window must be a positive integer.');
    }
    if (
      typeof options.plateau.minImprovement !== 'number' ||
      !Number.isFinite(options.plateau.minImprovement) ||
      options.plateau.minImprovement < 0
    ) {
      throw new ValidationError('optimize() plateau.minImprovement must be a finite number >= 0.');
    }
  }

  if (options.checkpoint !== undefined) {
    if (typeof options.checkpoint.key !== 'string' || options.checkpoint.key.trim().length === 0) {
      throw new ValidationError('optimize() checkpoint.key must be non-empty.');
    }
    if (options.checkpoint.dir !== undefined && options.checkpoint.dir.trim().length === 0) {
      throw new ValidationError('optimize() checkpoint.dir must be non-empty when provided.');
    }
  }

  return {
    ...options,
    mode: 'native',
    trialConcurrency,
  };
}

function createTrialConfig(
  config: CandidateConfig,
  trialNumber: number,
  totalRows: number,
  experimentRunId: string,
  rowLimit = totalRows
): TrialConfig {
  const limitedTotal = clamp(rowLimit, 0, totalRows);
  return {
    trial_id: `trial_${trialNumber}_${randomUUID()}`,
    trial_number: trialNumber,
    experiment_run_id: experimentRunId,
    config,
    dataset_subset: {
      indices: Array.from({ length: limitedTotal }, (_, index) => index),
      total: totalRows,
    },
  };
}

function normalizeDuration(result: NativeTrialFunctionResult, fallbackDuration: number): number {
  if (result.duration !== undefined && Number.isFinite(result.duration) && result.duration >= 0) {
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

function getTrialSubsetSize(trialConfig: TrialConfig): number {
  if (trialConfig.dataset_subset.inline_rows) {
    return trialConfig.dataset_subset.inline_rows.length;
  }

  if (trialConfig.dataset_subset.indices.length > 0) {
    return trialConfig.dataset_subset.indices.length;
  }

  return trialConfig.dataset_subset.total;
}

function extractEvaluatedExamples(
  metadata: Record<string, unknown> | undefined,
  trialConfig: TrialConfig
): number {
  const evaluatedRows = metadata?.['evaluatedRows'];
  if (typeof evaluatedRows === 'number' && Number.isFinite(evaluatedRows) && evaluatedRows >= 0) {
    return evaluatedRows;
  }

  return getTrialSubsetSize(trialConfig);
}

async function executeTrial(
  trialFn: NativeTrialFunction,
  trialConfig: TrialConfig,
  timeoutMs: number | undefined,
  signal: AbortSignal | undefined
): Promise<TrialOutcome> {
  const controller = new AbortController();
  const listeners: Array<() => void> = [];

  let timeoutId: NodeJS.Timeout | undefined;
  const start = Date.now();

  try {
    const trialPromise = Promise.resolve(
      TrialContext.run(trialConfig, async () => trialFn(trialConfig), controller.signal)
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
        (value): value is Promise<NativeTrialFunctionResult> => value !== undefined
      )
    );

    if (!rawResult || typeof rawResult !== 'object') {
      throw new ValidationError(
        'optimize() trial function must resolve to an object containing metrics.'
      );
    }

    const metricsParse = MetricsSchema.safeParse(rawResult.metrics);
    if (!metricsParse.success) {
      throw new ValidationError(
        `optimize() trial metrics are invalid: ${metricsParse.error.message}`
      );
    }

    const duration = normalizeDuration(rawResult, (Date.now() - start) / 1000);
    const normalizedMetrics = normalizeCostMetrics(metricsParse.data);
    return {
      status: 'completed',
      record: {
        trialId: trialConfig.trial_id,
        trialNumber: trialConfig.trial_number,
        config: { ...trialConfig.config },
        metrics: normalizedMetrics,
        duration,
        status: 'completed',
        metadata: rawResult.metadata,
      },
      actualCostUsd: extractTrialCost(normalizedMetrics),
      evaluatedExamples: extractEvaluatedExamples(rawResult.metadata, trialConfig),
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
        actualCostUsd: 0,
        evaluatedExamples: 0,
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
        actualCostUsd: 0,
        evaluatedExamples: 0,
      };
    }
    return {
      status: 'error',
      trialNumber: trialConfig.trial_number,
      errorMessage: toErrorMessage(error),
      actualCostUsd: 0,
      evaluatedExamples: 0,
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

async function executeTrialWithRepetitions(
  trialFn: NativeTrialFunction,
  trialConfig: TrialConfig,
  timeoutMs: number | undefined,
  signal: AbortSignal | undefined,
  spec: NormalizedOptimizationSpec
): Promise<TrialOutcome> {
  const repetitionCount = spec.execution.repsPerTrial;
  if (repetitionCount <= 1) {
    return executeTrial(trialFn, trialConfig, timeoutMs, signal);
  }

  const repetitionRecords: OptimizationTrialRecord[] = [];
  let actualCostUsd = 0;
  let evaluatedExamples = 0;

  for (let repetitionIndex = 0; repetitionIndex < repetitionCount; repetitionIndex += 1) {
    const repetitionOutcome = await executeTrial(
      trialFn,
      {
        ...trialConfig,
        metadata: {
          ...trialConfig.metadata,
          repetition_index: repetitionIndex + 1,
          repetition_count: repetitionCount,
        },
      },
      timeoutMs,
      signal
    );

    actualCostUsd += repetitionOutcome.actualCostUsd;
    evaluatedExamples += repetitionOutcome.evaluatedExamples;

    if (repetitionOutcome.status !== 'completed') {
      return {
        ...repetitionOutcome,
        actualCostUsd,
        evaluatedExamples,
      };
    }

    repetitionRecords.push(repetitionOutcome.record);
  }

  const aggregatedMetrics = aggregateRepetitionMetrics(
    repetitionRecords.map((record) => record.metrics),
    spec.execution.repsAggregation
  );
  const metricSamples =
    spec.promotionPolicy !== undefined
      ? mergeMetricSamples(
          repetitionRecords.map((record) => {
            const metadata =
              record.metadata !== null &&
              typeof record.metadata === 'object' &&
              !Array.isArray(record.metadata)
                ? (record.metadata as Record<string, unknown>)
                : undefined;
            return (
              asMetricSampleMap(metadata?.['metricSamples']) ??
              collectMetricSamples([record.metrics])
            );
          })
        )
      : undefined;
  const chanceConstraintCounts = mergeChanceConstraintCounts(repetitionRecords);
  const totalDuration = repetitionRecords.reduce((sum, record) => sum + record.duration, 0);
  const baseRecord = repetitionRecords[0]!;

  return {
    status: 'completed',
    record: {
      ...baseRecord,
      metrics: aggregatedMetrics,
      duration: totalDuration,
      status: 'completed',
      metadata: {
        ...baseRecord.metadata,
        evaluatedRows: evaluatedExamples,
        repsPerTrial: repetitionCount,
        repsAggregation: spec.execution.repsAggregation,
        ...(metricSamples ? { metricSamples } : {}),
        ...(chanceConstraintCounts ? { chanceConstraintCounts } : {}),
      },
    },
    actualCostUsd,
    evaluatedExamples,
  };
}

async function ensureCheckpointDirectory(path: string): Promise<void> {
  await mkdir(path, { recursive: true });
}

async function resolveCheckpointContext(
  spec: NormalizedOptimizationSpec,
  options: ValidatedOptimizeOptions
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
  const filename = `${createHash('sha256').update(options.checkpoint.key).digest('hex')}.json`;

  return {
    specHash,
    optionsHash,
    path: join(dir, filename),
  };
}

async function loadCheckpointState(
  checkpoint: CheckpointContext | undefined,
  options: ValidatedOptimizeOptions
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
      throw new ValidationError('optimize() checkpoint does not match the current spec/options.');
    }

    return {
      experimentRunId: parsed.experimentRunId,
      exhaustive: parsed.exhaustive,
      candidateConfigs: parsed.candidateConfigs,
      samplerState: parsed.samplerState,
      trials: parsed.trials ?? parsed.completedTrials ?? [],
      totalCostUsd: parsed.totalCostUsd ?? 0,
      totalExamplesEvaluated: parsed.totalExamplesEvaluated ?? 0,
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
  state: ResumeState
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
    trials: state.trials,
    totalCostUsd: state.totalCostUsd,
    totalExamplesEvaluated: state.totalExamplesEvaluated,
  };

  await writeFile(checkpoint.path, `${JSON.stringify(payload, null, 2)}\n`, 'utf8');
}

function buildReportingSummary(
  spec: NormalizedOptimizationSpec,
  trials: readonly OptimizationTrialRecord[],
  bestTrial: OptimizationTrialRecord | null,
  promotionDecision: OptimizationResult['promotionDecision'],
  totalExamplesEvaluated: number
): NativeOptimizationReportingSummary {
  const completedTrials = trials.filter((trial) => trial.status === 'completed');
  const rejectedTrials = trials.length - completedTrials.length;
  const trialDecisions = trials.flatMap((trial) =>
    trial.promotionDecision ? [trial.promotionDecision] : []
  );
  const latestMeaningfulPromotionDecision = [...trialDecisions]
    .reverse()
    .find(
      (decision) =>
        decision !== undefined &&
        (decision.method !== 'none' || decision.decision !== 'no_decision')
    );
  const summaryPromotionDecision =
    promotionDecision !== undefined &&
    (promotionDecision.method !== 'none' || promotionDecision.decision !== 'no_decision')
      ? promotionDecision
      : latestMeaningfulPromotionDecision;

  return {
    totalTrials: trials.length,
    completedTrials: completedTrials.length,
    rejectedTrials,
    evaluatedExamples: totalExamplesEvaluated,
    promotion: {
      applied: spec.promotionPolicy !== undefined,
      bestTrialId: bestTrial?.trialId,
      bestTrialNumber: bestTrial?.trialNumber,
      decision: summaryPromotionDecision?.decision,
      method: summaryPromotionDecision?.method,
      usedChanceConstraints:
        trialDecisions.some((decision) => decision.chanceResults.length > 0) ||
        (summaryPromotionDecision?.chanceResults.length ?? 0) > 0,
      usedStatisticalComparison:
        trialDecisions.some(
          (decision) =>
            decision.method === 'statistical' ||
            decision.objectiveResults.some((result) => result.method === 'statistical')
        ) ||
        summaryPromotionDecision?.method === 'statistical' ||
        summaryPromotionDecision?.objectiveResults.some(
          (result) => result.method === 'statistical'
        ) === true,
      usedTieBreakers:
        trialDecisions.some((decision) => /tie-breaker/i.test(decision.reason)) ||
        /tie-breaker/i.test(summaryPromotionDecision?.reason ?? ''),
    },
  };
}

function finalizeResult(
  trials: OptimizationTrialRecord[],
  spec: NormalizedOptimizationSpec,
  objectives: readonly NormalizedObjectiveDefinition[],
  stopReason: OptimizationResult['stopReason'],
  totalCostUsd: number,
  totalExamplesEvaluated: number,
  errorMessage?: string
): OptimizationResult {
  const orderedTrials = [...trials].sort((left, right) => left.trialNumber - right.trialNumber);
  const selection = spec.promotionPolicy
    ? selectBestTrialWithPromotionDecision(orderedTrials, objectives, spec.promotionPolicy)
    : {
        bestTrial: selectBestTrial(orderedTrials, objectives, undefined),
        promotionDecision: undefined,
      };
  const bestTrial = selection.bestTrial;
  const reporting = buildReportingSummary(
    spec,
    orderedTrials,
    bestTrial,
    selection.promotionDecision,
    totalExamplesEvaluated
  );

  return {
    mode: 'native',
    bestConfig: bestTrial?.config ?? null,
    bestMetrics: bestTrial?.metrics ?? null,
    trials: orderedTrials,
    promotionDecision: selection.promotionDecision,
    reporting,
    stopReason,
    totalCostUsd,
    errorMessage,
  };
}

function getMaxRowsForNextTrial(
  spec: NormalizedOptimizationSpec,
  totalRows: number,
  totalExamplesEvaluated: number,
  reservedExamples: number
): number {
  const maxTotalExamples = spec.execution.maxTotalExamples;
  if (maxTotalExamples === undefined) {
    return totalRows;
  }

  const remainingExamples = maxTotalExamples - totalExamplesEvaluated - reservedExamples;
  if (remainingExamples <= 0) {
    return 0;
  }

  return Math.min(totalRows, Math.floor(remainingExamples / spec.execution.repsPerTrial));
}

function hasExceededWallclock(spec: NormalizedOptimizationSpec, startedAt: number): boolean {
  return (
    spec.execution.maxWallclockMs !== undefined &&
    Date.now() - startedAt >= spec.execution.maxWallclockMs
  );
}

async function runCandidatePlan(
  trialFn: NativeTrialFunction,
  spec: NormalizedOptimizationSpec,
  options: ValidatedOptimizeOptions,
  plan: CandidatePlan,
  resume: ResumeState,
  evaluationRows: readonly unknown[],
  checkpoint: CheckpointContext | undefined
): Promise<OptimizationResult> {
  const trials = [...resume.trials];
  let totalCostUsd = resume.totalCostUsd;
  let totalExamplesEvaluated = resume.totalExamplesEvaluated;
  let stopReason: OptimizationResult['stopReason'] = plan.exhaustive ? 'completed' : 'maxTrials';
  let errorMessage: string | undefined;
  let committedTrialNumber = trials.length;
  const startedAt = Date.now();

  const abortController = new AbortController();
  if (options.signal) {
    if (options.signal.aborted) {
      abortController.abort();
      return finalizeResult(
        trials,
        spec,
        spec.objectives,
        'cancelled',
        totalCostUsd,
        totalExamplesEvaluated,
        'Optimization cancelled'
      );
    }
    options.signal.addEventListener('abort', () => abortController.abort(), {
      once: true,
    });
  }

  type ActiveTask = {
    trialNumber: number;
    reservedExamples: number;
    promise: Promise<TrialOutcome>;
  };
  const active = new Map<number, ActiveTask>();
  const buffered = new Map<number, TrialOutcome>();

  let nextIndex = committedTrialNumber;
  const maybeDispatch = (): void => {
    while (
      !abortController.signal.aborted &&
      active.size < options.trialConcurrency &&
      nextIndex < plan.configs.length
    ) {
      if (hasExceededWallclock(spec, startedAt)) {
        stopReason = 'timeout';
        errorMessage = 'Optimization wallclock budget exceeded';
        abortController.abort();
        break;
      }
      const rowLimit = getMaxRowsForNextTrial(
        spec,
        evaluationRows.length,
        totalExamplesEvaluated,
        [...active.values()].reduce((sum, task) => sum + task.reservedExamples, 0)
      );
      if (rowLimit <= 0) {
        stopReason = 'maxExamples';
        abortController.abort();
        break;
      }

      const trialNumber = nextIndex + 1;
      const trialConfig = createTrialConfig(
        plan.configs[nextIndex]!,
        trialNumber,
        evaluationRows.length,
        resume.experimentRunId,
        rowLimit
      );
      const trialReservedExamples = rowLimit * spec.execution.repsPerTrial;
      active.set(trialNumber, {
        trialNumber,
        reservedExamples: trialReservedExamples,
        promise: executeTrialWithRepetitions(
          trialFn,
          trialConfig,
          options.timeoutMs,
          abortController.signal,
          spec
        ).then((outcome) => applyPostTrialGuards(spec, outcome, toErrorMessage)),
      });
      nextIndex += 1;
    }
  };

  maybeDispatch();

  while (active.size > 0) {
    const { trialNumber, outcome } = await Promise.race(
      [...active.values()].map(async (task) => ({
        trialNumber: task.trialNumber,
        outcome: await task.promise,
      }))
    );
    active.delete(trialNumber);
    buffered.set(trialNumber, outcome);

    while (buffered.has(committedTrialNumber + 1)) {
      const current = buffered.get(committedTrialNumber + 1)!;
      buffered.delete(committedTrialNumber + 1);
      committedTrialNumber += 1;

      if (current.status !== 'completed') {
        if (current.status === 'rejected') {
          totalCostUsd += current.actualCostUsd;
          totalExamplesEvaluated += current.evaluatedExamples;
          trials.push(current.record);

          await saveCheckpointState(checkpoint, options, {
            ...resume,
            exhaustive: plan.exhaustive,
            candidateConfigs: plan.configs,
            trials: [...trials].sort((left, right) => left.trialNumber - right.trialNumber),
            totalCostUsd,
            totalExamplesEvaluated,
          });

          if (spec.budget?.maxCostUsd !== undefined && totalCostUsd >= spec.budget.maxCostUsd) {
            stopReason = 'budget';
            abortController.abort();
            break;
          }

          if (
            spec.execution.maxTotalExamples !== undefined &&
            totalExamplesEvaluated >= spec.execution.maxTotalExamples
          ) {
            stopReason = 'maxExamples';
            abortController.abort();
            break;
          }

          continue;
        }

        totalCostUsd += current.actualCostUsd;
        totalExamplesEvaluated += current.evaluatedExamples;
        stopReason = current.status;
        errorMessage = current.errorMessage;
        abortController.abort();
        break;
      }

      for (const objective of spec.objectives) {
        getObjectiveMetric(current.record, objective);
      }
      assertTrialCostMetricAvailable(spec, current.record);

      totalCostUsd += current.actualCostUsd;
      totalExamplesEvaluated += current.evaluatedExamples;
      trials.push(current.record);

      await saveCheckpointState(checkpoint, options, {
        ...resume,
        exhaustive: plan.exhaustive,
        candidateConfigs: plan.configs,
        trials: [...trials].sort((left, right) => left.trialNumber - right.trialNumber),
        totalCostUsd,
        totalExamplesEvaluated,
      });

      if (spec.budget?.maxCostUsd !== undefined && totalCostUsd >= spec.budget.maxCostUsd) {
        stopReason = 'budget';
        abortController.abort();
        break;
      }

      if (hasPlateau(trials, spec.objectives, options.plateau)) {
        stopReason = 'plateau';
        abortController.abort();
        break;
      }

      if (
        spec.execution.maxTotalExamples !== undefined &&
        totalExamplesEvaluated >= spec.execution.maxTotalExamples
      ) {
        stopReason = 'maxExamples';
        abortController.abort();
        break;
      }

      if (hasExceededWallclock(spec, startedAt)) {
        stopReason = 'timeout';
        errorMessage = 'Optimization wallclock budget exceeded';
        abortController.abort();
        break;
      }
    }

    if (abortController.signal.aborted) {
      break;
    }

    maybeDispatch();
  }

  await Promise.allSettled([...active.values()].map((task) => task.promise));
  return finalizeResult(
    trials,
    spec,
    spec.objectives,
    stopReason,
    totalCostUsd,
    totalExamplesEvaluated,
    errorMessage
  );
}

async function runBayesianPlan(
  trialFn: NativeTrialFunction,
  spec: NormalizedOptimizationSpec,
  options: ValidatedOptimizeOptions,
  resume: ResumeState,
  evaluationRows: readonly unknown[],
  checkpoint: CheckpointContext | undefined
): Promise<OptimizationResult> {
  const trials = [...resume.trials].sort((left, right) => left.trialNumber - right.trialNumber);
  let totalCostUsd = resume.totalCostUsd;
  let totalExamplesEvaluated = resume.totalExamplesEvaluated;
  let stopReason: OptimizationResult['stopReason'] = 'maxTrials';
  let errorMessage: string | undefined;
  const startedAt = Date.now();
  const random = resume.samplerState
    ? new PythonRandom(resume.samplerState)
    : new PythonRandom(options.randomSeed ?? 0);

  for (let index = trials.length; index < options.maxTrials; index += 1) {
    if (hasExceededWallclock(spec, startedAt)) {
      stopReason = 'timeout';
      errorMessage = 'Optimization wallclock budget exceeded';
      break;
    }
    if (options.signal?.aborted) {
      stopReason = 'cancelled';
      errorMessage = 'Optimization cancelled';
      break;
    }

    const rowLimit = getMaxRowsForNextTrial(spec, evaluationRows.length, totalExamplesEvaluated, 0);
    if (rowLimit <= 0) {
      stopReason = 'maxExamples';
      break;
    }

    const suggestion = suggestBayesianConfig(
      spec,
      trials,
      random,
      options.maxTrials,
      (currentSpec, config) => evaluatePreTrialConstraints(currentSpec, config, toErrorMessage)
    );
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
      rowLimit
    );

    const outcome = applyPostTrialGuards(
      spec,
      await executeTrialWithRepetitions(
        trialFn,
        trialConfig,
        options.timeoutMs,
        options.signal,
        spec
      ),
      toErrorMessage
    );

    if (outcome.status !== 'completed') {
      if (outcome.status === 'rejected') {
        totalCostUsd += outcome.actualCostUsd;
        totalExamplesEvaluated += outcome.evaluatedExamples;
        trials.push(outcome.record);

        await saveCheckpointState(checkpoint, options, {
          ...resume,
          exhaustive: suggestion.exhaustive,
          samplerState: random.serialize(),
          trials: [...trials],
          totalCostUsd,
          totalExamplesEvaluated,
        });

        if (spec.budget?.maxCostUsd !== undefined && totalCostUsd >= spec.budget.maxCostUsd) {
          stopReason = 'budget';
          break;
        }

        if (
          spec.execution.maxTotalExamples !== undefined &&
          totalExamplesEvaluated >= spec.execution.maxTotalExamples
        ) {
          stopReason = 'maxExamples';
          break;
        }

        continue;
      }

      totalCostUsd += outcome.actualCostUsd;
      totalExamplesEvaluated += outcome.evaluatedExamples;
      stopReason = outcome.status;
      errorMessage = outcome.errorMessage;
      break;
    }

    for (const objective of spec.objectives) {
      getObjectiveMetric(outcome.record, objective);
    }
    assertTrialCostMetricAvailable(spec, outcome.record);

    totalCostUsd += outcome.actualCostUsd;
    totalExamplesEvaluated += outcome.evaluatedExamples;
    trials.push(outcome.record);

    await saveCheckpointState(checkpoint, options, {
      ...resume,
      exhaustive: suggestion.exhaustive,
      samplerState: random.serialize(),
      trials: [...trials],
      totalCostUsd,
      totalExamplesEvaluated,
    });

    if (spec.budget?.maxCostUsd !== undefined && totalCostUsd >= spec.budget.maxCostUsd) {
      stopReason = 'budget';
      break;
    }

    if (hasPlateau(trials, spec.objectives, options.plateau)) {
      stopReason = 'plateau';
      break;
    }

    if (
      spec.execution.maxTotalExamples !== undefined &&
      totalExamplesEvaluated >= spec.execution.maxTotalExamples
    ) {
      stopReason = 'maxExamples';
      break;
    }

    if (hasExceededWallclock(spec, startedAt)) {
      stopReason = 'timeout';
      errorMessage = 'Optimization wallclock budget exceeded';
      break;
    }
  }

  return finalizeResult(
    trials,
    spec,
    spec.objectives,
    stopReason,
    totalCostUsd,
    totalExamplesEvaluated,
    errorMessage
  );
}

export async function runNativeOptimization(
  trialFn: NativeTrialFunction,
  spec: NormalizedOptimizationSpec,
  rawOptions: NativeOptimizeOptions
): Promise<OptimizationResult> {
  const options = validateOptimizeOptions(rawOptions);
  const evaluationRows = await resolveEvaluationRows(spec);

  if (!Array.isArray(evaluationRows) || evaluationRows.length === 0) {
    throw new ValidationError('optimize() requires evaluation data to be a non-empty array.');
  }

  const checkpoint = await resolveCheckpointContext(spec, options);
  const resume = (await loadCheckpointState(checkpoint, options)) ?? {
    experimentRunId: `native_${randomUUID()}`,
    trials: [],
    totalCostUsd: 0,
    totalExamplesEvaluated: 0,
  };

  if (options.algorithm === 'bayesian') {
    return runBayesianPlan(trialFn, spec, options, resume, evaluationRows, checkpoint);
  }

  const builtPlan = buildCandidatePlan(spec, options);
  const plan = {
    ...builtPlan,
    configs: resume.candidateConfigs ?? builtPlan.configs,
    exhaustive: resume.exhaustive ?? builtPlan.exhaustive,
  };

  if (plan.configs.length === 0) {
    throw new ValidationError('No valid configurations satisfy the configured constraints.');
  }

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
    checkpoint
  );
}
