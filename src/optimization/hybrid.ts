import { TrialCancelledError, TrialContext } from '../core/context.js';
import { CancelledError, TimeoutError, ValidationError } from '../core/errors.js';
import { MetricsSchema, type Metrics, type TrialConfig } from '../dtos/trial.js';
import { resolveEvaluationRows } from './agent.js';
import { stableValueEquals } from './stable-value.js';
import type {
  BandedObjectiveDefinition,
  HybridOptimizeOptions,
  NativeTrialFunctionResult,
  NormalizedObjectiveDefinition,
  NormalizedOptimizationSpec,
  ObjectiveDefinition,
  ObjectiveInput,
  OptimizationConvergencePoint,
  OptimizationSessionCreateRequest,
  OptimizationSessionCreationResponse,
  OptimizationSessionDatasetSubset,
  OptimizationSessionDeleteOptions,
  OptimizationSessionDeleteResponse,
  OptimizationSessionFinalizeOptions,
  OptimizationSessionFinalizationResponse,
  OptimizationSessionListOptions,
  OptimizationSessionListResponse,
  OptimizationSessionNextTrialOptions,
  OptimizationSessionNextTrialResponse,
  OptimizationSessionRequestOptions,
  OptimizationSessionStatusMetadata,
  OptimizationSessionStatusSummary,
  OptimizationSessionStatusResponse,
  OptimizationServiceStatusResponse,
  OptimizationReportingSummary,
  OptimizationReportingTrialHistoryEntry,
  OptimizationResult,
  OptimizationSpec,
  OptimizationSessionSubmitResultResponse,
  OptimizationSessionTrialResultInput,
  OptimizationSessionTrialSuggestion,
  OptimizationTrialRecord,
  ParameterDefinition,
} from './types.js';

type NativeTrialFunction = (trialConfig: TrialConfig) => Promise<NativeTrialFunctionResult>;

type TrialFailureStatus = 'timeout' | 'error' | 'cancelled';

type TrialOutcome =
  | {
      status: 'completed';
      record: OptimizationTrialRecord;
    }
  | {
      status: TrialFailureStatus;
      trialNumber: number;
      errorMessage: string;
    };

interface ValidatedHybridOptimizeOptions extends HybridOptimizeOptions {
  mode?: 'hybrid';
  algorithm: 'optuna';
  backendUrl: string;
  apiKey: string;
  requestTimeoutMs: number;
}

interface ValidatedHybridRequestOptions {
  backendUrl: string;
  apiKey: string;
  requestTimeoutMs: number;
  signal?: AbortSignal;
}

interface HybridConnectionOptions {
  backendUrl?: string;
  apiKey?: string;
}

interface SerializedHybridObjective {
  metric: string;
  direction?: 'maximize' | 'minimize';
  band?: {
    low: number;
    high: number;
  };
  test?: 'TOST';
  alpha?: number;
  weight?: number;
}

interface HybridSessionCreateResponse {
  session_id: string;
  status: string;
  optimization_strategy?: Record<string, unknown>;
  metadata?: Record<string, unknown>;
}

interface HybridSessionSuggestionPayload {
  trial_id: string;
  session_id: string;
  trial_number: number;
  config: Record<string, unknown>;
  dataset_subset: {
    indices: number[];
    selection_strategy?: string;
    confidence_level?: number;
    estimated_representativeness?: number;
    metadata?: Record<string, unknown>;
  };
  exploration_type?: string;
  priority?: number;
  estimated_duration?: number;
  metadata?: Record<string, unknown>;
}

interface HybridNextTrialResponse {
  suggestion: HybridSessionSuggestionPayload | null;
  should_continue: boolean;
  reason?: string | null;
  stop_reason?: string | null;
  session_status?: string;
  metadata?: Record<string, unknown>;
}

interface HybridFinalizationResponse {
  session_id: string;
  best_config?: Record<string, unknown>;
  best_metrics?: Record<string, unknown>;
  total_trials?: number;
  successful_trials?: number;
  total_duration?: number;
  cost_savings?: number;
  stop_reason?: string | null;
  convergence_history?: unknown[];
  full_history?: unknown[];
  metadata?: Record<string, unknown>;
}

interface HybridSessionListPayload {
  sessions?: unknown[];
  total?: number;
  metadata?: Record<string, unknown>;
}

interface HybridSubmitResultsResponse {
  success?: boolean;
  continue_optimization?: boolean;
  message?: string;
  [key: string]: unknown;
}

interface HybridSubmittedResult {
  session_id: string;
  trial_id: string;
  metrics: Metrics;
  duration: number;
  status: 'completed' | 'failed' | 'cancelled' | 'timeout';
  error_message: string | null;
  metadata: Record<string, unknown>;
}

interface HybridStopContext {
  backendReason?: string;
  failedTrialStopReason?: 'timeout' | 'error';
}

const DEFAULT_REQUEST_TIMEOUT_MS = 30_000;
function isPlainObject(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

function toSnakeCaseKey(key: string): string {
  return key
    .replace(/([a-z0-9])([A-Z])/g, '$1_$2')
    .replace(/[-\s]+/g, '_')
    .toLowerCase();
}

function serializeSnakeCaseObject(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map((item) => serializeSnakeCaseObject(item));
  }

  if (!isPlainObject(value)) {
    return value;
  }

  return Object.fromEntries(
    Object.entries(value).map(([key, nestedValue]) => [
      toSnakeCaseKey(key),
      serializeSnakeCaseObject(nestedValue),
    ])
  );
}

function toErrorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }
  return String(error);
}

function normalizeDuration(result: NativeTrialFunctionResult, fallbackDuration: number): number {
  if (result.duration !== undefined && Number.isFinite(result.duration) && result.duration >= 0) {
    return result.duration;
  }

  return fallbackDuration;
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

function normalizeBackendStopReason(
  context: HybridStopContext,
  trialCount: number,
  maxTrials: number
): OptimizationResult['stopReason'] {
  const normalized = context.backendReason?.trim().toLowerCase().replace(/[_-]+/g, ' ');

  if (normalized) {
    const exactMappings: Record<string, OptimizationResult['stopReason']> = {
      'budget exhausted': 'budget',
      'max trials reached': 'maxTrials',
      'max wallclock reached': 'timeout',
      'search complete': 'completed',
      finalized: 'completed',
    };

    if (normalized in exactMappings) {
      return exactMappings[normalized]!;
    }

    if (normalized.includes('cancel')) return 'cancelled';
    if (normalized.includes('timeout')) return 'timeout';
    if (normalized.includes('wallclock') || normalized.includes('elapsed')) {
      return 'timeout';
    }
    if (normalized.includes('budget') || normalized.includes('cost')) {
      return 'budget';
    }
    if (
      normalized.includes('plateau') ||
      normalized.includes('converg') ||
      normalized.includes('stagnat')
    ) {
      return 'plateau';
    }
    if (
      normalized.includes('max trial') ||
      normalized.includes('trial limit') ||
      normalized.includes('limit reached')
    ) {
      return 'maxTrials';
    }
    if (
      normalized.includes('complete') ||
      normalized.includes('exhaust') ||
      normalized.includes('no more')
    ) {
      return 'completed';
    }
  }

  if (context.failedTrialStopReason) {
    return context.failedTrialStopReason;
  }

  return trialCount >= maxTrials ? 'maxTrials' : 'completed';
}

function resolveBackendStopReason(
  value: string | null | undefined,
  fallback: string | null | undefined
): string | undefined {
  if (typeof value === 'string' && value.trim().length > 0) {
    return value;
  }
  if (typeof fallback === 'string' && fallback.trim().length > 0) {
    return fallback;
  }
  return undefined;
}

function updateTotalCost(
  spec: NormalizedOptimizationSpec,
  trial: OptimizationTrialRecord,
  totalCostUsd: number
): number {
  if (spec.budget?.maxCostUsd === undefined) {
    const trialCost = trial.metrics['cost'];
    return typeof trialCost === 'number' && Number.isFinite(trialCost)
      ? totalCostUsd + trialCost
      : totalCostUsd;
  }

  const costMetric = trial.metrics['cost'];
  if (typeof costMetric !== 'number' || !Number.isFinite(costMetric)) {
    throw new ValidationError(
      'budget.maxCostUsd requires every trial to return numeric metrics.cost.'
    );
  }

  return totalCostUsd + costMetric;
}

function getObjectiveMetric(
  trial: OptimizationTrialRecord,
  objective: NormalizedObjectiveDefinition
): number {
  const value = trial.metrics[objective.metric];
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new ValidationError(
      `Trial "${trial.trialId}" is missing numeric metric "${objective.metric}".`
    );
  }
  return value;
}

export function objectiveScoreValue(
  value: number,
  objective: NormalizedObjectiveDefinition
): number {
  const maybeBanded = objective as NormalizedObjectiveDefinition & {
    kind?: 'banded';
    band?: { low: number; high: number };
  };

  if (
    (maybeBanded.direction === 'band' || maybeBanded.kind === 'banded') &&
    maybeBanded.band !== undefined
  ) {
    if (value < maybeBanded.band.low) {
      return -(maybeBanded.band.low - value);
    }
    if (value > maybeBanded.band.high) {
      return -(value - maybeBanded.band.high);
    }
    return 0;
  }

  return objective.direction === 'minimize' ? -value : value;
}

function selectBestTrial(
  trials: OptimizationTrialRecord[],
  objectives: readonly NormalizedObjectiveDefinition[]
): OptimizationTrialRecord | null {
  if (trials.length === 0) return null;

  const ranges = objectives.map((objective) => {
    const values = trials.map((trial) =>
      objectiveScoreValue(getObjectiveMetric(trial, objective), objective)
    );
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
      const value = objectiveScoreValue(
        getObjectiveMetric(trial, range.objective),
        range.objective
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

export function normalizeBackendApiBase(rawUrl: string): string {
  if (typeof rawUrl !== 'string' || rawUrl.trim().length === 0) {
    throw new ValidationError(
      'Hybrid optimize() requires backendUrl or TRAIGENT_BACKEND_URL / TRAIGENT_API_URL.'
    );
  }

  let url: URL;
  try {
    url = new URL(rawUrl);
  } catch {
    throw new ValidationError(`Hybrid optimize() received an invalid backendUrl: "${rawUrl}".`);
  }

  const pathname = url.pathname.replace(/\/+$/, '');
  if (pathname === '' || pathname === '/') {
    url.pathname = '/api/v1';
  } else if (pathname === '/api/v1' || pathname.endsWith('/api/v1')) {
    url.pathname = pathname;
  } else {
    throw new ValidationError(
      'Hybrid optimize() backendUrl must be a backend origin or an /api/v1 base URL.'
    );
  }

  url.search = '';
  url.hash = '';
  return url.toString().replace(/\/$/, '');
}

function resolveBackendUrl(options: HybridConnectionOptions): string {
  return (
    options.backendUrl ??
    process.env['TRAIGENT_BACKEND_URL'] ??
    process.env['TRAIGENT_API_URL'] ??
    ''
  );
}

function resolveApiKey(options: HybridConnectionOptions): string {
  return options.apiKey ?? process.env['TRAIGENT_API_KEY'] ?? '';
}

function deriveServiceBaseUrl(apiBase: string): string {
  const url = new URL(apiBase);
  if (url.pathname.endsWith('/api/v1')) {
    url.pathname = url.pathname.replace(/\/api\/v1$/, '') || '/';
  }
  url.search = '';
  url.hash = '';
  return url.toString().replace(/\/$/, '');
}

function validateStringOption(value: unknown, message: string): asserts value is string {
  if (typeof value !== 'string' || value.trim().length === 0) {
    throw new ValidationError(message);
  }
}

function validateHybridOptimizeOptions(
  options: HybridOptimizeOptions
): ValidatedHybridOptimizeOptions {
  if (!options || typeof options !== 'object') {
    throw new ValidationError('optimize() options are required.');
  }

  if (options.mode !== undefined && options.mode !== 'hybrid') {
    throw new ValidationError(
      'Hybrid optimize() only supports mode: "hybrid". Use mode: "native" for local execution.'
    );
  }

  if (options.algorithm !== 'optuna') {
    throw new ValidationError(
      'Hybrid optimize() requires algorithm "optuna". Set mode: "native" to use grid, random, or bayesian locally.'
    );
  }

  if (!Number.isInteger(options.maxTrials) || options.maxTrials <= 0) {
    throw new ValidationError('Hybrid optimize() requires maxTrials to be a positive integer.');
  }

  if (
    options.timeoutMs !== undefined &&
    (!Number.isInteger(options.timeoutMs) || options.timeoutMs <= 0)
  ) {
    throw new ValidationError(
      'Hybrid optimize() timeoutMs must be a positive integer when provided.'
    );
  }

  if (
    options.requestTimeoutMs !== undefined &&
    (!Number.isInteger(options.requestTimeoutMs) || options.requestTimeoutMs <= 0)
  ) {
    throw new ValidationError(
      'Hybrid optimize() requestTimeoutMs must be a positive integer when provided.'
    );
  }

  if (options.userId !== undefined) {
    validateStringOption(options.userId, 'Hybrid optimize() userId must be non-empty.');
  }

  if (options.billingTier !== undefined) {
    validateStringOption(options.billingTier, 'Hybrid optimize() billingTier must be non-empty.');
  }

  if (options.optimizationStrategy !== undefined && !isPlainObject(options.optimizationStrategy)) {
    throw new ValidationError(
      'Hybrid optimize() optimizationStrategy must be an object when provided.'
    );
  }

  if (options.datasetMetadata !== undefined && !isPlainObject(options.datasetMetadata)) {
    throw new ValidationError('Hybrid optimize() datasetMetadata must be an object when provided.');
  }

  if (options.includeFullHistory !== undefined && typeof options.includeFullHistory !== 'boolean') {
    throw new ValidationError(
      'Hybrid optimize() includeFullHistory must be a boolean when provided.'
    );
  }

  const nativeOnlyKeys = ['trialConcurrency', 'plateau', 'checkpoint', 'randomSeed'] as const;
  for (const key of nativeOnlyKeys) {
    if (key in (options as unknown as Record<string, unknown>)) {
      throw new ValidationError(`Hybrid optimize() does not support native option "${key}".`);
    }
  }

  const unresolvedBackendUrl = resolveBackendUrl(options);
  validateStringOption(
    unresolvedBackendUrl,
    'Hybrid optimize() requires backendUrl, TRAIGENT_BACKEND_URL, or TRAIGENT_API_URL.'
  );
  const backendUrl = normalizeBackendApiBase(unresolvedBackendUrl);
  const apiKey = resolveApiKey(options);
  validateStringOption(apiKey, 'Hybrid optimize() requires apiKey or TRAIGENT_API_KEY.');

  return {
    ...options,
    backendUrl,
    apiKey,
    requestTimeoutMs: options.requestTimeoutMs ?? DEFAULT_REQUEST_TIMEOUT_MS,
  };
}

function validateHybridRequestOptions(
  options: OptimizationSessionRequestOptions | undefined
): ValidatedHybridRequestOptions {
  if (options !== undefined && typeof options !== 'object') {
    throw new ValidationError('Session request options must be an object.');
  }

  if (
    options?.requestTimeoutMs !== undefined &&
    (!Number.isInteger(options.requestTimeoutMs) || options.requestTimeoutMs <= 0)
  ) {
    throw new ValidationError(
      'Session request options requestTimeoutMs must be a positive integer when provided.'
    );
  }

  const unresolvedBackendUrl = resolveBackendUrl(options ?? {});
  validateStringOption(
    unresolvedBackendUrl,
    'Session requests require backendUrl, TRAIGENT_BACKEND_URL, or TRAIGENT_API_URL.'
  );
  const apiKey = resolveApiKey(options ?? {});
  validateStringOption(apiKey, 'Session requests require apiKey or TRAIGENT_API_KEY.');

  return {
    backendUrl: normalizeBackendApiBase(unresolvedBackendUrl),
    apiKey,
    requestTimeoutMs: options?.requestTimeoutMs ?? DEFAULT_REQUEST_TIMEOUT_MS,
    signal: options?.signal,
  };
}

function validateHybridListOptions(
  options: OptimizationSessionListOptions | undefined
): ValidatedHybridRequestOptions & {
  pattern?: string;
  status?: string;
} {
  if (options?.pattern !== undefined) {
    validateStringOption(
      options.pattern,
      'Session list pattern must be a non-empty string when provided.'
    );
  }

  if (options?.status !== undefined) {
    validateStringOption(
      options.status,
      'Session list status must be a non-empty string when provided.'
    );
  }

  const resolved = validateHybridRequestOptions(options);
  return {
    ...resolved,
    pattern: options?.pattern,
    status: options?.status,
  };
}

function validateSessionCreateRequest(request: OptimizationSessionCreateRequest): void {
  if (!request || typeof request !== 'object') {
    throw new ValidationError('Session creation request must be an object.');
  }

  validateStringOption(request.functionName, 'Session creation requires a non-empty functionName.');

  if (
    !request.configurationSpace ||
    typeof request.configurationSpace !== 'object' ||
    Array.isArray(request.configurationSpace) ||
    Object.keys(request.configurationSpace).length === 0
  ) {
    throw new ValidationError('Session creation requires a non-empty configurationSpace object.');
  }

  if (!Array.isArray(request.objectives) || request.objectives.length === 0) {
    throw new ValidationError('Session creation requires a non-empty objectives array.');
  }

  if (
    request.maxTrials !== undefined &&
    (!Number.isInteger(request.maxTrials) || request.maxTrials <= 0)
  ) {
    throw new ValidationError(
      'Session creation maxTrials must be a positive integer when provided.'
    );
  }

  if (request.datasetMetadata !== undefined && !isPlainObject(request.datasetMetadata)) {
    throw new ValidationError('Session creation datasetMetadata must be an object when provided.');
  }

  if (request.userId !== undefined) {
    validateStringOption(
      request.userId,
      'Session creation userId must be non-empty when provided.'
    );
  }

  if (request.billingTier !== undefined) {
    validateStringOption(
      request.billingTier,
      'Session creation billingTier must be non-empty when provided.'
    );
  }

  const objectFields = [
    ['budget', request.budget],
    ['constraints', request.constraints],
    ['defaultConfig', request.defaultConfig],
    ['promotionPolicy', request.promotionPolicy],
    ['optimizationStrategy', request.optimizationStrategy],
    ['metadata', request.metadata],
  ] as const;

  for (const [label, value] of objectFields) {
    if (value !== undefined && !isPlainObject(value)) {
      throw new ValidationError(`Session creation ${label} must be an object when provided.`);
    }
  }
}

function validateHybridNextTrialOptions(
  sessionId: string,
  options: OptimizationSessionNextTrialOptions | undefined
): ValidatedHybridRequestOptions & {
  previousResults?: readonly OptimizationSessionTrialResultInput[];
  requestMetadata?: Record<string, unknown>;
} {
  validateStringOption(sessionId, 'Next-trial requests require a non-empty sessionId.');

  if (options?.previousResults !== undefined && !Array.isArray(options.previousResults)) {
    throw new ValidationError('Next-trial previousResults must be an array when provided.');
  }

  if (options?.requestMetadata !== undefined && !isPlainObject(options.requestMetadata)) {
    throw new ValidationError('Next-trial requestMetadata must be an object when provided.');
  }

  return {
    ...validateHybridRequestOptions(options),
    previousResults: options?.previousResults,
    requestMetadata: options?.requestMetadata,
  };
}

function validateSessionTrialResultInput(
  sessionId: string,
  result: OptimizationSessionTrialResultInput
): OptimizationSessionTrialResultInput {
  validateStringOption(sessionId, 'Trial result submission requires a non-empty sessionId.');

  if (!result || typeof result !== 'object') {
    throw new ValidationError('Trial result submission must be an object.');
  }

  validateStringOption(result.trialId, 'Trial result submission requires a non-empty trialId.');

  const metricsParse = MetricsSchema.safeParse(result.metrics);
  if (!metricsParse.success) {
    throw new ValidationError('Trial result submission metrics must be valid.');
  }

  if (!Number.isFinite(result.duration) || result.duration < 0) {
    throw new ValidationError('Trial result submission duration must be a non-negative number.');
  }

  if (result.metadata !== undefined && !isPlainObject(result.metadata)) {
    throw new ValidationError('Trial result submission metadata must be an object when provided.');
  }

  if (
    result.outputsSample !== undefined &&
    result.outputsSample !== null &&
    !Array.isArray(result.outputsSample)
  ) {
    throw new ValidationError(
      'Trial result submission outputsSample must be an array when provided.'
    );
  }

  return {
    ...result,
    metrics: metricsParse.data,
  };
}

function serializeHybridObjective(
  objective: ObjectiveDefinition | BandedObjectiveDefinition
): SerializedHybridObjective {
  const serialized: SerializedHybridObjective = {
    metric: objective.metric,
  };

  if ('band' in objective && objective.band !== undefined) {
    const bandTarget = objective.band as {
      low?: number;
      high?: number;
      center?: number;
      tol?: number;
      test?: 'TOST';
      alpha?: number;
    };
    const low =
      typeof bandTarget.low === 'number'
        ? bandTarget.low
        : Number(bandTarget.center) - Number(bandTarget.tol);
    const high =
      typeof bandTarget.high === 'number'
        ? bandTarget.high
        : Number(bandTarget.center) + Number(bandTarget.tol);
    serialized.band = { low, high };
    serialized.test =
      'test' in objective && objective.test !== undefined
        ? objective.test
        : (bandTarget.test ?? 'TOST');
    serialized.alpha =
      'alpha' in objective && objective.alpha !== undefined ? objective.alpha : bandTarget.alpha;
  } else {
    const direction = 'direction' in objective ? objective.direction : undefined;
    if (direction === 'maximize' || direction === 'minimize') {
      serialized.direction = direction;
    }
  }

  if (objective.weight !== undefined) {
    serialized.weight = objective.weight;
  }

  return serialized;
}

function serializeHybridObjectives(
  objectivesInput: readonly ObjectiveInput[]
): Array<string | SerializedHybridObjective> {
  const objectives: Array<string | SerializedHybridObjective> = [];

  for (const objective of objectivesInput) {
    if (typeof objective === 'string') {
      objectives.push(objective);
      continue;
    }

    objectives.push(serializeHybridObjective(objective));
  }

  return objectives;
}

function serializeSessionBudget(
  spec: NormalizedOptimizationSpec
): Record<string, unknown> | undefined {
  if (!spec.budget) {
    return undefined;
  }

  return serializeSnakeCaseObject(spec.budget) as Record<string, unknown>;
}

function serializePromotionPolicy(
  spec: NormalizedOptimizationSpec
): Record<string, unknown> | undefined {
  if (!spec.promotionPolicy) {
    return undefined;
  }
  return serializeSnakeCaseObject(spec.promotionPolicy) as Record<string, unknown>;
}

function serializeSessionConstraints(
  spec: NormalizedOptimizationSpec,
  specInput?: OptimizationSpec
): Record<string, unknown> | undefined {
  if (specInput?.constraints !== undefined && !Array.isArray(specInput.constraints)) {
    return serializeSnakeCaseObject(specInput.constraints) as Record<string, unknown>;
  }
  if (!spec.constraints) {
    return undefined;
  }

  return serializeSnakeCaseObject(spec.constraints) as Record<string, unknown>;
}

export function serializeSessionConfigurationSpace(
  configurationSpace: Record<string, ParameterDefinition>
): Record<string, Record<string, unknown>> {
  const encodeCategoricalChoices = (
    values: readonly unknown[],
    defaultValue: unknown
  ): Record<string, unknown> => {
    const requiresEncoding = values.some((value) => typeof value === 'object' && value !== null);
    if (!requiresEncoding) {
      return {
        choices: [...values],
        ...(defaultValue !== undefined ? { default: defaultValue } : {}),
      };
    }

    const valueMap = Object.fromEntries(values.map((value, index) => [`choice_${index}`, value]));
    const encodedChoices = Object.keys(valueMap);
    const encodedDefault =
      defaultValue === undefined
        ? undefined
        : encodedChoices.find((choice) => stableValueEquals(valueMap[choice], defaultValue));

    return {
      choices: encodedChoices,
      value_map: valueMap,
      ...(encodedDefault !== undefined ? { default: encodedDefault } : {}),
    };
  };

  const entries = Object.entries(configurationSpace);

  return Object.fromEntries(
    entries.map(([name, definition]) => {
      switch (definition.type) {
        case 'enum': {
          const categorical = encodeCategoricalChoices(definition.values, definition.default);
          return [
            name,
            {
              type: 'categorical',
              ...categorical,
              ...(definition.conditions !== undefined
                ? { conditions: { ...definition.conditions } }
                : {}),
            },
          ];
        }
        case 'int':
          return [
            name,
            {
              type: 'int',
              low: definition.min,
              high: definition.max,
              ...(definition.step !== undefined ? { step: definition.step } : {}),
              ...(definition.scale === 'log' ? { log: true } : {}),
              ...(definition.conditions !== undefined
                ? { conditions: { ...definition.conditions } }
                : {}),
              ...(definition.default !== undefined ? { default: definition.default } : {}),
            },
          ];
        case 'float':
          return [
            name,
            {
              type: 'float',
              low: definition.min,
              high: definition.max,
              ...(definition.step !== undefined ? { step: definition.step } : {}),
              ...(definition.scale === 'log' ? { log: true } : {}),
              ...(definition.conditions !== undefined
                ? { conditions: { ...definition.conditions } }
                : {}),
              ...(definition.default !== undefined ? { default: definition.default } : {}),
            },
          ];
        default:
          throw new ValidationError(
            `Unsupported parameter type for "${name}" in hybrid optimize().`
          );
      }
    })
  );
}

function serializeSessionCreateRequest(
  request: OptimizationSessionCreateRequest
): Record<string, unknown> {
  validateSessionCreateRequest(request);

  return {
    function_name: request.functionName,
    configuration_space: serializeSessionConfigurationSpace(request.configurationSpace),
    objectives: serializeHybridObjectives(request.objectives),
    dataset_metadata: request.datasetMetadata,
    max_trials: request.maxTrials ?? 10,
    budget: request.budget
      ? (serializeSnakeCaseObject(request.budget) as Record<string, unknown>)
      : undefined,
    constraints: request.constraints
      ? (serializeSnakeCaseObject(request.constraints) as Record<string, unknown>)
      : undefined,
    default_config: request.defaultConfig,
    promotion_policy: request.promotionPolicy
      ? (serializeSnakeCaseObject(request.promotionPolicy) as Record<string, unknown>)
      : undefined,
    optimization_strategy: request.optimizationStrategy,
    user_id: request.userId,
    billing_tier: request.billingTier,
    metadata: request.metadata,
  };
}

function serializeTrialResultInput(
  sessionId: string,
  result: OptimizationSessionTrialResultInput
): HybridSubmittedResult {
  const validated = validateSessionTrialResultInput(sessionId, result);
  return {
    session_id: validated.sessionId ?? sessionId,
    trial_id: validated.trialId,
    metrics: validated.metrics,
    duration: validated.duration,
    status: validated.status ?? 'completed',
    error_message: validated.errorMessage ?? null,
    metadata: validated.metadata ?? {},
    ...(validated.outputsSample !== undefined
      ? {
          outputs_sample: validated.outputsSample === null ? null : [...validated.outputsSample],
        }
      : {}),
  };
}

function createTrialConfigFromSuggestion(
  suggestion: HybridSessionSuggestionPayload,
  totalRows: number,
  experimentRunId: string,
  defaultConfig?: Record<string, unknown>
): TrialConfig {
  if (!Array.isArray(suggestion.dataset_subset?.indices)) {
    throw new ValidationError(
      `Hybrid optimize() suggestion "${suggestion.trial_id}" is missing dataset_subset.indices.`
    );
  }

  for (const index of suggestion.dataset_subset.indices) {
    if (!Number.isInteger(index) || index < 0 || index >= totalRows) {
      throw new ValidationError(
        `Hybrid optimize() suggestion "${suggestion.trial_id}" contains an out-of-range dataset index.`
      );
    }
  }

  return {
    trial_id: suggestion.trial_id,
    trial_number: suggestion.trial_number,
    experiment_run_id: experimentRunId,
    config: {
      ...(defaultConfig ?? {}),
      ...suggestion.config,
    },
    dataset_subset: {
      indices: [...suggestion.dataset_subset.indices],
      total: totalRows,
    },
    metadata: {
      session_id: suggestion.session_id,
      exploration_type: suggestion.exploration_type,
      priority: suggestion.priority,
      estimated_duration: suggestion.estimated_duration,
      selection_strategy: suggestion.dataset_subset.selection_strategy,
      confidence_level: suggestion.dataset_subset.confidence_level,
      estimated_representativeness: suggestion.dataset_subset.estimated_representativeness,
      ...(suggestion.dataset_subset.metadata ?? {}),
      ...(suggestion.metadata ?? {}),
    },
  };
}

function buildSubmittedResult(
  sessionId: string,
  trialId: string,
  outcome: TrialOutcome
): HybridSubmittedResult {
  if (outcome.status === 'completed') {
    return {
      session_id: sessionId,
      trial_id: trialId,
      metrics: outcome.record.metrics,
      duration: outcome.record.duration,
      status: 'completed',
      error_message: null,
      metadata: outcome.record.metadata ?? {},
    };
  }

  return {
    session_id: sessionId,
    trial_id: trialId,
    metrics: {},
    duration: 0,
    status: 'failed',
    error_message: outcome.errorMessage,
    metadata:
      outcome.status === 'cancelled'
        ? { cancelled: true }
        : outcome.status === 'timeout'
          ? { timeout: true }
          : {},
  };
}

function finalizeHybridResult(
  spec: NormalizedOptimizationSpec,
  trials: OptimizationTrialRecord[],
  stopReason: OptimizationResult['stopReason'],
  totalCostUsd: number,
  sessionId: string | undefined,
  errorMessage: string | undefined,
  metadata: Record<string, unknown> | undefined,
  finalized: HybridFinalizationResponse | undefined
): OptimizationResult {
  const orderedTrials = [...trials].sort((left, right) => left.trialNumber - right.trialNumber);
  const bestTrial = selectBestTrial(orderedTrials, spec.objectives);
  const bestMetricsParse = MetricsSchema.safeParse(finalized?.best_metrics);
  const bestConfig = isPlainObject(finalized?.best_config)
    ? finalized?.best_config
    : bestTrial?.config;
  const reporting = createReportingSummary(finalized);

  return {
    mode: 'hybrid',
    sessionId,
    bestConfig: bestConfig ?? null,
    bestMetrics: bestMetricsParse.success ? bestMetricsParse.data : (bestTrial?.metrics ?? null),
    trials: orderedTrials,
    stopReason,
    totalCostUsd,
    reporting,
    metadata,
    errorMessage,
  };
}

function toOptionalFiniteNumber(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
}

function createReportingSummary(
  finalized: HybridFinalizationResponse | undefined
): OptimizationReportingSummary | undefined {
  if (!finalized) {
    return undefined;
  }

  const reporting: OptimizationReportingSummary = {
    totalTrials: toOptionalFiniteNumber(finalized.total_trials),
    successfulTrials: toOptionalFiniteNumber(finalized.successful_trials),
    totalDuration: toOptionalFiniteNumber(finalized.total_duration),
    costSavings: toOptionalFiniteNumber(finalized.cost_savings),
    convergenceHistory: Array.isArray(finalized.convergence_history)
      ? normalizeConvergenceHistory(finalized.convergence_history)
      : undefined,
    fullHistory: Array.isArray(finalized.full_history)
      ? normalizeFullHistory(finalized.full_history)
      : undefined,
  };

  if (
    reporting.totalTrials === undefined &&
    reporting.successfulTrials === undefined &&
    reporting.totalDuration === undefined &&
    reporting.costSavings === undefined &&
    reporting.convergenceHistory === undefined &&
    reporting.fullHistory === undefined
  ) {
    return undefined;
  }

  return reporting;
}

function normalizeConvergenceHistory(
  entries: unknown[]
): readonly OptimizationConvergencePoint[] | undefined {
  const normalized = entries.filter((entry): entry is OptimizationConvergencePoint =>
    isPlainObject(entry)
  );
  return normalized.length > 0 ? normalized : undefined;
}

function normalizeFullHistory(
  entries: unknown[]
): readonly OptimizationReportingTrialHistoryEntry[] | undefined {
  const normalized = entries.filter(
    (entry): entry is OptimizationReportingTrialHistoryEntry =>
      isPlainObject(entry) &&
      typeof entry['session_id'] === 'string' &&
      typeof entry['trial_id'] === 'string' &&
      isPlainObject(entry['metrics']) &&
      typeof entry['duration'] === 'number' &&
      Number.isFinite(entry['duration']) &&
      typeof entry['status'] === 'string'
  );
  return normalized.length > 0 ? normalized : undefined;
}

interface HybridFetchResponse {
  ok: boolean;
  status: number;
  json: () => Promise<unknown>;
  text: () => Promise<string>;
}

interface HybridSessionStatusPayload {
  session_id?: string;
  status?: string;
  progress?: Record<string, unknown>;
  metadata?: Record<string, unknown>;
  [key: string]: unknown;
}

interface HybridSuccessEnvelope<T> {
  success?: boolean;
  message?: string;
  data?: T;
  meta?: Record<string, unknown>;
}

interface HybridErrorPayload {
  error?: string;
  message?: string;
  error_code?: string;
}

function redactSensitiveText(text: string, apiKey: string, apiBase: string): string {
  if (text === '') {
    return text;
  }

  const redactions = new Set<string>([apiKey, `Bearer ${apiKey}`, apiBase, `${apiBase}/sessions`]);

  let sanitized = text;
  for (const secret of redactions) {
    if (secret !== '') {
      sanitized = sanitized.split(secret).join('[REDACTED]');
    }
  }

  return sanitized
    .replace(/Authorization\s*:\s*Bearer\s+[^\s",]+/gi, '[REDACTED_AUTH_HEADER]')
    .replace(/X-API-Key\s*:\s*[^\s",]+/gi, '[REDACTED_API_KEY_HEADER]')
    .replace(/Authorization/gi, '[REDACTED_AUTH_HEADER]')
    .replace(/X-API-Key/gi, '[REDACTED_API_KEY_HEADER]')
    .replace(/Bearer\s+[A-Za-z0-9._-]{8,}/g, 'Bearer [REDACTED]');
}

function parseHybridErrorPayload(raw: string): HybridErrorPayload | null {
  try {
    const parsed = JSON.parse(raw) as unknown;
    return isPlainObject(parsed) ? (parsed as HybridErrorPayload) : null;
  } catch {
    return null;
  }
}

function classifyHybridCompatibilityError(
  method: string,
  path: string,
  status: number,
  errorText: string
): ValidationError | null {
  if (method !== 'POST' || path !== '/sessions' || status !== 400) {
    return null;
  }

  const payload = parseHybridErrorPayload(errorText);
  const message = payload?.error ?? payload?.message ?? errorText;

  if (
    message.includes(
      'Missing required fields: problem_statement, dataset, search_space, optimization_config'
    )
  ) {
    return new ValidationError(
      'Hybrid optimize() is pointed at a legacy TraiGent /sessions API that expects problem_statement, dataset, search_space, and optimization_config. This JS client requires the typed interactive session contract for backend-guided optimization.'
    );
  }

  return null;
}

class HybridSessionClient {
  constructor(
    private readonly apiBase: string,
    private readonly apiKey: string,
    private readonly requestTimeoutMs: number
  ) {}

  async createSession(
    payload: Record<string, unknown>,
    signal: AbortSignal | undefined
  ): Promise<HybridSessionCreateResponse> {
    return this.requestJson<HybridSessionCreateResponse>('POST', '/sessions', {
      body: payload,
      signal,
    });
  }

  async getNextTrial(
    sessionId: string,
    payload: Record<string, unknown>,
    signal: AbortSignal | undefined
  ): Promise<HybridNextTrialResponse> {
    return this.requestJson<HybridNextTrialResponse>('POST', `/sessions/${sessionId}/next-trial`, {
      body: payload,
      signal,
    });
  }

  async submitResult(
    sessionId: string,
    payload: HybridSubmittedResult,
    signal: AbortSignal | undefined
  ): Promise<
    HybridSubmitResultsResponse | HybridSuccessEnvelope<HybridSubmitResultsResponse> | undefined
  > {
    return this.requestJson('POST', `/sessions/${sessionId}/results`, {
      body: payload,
      signal,
    });
  }

  async finalizeSession(
    sessionId: string,
    includeFullHistory: boolean,
    signal: AbortSignal | undefined
  ): Promise<HybridFinalizationResponse> {
    const response = await this.requestJson<
      HybridFinalizationResponse | HybridSuccessEnvelope<HybridFinalizationResponse>
    >('POST', `/sessions/${sessionId}/finalize`, {
      body: {
        session_id: sessionId,
        include_full_history: includeFullHistory,
        metadata: {
          sdk: 'js',
          mode: 'hybrid',
        },
      },
      signal,
    });
    return normalizeHybridFinalizationPayload(sessionId, response);
  }

  async getSessionStatus(
    sessionId: string,
    signal: AbortSignal | undefined
  ): Promise<HybridSessionStatusPayload> {
    return this.requestJson<HybridSessionStatusPayload>('GET', `/sessions/${sessionId}`, {
      signal,
    });
  }

  async listSessions(
    pattern: string | undefined,
    status: string | undefined,
    signal: AbortSignal | undefined
  ): Promise<HybridSessionListPayload> {
    const params = new URLSearchParams();
    if (pattern) {
      params.set('pattern', pattern);
    }
    if (status) {
      params.set('status', status);
    }
    const query = params.toString();
    const suffix = query !== '' ? `?${query}` : '';
    return this.requestJson<HybridSessionListPayload>('GET', `/sessions${suffix}`, {
      signal,
    });
  }

  async deleteSession(
    sessionId: string,
    cascade: boolean,
    signal: AbortSignal | undefined
  ): Promise<OptimizationSessionDeleteResponse> {
    const encodedCascade = cascade ? 'true' : 'false';
    const response = await this.requestJson<
      OptimizationSessionDeleteResponse | HybridSuccessEnvelope<Record<string, unknown>> | undefined
    >('DELETE', `/sessions/${sessionId}?cascade=${encodedCascade}`, { signal });
    return normalizeSessionDeleteResponse(sessionId, response);
  }

  async getServiceStatus(
    signal: AbortSignal | undefined
  ): Promise<OptimizationServiceStatusResponse> {
    return this.requestUrlJson<OptimizationServiceStatusResponse>(
      'GET',
      `${deriveServiceBaseUrl(this.apiBase)}/health`,
      { signal }
    );
  }

  private async requestJson<T>(
    method: string,
    path: string,
    options: {
      body?: unknown;
      signal?: AbortSignal;
    }
  ): Promise<T> {
    return this.requestUrlJson<T>(method, `${this.apiBase}${path}`, options);
  }

  private async requestUrlJson<T>(
    method: string,
    url: string,
    options: {
      body?: unknown;
      signal?: AbortSignal;
    }
  ): Promise<T> {
    const controller = new AbortController();
    const listeners: Array<() => void> = [];
    let timeoutId: NodeJS.Timeout | undefined;
    let timedOut = false;
    const requestPath = new URL(url).pathname.replace(/^\/api\/v1/, '') || '/';

    try {
      if (options.signal) {
        const onAbort = () => controller.abort();
        if (options.signal.aborted) {
          throw new CancelledError('Optimization cancelled');
        }
        options.signal.addEventListener('abort', onAbort, { once: true });
        listeners.push(() => options.signal?.removeEventListener('abort', onAbort));
      }

      timeoutId = setTimeout(() => {
        timedOut = true;
        controller.abort();
      }, this.requestTimeoutMs);

      const response = (await fetch(url, {
        method,
        headers: {
          // Some backend routes accept bearer auth while others still validate X-API-Key.
          // Send both for compatibility across the current typed-session surface.
          Authorization: `Bearer ${this.apiKey}`,
          'X-API-Key': this.apiKey,
          'Content-Type': 'application/json',
        },
        body: options.body === undefined ? undefined : JSON.stringify(options.body),
        signal: controller.signal,
      })) as HybridFetchResponse;

      if (!response.ok) {
        const rawErrorText = await response.text();
        const errorText = redactSensitiveText(rawErrorText, this.apiKey, this.apiBase);
        const compatibilityError = classifyHybridCompatibilityError(
          method,
          requestPath,
          response.status,
          errorText
        );
        if (compatibilityError) {
          throw compatibilityError;
        }
        throw new Error(
          `Hybrid optimize() request failed (${method} ${requestPath}) with HTTP ${response.status}: ${errorText || 'No response body'}`
        );
      }

      if (response.status === 204) {
        return undefined as T;
      }

      return (await response.json()) as T;
    } catch (error) {
      if (error instanceof CancelledError) {
        throw error;
      }

      if (timedOut) {
        throw new TimeoutError(
          `Hybrid optimize() request timeout (${method} ${requestPath})`,
          this.requestTimeoutMs
        );
      }

      if (options.signal?.aborted || (error instanceof Error && error.name === 'AbortError')) {
        throw new CancelledError('Optimization cancelled');
      }

      throw error;
    } finally {
      if (timeoutId) {
        clearTimeout(timeoutId);
      }
      for (const removeListener of listeners) {
        removeListener();
      }
    }
  }
}

function normalizeServiceStatusResponse(
  payload: OptimizationServiceStatusResponse
): OptimizationServiceStatusResponse {
  return {
    ...payload,
    status: typeof payload.status === 'string' ? payload.status : 'unknown',
    error: typeof payload.error === 'string' ? payload.error : undefined,
  };
}

function normalizeOptimizationSessionLifecycleStatus(value: unknown): string | undefined {
  if (typeof value !== 'string') {
    return undefined;
  }

  const normalized = value.trim().toLowerCase();
  switch (normalized) {
    case 'pending':
    case 'queued':
    case 'created':
    case 'active':
    case 'paused':
    case 'running':
    case 'completed':
    case 'failed':
    case 'cancelled':
    case 'expired':
      return normalized;
    default:
      return value;
  }
}

function normalizeSessionStatusResponse(
  sessionId: string,
  payload: HybridSessionStatusPayload | HybridSuccessEnvelope<HybridSessionStatusPayload>
): OptimizationSessionStatusResponse {
  const source = unwrapSuccessEnvelope(payload);
  const metadata = isPlainObject(source['metadata'])
    ? (source['metadata'] as OptimizationSessionStatusMetadata)
    : undefined;
  const createdAt =
    source['created_at'] ??
    (metadata && 'created_at' in metadata ? metadata.created_at : undefined);
  const functionName =
    typeof source['function_name'] === 'string'
      ? source['function_name']
      : typeof metadata?.function_name === 'string'
        ? metadata.function_name
        : undefined;
  const datasetSize =
    typeof source['dataset_size'] === 'number' && Number.isFinite(source['dataset_size'])
      ? source['dataset_size']
      : typeof metadata?.dataset_size === 'number' && Number.isFinite(metadata.dataset_size)
        ? metadata.dataset_size
        : undefined;
  const objectives = Array.isArray(source['objectives'])
    ? source['objectives'].filter((objective): objective is string => typeof objective === 'string')
    : Array.isArray(metadata?.objectives)
      ? metadata.objectives.filter(
          (objective): objective is string => typeof objective === 'string'
        )
      : undefined;
  const experimentId =
    typeof source['experiment_id'] === 'string'
      ? source['experiment_id']
      : typeof metadata?.experiment_id === 'string'
        ? metadata.experiment_id
        : undefined;
  const experimentRunId =
    typeof source['experiment_run_id'] === 'string'
      ? source['experiment_run_id']
      : typeof metadata?.experiment_run_id === 'string'
        ? metadata.experiment_run_id
        : undefined;
  return {
    ...source,
    sessionId:
      typeof source['session_id'] === 'string' && source['session_id'].length > 0
        ? source['session_id']
        : sessionId,
    status: normalizeOptimizationSessionLifecycleStatus(source['status']),
    progress: normalizeSessionProgress(source['progress']),
    createdAt:
      typeof createdAt === 'string' || typeof createdAt === 'number' ? createdAt : undefined,
    functionName,
    datasetSize,
    objectives,
    experimentId,
    experimentRunId,
    metadata,
  };
}

function normalizeSessionCreationResponse(
  payload: HybridSessionCreateResponse | HybridSuccessEnvelope<HybridSessionCreateResponse>
): OptimizationSessionCreationResponse {
  const source = unwrapSuccessEnvelope(payload);
  const sessionId = source['session_id'];
  if (typeof sessionId !== 'string' || sessionId.length === 0) {
    throw new ValidationError('Session creation response is missing a valid session_id.');
  }

  return {
    ...source,
    sessionId,
    status: normalizeOptimizationSessionLifecycleStatus(source['status']),
    optimizationStrategy: isPlainObject(source['optimization_strategy'])
      ? (source['optimization_strategy'] as Record<string, unknown>)
      : undefined,
    estimatedDuration: toOptionalFiniteNumber(source['estimated_duration']),
    billingEstimate: isPlainObject(source['billing_estimate'])
      ? (source['billing_estimate'] as Record<string, unknown>)
      : undefined,
    metadata: isPlainObject(source['metadata'])
      ? (source['metadata'] as Record<string, unknown>)
      : undefined,
  };
}

function normalizeSessionDatasetSubset(value: unknown): OptimizationSessionDatasetSubset {
  if (!isPlainObject(value) || !Array.isArray(value['indices'])) {
    throw new ValidationError(
      'Next-trial response suggestion is missing a valid dataset_subset.indices array.'
    );
  }

  return {
    indices: value['indices'].filter((index): index is number => Number.isInteger(index)),
    selectionStrategy:
      typeof value['selection_strategy'] === 'string' ? value['selection_strategy'] : undefined,
    confidenceLevel: toOptionalFiniteNumber(value['confidence_level']),
    estimatedRepresentativeness: toOptionalFiniteNumber(value['estimated_representativeness']),
    metadata: isPlainObject(value['metadata'])
      ? (value['metadata'] as Record<string, unknown>)
      : undefined,
  };
}

function normalizeSessionTrialSuggestion(value: unknown): OptimizationSessionTrialSuggestion {
  if (!isPlainObject(value)) {
    throw new ValidationError('Next-trial response suggestion must be an object.');
  }

  if (
    typeof value['trial_id'] !== 'string' ||
    value['trial_id'].length === 0 ||
    typeof value['session_id'] !== 'string' ||
    value['session_id'].length === 0 ||
    !Number.isInteger(value['trial_number']) ||
    !isPlainObject(value['config'])
  ) {
    throw new ValidationError('Next-trial response suggestion is missing required trial fields.');
  }

  return {
    trialId: value['trial_id'],
    sessionId: value['session_id'],
    trialNumber: value['trial_number'] as number,
    config: value['config'] as TrialConfig['config'],
    datasetSubset: normalizeSessionDatasetSubset(value['dataset_subset']),
    explorationType:
      typeof value['exploration_type'] === 'string' ? value['exploration_type'] : undefined,
    priority:
      typeof value['priority'] === 'number' && Number.isFinite(value['priority'])
        ? value['priority']
        : undefined,
    estimatedDuration: toOptionalFiniteNumber(value['estimated_duration']),
    metadata: isPlainObject(value['metadata'])
      ? (value['metadata'] as Record<string, unknown>)
      : undefined,
  };
}

function normalizeNextTrialResponse(
  payload: HybridNextTrialResponse | HybridSuccessEnvelope<HybridNextTrialResponse>
): OptimizationSessionNextTrialResponse {
  const source = unwrapSuccessEnvelope(payload);
  const suggestionValue = source['suggestion'];

  return {
    ...source,
    suggestion:
      suggestionValue === null || suggestionValue === undefined
        ? null
        : normalizeSessionTrialSuggestion(suggestionValue),
    shouldContinue:
      typeof source['should_continue'] === 'boolean'
        ? source['should_continue']
        : typeof source['shouldContinue'] === 'boolean'
          ? source['shouldContinue']
          : false,
    reason:
      typeof source['reason'] === 'string' || source['reason'] === null
        ? (source['reason'] as string | null)
        : undefined,
    stopReason:
      typeof source['stop_reason'] === 'string' || source['stop_reason'] === null
        ? (source['stop_reason'] as string | null)
        : typeof source['stopReason'] === 'string' || source['stopReason'] === null
          ? (source['stopReason'] as string | null)
          : undefined,
    session_status: normalizeOptimizationSessionLifecycleStatus(
      typeof source['session_status'] === 'string'
        ? source['session_status']
        : typeof source['sessionStatus'] === 'string'
          ? source['sessionStatus']
          : undefined
    ),
    sessionStatus: normalizeOptimizationSessionLifecycleStatus(
      typeof source['session_status'] === 'string'
        ? source['session_status']
        : typeof source['sessionStatus'] === 'string'
          ? source['sessionStatus']
          : undefined
    ),
    metadata: isPlainObject(source['metadata'])
      ? (source['metadata'] as Record<string, unknown>)
      : undefined,
  };
}

function normalizeSessionListResponse(
  payload: HybridSessionListPayload | HybridSuccessEnvelope<HybridSessionListPayload>
): OptimizationSessionListResponse {
  const source = unwrapSuccessEnvelope(payload);
  const rawSessions = Array.isArray(source['sessions']) ? source['sessions'] : [];
  const sessions = rawSessions.flatMap((entry) => {
    if (!isPlainObject(entry)) {
      return [];
    }

    const record = entry as Record<string, unknown>;
    const entrySessionId =
      typeof record['session_id'] === 'string' && record['session_id'].length > 0
        ? record['session_id']
        : typeof record['sessionId'] === 'string' && record['sessionId'].length > 0
          ? record['sessionId']
          : undefined;

    if (!entrySessionId) {
      return [];
    }

    return [normalizeSessionStatusResponse(entrySessionId, record)];
  });

  return {
    ...source,
    sessions,
    total:
      typeof source['total'] === 'number' && Number.isFinite(source['total'])
        ? source['total']
        : sessions.length,
  };
}

function normalizeSessionDeleteResponse(
  sessionId: string,
  payload:
    | OptimizationSessionDeleteResponse
    | HybridSuccessEnvelope<Record<string, unknown>>
    | undefined
): OptimizationSessionDeleteResponse {
  if (!payload) {
    return {
      success: true,
      sessionId,
    };
  }

  const source = unwrapSuccessEnvelope(payload);
  const deleted =
    typeof source['deleted'] === 'boolean'
      ? source['deleted']
      : typeof source['success'] === 'boolean'
        ? source['success']
        : true;
  const normalizedSessionId =
    typeof source['session_id'] === 'string'
      ? source['session_id']
      : typeof source['sessionId'] === 'string'
        ? source['sessionId']
        : sessionId;
  const message =
    typeof payload.message === 'string'
      ? payload.message
      : typeof source['message'] === 'string'
        ? source['message']
        : undefined;
  const metadata = isPlainObject(source['metadata'])
    ? (source['metadata'] as Record<string, unknown>)
    : undefined;

  return {
    ...(isPlainObject(source) ? source : {}),
    success: deleted,
    deleted,
    cascade: typeof source['cascade'] === 'boolean' ? source['cascade'] : undefined,
    sessionId: normalizedSessionId,
    message,
    metadata,
  };
}

function normalizeHybridFinalizationPayload(
  sessionId: string,
  payload: HybridFinalizationResponse | HybridSuccessEnvelope<HybridFinalizationResponse>
): HybridFinalizationResponse {
  const source = unwrapSuccessEnvelope(payload);

  return {
    ...source,
    session_id:
      typeof source['session_id'] === 'string' && source['session_id'].length > 0
        ? source['session_id']
        : sessionId,
    best_config: isPlainObject(source['best_config'])
      ? (source['best_config'] as Record<string, unknown>)
      : undefined,
    best_metrics: isPlainObject(source['best_metrics'])
      ? (source['best_metrics'] as Record<string, unknown>)
      : undefined,
    metadata: isPlainObject(source['metadata'])
      ? (source['metadata'] as Record<string, unknown>)
      : undefined,
  };
}

function unwrapSuccessEnvelope(payload: unknown): Record<string, unknown> {
  const wrappedData = isPlainObject(
    (payload as HybridSuccessEnvelope<Record<string, unknown>>).data
  )
    ? ((payload as HybridSuccessEnvelope<Record<string, unknown>>).data as Record<string, unknown>)
    : undefined;

  return (wrappedData ?? (payload as unknown as Record<string, unknown>)) as Record<
    string,
    unknown
  >;
}

function normalizeSessionProgress(value: unknown): OptimizationSessionStatusSummary | undefined {
  if (!isPlainObject(value)) {
    return undefined;
  }

  const source = value as Record<string, unknown>;
  const progress: OptimizationSessionStatusSummary = {};

  const completed = source['completed'];
  if (typeof completed === 'number' && Number.isFinite(completed)) {
    progress.completed = completed;
  }

  const total = source['total'];
  if (typeof total === 'number' && Number.isFinite(total)) {
    progress.total = total;
  }

  const failed = source['failed'];
  if (typeof failed === 'number' && Number.isFinite(failed)) {
    progress.failed = failed;
  }

  for (const [key, nestedValue] of Object.entries(source)) {
    if (key === 'completed' || key === 'total' || key === 'failed') {
      continue;
    }
    progress[key] = nestedValue;
  }

  return Object.keys(progress).length > 0 ? progress : undefined;
}

function normalizeSessionFinalizationResponse(
  sessionId: string,
  payload: HybridFinalizationResponse | HybridSuccessEnvelope<HybridFinalizationResponse>
): OptimizationSessionFinalizationResponse {
  const finalized = normalizeHybridFinalizationPayload(sessionId, payload);
  const bestMetricsParse = MetricsSchema.safeParse(finalized.best_metrics);
  return {
    sessionId: finalized.session_id,
    bestConfig: isPlainObject(finalized.best_config) ? finalized.best_config : undefined,
    bestMetrics: bestMetricsParse.success ? bestMetricsParse.data : null,
    stopReason:
      typeof finalized.stop_reason === 'string' || finalized.stop_reason === null
        ? finalized.stop_reason
        : undefined,
    reporting: createReportingSummary(finalized),
    metadata: finalized.metadata,
  };
}

function normalizeSessionSubmitResultResponse(
  payload:
    | HybridSubmitResultsResponse
    | HybridSuccessEnvelope<HybridSubmitResultsResponse>
    | undefined
): OptimizationSessionSubmitResultResponse {
  if (!payload) {
    return { success: true };
  }

  const source = unwrapSuccessEnvelope(payload);
  const success = typeof source['success'] === 'boolean' ? source['success'] : true;

  return {
    ...source,
    success,
    continueOptimization:
      typeof source['continue_optimization'] === 'boolean'
        ? source['continue_optimization']
        : typeof source['continueOptimization'] === 'boolean'
          ? source['continueOptimization']
          : undefined,
    message: typeof source['message'] === 'string' ? source['message'] : undefined,
  };
}

function validateHybridFinalizeOptions(
  options: OptimizationSessionFinalizeOptions | undefined
): ValidatedHybridRequestOptions & { includeFullHistory: boolean } {
  if (
    options?.includeFullHistory !== undefined &&
    typeof options.includeFullHistory !== 'boolean'
  ) {
    throw new ValidationError(
      'Session finalization includeFullHistory must be a boolean when provided.'
    );
  }

  const resolved = validateHybridRequestOptions(options);
  return {
    ...resolved,
    includeFullHistory: options?.includeFullHistory ?? false,
  };
}

export async function getOptimizationSessionStatus(
  sessionId: string,
  options?: OptimizationSessionRequestOptions
): Promise<OptimizationSessionStatusResponse> {
  validateStringOption(sessionId, 'Session status requires a non-empty sessionId.');
  const resolved = validateHybridRequestOptions(options);
  const client = new HybridSessionClient(
    resolved.backendUrl,
    resolved.apiKey,
    resolved.requestTimeoutMs
  );
  const payload = await client.getSessionStatus(sessionId, resolved.signal);
  return normalizeSessionStatusResponse(sessionId, payload);
}

export async function checkOptimizationServiceStatus(
  options?: OptimizationSessionRequestOptions
): Promise<OptimizationServiceStatusResponse> {
  const resolved = validateHybridRequestOptions(options);
  const client = new HybridSessionClient(
    resolved.backendUrl,
    resolved.apiKey,
    resolved.requestTimeoutMs
  );

  try {
    const payload = await client.getServiceStatus(resolved.signal);
    return normalizeServiceStatusResponse(payload);
  } catch (error) {
    return {
      status: 'unavailable',
      error: toErrorMessage(error),
    };
  }
}

export async function createOptimizationSession(
  request: OptimizationSessionCreateRequest,
  options?: OptimizationSessionRequestOptions
): Promise<OptimizationSessionCreationResponse> {
  const resolved = validateHybridRequestOptions(options);
  const client = new HybridSessionClient(
    resolved.backendUrl,
    resolved.apiKey,
    resolved.requestTimeoutMs
  );
  const payload = await client.createSession(
    serializeSessionCreateRequest(request),
    resolved.signal
  );
  return normalizeSessionCreationResponse(payload);
}

export async function getNextOptimizationTrial(
  sessionId: string,
  options?: OptimizationSessionNextTrialOptions
): Promise<OptimizationSessionNextTrialResponse> {
  const resolved = validateHybridNextTrialOptions(sessionId, options);
  const client = new HybridSessionClient(
    resolved.backendUrl,
    resolved.apiKey,
    resolved.requestTimeoutMs
  );
  const payload = await client.getNextTrial(
    sessionId,
    {
      ...(resolved.previousResults
        ? {
            previous_results: resolved.previousResults.map((entry) =>
              serializeTrialResultInput(sessionId, entry)
            ),
          }
        : {}),
      ...(resolved.requestMetadata ? { request_metadata: resolved.requestMetadata } : {}),
    },
    resolved.signal
  );
  return normalizeNextTrialResponse(payload);
}

export async function listOptimizationSessions(
  options?: OptimizationSessionListOptions
): Promise<OptimizationSessionListResponse> {
  const resolved = validateHybridListOptions(options);
  const client = new HybridSessionClient(
    resolved.backendUrl,
    resolved.apiKey,
    resolved.requestTimeoutMs
  );
  const payload = await client.listSessions(resolved.pattern, resolved.status, resolved.signal);
  return normalizeSessionListResponse(payload);
}

export async function deleteOptimizationSession(
  sessionId: string,
  options?: OptimizationSessionDeleteOptions
): Promise<OptimizationSessionDeleteResponse> {
  validateStringOption(sessionId, 'Session deletion requires a non-empty sessionId.');
  const resolved = validateHybridRequestOptions(options);
  const client = new HybridSessionClient(
    resolved.backendUrl,
    resolved.apiKey,
    resolved.requestTimeoutMs
  );
  const response = await client.deleteSession(
    sessionId,
    options?.cascade ?? false,
    resolved.signal
  );
  return response;
}

export async function submitOptimizationTrialResult(
  sessionId: string,
  result: OptimizationSessionTrialResultInput,
  options?: OptimizationSessionRequestOptions
): Promise<OptimizationSessionSubmitResultResponse> {
  const resolved = validateHybridRequestOptions(options);
  validateStringOption(sessionId, 'Trial result submission requires a non-empty sessionId.');
  const client = new HybridSessionClient(
    resolved.backendUrl,
    resolved.apiKey,
    resolved.requestTimeoutMs
  );
  const payload = await client.submitResult(
    sessionId,
    serializeTrialResultInput(sessionId, result),
    resolved.signal
  );
  return normalizeSessionSubmitResultResponse(payload);
}

export async function finalizeOptimizationSession(
  sessionId: string,
  options?: OptimizationSessionFinalizeOptions
): Promise<OptimizationSessionFinalizationResponse> {
  validateStringOption(sessionId, 'Session finalization requires a non-empty sessionId.');
  const resolved = validateHybridFinalizeOptions(options);
  const client = new HybridSessionClient(
    resolved.backendUrl,
    resolved.apiKey,
    resolved.requestTimeoutMs
  );
  const response = await client.finalizeSession(
    sessionId,
    resolved.includeFullHistory,
    resolved.signal
  );
  return normalizeSessionFinalizationResponse(sessionId, response);
}

export async function runHybridOptimization(
  trialFn: NativeTrialFunction,
  spec: NormalizedOptimizationSpec,
  specInput: OptimizationSpec,
  rawOptions: HybridOptimizeOptions,
  functionName?: string
): Promise<OptimizationResult> {
  const options = validateHybridOptimizeOptions(rawOptions);
  const evaluationRows = await resolveEvaluationRows(spec);

  if (!Array.isArray(evaluationRows) || evaluationRows.length === 0) {
    throw new ValidationError('optimize() requires evaluation data to be a non-empty array.');
  }

  const objectives = serializeHybridObjectives(specInput.objectives);
  const configurationSpace = serializeSessionConfigurationSpace(spec.configurationSpace);
  const budget = serializeSessionBudget(spec);
  const constraints = serializeSessionConstraints(spec, specInput);
  const promotionPolicy = serializePromotionPolicy(spec);

  if (options.datasetMetadata?.['size'] !== undefined) {
    if (
      typeof options.datasetMetadata['size'] !== 'number' ||
      !Number.isFinite(options.datasetMetadata['size']) ||
      options.datasetMetadata['size'] <= 0
    ) {
      throw new ValidationError(
        'Hybrid optimize() datasetMetadata.size must be a positive number when provided.'
      );
    }
    if (options.datasetMetadata['size'] !== evaluationRows.length) {
      throw new ValidationError(
        'Hybrid optimize() datasetMetadata.size must match the loaded evaluation dataset size.'
      );
    }
  }

  const client = new HybridSessionClient(
    options.backendUrl,
    options.apiKey,
    options.requestTimeoutMs
  );
  const completedTrials: OptimizationTrialRecord[] = [];
  const previousResults: HybridSubmittedResult[] = [];
  let totalCostUsd = 0;
  let sessionId: string | undefined;
  let backendReason: string | undefined;
  let failedTrialStopReason: 'timeout' | 'error' | undefined;
  let finalization: HybridFinalizationResponse | undefined;
  let finalizationError: string | undefined;
  let stopReason: OptimizationResult['stopReason'] | undefined;
  let errorMessage: string | undefined;
  let optimizationStrategyMetadata: Record<string, unknown> | undefined;

  try {
    const createResponse = await client.createSession(
      {
        function_name:
          functionName && functionName.length > 0 ? functionName : 'anonymous_js_trial',
        configuration_space: configurationSpace,
        objectives,
        dataset_metadata: {
          size: evaluationRows.length,
          ...(options.datasetMetadata ?? {}),
        },
        max_trials: options.maxTrials,
        ...(budget ? { budget } : {}),
        ...(constraints ? { constraints } : {}),
        ...(promotionPolicy ? { promotion_policy: promotionPolicy } : {}),
        ...(spec.defaultConfig ? { default_config: { ...spec.defaultConfig } } : {}),
        optimization_strategy: {
          algorithm: 'optuna',
          ...(serializeSnakeCaseObject(options.optimizationStrategy ?? {}) as Record<
            string,
            unknown
          >),
        },
        user_id: options.userId,
        billing_tier: options.billingTier ?? 'standard',
        metadata: {
          sdk: 'js',
          mode: 'hybrid',
        },
      },
      options.signal
    );

    sessionId = createResponse.session_id;
    optimizationStrategyMetadata = createResponse.optimization_strategy;

    while (stopReason === undefined) {
      if (options.signal?.aborted) {
        stopReason = 'cancelled';
        errorMessage = 'Optimization cancelled';
        break;
      }

      const nextResponse = await client.getNextTrial(
        sessionId,
        {
          session_id: sessionId,
          // The typed backend session is authoritative and persists full trial
          // state server-side, so the client only needs to send recent results
          // as context instead of replaying the entire history every turn.
          previous_results: previousResults.slice(-5),
          request_metadata: {
            dataset_size: evaluationRows.length,
            completed_trials: previousResults.length,
          },
        },
        options.signal
      );

      if (!nextResponse.should_continue || !nextResponse.suggestion) {
        backendReason = resolveBackendStopReason(nextResponse.stop_reason, nextResponse.reason);
        stopReason = normalizeBackendStopReason(
          {
            backendReason,
            failedTrialStopReason,
          },
          previousResults.length,
          options.maxTrials
        );
        break;
      }

      const trialConfig = createTrialConfigFromSuggestion(
        nextResponse.suggestion,
        evaluationRows.length,
        sessionId,
        spec.defaultConfig
      );
      const outcome = await executeTrial(trialFn, trialConfig, options.timeoutMs, options.signal);
      const submission = buildSubmittedResult(sessionId, trialConfig.trial_id, outcome);

      if (outcome.status === 'completed') {
        totalCostUsd = updateTotalCost(spec, outcome.record, totalCostUsd);
        completedTrials.push(outcome.record);
        failedTrialStopReason = undefined;
      } else if (outcome.status === 'timeout' || outcome.status === 'error') {
        failedTrialStopReason = outcome.status;
      } else {
        stopReason = 'cancelled';
        errorMessage = outcome.errorMessage;
      }

      await client.submitResult(
        sessionId,
        submission,
        outcome.status === 'cancelled' ? undefined : options.signal
      );
      previousResults.push(submission);

      if (stopReason === 'cancelled') {
        break;
      }
    }
  } catch (error) {
    if (error instanceof ValidationError) {
      throw error;
    }

    if (error instanceof CancelledError || error instanceof TrialCancelledError) {
      stopReason = 'cancelled';
      errorMessage = toErrorMessage(error);
    } else if (error instanceof TimeoutError) {
      stopReason = 'timeout';
      errorMessage = error.message;
    } else {
      stopReason = 'error';
      errorMessage = toErrorMessage(error);
    }
  } finally {
    if (sessionId) {
      try {
        finalization = await client.finalizeSession(
          sessionId,
          options.includeFullHistory ?? false,
          undefined
        );
        backendReason = resolveBackendStopReason(finalization.stop_reason, backendReason);
        if ((!stopReason || stopReason === 'completed') && backendReason) {
          stopReason = normalizeBackendStopReason(
            {
              backendReason,
              failedTrialStopReason,
            },
            previousResults.length,
            options.maxTrials
          );
        }
      } catch (error) {
        finalizationError = toErrorMessage(error);
        if (!stopReason) {
          stopReason = error instanceof TimeoutError ? 'timeout' : 'error';
          errorMessage = finalizationError;
        }
      }
    }
  }

  return finalizeHybridResult(
    spec,
    completedTrials,
    stopReason ?? 'completed',
    totalCostUsd,
    sessionId,
    errorMessage,
    {
      backendReason,
      finalizeError: finalizationError,
      optimizationStrategy: optimizationStrategyMetadata,
      finalization: finalization?.metadata,
    },
    finalization
  );
}
