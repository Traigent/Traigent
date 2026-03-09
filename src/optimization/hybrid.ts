import { TrialCancelledError, TrialContext } from "../core/context.js";
import {
  CancelledError,
  TimeoutError,
  ValidationError,
} from "../core/errors.js";
import {
  MetricsSchema,
  type Metrics,
  type TrialConfig,
} from "../dtos/trial.js";
import { stableValueEquals } from "./stable-value.js";
import type {
  BandedObjectiveDefinition,
  HybridOptimizeOptions,
  NativeTrialFunctionResult,
  NormalizedObjectiveDefinition,
  NormalizedOptimizationSpec,
  ObjectiveDefinition,
  OptimizationResult,
  OptimizationSpec,
  OptimizationTrialRecord,
} from "./types.js";

type NativeTrialFunction = (
  trialConfig: TrialConfig,
) => Promise<NativeTrialFunctionResult>;

type TrialFailureStatus = "timeout" | "error" | "cancelled";

type TrialOutcome =
  | {
      status: "completed";
      record: OptimizationTrialRecord;
    }
  | {
      status: TrialFailureStatus;
      trialNumber: number;
      errorMessage: string;
    };

interface ValidatedHybridOptimizeOptions extends HybridOptimizeOptions {
  mode?: "hybrid";
  algorithm: "optuna";
  backendUrl: string;
  apiKey: string;
  requestTimeoutMs: number;
}

interface SerializedHybridObjective {
  metric: string;
  direction?: "maximize" | "minimize";
  band?: {
    low: number;
    high: number;
  };
  test?: "TOST";
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

interface HybridSubmittedResult {
  session_id: string;
  trial_id: string;
  metrics: Metrics;
  duration: number;
  status: "completed" | "failed";
  error_message: string | null;
  metadata: Record<string, unknown>;
}

interface HybridStopContext {
  backendReason?: string;
  failedTrialStopReason?: "timeout" | "error";
}

const DEFAULT_REQUEST_TIMEOUT_MS = 30_000;
function isPlainObject(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function toSnakeCaseKey(key: string): string {
  return key
    .replace(/([a-z0-9])([A-Z])/g, "$1_$2")
    .replace(/[-\s]+/g, "_")
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
    ]),
  );
}

function toErrorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }
  return String(error);
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

async function resolveEvaluationRows(
  spec: NormalizedOptimizationSpec,
): Promise<readonly unknown[]> {
  if (spec.evaluation?.data) return spec.evaluation.data;
  if (spec.evaluation?.loadData) return spec.evaluation.loadData();

  throw new ValidationError(
    "optimize() requires spec.evaluation.data or spec.evaluation.loadData.",
  );
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
              reject(new TimeoutError("Trial timeout", timeoutMs));
            }, timeoutMs);
          });

    const cancelPromise =
      signal === undefined
        ? undefined
        : new Promise<never>((_, reject) => {
            const onAbort = () => {
              controller.abort();
              reject(new CancelledError("Optimization cancelled"));
            };

            if (signal.aborted) {
              onAbort();
              return;
            }

            signal.addEventListener("abort", onAbort, { once: true });
            listeners.push(() => signal.removeEventListener("abort", onAbort));
          });

    const rawResult = await Promise.race(
      [trialPromise, timeoutPromise, cancelPromise].filter(
        (value): value is Promise<NativeTrialFunctionResult> =>
          value !== undefined,
      ),
    );

    if (!rawResult || typeof rawResult !== "object") {
      throw new ValidationError(
        "optimize() trial function must resolve to an object containing metrics.",
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
      status: "completed",
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
        status: "timeout",
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
        status: "cancelled",
        trialNumber: trialConfig.trial_number,
        errorMessage: toErrorMessage(error),
      };
    }
    return {
      status: "error",
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
  maxTrials: number,
): OptimizationResult["stopReason"] {
  const normalized = context.backendReason
    ?.trim()
    .toLowerCase()
    .replace(/[_-]+/g, " ");

  if (normalized) {
    const exactMappings: Record<string, OptimizationResult["stopReason"]> = {
      "budget exhausted": "budget",
      "max trials reached": "maxTrials",
      "max wallclock reached": "timeout",
      "search complete": "completed",
      finalized: "completed",
    };

    if (normalized in exactMappings) {
      return exactMappings[normalized]!;
    }

    if (normalized.includes("cancel")) return "cancelled";
    if (normalized.includes("timeout")) return "timeout";
    if (normalized.includes("wallclock") || normalized.includes("elapsed")) {
      return "timeout";
    }
    if (normalized.includes("budget") || normalized.includes("cost")) {
      return "budget";
    }
    if (
      normalized.includes("plateau") ||
      normalized.includes("converg") ||
      normalized.includes("stagnat")
    ) {
      return "plateau";
    }
    if (
      normalized.includes("max trial") ||
      normalized.includes("trial limit") ||
      normalized.includes("limit reached")
    ) {
      return "maxTrials";
    }
    if (
      normalized.includes("complete") ||
      normalized.includes("exhaust") ||
      normalized.includes("no more")
    ) {
      return "completed";
    }
  }

  if (context.failedTrialStopReason) {
    return context.failedTrialStopReason;
  }

  return trialCount >= maxTrials ? "maxTrials" : "completed";
}

function resolveBackendStopReason(
  value: string | null | undefined,
  fallback: string | null | undefined,
): string | undefined {
  if (typeof value === "string" && value.trim().length > 0) {
    return value;
  }
  if (typeof fallback === "string" && fallback.trim().length > 0) {
    return fallback;
  }
  return undefined;
}

function updateTotalCost(
  spec: NormalizedOptimizationSpec,
  trial: OptimizationTrialRecord,
  totalCostUsd: number,
): number {
  if (spec.budget?.maxCostUsd === undefined) {
    const trialCost = trial.metrics["cost"];
    return typeof trialCost === "number" && Number.isFinite(trialCost)
      ? totalCostUsd + trialCost
      : totalCostUsd;
  }

  const costMetric = trial.metrics["cost"];
  if (typeof costMetric !== "number" || !Number.isFinite(costMetric)) {
    throw new ValidationError(
      "budget.maxCostUsd requires every trial to return numeric metrics.cost.",
    );
  }

  return totalCostUsd + costMetric;
}

function getObjectiveMetric(
  trial: OptimizationTrialRecord,
  objective: NormalizedObjectiveDefinition,
): number {
  const value = trial.metrics[objective.metric];
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new ValidationError(
      `Trial "${trial.trialId}" is missing numeric metric "${objective.metric}".`,
    );
  }
  return value;
}

export function objectiveScoreValue(
  value: number,
  objective: NormalizedObjectiveDefinition,
): number {
  if (objective.kind === "banded") {
    if (value < objective.band.low) {
      return -(objective.band.low - value);
    }
    if (value > objective.band.high) {
      return -(value - objective.band.high);
    }
    return 0;
  }

  return objective.direction === "minimize" ? -value : value;
}

function selectBestTrial(
  trials: OptimizationTrialRecord[],
  objectives: readonly NormalizedObjectiveDefinition[],
): OptimizationTrialRecord | null {
  if (trials.length === 0) return null;

  const ranges = objectives.map((objective) => {
    const values = trials.map((trial) =>
      objectiveScoreValue(getObjectiveMetric(trial, objective), objective),
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

export function normalizeBackendApiBase(rawUrl: string): string {
  if (typeof rawUrl !== "string" || rawUrl.trim().length === 0) {
    throw new ValidationError(
      "Hybrid optimize() requires backendUrl or TRAIGENT_BACKEND_URL / TRAIGENT_API_URL.",
    );
  }

  let url: URL;
  try {
    url = new URL(rawUrl);
  } catch {
    throw new ValidationError(
      `Hybrid optimize() received an invalid backendUrl: "${rawUrl}".`,
    );
  }

  const pathname = url.pathname.replace(/\/+$/, "");
  if (pathname === "" || pathname === "/") {
    url.pathname = "/api/v1";
  } else if (pathname === "/api/v1") {
    url.pathname = "/api/v1";
  } else {
    throw new ValidationError(
      "Hybrid optimize() backendUrl must be a backend origin or the /api/v1 base URL.",
    );
  }

  url.search = "";
  url.hash = "";
  return url.toString().replace(/\/$/, "");
}

function resolveBackendUrl(options: HybridOptimizeOptions): string {
  return (
    options.backendUrl ??
    process.env["TRAIGENT_BACKEND_URL"] ??
    process.env["TRAIGENT_API_URL"] ??
    ""
  );
}

function resolveApiKey(options: HybridOptimizeOptions): string {
  return options.apiKey ?? process.env["TRAIGENT_API_KEY"] ?? "";
}

function validateStringOption(
  value: unknown,
  message: string,
): asserts value is string {
  if (typeof value !== "string" || value.trim().length === 0) {
    throw new ValidationError(message);
  }
}

function validateHybridOptimizeOptions(
  options: HybridOptimizeOptions,
): ValidatedHybridOptimizeOptions {
  if (!options || typeof options !== "object") {
    throw new ValidationError("optimize() options are required.");
  }

  if (options.mode !== undefined && options.mode !== "hybrid") {
    throw new ValidationError(
      'Hybrid optimize() only supports mode: "hybrid". Use mode: "native" for local execution.',
    );
  }

  if (options.algorithm !== "optuna") {
    throw new ValidationError(
      'Hybrid optimize() requires algorithm "optuna". Set mode: "native" to use grid, random, or bayesian locally.',
    );
  }

  if (!Number.isInteger(options.maxTrials) || options.maxTrials <= 0) {
    throw new ValidationError(
      "Hybrid optimize() requires maxTrials to be a positive integer.",
    );
  }

  if (
    options.timeoutMs !== undefined &&
    (!Number.isInteger(options.timeoutMs) || options.timeoutMs <= 0)
  ) {
    throw new ValidationError(
      "Hybrid optimize() timeoutMs must be a positive integer when provided.",
    );
  }

  if (
    options.requestTimeoutMs !== undefined &&
    (!Number.isInteger(options.requestTimeoutMs) ||
      options.requestTimeoutMs <= 0)
  ) {
    throw new ValidationError(
      "Hybrid optimize() requestTimeoutMs must be a positive integer when provided.",
    );
  }

  if (options.userId !== undefined) {
    validateStringOption(
      options.userId,
      "Hybrid optimize() userId must be non-empty.",
    );
  }

  if (options.billingTier !== undefined) {
    validateStringOption(
      options.billingTier,
      "Hybrid optimize() billingTier must be non-empty.",
    );
  }

  if (
    options.optimizationStrategy !== undefined &&
    !isPlainObject(options.optimizationStrategy)
  ) {
    throw new ValidationError(
      "Hybrid optimize() optimizationStrategy must be an object when provided.",
    );
  }

  if (
    options.datasetMetadata !== undefined &&
    !isPlainObject(options.datasetMetadata)
  ) {
    throw new ValidationError(
      "Hybrid optimize() datasetMetadata must be an object when provided.",
    );
  }

  const nativeOnlyKeys = [
    "trialConcurrency",
    "plateau",
    "checkpoint",
    "randomSeed",
  ] as const;
  for (const key of nativeOnlyKeys) {
    if (key in (options as unknown as Record<string, unknown>)) {
      throw new ValidationError(
        `Hybrid optimize() does not support native option "${key}".`,
      );
    }
  }

  const unresolvedBackendUrl = resolveBackendUrl(options);
  validateStringOption(
    unresolvedBackendUrl,
    "Hybrid optimize() requires backendUrl, TRAIGENT_BACKEND_URL, or TRAIGENT_API_URL.",
  );
  const backendUrl = normalizeBackendApiBase(unresolvedBackendUrl);
  const apiKey = resolveApiKey(options);
  validateStringOption(
    apiKey,
    "Hybrid optimize() requires apiKey or TRAIGENT_API_KEY.",
  );

  return {
    ...options,
    backendUrl,
    apiKey,
    requestTimeoutMs: options.requestTimeoutMs ?? DEFAULT_REQUEST_TIMEOUT_MS,
  };
}

function serializeHybridObjective(
  objective: ObjectiveDefinition | BandedObjectiveDefinition,
): SerializedHybridObjective {
  const serialized: SerializedHybridObjective =
    "band" in objective
      ? {
          metric: objective.metric,
          band: {
            low:
              "low" in objective.band && typeof objective.band.low === "number"
                ? objective.band.low
                : Number(objective.band.center) - Number(objective.band.tol),
            high:
              "high" in objective.band && typeof objective.band.high === "number"
                ? objective.band.high
                : Number(objective.band.center) + Number(objective.band.tol),
          },
          test: objective.test ?? "TOST",
          alpha: objective.alpha,
        }
      : {
          metric: objective.metric,
          direction: objective.direction,
        };

  if (objective.weight !== undefined) {
    serialized.weight = objective.weight;
  }

  return serialized;
}

function serializeHybridObjectives(
  specInput: OptimizationSpec,
): Array<string | SerializedHybridObjective> {
  const objectives: Array<string | SerializedHybridObjective> = [];

  for (const objective of specInput.objectives) {
    if (typeof objective === "string") {
      objectives.push(objective);
      continue;
    }

    objectives.push(serializeHybridObjective(objective));
  }

  return objectives;
}

function serializeSessionBudget(
  spec: NormalizedOptimizationSpec,
): Record<string, unknown> | undefined {
  if (!spec.budget) {
    return undefined;
  }

  return serializeSnakeCaseObject(spec.budget) as Record<string, unknown>;
}

function serializePromotionPolicy(
  spec: NormalizedOptimizationSpec,
): Record<string, unknown> | undefined {
  if (!spec.promotionPolicy) {
    return undefined;
  }
  return serializeSnakeCaseObject(spec.promotionPolicy) as Record<string, unknown>;
}

function serializeSessionConstraints(
  spec: NormalizedOptimizationSpec,
): Record<string, unknown> | undefined {
  if (!spec.constraints) {
    return undefined;
  }

  return serializeSnakeCaseObject(spec.constraints) as Record<string, unknown>;
}

export function serializeSessionConfigurationSpace(
  spec: NormalizedOptimizationSpec,
): Record<string, Record<string, unknown>> {
  const encodeCategoricalChoices = (
    values: readonly unknown[],
    defaultValue: unknown,
  ): Record<string, unknown> => {
    const requiresEncoding = values.some(
      (value) => typeof value === "object" && value !== null,
    );
    if (!requiresEncoding) {
      return {
        choices: [...values],
        ...(defaultValue !== undefined ? { default: defaultValue } : {}),
      };
    }

    const valueMap = Object.fromEntries(
      values.map((value, index) => [`choice_${index}`, value]),
    );
    const encodedChoices = Object.keys(valueMap);
    const encodedDefault =
      defaultValue === undefined
        ? undefined
        : encodedChoices.find(
            (choice) => stableValueEquals(valueMap[choice], defaultValue),
          );

    return {
      choices: encodedChoices,
      value_map: valueMap,
      ...(encodedDefault !== undefined ? { default: encodedDefault } : {}),
    };
  };

  const entries = Object.entries(spec.configurationSpace);

  return Object.fromEntries(
    entries.map(([name, definition]) => {
      switch (definition.type) {
        case "enum": {
          const categorical = encodeCategoricalChoices(
            definition.values,
            definition.default,
          );
          return [
            name,
            {
              type: "categorical",
              ...categorical,
              ...(definition.conditions !== undefined
                ? { conditions: { ...definition.conditions } }
                : {}),
            },
          ];
        }
        case "int":
          return [
            name,
            {
              type: "int",
              low: definition.min,
              high: definition.max,
              ...(definition.step !== undefined
                ? { step: definition.step }
                : {}),
              ...(definition.scale === "log" ? { log: true } : {}),
              ...(definition.conditions !== undefined
                ? { conditions: { ...definition.conditions } }
                : {}),
              ...(definition.default !== undefined
                ? { default: definition.default }
                : {}),
            },
          ];
        case "float":
          return [
            name,
            {
              type: "float",
              low: definition.min,
              high: definition.max,
              ...(definition.step !== undefined
                ? { step: definition.step }
                : {}),
              ...(definition.scale === "log" ? { log: true } : {}),
              ...(definition.conditions !== undefined
                ? { conditions: { ...definition.conditions } }
                : {}),
              ...(definition.default !== undefined
                ? { default: definition.default }
                : {}),
            },
          ];
        default:
          throw new ValidationError(
            `Unsupported parameter type for "${name}" in hybrid optimize().`,
          );
      }
    }),
  );
}

function createTrialConfigFromSuggestion(
  suggestion: HybridSessionSuggestionPayload,
  totalRows: number,
  experimentRunId: string,
  defaultConfig?: Record<string, unknown>,
): TrialConfig {
  if (!Array.isArray(suggestion.dataset_subset?.indices)) {
    throw new ValidationError(
      `Hybrid optimize() suggestion "${suggestion.trial_id}" is missing dataset_subset.indices.`,
    );
  }

  for (const index of suggestion.dataset_subset.indices) {
    if (!Number.isInteger(index) || index < 0 || index >= totalRows) {
      throw new ValidationError(
        `Hybrid optimize() suggestion "${suggestion.trial_id}" contains an out-of-range dataset index.`,
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
      estimated_representativeness:
        suggestion.dataset_subset.estimated_representativeness,
      ...(suggestion.dataset_subset.metadata ?? {}),
      ...(suggestion.metadata ?? {}),
    },
  };
}

function buildSubmittedResult(
  sessionId: string,
  trialId: string,
  outcome: TrialOutcome,
): HybridSubmittedResult {
  if (outcome.status === "completed") {
    return {
      session_id: sessionId,
      trial_id: trialId,
      metrics: outcome.record.metrics,
      duration: outcome.record.duration,
      status: "completed",
      error_message: null,
      metadata: outcome.record.metadata ?? {},
    };
  }

  return {
    session_id: sessionId,
    trial_id: trialId,
    metrics: {},
    duration: 0,
    status: "failed",
    error_message: outcome.errorMessage,
    metadata:
      outcome.status === "cancelled"
        ? { cancelled: true }
        : outcome.status === "timeout"
          ? { timeout: true }
          : {},
  };
}

function finalizeHybridResult(
  spec: NormalizedOptimizationSpec,
  trials: OptimizationTrialRecord[],
  stopReason: OptimizationResult["stopReason"],
  totalCostUsd: number,
  sessionId: string | undefined,
  errorMessage: string | undefined,
  metadata: Record<string, unknown> | undefined,
  finalized: HybridFinalizationResponse | undefined,
): OptimizationResult {
  const orderedTrials = [...trials].sort(
    (left, right) => left.trialNumber - right.trialNumber,
  );
  const bestTrial = selectBestTrial(orderedTrials, spec.objectives);
  const bestMetricsParse = MetricsSchema.safeParse(finalized?.best_metrics);
  const bestConfig = isPlainObject(finalized?.best_config)
    ? finalized?.best_config
    : bestTrial?.config;

  return {
    mode: "hybrid",
    sessionId,
    bestConfig: bestConfig ?? null,
    bestMetrics: bestMetricsParse.success
      ? bestMetricsParse.data
      : (bestTrial?.metrics ?? null),
    trials: orderedTrials,
    stopReason,
    totalCostUsd,
    metadata,
    errorMessage,
  };
}

interface HybridFetchResponse {
  ok: boolean;
  status: number;
  json: () => Promise<unknown>;
  text: () => Promise<string>;
}

interface HybridErrorPayload {
  error?: string;
  message?: string;
  error_code?: string;
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
  errorText: string,
): ValidationError | null {
  if (method !== "POST" || path !== "/sessions" || status !== 400) {
    return null;
  }

  const payload = parseHybridErrorPayload(errorText);
  const message = payload?.error ?? payload?.message ?? errorText;

  if (
    message.includes(
      "Missing required fields: problem_statement, dataset, search_space, optimization_config",
    )
  ) {
    return new ValidationError(
      "Hybrid optimize() is pointed at a legacy TraiGent /sessions API that expects problem_statement, dataset, search_space, and optimization_config. This JS client requires the typed interactive session contract for backend-guided optimization.",
    );
  }

  return null;
}

class HybridSessionClient {
  constructor(
    private readonly apiBase: string,
    private readonly apiKey: string,
    private readonly requestTimeoutMs: number,
  ) {}

  async createSession(
    payload: Record<string, unknown>,
    signal: AbortSignal | undefined,
  ): Promise<HybridSessionCreateResponse> {
    return this.requestJson<HybridSessionCreateResponse>("POST", "/sessions", {
      body: payload,
      signal,
    });
  }

  async getNextTrial(
    sessionId: string,
    payload: Record<string, unknown>,
    signal: AbortSignal | undefined,
  ): Promise<HybridNextTrialResponse> {
    return this.requestJson<HybridNextTrialResponse>(
      "POST",
      `/sessions/${sessionId}/next-trial`,
      {
        body: payload,
        signal,
      },
    );
  }

  async submitResult(
    sessionId: string,
    payload: HybridSubmittedResult,
    signal: AbortSignal | undefined,
  ): Promise<void> {
    await this.requestJson("POST", `/sessions/${sessionId}/results`, {
      body: payload,
      signal,
    });
  }

  async finalizeSession(
    sessionId: string,
    signal: AbortSignal | undefined,
  ): Promise<HybridFinalizationResponse> {
    return this.requestJson<HybridFinalizationResponse>(
      "POST",
      `/sessions/${sessionId}/finalize`,
      {
        body: {
          session_id: sessionId,
          include_full_history: false,
          metadata: {
            sdk: "js",
            mode: "hybrid",
          },
        },
        signal,
      },
    );
  }

  private async requestJson<T>(
    method: string,
    path: string,
    options: {
      body?: unknown;
      signal?: AbortSignal;
    },
  ): Promise<T> {
    const url = `${this.apiBase}${path}`;
    const controller = new AbortController();
    const listeners: Array<() => void> = [];
    let timeoutId: NodeJS.Timeout | undefined;
    let timedOut = false;

    try {
      if (options.signal) {
        const onAbort = () => controller.abort();
        if (options.signal.aborted) {
          throw new CancelledError("Optimization cancelled");
        }
        options.signal.addEventListener("abort", onAbort, { once: true });
        listeners.push(() =>
          options.signal?.removeEventListener("abort", onAbort),
        );
      }

      timeoutId = setTimeout(() => {
        timedOut = true;
        controller.abort();
      }, this.requestTimeoutMs);

      const response = (await fetch(url, {
        method,
        headers: {
          Authorization: `Bearer ${this.apiKey}`,
          "Content-Type": "application/json",
        },
        body:
          options.body === undefined ? undefined : JSON.stringify(options.body),
        signal: controller.signal,
      })) as HybridFetchResponse;

      if (!response.ok) {
        const errorText = await response.text();
        const compatibilityError = classifyHybridCompatibilityError(
          method,
          path,
          response.status,
          errorText,
        );
        if (compatibilityError) {
          throw compatibilityError;
        }
        throw new Error(
          `Hybrid optimize() request failed (${method} ${path}) with HTTP ${response.status}: ${errorText || "No response body"}`,
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
          `Hybrid optimize() request timeout (${method} ${path})`,
          this.requestTimeoutMs,
        );
      }

      if (
        options.signal?.aborted ||
        (error instanceof Error && error.name === "AbortError")
      ) {
        throw new CancelledError("Optimization cancelled");
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

export async function runHybridOptimization(
  trialFn: NativeTrialFunction,
  spec: NormalizedOptimizationSpec,
  specInput: OptimizationSpec,
  rawOptions: HybridOptimizeOptions,
  functionName?: string,
): Promise<OptimizationResult> {
  const options = validateHybridOptimizeOptions(rawOptions);
  const evaluationRows = await resolveEvaluationRows(spec);

  if (!Array.isArray(evaluationRows) || evaluationRows.length === 0) {
    throw new ValidationError(
      "optimize() requires evaluation data to be a non-empty array.",
    );
  }

  const objectives = serializeHybridObjectives(specInput);
  const configurationSpace = serializeSessionConfigurationSpace(spec);
  const budget = serializeSessionBudget(spec);
  const constraints = serializeSessionConstraints(spec);
  const promotionPolicy = serializePromotionPolicy(spec);

  if (options.datasetMetadata?.["size"] !== undefined) {
    if (
      typeof options.datasetMetadata["size"] !== "number" ||
      !Number.isFinite(options.datasetMetadata["size"]) ||
      options.datasetMetadata["size"] <= 0
    ) {
      throw new ValidationError(
        "Hybrid optimize() datasetMetadata.size must be a positive number when provided.",
      );
    }
    if (options.datasetMetadata["size"] !== evaluationRows.length) {
      throw new ValidationError(
        "Hybrid optimize() datasetMetadata.size must match the loaded evaluation dataset size.",
      );
    }
  }

  const client = new HybridSessionClient(
    options.backendUrl,
    options.apiKey,
    options.requestTimeoutMs,
  );
  const completedTrials: OptimizationTrialRecord[] = [];
  const previousResults: HybridSubmittedResult[] = [];
  let totalCostUsd = 0;
  let sessionId: string | undefined;
  let backendReason: string | undefined;
  let failedTrialStopReason: "timeout" | "error" | undefined;
  let finalization: HybridFinalizationResponse | undefined;
  let finalizationError: string | undefined;
  let stopReason: OptimizationResult["stopReason"] | undefined;
  let errorMessage: string | undefined;
  let optimizationStrategyMetadata: Record<string, unknown> | undefined;

  try {
    const createResponse = await client.createSession(
      {
        function_name:
          functionName && functionName.length > 0
            ? functionName
            : "anonymous_js_trial",
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
          algorithm: "optuna",
          ...(serializeSnakeCaseObject(
            options.optimizationStrategy ?? {},
          ) as Record<string, unknown>),
        },
        user_id: options.userId,
        billing_tier: options.billingTier ?? "standard",
        metadata: {
          sdk: "js",
          mode: "hybrid",
        },
      },
      options.signal,
    );

    sessionId = createResponse.session_id;
    optimizationStrategyMetadata = createResponse.optimization_strategy;

    while (stopReason === undefined) {
      if (options.signal?.aborted) {
        stopReason = "cancelled";
        errorMessage = "Optimization cancelled";
        break;
      }

      const nextResponse = await client.getNextTrial(
        sessionId,
        {
          session_id: sessionId,
          previous_results: previousResults.slice(-5),
          request_metadata: {
            dataset_size: evaluationRows.length,
            completed_trials: previousResults.length,
          },
        },
        options.signal,
      );

      if (!nextResponse.should_continue || !nextResponse.suggestion) {
        backendReason = resolveBackendStopReason(
          nextResponse.stop_reason,
          nextResponse.reason,
        );
        stopReason = normalizeBackendStopReason(
          {
            backendReason,
            failedTrialStopReason,
          },
          previousResults.length,
          options.maxTrials,
        );
        break;
      }

      const trialConfig = createTrialConfigFromSuggestion(
        nextResponse.suggestion,
        evaluationRows.length,
        sessionId,
        spec.defaultConfig,
      );
      const outcome = await executeTrial(
        trialFn,
        trialConfig,
        options.timeoutMs,
        options.signal,
      );
      const submission = buildSubmittedResult(
        sessionId,
        trialConfig.trial_id,
        outcome,
      );

      if (outcome.status === "completed") {
        totalCostUsd = updateTotalCost(spec, outcome.record, totalCostUsd);
        completedTrials.push(outcome.record);
        failedTrialStopReason = undefined;
      } else if (outcome.status === "timeout" || outcome.status === "error") {
        failedTrialStopReason = outcome.status;
      } else {
        stopReason = "cancelled";
        errorMessage = outcome.errorMessage;
      }

      await client.submitResult(
        sessionId,
        submission,
        outcome.status === "cancelled" ? undefined : options.signal,
      );
      previousResults.push(submission);

      if (stopReason === "cancelled") {
        break;
      }
    }
  } catch (error) {
    if (error instanceof ValidationError) {
      throw error;
    }

    if (
      error instanceof CancelledError ||
      error instanceof TrialCancelledError
    ) {
      stopReason = "cancelled";
      errorMessage = toErrorMessage(error);
    } else if (error instanceof TimeoutError) {
      stopReason = "timeout";
      errorMessage = error.message;
    } else {
      stopReason = "error";
      errorMessage = toErrorMessage(error);
    }
  } finally {
    if (sessionId) {
      try {
        finalization = await client.finalizeSession(sessionId, undefined);
        backendReason = resolveBackendStopReason(
          finalization.stop_reason,
          backendReason,
        );
        if ((!stopReason || stopReason === "completed") && backendReason) {
          stopReason = normalizeBackendStopReason(
            {
              backendReason,
              failedTrialStopReason,
            },
            previousResults.length,
            options.maxTrials,
          );
        }
      } catch (error) {
        finalizationError = toErrorMessage(error);
        if (!stopReason) {
          stopReason = error instanceof TimeoutError ? "timeout" : "error";
          errorMessage = finalizationError;
        }
      }
    }
  }

  return finalizeHybridResult(
    spec,
    completedTrials,
    stopReason ?? "completed",
    totalCostUsd,
    sessionId,
    errorMessage,
    {
      backendReason,
      finalizeError: finalizationError,
      optimizationStrategy: optimizationStrategyMetadata,
      finalization: finalization?.metadata,
    },
    finalization,
  );
}
