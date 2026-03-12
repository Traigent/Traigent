import type { Metrics, TrialConfig } from "../dtos/trial.js";

export type ObjectiveDirection = "maximize" | "minimize";
export type BuiltInObjectiveName = "accuracy" | "cost" | "latency";
export type ParamScale = "linear" | "log";
export type InjectionMode = "context" | "parameter" | "seamless";
export type ExecutionContract = "agent" | "trial";
export type ParameterConditionValue = string | number | boolean;
export type ParameterConditions = Record<string, ParameterConditionValue>;
export type AggregationStrategy = "mean" | "median" | "sum" | "min" | "max";
export type FrameworkTarget = "openai" | "langchain" | "vercel-ai";

export interface FrameworkAutoOverrideStatus {
  autoOverrideFrameworks: boolean;
  requestedTargets?: readonly FrameworkTarget[];
  activeTargets: readonly FrameworkTarget[];
  selectedTargets: readonly FrameworkTarget[];
  enabled: boolean;
  reason: string;
}

export interface SeamlessResolution {
  path: "framework";
  reason: string;
  experimental: boolean;
  targets?: readonly FrameworkTarget[];
}

export interface BandTarget {
  low?: number;
  high?: number;
  center?: number;
  tol?: number;
}

export interface ObjectiveDefinition {
  metric: string;
  direction: ObjectiveDirection;
  weight?: number;
}

export interface BandedObjectiveDefinition {
  metric: string;
  band: BandTarget;
  test?: "TOST";
  alpha?: number;
  weight?: number;
}

export type ObjectiveInput =
  | BuiltInObjectiveName
  | ObjectiveDefinition
  | BandedObjectiveDefinition;

interface ConditionalParameterDefinition<T> {
  conditions?: ParameterConditions;
  default?: T;
}

export interface EnumParamDefinition<
  T = unknown,
> extends ConditionalParameterDefinition<T> {
  type: "enum";
  values: readonly T[];
}

export interface FloatParamDefinition extends ConditionalParameterDefinition<number> {
  type: "float";
  min: number;
  max: number;
  scale?: ParamScale;
  step?: number;
}

export interface IntParamDefinition extends ConditionalParameterDefinition<number> {
  type: "int";
  min: number;
  max: number;
  scale?: ParamScale;
  step?: number;
}

export type ParameterDefinition =
  | EnumParamDefinition
  | FloatParamDefinition
  | IntParamDefinition;

export interface OptimizationBudget {
  maxCostUsd?: number;
  maxTrials?: number;
  maxWallclockMs?: number;
}

export interface ChanceConstraintDefinition {
  name: string;
  threshold: number;
  confidence: number;
}

export interface PromotionPolicy {
  dominance?: "epsilon_pareto";
  alpha?: number;
  minEffect?: Record<string, number>;
  adjust?: "none" | "BH";
  chanceConstraints?: readonly ChanceConstraintDefinition[];
  tieBreakers?: Record<string, "maximize" | "minimize">;
}

export interface StructuralConstraintDefinition {
  id?: string;
  require?: string;
  when?: string;
  then?: string;
  errorMessage?: string;
}

export interface DerivedConstraintDefinition {
  id?: string;
  require: string;
  errorMessage?: string;
}

export interface OptimizationConstraints {
  structural?: readonly StructuralConstraintDefinition[];
  derived?: readonly DerivedConstraintDefinition[];
}

export interface TvlLoadOptions {
  path?: string;
  source?: string;
}

export interface LoadedTvlOptimizationSpec {
  spec: OptimizationSpec;
  metadata: {
    path?: string;
    strategyType?: string;
  };
}

export type EvaluationScoringFunction<Row = unknown, Output = unknown> = (
  output: Output,
  expectedOutput: unknown,
  runtimeMetrics: Metrics,
  row: Row,
) => number | null | Promise<number | null>;

export type EvaluationMetricFunction<Row = unknown, Output = unknown> = (
  output: Output,
  expectedOutput: unknown,
  runtimeMetrics: Metrics,
  row: Row,
) => number | null | Promise<number | null>;

export interface EvaluationContext<Row = unknown, Output = unknown> {
  output: Output;
  expectedOutput: unknown;
  runtimeMetrics: Metrics;
  row: Row;
  config: TrialConfig["config"];
}

export interface EvaluationAggregationMap {
  default?: AggregationStrategy;
  [metric: string]: AggregationStrategy | undefined;
}

export type AgentCustomEvaluator<Row = unknown, Output = unknown> =
  | ((context: EvaluationContext<Row, Output>) => Metrics | Promise<Metrics>)
  | ((
      agentFn: (input: unknown) => unknown | Promise<unknown>,
      config: TrialConfig["config"],
      row: Row,
    ) => Metrics | Promise<Metrics>);

export interface EvaluationSpec {
  data?: readonly unknown[];
  loadData?: () => Promise<readonly unknown[]>;
  scoringFunction?: EvaluationScoringFunction;
  metricFunctions?: Record<string, EvaluationMetricFunction>;
  customEvaluator?: AgentCustomEvaluator;
  inputField?: string;
  expectedField?: string;
  aggregation?: AggregationStrategy | EvaluationAggregationMap;
}

export interface InjectionSpec {
  mode?: InjectionMode;
  autoOverrideFrameworks?: boolean;
  frameworkTargets?: readonly FrameworkTarget[];
}

export interface OptimizationExecutionSpec {
  mode?: "native" | "hybrid";
  algorithm?: "grid" | "random" | "bayesian" | "optuna";
  /** Backend origin (for example, https://host:5000) or any path-prefixed /api/v1 base URL. */
  backendUrl?: string;
  apiKey?: string;
  timeoutMs?: number;
  requestTimeoutMs?: number;
  trialConcurrency?: number;
  contract?: ExecutionContract;
}

export interface OptimizationSpec {
  configurationSpace: Record<string, ParameterDefinition>;
  objectives: readonly ObjectiveInput[];
  budget?: OptimizationBudget;
  constraints?: OptimizationConstraints;
  defaultConfig?: Record<string, unknown>;
  promotionPolicy?: PromotionPolicy;
  execution?: OptimizationExecutionSpec;
  autoLoadBest?: boolean;
  loadFrom?: string;
  evaluation?: EvaluationSpec;
  injection?: InjectionSpec;
}

export interface NormalizedStandardObjectiveDefinition {
  kind: "standard";
  metric: string;
  direction: ObjectiveDirection;
  weight: number;
}

export interface NormalizedBandedObjectiveDefinition {
  kind: "banded";
  metric: string;
  band: {
    low: number;
    high: number;
  };
  bandTest: "TOST";
  bandAlpha: number;
  weight: number;
}

export type NormalizedObjectiveDefinition =
  | NormalizedStandardObjectiveDefinition
  | NormalizedBandedObjectiveDefinition;

export interface NormalizedOptimizationSpec {
  configurationSpace: Record<string, ParameterDefinition>;
  objectives: readonly NormalizedObjectiveDefinition[];
  budget?: OptimizationBudget;
  constraints?: OptimizationConstraints;
  defaultConfig?: Record<string, unknown>;
  promotionPolicy?: PromotionPolicy;
  execution?: OptimizationExecutionSpec;
  autoLoadBest?: boolean;
  loadFrom?: string;
  evaluation?: EvaluationSpec;
  injection?: Required<Pick<InjectionSpec, "mode">> & InjectionSpec;
}

export interface NativeOptimizeOptions {
  mode?: "native";
  algorithm: "grid" | "random" | "bayesian";
  maxTrials: number;
  randomSeed?: number;
  timeoutMs?: number;
  trialConcurrency?: number;
  signal?: AbortSignal;
  plateau?: {
    window: number;
    minImprovement: number;
  };
  checkpoint?: {
    key: string;
    dir?: string;
    resume?: boolean;
  };
}

export interface HybridOptimizeOptions {
  mode?: "hybrid";
  algorithm: "optuna";
  maxTrials: number;
  /** Backend origin (for example, https://host:5000) or any path-prefixed /api/v1 base URL. */
  backendUrl?: string;
  apiKey?: string;
  userId?: string;
  billingTier?: string;
  optimizationStrategy?: Record<string, unknown>;
  datasetMetadata?: Record<string, unknown>;
  includeFullHistory?: boolean;
  timeoutMs?: number;
  requestTimeoutMs?: number;
  signal?: AbortSignal;
}

export interface OptimizationSessionRequestOptions {
  /** Backend origin (for example, https://host:5000) or any path-prefixed /api/v1 base URL. */
  backendUrl?: string;
  apiKey?: string;
  requestTimeoutMs?: number;
  signal?: AbortSignal;
}

export interface OptimizationServiceStatusResponse {
  status: string;
  error?: string;
  [key: string]: unknown;
}

export interface OptimizationSessionCreateRequest {
  functionName: string;
  configurationSpace: Record<string, ParameterDefinition>;
  objectives: readonly ObjectiveInput[];
  datasetMetadata?: Record<string, unknown>;
  maxTrials?: number;
  budget?: OptimizationBudget;
  constraints?: OptimizationConstraints;
  defaultConfig?: Record<string, unknown>;
  promotionPolicy?: PromotionPolicy;
  optimizationStrategy?: Record<string, unknown>;
  userId?: string;
  billingTier?: string;
  metadata?: Record<string, unknown>;
}

export interface OptimizationSessionCreationResponse {
  sessionId: string;
  status?: OptimizationSessionLifecycleStatus | string;
  optimizationStrategy?: Record<string, unknown>;
  estimatedDuration?: number;
  billingEstimate?: Record<string, unknown>;
  metadata?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface OptimizationSessionDatasetSubset {
  indices: readonly number[];
  selectionStrategy?: string;
  confidenceLevel?: number;
  estimatedRepresentativeness?: number;
  metadata?: Record<string, unknown>;
}

export interface OptimizationSessionTrialSuggestion {
  trialId: string;
  sessionId: string;
  trialNumber: number;
  config: TrialConfig["config"];
  datasetSubset: OptimizationSessionDatasetSubset;
  explorationType?: string;
  priority?: number;
  estimatedDuration?: number;
  metadata?: Record<string, unknown>;
}

export interface OptimizationSessionTrialResultInput {
  sessionId?: string;
  trialId: string;
  metrics: Metrics;
  duration: number;
  status?: "completed" | "failed" | "cancelled" | "timeout";
  outputsSample?: readonly unknown[] | null;
  errorMessage?: string | null;
  metadata?: Record<string, unknown>;
}

export interface OptimizationSessionNextTrialOptions
  extends OptimizationSessionRequestOptions {
  previousResults?: readonly OptimizationSessionTrialResultInput[];
  requestMetadata?: Record<string, unknown>;
}

export interface OptimizationSessionNextTrialResponse {
  suggestion: OptimizationSessionTrialSuggestion | null;
  shouldContinue: boolean;
  reason?: string | null;
  stopReason?: string | null;
  sessionStatus?: OptimizationSessionLifecycleStatus | string;
  metadata?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface OptimizationSessionSubmitResultResponse {
  success: boolean;
  continueOptimization?: boolean;
  message?: string;
  [key: string]: unknown;
}

export interface OptimizationSessionListOptions
  extends OptimizationSessionRequestOptions {
  /**
   * Backend-defined session-id filter forwarded verbatim as the `pattern`
   * query parameter. The current backend treats this as a substring-style match.
   */
  pattern?: string;
  status?: OptimizationSessionLifecycleStatus | string;
}

export interface OptimizationSessionDeleteOptions
  extends OptimizationSessionRequestOptions {
  /**
   * When true, delete related backend session artifacts in addition to the
   * session itself. Defaults to false for the public helper.
   */
  cascade?: boolean;
}

export interface OptimizationSessionFinalizeOptions
  extends OptimizationSessionRequestOptions {
  includeFullHistory?: boolean;
}

export type OptimizationSessionLifecycleStatus =
  | "pending"
  | "created"
  | "active"
  | "paused"
  | "completed"
  | "failed"
  | "cancelled"
  | "PENDING"
  | "CREATED"
  | "ACTIVE"
  | "PAUSED"
  | "COMPLETED"
  | "FAILED"
  | "CANCELLED";

export interface OptimizationSessionStatusSummary {
  completed?: number;
  total?: number;
  failed?: number;
  [key: string]: unknown;
}

export interface OptimizationSessionStatusMetadata {
  function_name?: string;
  dataset_size?: number;
  objectives?: readonly string[];
  created_at?: string | number;
  experiment_id?: string;
  experiment_run_id?: string;
  [key: string]: unknown;
}

export interface OptimizationSessionStatusResponse {
  sessionId: string;
  status?: OptimizationSessionLifecycleStatus | string;
  progress?: OptimizationSessionStatusSummary;
  /** Top-level normalized alias for backend `created_at` when present. */
  createdAt?: string | number;
  /** Top-level normalized alias for backend `function_name` when present. */
  functionName?: string;
  /** Top-level normalized alias for backend `dataset_size` when present. */
  datasetSize?: number;
  /** Top-level normalized alias for backend `objectives` when present. */
  objectives?: readonly string[];
  /** Top-level normalized alias for backend `experiment_id` when present. */
  experimentId?: string;
  /** Top-level normalized alias for backend `experiment_run_id` when present. */
  experimentRunId?: string;
  metadata?: OptimizationSessionStatusMetadata;
  [key: string]: unknown;
}

export interface OptimizationSessionListResponse {
  sessions: readonly OptimizationSessionStatusResponse[];
  /**
   * Backend-reported total count before SDK-side filtering of malformed session
   * entries. This may exceed `sessions.length` if the backend returned entries
   * without a usable `session_id`.
   */
  total: number;
  [key: string]: unknown;
}

export interface OptimizationSessionDeleteResponse {
  success: boolean;
  sessionId: string;
  deleted?: boolean;
  cascade?: boolean;
  message?: string;
  metadata?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface OptimizationSessionFinalizationResponse {
  sessionId: string;
  bestConfig?: TrialConfig["config"];
  bestMetrics?: Metrics | null;
  stopReason?: string | null;
  reporting?: OptimizationReportingSummary;
  metadata?: Record<string, unknown>;
}

export type OptimizeOptions = NativeOptimizeOptions | HybridOptimizeOptions;

export interface OptimizationTrialRecord {
  trialId: string;
  trialNumber: number;
  config: TrialConfig["config"];
  metrics: Metrics;
  duration: number;
  metadata?: Record<string, unknown>;
}

export interface OptimizationConvergencePoint {
  trial?: number;
  score?: number;
  [key: string]: unknown;
}

export interface OptimizationReportingTrialHistoryEntry {
  session_id: string;
  trial_id: string;
  metrics: Metrics;
  duration: number;
  status: string;
  outputs_sample?: readonly unknown[] | null;
  error_message?: string | null;
  metadata?: Record<string, unknown>;
}

export interface OptimizationReportingSummary {
  totalTrials?: number;
  successfulTrials?: number;
  totalDuration?: number;
  costSavings?: number;
  convergenceHistory?: readonly OptimizationConvergencePoint[];
  fullHistory?: readonly OptimizationReportingTrialHistoryEntry[];
}

export interface OptimizationResult {
  mode: "native" | "hybrid";
  bestConfig: TrialConfig["config"] | null;
  bestMetrics: Metrics | null;
  trials: OptimizationTrialRecord[];
  stopReason:
    | "completed"
    | "maxTrials"
    | "budget"
    | "timeout"
    | "error"
    | "plateau"
    | "cancelled";
  totalCostUsd: number;
  sessionId?: string;
  reporting?: OptimizationReportingSummary;
  metadata?: Record<string, unknown>;
  errorMessage?: string;
}

export interface HybridTunableDefinition {
  name: string;
  type: "enum" | "float" | "int";
  domain: {
    values?: unknown[];
    range?: [number, number];
  };
  scale?: ParamScale;
}

export interface HybridConfigSpace {
  tunables: HybridTunableDefinition[];
  constraints: OptimizationConstraints;
}

export interface NativeTrialFunctionResult {
  metrics: Metrics;
  output?: unknown;
  metadata?: Record<string, unknown>;
  duration?: number;
}

export type NativeOptimizedFunction<T extends (...args: any[]) => any> = T & {
  optimize(options: OptimizeOptions): Promise<OptimizationResult>;
  applyBestConfig(
    result: OptimizationResult,
  ): TrialConfig["config"] | undefined;
  currentConfig(): TrialConfig["config"] | undefined;
  /**
   * Returns the current framework auto-override status for this optimized
   * function. The result reflects the current framework registry state, not a
   * historical snapshot from a previous optimization run.
   */
  frameworkAutoOverrideStatus(): FrameworkAutoOverrideStatus;
  /**
   * Returns the resolved seamless path for this optimized function when a
   * seamless interception path is currently active.
   *
   * Returns `undefined` when:
   * - the function is not configured with `injection.mode = "seamless"`, or
   * - seamless mode is configured but no active framework targets are currently
   *   registered for interception
   *
   * Use `frameworkAutoOverrideStatus()` to distinguish those cases and inspect
   * why framework interception is or is not enabled.
   */
  seamlessResolution(): SeamlessResolution | undefined;
};
