import type { Metrics, TrialConfig } from '../dtos/trial.js';

export type ObjectiveDirection = 'maximize' | 'minimize' | 'band';
export type BuiltInObjectiveName = 'accuracy' | 'cost' | 'latency';
export type ParamScale = 'linear' | 'log';
export type InjectionMode = 'context' | 'parameter' | 'seamless';
export type ExecutionMode = 'native' | 'hybrid';
export type ExecutionContract = 'agent' | 'trial';
export type ParameterConditionValue = string | number | boolean;
export type ParameterConditions = Record<string, ParameterConditionValue>;
export type AggregationStrategy = 'mean' | 'median' | 'sum' | 'min' | 'max';
export type RepetitionAggregationStrategy = 'mean' | 'median' | 'min' | 'max';
export type FrameworkTarget = 'openai' | 'langchain' | 'vercel-ai';

export interface FrameworkAutoOverrideStatus {
  autoOverrideFrameworks: boolean;
  requestedTargets?: readonly FrameworkTarget[];
  activeTargets: readonly FrameworkTarget[];
  selectedTargets: readonly FrameworkTarget[];
  enabled: boolean;
  reason: string;
}

export interface SeamlessResolution {
  path: 'framework' | 'pretransformed' | 'runtime';
  reason: string;
  experimental: boolean;
  targets?: readonly FrameworkTarget[];
}

export type NativeTvlSupportStatus =
  | 'supported'
  | 'supported-with-reduced-semantics'
  | 'hybrid-only'
  | 'unsupported';

export interface NativeTvlCompatibilityItem {
  feature: string;
  status: NativeTvlSupportStatus;
  reason: string;
  used: boolean;
}

export interface NativeTvlCompatibilityReport {
  scope: 'native';
  items: NativeTvlCompatibilityItem[];
  usedFeatures: string[];
  warnings: string[];
}

export interface ObjectiveDefinition {
  metric: string;
  direction?: ObjectiveDirection;
  weight?: number;
  band?: {
    low: number;
    high: number;
    test?: 'TOST';
    alpha?: number;
  };
}

export type ObjectiveInput = BuiltInObjectiveName | ObjectiveDefinition;

interface ConditionalParameterDefinition<T> {
  conditions?: ParameterConditions;
  default?: T;
}

export interface EnumParamDefinition<T = unknown> extends ConditionalParameterDefinition<T> {
  type: 'enum';
  values: readonly T[];
}

export interface FloatParamDefinition extends ConditionalParameterDefinition<number> {
  type: 'float';
  min: number;
  max: number;
  scale?: ParamScale;
  step?: number;
}

export interface IntParamDefinition extends ConditionalParameterDefinition<number> {
  type: 'int';
  min: number;
  max: number;
  scale?: ParamScale;
  step?: number;
}

export type ParameterDefinition = EnumParamDefinition | FloatParamDefinition | IntParamDefinition;

export interface OptimizationBudget {
  maxCostUsd?: number;
  maxTrials?: number;
  maxWallclockMs?: number;
}

export interface StructuralConstraintDefinition {
  when?: string;
  then?: string;
  require?: string;
  errorMessage?: string;
  id?: string;
}

export interface DerivedConstraintDefinition {
  require: string;
  errorMessage?: string;
  id?: string;
}

export interface OptimizationConstraints {
  structural?: readonly StructuralConstraintDefinition[];
  derived?: readonly DerivedConstraintDefinition[];
}

export type EvaluationScoringFunction<Row = unknown, Output = unknown> = (
  output: Output,
  expectedOutput: unknown,
  runtimeMetrics: Metrics,
  row: Row
) => number | null | Promise<number | null>;

export type EvaluationMetricFunction<Row = unknown, Output = unknown> = (
  output: Output,
  expectedOutput: unknown,
  runtimeMetrics: Metrics,
  row: Row
) => number | null | Promise<number | null>;

export interface EvaluationContext<Row = unknown, Output = unknown> {
  output: Output;
  expectedOutput: unknown;
  runtimeMetrics: Metrics;
  row: Row;
  config: TrialConfig['config'];
}

/**
 * Optimization constraints may run either before execution (config-only) or
 * after execution (config + metrics). If a post-trial constraint uses default
 * parameters or rest arguments, set `constraint.requiresMetrics = true` so the
 * native runtime does not rely on `function.length` inference.
 */
export type OptimizationConstraint = ((
  config: TrialConfig['config'],
  metrics?: Metrics
) => boolean) & {
  requiresMetrics?: boolean;
};

export type SafetyConstraint = ((config: TrialConfig['config'], metrics: Metrics) => boolean) & {
  requiresMetrics?: boolean;
};

export interface EvaluationAggregationMap {
  default?: AggregationStrategy;
  [metric: string]: AggregationStrategy | undefined;
}

export type AgentCustomEvaluator<Row = unknown, Output = unknown> =
  | ((context: EvaluationContext<Row, Output>) => Metrics | Promise<Metrics>)
  | ((
      agentFn: (input: unknown) => unknown | Promise<unknown>,
      config: TrialConfig['config'],
      row: Row
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

export interface ExecutionSpec {
  mode?: ExecutionMode;
  contract?: ExecutionContract;
  maxTotalExamples?: number;
  maxWallclockMs?: number;
  exampleConcurrency?: number;
  repsPerTrial?: number;
  repsAggregation?: RepetitionAggregationStrategy;
}

export interface NormalizedExecutionSpec {
  mode: ExecutionMode;
  contract: ExecutionContract;
  maxTotalExamples?: number;
  maxWallclockMs?: number;
  exampleConcurrency: number;
  repsPerTrial: number;
  repsAggregation: RepetitionAggregationStrategy;
}

export interface OptimizationSpec {
  configurationSpace: Record<string, ParameterDefinition>;
  objectives: readonly ObjectiveInput[];
  budget?: OptimizationBudget;
  defaultConfig?: TrialConfig['config'];
  promotionPolicy?: TvlPromotionPolicy;
  constraints?: readonly OptimizationConstraint[] | OptimizationConstraints;
  safetyConstraints?: readonly SafetyConstraint[];
  evaluation?: EvaluationSpec;
  injection?: InjectionSpec;
  execution?: ExecutionSpec;
}

export interface NormalizedObjectiveDefinition {
  metric: string;
  direction: ObjectiveDirection;
  weight: number;
  band?: {
    low: number;
    high: number;
    test: 'TOST';
    alpha: number;
  };
}

export interface NormalizedOptimizationSpec {
  configurationSpace: Record<string, ParameterDefinition>;
  objectives: readonly NormalizedObjectiveDefinition[];
  budget?: OptimizationBudget;
  defaultConfig: TrialConfig['config'];
  promotionPolicy?: TvlPromotionPolicy;
  constraints: readonly OptimizationConstraint[];
  safetyConstraints: readonly SafetyConstraint[];
  evaluation?: EvaluationSpec;
  injection: Required<Pick<InjectionSpec, 'mode'>> & InjectionSpec;
  execution: NormalizedExecutionSpec;
}

export interface NativeOptimizeOptions {
  mode?: 'native';
  algorithm: 'grid' | 'random' | 'bayesian';
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
  mode?: 'hybrid';
  algorithm: 'optuna';
  maxTrials: number;
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
  /** Backend origin or a path-prefixed `/api/v1` base URL. */
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

export interface BandTarget {
  low?: number;
  high?: number;
  center?: number;
  tol?: number;
}

export interface BandedObjectiveDefinition {
  metric: string;
  band: BandTarget;
  test?: 'TOST';
  alpha?: number;
  weight?: number;
}

export interface OptimizationSessionCreateRequest {
  functionName: string;
  configurationSpace: Record<string, ParameterDefinition>;
  objectives: readonly ObjectiveInput[];
  datasetMetadata?: Record<string, unknown>;
  maxTrials?: number;
  budget?: OptimizationBudget;
  constraints?: readonly OptimizationConstraint[];
  defaultConfig?: Record<string, unknown>;
  promotionPolicy?: TvlPromotionPolicy;
  optimizationStrategy?: Record<string, unknown>;
  userId?: string;
  billingTier?: string;
  metadata?: Record<string, unknown>;
}

export type OptimizationSessionLifecycleStatus =
  | 'pending'
  | 'created'
  | 'active'
  | 'paused'
  | 'completed'
  | 'failed'
  | 'cancelled'
  | 'PENDING'
  | 'CREATED'
  | 'ACTIVE'
  | 'PAUSED'
  | 'COMPLETED'
  | 'FAILED'
  | 'CANCELLED';

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
  config: TrialConfig['config'];
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
  status?: 'completed' | 'failed' | 'cancelled' | 'timeout';
  outputsSample?: readonly unknown[] | null;
  errorMessage?: string | null;
  metadata?: Record<string, unknown>;
}

export interface OptimizationSessionNextTrialOptions extends OptimizationSessionRequestOptions {
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

export interface OptimizationSessionListOptions extends OptimizationSessionRequestOptions {
  /**
   * Backend-defined `pattern` query parameter. The current backend applies
   * substring-style matching.
   */
  pattern?: string;
  status?: OptimizationSessionLifecycleStatus | string;
}

export interface OptimizationSessionDeleteOptions extends OptimizationSessionRequestOptions {
  /**
   * When true, delete related backend session artifacts in addition to the
   * session itself. Defaults to false for the public helper.
   */
  cascade?: boolean;
}

export interface OptimizationSessionFinalizeOptions extends OptimizationSessionRequestOptions {
  includeFullHistory?: boolean;
}

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
  createdAt?: string | number;
  functionName?: string;
  datasetSize?: number;
  objectives?: readonly string[];
  experimentId?: string;
  experimentRunId?: string;
  metadata?: OptimizationSessionStatusMetadata;
  [key: string]: unknown;
}

export interface OptimizationSessionListResponse {
  sessions: readonly OptimizationSessionStatusResponse[];
  /**
   * Backend-reported total count before SDK-side filtering of malformed entries.
   * This may exceed `sessions.length`.
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

export interface OptimizationSessionFinalizationResponse {
  sessionId: string;
  bestConfig?: TrialConfig['config'];
  bestMetrics?: Metrics | null;
  stopReason?: string | null;
  reporting?: OptimizationReportingSummary;
  metadata?: Record<string, unknown>;
}

export type OptimizeOptions = NativeOptimizeOptions | HybridOptimizeOptions;

export interface OptimizationTrialRecord {
  trialId: string;
  trialNumber: number;
  config: TrialConfig['config'];
  metrics: Metrics;
  duration: number;
  status?: 'completed' | 'rejected';
  errorMessage?: string;
  metadata?: Record<string, unknown>;
  promotionDecision?: PromotionDecision;
}

export interface OptimizationResult {
  mode: 'native' | 'hybrid';
  bestConfig: TrialConfig['config'] | null;
  bestMetrics: Metrics | null;
  trials: OptimizationTrialRecord[];
  promotionDecision?: PromotionDecision;
  reporting?: NativeOptimizationReportingSummary | OptimizationReportingSummary;
  stopReason:
    | 'completed'
    | 'maxTrials'
    | 'maxExamples'
    | 'budget'
    | 'timeout'
    | 'error'
    | 'plateau'
    | 'cancelled';
  totalCostUsd: number;
  sessionId?: string;
  metadata?: Record<string, unknown>;
  errorMessage?: string;
}

export interface NativePromotionReportingSummary {
  applied: boolean;
  bestTrialId?: string;
  bestTrialNumber?: number;
  decision?: PromotionDecision['decision'];
  method?: PromotionDecision['method'];
  usedChanceConstraints: boolean;
  usedStatisticalComparison: boolean;
  usedTieBreakers: boolean;
}

export interface NativeOptimizationReportingSummary {
  totalTrials: number;
  completedTrials: number;
  rejectedTrials: number;
  evaluatedExamples: number;
  promotion: NativePromotionReportingSummary;
}

export interface PromotionObjectiveResult {
  name: string;
  direction: ObjectiveDirection;
  candidateBetter: boolean;
  effectSize: number;
  epsilon: number;
  pValue?: number;
  adjustedPValue?: number;
  candidateMean?: number;
  incumbentMean?: number;
  method: 'deterministic' | 'statistical';
}

export interface PromotionDecision {
  decision: 'promote' | 'reject' | 'no_decision';
  reason: string;
  objectiveResults: PromotionObjectiveResult[];
  chanceResults: PromotionChanceConstraintResult[];
  adjustedPValues: Record<string, number>;
  dominanceSatisfied: boolean;
  method: 'none' | 'deterministic' | 'statistical' | 'chance-constraints';
  candidateTrialId?: string;
  incumbentTrialId?: string;
}

export interface TvlPromotionPolicy {
  dominance?: 'epsilon_pareto';
  alpha?: number;
  minEffect?: Record<string, number>;
  adjust?: 'none' | 'BH';
  tieBreakers?: Record<string, 'maximize' | 'minimize'>;
  chanceConstraints?: Array<{
    name: string;
    threshold: number;
    confidence: number;
  }>;
}

export interface PromotionChanceConstraintResult {
  name: string;
  satisfied: boolean;
  observedRate: number;
  lowerBound: number;
  threshold: number;
  confidence: number;
}

export interface TvlSpecArtifact {
  spec: OptimizationSpec;
  optimizeOptions?: Partial<Pick<NativeOptimizeOptions, 'algorithm' | 'maxTrials'>>;
  tvlVersion?: string;
  moduleId?: string;
  nativeCompatibility: NativeTvlCompatibilityReport;
  metadata: Record<string, unknown>;
  promotionPolicy?: TvlPromotionPolicy;
}

export interface TvlLoadOptions {
  path?: string;
  source?: string;
}

export interface HybridTunableDefinition {
  name: string;
  type: 'enum' | 'float' | 'int';
  domain: {
    values?: unknown[];
    range?: [number, number];
  };
  scale?: ParamScale;
}

export interface HybridConfigSpace {
  tunables: HybridTunableDefinition[];
  constraints: Record<string, never>;
}

export interface NativeTrialFunctionResult {
  metrics: Metrics;
  output?: unknown;
  metadata?: Record<string, unknown>;
  duration?: number;
}

export type NativeOptimizedFunction<T extends (...args: any[]) => any> = T & {
  optimize(options: OptimizeOptions): Promise<OptimizationResult>;
  applyBestConfig(result: OptimizationResult): TrialConfig['config'] | undefined;
  currentConfig(): TrialConfig['config'] | undefined;
  seamlessResolution(): SeamlessResolution | undefined;
};
