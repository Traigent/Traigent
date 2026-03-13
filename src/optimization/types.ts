import type { Metrics, TrialConfig } from '../dtos/trial.js';

export type ObjectiveDirection = 'maximize' | 'minimize' | 'band';
export type BuiltInObjectiveName = 'accuracy' | 'cost' | 'latency';
export type ParamScale = 'linear' | 'log';
export type InjectionMode = 'context' | 'parameter' | 'seamless';
export type ExecutionMode = 'native' | 'hybrid';
export type ExecutionContract = 'agent' | 'trial';
export type AggregationStrategy = 'mean' | 'median' | 'sum' | 'min' | 'max';
export type RepetitionAggregationStrategy =
  | 'mean'
  | 'median'
  | 'min'
  | 'max';
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

export interface EnumParamDefinition<T = unknown> {
  type: 'enum';
  values: readonly T[];
}

export interface FloatParamDefinition {
  type: 'float';
  min: number;
  max: number;
  scale?: ParamScale;
  step?: number;
}

export interface IntParamDefinition {
  type: 'int';
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
  config: TrialConfig['config'];
}

/**
 * Optimization constraints may run either before execution (config-only) or
 * after execution (config + metrics). If a post-trial constraint uses default
 * parameters or rest arguments, set `constraint.requiresMetrics = true` so the
 * native runtime does not rely on `function.length` inference.
 */
export type OptimizationConstraint = ((config: TrialConfig['config'], metrics?: Metrics) => boolean) & {
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
  constraints?: readonly OptimizationConstraint[];
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
  bestConfig: TrialConfig['config'] | null;
  bestMetrics: Metrics | null;
  trials: OptimizationTrialRecord[];
  promotionDecision?: PromotionDecision;
  reporting: NativeOptimizationReportingSummary;
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
  optimize(options: NativeOptimizeOptions): Promise<OptimizationResult>;
  applyBestConfig(
    result: OptimizationResult,
  ): TrialConfig['config'] | undefined;
  currentConfig(): TrialConfig['config'] | undefined;
  seamlessResolution(): SeamlessResolution | undefined;
};
