import type { Metrics, TrialConfig } from "../dtos/trial.js";

export type ObjectiveDirection = "maximize" | "minimize";
export type BuiltInObjectiveName = "accuracy" | "cost" | "latency";
export type ParamScale = "linear" | "log";
export type ParameterConditionValue = string | number | boolean;
export type ParameterConditions = Record<string, ParameterConditionValue>;

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

export interface EvaluationSpec {
  data?: readonly unknown[];
  loadData?: () => Promise<readonly unknown[]>;
}

export interface OptimizationExecutionSpec {
  mode?: "native" | "hybrid";
  algorithm?: "grid" | "random" | "bayesian" | "optuna";
  backendUrl?: string;
  apiKey?: string;
  timeoutMs?: number;
  requestTimeoutMs?: number;
  trialConcurrency?: number;
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
  backendUrl?: string;
  apiKey?: string;
  userId?: string;
  billingTier?: string;
  optimizationStrategy?: Record<string, unknown>;
  datasetMetadata?: Record<string, unknown>;
  timeoutMs?: number;
  requestTimeoutMs?: number;
  signal?: AbortSignal;
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
};
