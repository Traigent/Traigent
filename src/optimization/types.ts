import type { Metrics, TrialConfig } from '../dtos/trial.js';

export type ObjectiveDirection = 'maximize' | 'minimize';
export type BuiltInObjectiveName = 'accuracy' | 'cost' | 'latency';
export type ParamScale = 'linear' | 'log';

export interface ObjectiveDefinition {
  metric: string;
  direction: ObjectiveDirection;
  weight?: number;
}

export type ObjectiveInput = BuiltInObjectiveName | ObjectiveDefinition;

export interface EnumParamDefinition<T = string | number | boolean> {
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

export interface EvaluationSpec {
  data?: readonly unknown[];
  loadData?: () => Promise<readonly unknown[]>;
}

export interface OptimizationSpec {
  configurationSpace: Record<string, ParameterDefinition>;
  objectives: readonly ObjectiveInput[];
  budget?: OptimizationBudget;
  evaluation?: EvaluationSpec;
}

export interface NormalizedObjectiveDefinition {
  metric: string;
  direction: ObjectiveDirection;
  weight: number;
}

export interface NormalizedOptimizationSpec {
  configurationSpace: Record<string, ParameterDefinition>;
  objectives: readonly NormalizedObjectiveDefinition[];
  budget?: OptimizationBudget;
  evaluation?: EvaluationSpec;
}

export interface NativeOptimizeOptions {
  algorithm: 'grid' | 'random' | 'bayesian';
  maxTrials: number;
  randomSeed?: number;
  timeoutMs?: number;
}

export interface OptimizationTrialRecord {
  trialId: string;
  trialNumber: number;
  config: TrialConfig['config'];
  metrics: Metrics;
  duration: number;
  metadata?: Record<string, unknown>;
}

export interface OptimizationResult {
  bestConfig: TrialConfig['config'] | null;
  bestMetrics: Metrics | null;
  trials: OptimizationTrialRecord[];
  stopReason: 'completed' | 'maxTrials' | 'budget' | 'timeout' | 'error';
  totalCostUsd: number;
  errorMessage?: string;
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
};
