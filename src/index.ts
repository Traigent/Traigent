/**
 * @traigent/sdk - TypeScript SDK for Traigent LLM optimization platform.
 *
 * This SDK enables JavaScript/TypeScript LLM applications to participate
 * in Traigent's optimization loop via a Python-to-Node.js bridge.
 *
 * @example
 * ```typescript
 * import { TrialContext, getTrialConfig } from '@traigent/sdk';
 *
 * // Within a trial, access the current configuration
 * const config = getTrialConfig();
 * const model = config.model as string;
 * const temperature = config.temperature as number;
 * ```
 *
 * @packageDocumentation
 */

// Core exports
export {
  TrialContext,
  TrialContextError,
  TrialCancelledError,
  getTrialConfig,
  getTrialParam,
  wrapCallback,
  bindContext,
} from './core/context.js';

export type {
  ProgressCallback,
  ProgressInfo,
  TrialFunctionResult,
  TrialFunction,
  TraigentSDKConfig,
  TokenUsage,
  CostBreakdown,
  LLMMetrics,
} from './core/types.js';

// Optimization exports
export {
  optimize,
  checkOptimizationServiceStatus,
  createOptimizationSession,
  deleteOptimizationSession,
  finalizeOptimizationSession,
  getNextOptimizationTrial,
  getOptimizationSessionStatus,
  listOptimizationSessions,
  param,
  submitOptimizationTrialResult,
  getOptimizationSpec,
  getNativeTvlCompatibilityReport,
  toHybridConfigSpace,
  loadTvlSpec,
  parseTvlSpec,
} from './optimization/index.js';

export type {
  AggregationStrategy,
  BuiltInObjectiveName,
  EvaluationSpec,
  EvaluationAggregationMap,
  EvaluationContext,
  EvaluationMetricFunction,
  EvaluationScoringFunction,
  ExecutionContract,
  ExecutionMode,
  ExecutionSpec,
  FloatParamDefinition,
  FrameworkTarget,
  NativeTvlCompatibilityItem,
  NativeTvlCompatibilityReport,
  NativeTvlSupportStatus,
  HybridOptimizeOptions,
  HybridConfigSpace,
  HybridTunableDefinition,
  InjectionMode,
  InjectionSpec,
  IntParamDefinition,
  NativeOptimizedFunction,
  NativeOptimizeOptions,
  NativeTrialFunctionResult,
  ObjectiveDefinition,
  ObjectiveDirection,
  ObjectiveInput,
  OptimizeOptions,
  OptimizationBudget,
  OptimizationResult,
  OptimizationSpec,
  OptimizationTrialRecord,
  ParameterDefinition,
  ParamScale,
  PromotionChanceConstraintResult,
  PromotionDecision,
  PromotionObjectiveResult,
  SeamlessResolution,
  TvlLoadOptions,
  TvlPromotionPolicy,
  TvlSpecArtifact,
} from './optimization/index.js';

export {
  autoWrapFrameworkTarget,
  autoWrapFrameworkTargets,
  discoverFrameworkTargets,
  prepareFrameworkTargets,
} from './integrations/auto-wrap.js';
export {
  getRegisteredFrameworkTargets,
  describeFrameworkAutoOverride,
} from './integrations/registry.js';
export { createTraigentOpenAI } from './integrations/openai/index.js';
export { withTraigentModel } from './integrations/langchain/index.js';
export { withTraigent } from './integrations/vercel-ai/index.js';

// DTO exports
export {
  // Trial schemas
  DatasetSubsetSchema,
  TrialConfigSchema,
  TrialStatusSchema,
  ErrorCodeSchema,
  MetricsSchema,
  TrialResultPayloadSchema,
  // Trial helpers
  createSuccessResult,
  createFailureResult,
  // Measures schemas
  MeasuresDictSchema,
  // Measures helpers
  sanitizeMeasures,
  createEmptyMeasures,
  mergeMeasures,
  prefixMeasures,
  // Constants
  MAX_MEASURES_KEYS,
  MEASURE_KEY_PATTERN,
} from './dtos/index.js';

export type {
  DatasetSubset,
  TrialConfig,
  TrialStatus,
  ErrorCode,
  Metrics,
  TrialResultPayload,
  MeasuresDict,
} from './dtos/index.js';

// Protocol exports (for advanced usage)
export { PROTOCOL_VERSION } from './cli/protocol.js';

export {
  discoverTunedVariables,
  discoverTunedVariablesFromSource,
  discoverTunedVariablesFromFile,
} from './tuned-variables/index.js';

export type {
  DiscoverTunedVariablesOptions,
  DiscoveredTunedVariable,
  TunedVariableConfidence,
  TunedVariableDiscoveryResult,
  TunedVariableValueKind,
} from './tuned-variables/index.js';

export type {
  DiscoveredFrameworkTarget,
  PreparedFrameworkTargets,
  PrepareFrameworkTargetsOptions,
} from './integrations/auto-wrap.js';
