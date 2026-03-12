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
export {
  recordRuntimeMetrics,
  withRuntimeMetricsCollector,
} from "./core/runtime-metrics.js";

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
  getNextOptimizationTrial,
  param,
  finalizeOptimizationSession,
  getOptimizationSessionStatus,
  deleteOptimizationSession,
  listOptimizationSessions,
  submitOptimizationTrialResult,
  getOptimizationSpec,
  toHybridConfigSpace,
  loadTvlSpec,
  parseTvlSpec,
} from './optimization/index.js';

export {
  autoWrapFrameworkTarget,
  autoWrapFrameworkTargets,
  discoverFrameworkTargets,
  prepareFrameworkTargets,
} from "./integrations/auto-wrap.js";
export type {
  DiscoveredFrameworkTarget,
  PreparedFrameworkTargets,
  PrepareFrameworkTargetsOptions,
} from "./integrations/auto-wrap.js";
export {
  describeFrameworkAutoOverride,
  getRegisteredFrameworkTargets,
} from "./integrations/registry.js";

export type {
  AggregationStrategy,
  AgentCustomEvaluator,
  BandTarget,
  BandedObjectiveDefinition,
  BuiltInObjectiveName,
  ChanceConstraintDefinition,
  DerivedConstraintDefinition,
  EvaluationAggregationMap,
  EvaluationContext,
  EvaluationMetricFunction,
  EvaluationScoringFunction,
  EvaluationSpec,
  FloatParamDefinition,
  FrameworkAutoOverrideStatus,
  FrameworkTarget,
  HybridOptimizeOptions,
  HybridConfigSpace,
  HybridTunableDefinition,
  InjectionMode,
  InjectionSpec,
  IntParamDefinition,
  LoadedTvlOptimizationSpec,
  NativeOptimizedFunction,
  NativeOptimizeOptions,
  NativeTrialFunctionResult,
  ObjectiveDefinition,
  ObjectiveDirection,
  ObjectiveInput,
  OptimizationExecutionSpec,
  OptimizationReportingSummary,
  OptimizationServiceStatusResponse,
  OptimizeOptions,
  OptimizationBudget,
  OptimizationConstraints,
  OptimizationSessionDeleteOptions,
  OptimizationSessionDeleteResponse,
  OptimizationSessionCreateRequest,
  OptimizationSessionCreationResponse,
  OptimizationSessionDatasetSubset,
  OptimizationSessionFinalizeOptions,
  OptimizationSessionFinalizationResponse,
  OptimizationSessionListOptions,
  OptimizationSessionListResponse,
  OptimizationSessionNextTrialOptions,
  OptimizationSessionNextTrialResponse,
  OptimizationSessionRequestOptions,
  OptimizationSessionStatusResponse,
  OptimizationSessionStatusSummary,
  OptimizationSessionSubmitResultResponse,
  OptimizationSessionTrialResultInput,
  OptimizationSessionTrialSuggestion,
  ExecutionContract,
  PromotionPolicy,
  OptimizationResult,
  OptimizationSpec,
  OptimizationTrialRecord,
  ParameterConditions,
  ParameterConditionValue,
  ParameterDefinition,
  ParamScale,
  SeamlessResolution,
  StructuralConstraintDefinition,
  TvlLoadOptions,
} from './optimization/index.js';

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
