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
  param,
  getOptimizationSpec,
  toHybridConfigSpace,
  loadTvlSpec,
  parseTvlSpec,
} from './optimization/index.js';

export type {
  BandTarget,
  BandedObjectiveDefinition,
  BuiltInObjectiveName,
  ChanceConstraintDefinition,
  DerivedConstraintDefinition,
  EvaluationSpec,
  FloatParamDefinition,
  HybridOptimizeOptions,
  HybridConfigSpace,
  HybridTunableDefinition,
  IntParamDefinition,
  LoadedTvlOptimizationSpec,
  NativeOptimizedFunction,
  NativeOptimizeOptions,
  NativeTrialFunctionResult,
  ObjectiveDefinition,
  ObjectiveDirection,
  ObjectiveInput,
  OptimizationExecutionSpec,
  OptimizeOptions,
  OptimizationBudget,
  OptimizationConstraints,
  PromotionPolicy,
  OptimizationResult,
  OptimizationSpec,
  OptimizationTrialRecord,
  ParameterConditions,
  ParameterConditionValue,
  ParameterDefinition,
  ParamScale,
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
