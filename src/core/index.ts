/**
 * Core module exports.
 */
export {
  TrialContext,
  TrialContextError,
  getTrialConfig,
  getTrialParam,
  wrapCallback,
  bindContext,
} from './context.js';
export {
  recordRuntimeMetrics,
  withRuntimeMetricsCollector,
} from './runtime-metrics.js';

export type {
  ProgressCallback,
  ProgressInfo,
  TrialFunctionResult,
  TrialFunction,
  TraigentSDKConfig,
  TokenUsage,
  CostBreakdown,
  LLMMetrics,
} from './types.js';
