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
  TraigentError,
  TimeoutError,
  CancelledError,
  ValidationError,
  DatasetMismatchError,
  BusyError,
  UnsupportedActionError,
  PayloadTooLargeError,
  isTraigentError,
  getErrorCode,
  isRetryable,
} from './errors.js';

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
