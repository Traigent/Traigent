/**
 * DTO exports for trial configuration and results.
 */
export {
  // Trial schemas
  DatasetSubsetSchema,
  TrialConfigSchema,
  TrialStatusSchema,
  ErrorCodeSchema,
  MetricsSchema,
  TrialResultPayloadSchema,
  // Trial types
  type DatasetSubset,
  type TrialConfig,
  type TrialStatus,
  type ErrorCode,
  type Metrics,
  type TrialResultPayload,
  // Trial helpers
  createSuccessResult,
  createFailureResult,
} from './trial.js';

export {
  // Measures schemas
  MeasureKeySchema,
  MeasureValueSchema,
  MeasuresDictSchema,
  // Measures types
  type MeasuresDict,
  // Measures constants
  MAX_MEASURES_KEYS,
  MEASURE_KEY_PATTERN,
  // Measures helpers
  sanitizeMeasures,
  createEmptyMeasures,
  mergeMeasures,
  prefixMeasures,
} from './measures.js';
