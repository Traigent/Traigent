/**
 * Trial/runtime DTOs with Zod validation schemas.
 *
 * These match the JS/Python SDK bridge/runtime contract and align where
 * possible with canonical TraigentSchema value constraints (for example the
 * metric-value dictionary rules), but TrialConfig and TrialResultPayload are
 * SDK runtime transport objects rather than 1:1 copies of persisted platform
 * schemas such as configuration_run or hybrid_session.
 */
import { z } from 'zod';

/** Maximum inline rows allowed (to prevent memory issues) */
export const MAX_INLINE_ROWS = 100;

/** Maximum inline payload size (1MB) */
export const MAX_INLINE_PAYLOAD_SIZE = 1024 * 1024;

/**
 * Dataset subset information.
 *
 * Supports two modes:
 * 1. Indices mode: Pass indices to select from locally-loaded dataset
 * 2. Inline mode: Pass data rows directly (for small datasets or single queries)
 */
export const DatasetSubsetSchema = z.object({
  /** Indices of examples to evaluate in this trial (indices mode) */
  indices: z.array(z.number().int().nonnegative()),
  /** Total number of examples in the full dataset */
  total: z.number().int().positive(),
  /** Optional hash for reproducibility verification */
  hash: z.string().optional(),
  /** Inline data rows for direct data passing (v1.1) - max 100 rows */
  inline_rows: z.array(z.record(z.string(), z.unknown())).max(MAX_INLINE_ROWS).optional(),
});

export type DatasetSubset = z.infer<typeof DatasetSubsetSchema>;

/**
 * Trial configuration sent from Python orchestrator to JS runtime.
 */
export const TrialConfigSchema = z.object({
  /** Unique identifier for this trial */
  trial_id: z.string().min(1),
  /** Sequential trial number within the optimization run */
  trial_number: z.number().int().nonnegative(),
  /** Parent experiment run ID */
  experiment_run_id: z.string().min(1),
  /** Configuration parameters to test (model, temperature, etc.) */
  config: z.record(z.string(), z.unknown()),
  /** Dataset subset for this trial */
  dataset_subset: DatasetSubsetSchema,
  /** Optional metadata */
  metadata: z.record(z.string(), z.unknown()).optional(),
});

export type TrialConfig = z.infer<typeof TrialConfigSchema>;

/**
 * Trial status enum matching Python SDK.
 */
export const TrialStatusSchema = z.enum([
  'pending',
  'running',
  'completed',
  'failed',
  'cancelled',
  'pruned',
]);

export type TrialStatus = z.infer<typeof TrialStatusSchema>;

/**
 * Structured error codes for retry logic.
 * IMPORTANT: All error codes MUST be in this enum - no ad-hoc strings allowed.
 */
export const ErrorCodeSchema = z.enum([
  'TIMEOUT',
  'VALIDATION_ERROR',
  'LLM_ERROR',
  'RATE_LIMIT',
  'BUDGET_EXCEEDED',
  'INTERNAL_ERROR',
  'USER_FUNCTION_ERROR',
  'CANCELLED',
  // New error codes (v1.1)
  'BUSY', // Trial already running
  'DATASET_MISMATCH', // Hash verification failed
  'UNSUPPORTED_ACTION', // Unknown protocol action
  'PAYLOAD_TOO_LARGE', // Request exceeds size limit
  'MODULE_LOAD_ERROR', // User module failed to load
  'PROTOCOL_ERROR', // NDJSON parse or protocol error
]);

export type ErrorCode = z.infer<typeof ErrorCodeSchema>;

/**
 * Metrics dictionary with Python identifier key validation.
 * Keys must match: ^[a-zA-Z_][a-zA-Z0-9_]*$
 */
export const MetricsSchema = z.record(
  z.string().regex(/^[a-zA-Z_]\w*$/, {
    message: 'Metric keys must be valid Python identifiers',
  }),
  z.number().finite().nullable()
);

export type Metrics = z.infer<typeof MetricsSchema>;

/**
 * Trial result payload sent from JS runtime back to Python orchestrator.
 */
export const TrialResultPayloadSchema = z.object({
  /** Trial ID (must match the request) */
  trial_id: z.string().min(1),
  /** Trial completion status */
  status: TrialStatusSchema,
  /** Computed metrics (accuracy, latency, cost, etc.) */
  metrics: MetricsSchema,
  /** Execution duration in seconds */
  duration: z.number().nonnegative(),
  /** Error message if status is 'failed' */
  error_message: z.string().nullable(),
  /** Structured error code for retry logic */
  error_code: ErrorCodeSchema.nullable().optional(),
  /** Whether the error is retryable */
  retryable: z.boolean().optional().default(false),
  /** Optional result metadata */
  metadata: z.record(z.string(), z.unknown()).optional(),
  /** Warnings generated during trial execution (v1.1) */
  warnings: z.array(z.string()).optional(),
  /** Whether metrics were sanitized (invalid values dropped) (v1.1) */
  metrics_sanitized: z.boolean().optional(),
});

export type TrialResultPayload = z.infer<typeof TrialResultPayloadSchema>;

/**
 * Create a successful trial result.
 */
export function createSuccessResult(
  trialId: string,
  metrics: Metrics,
  duration: number,
  metadata?: Record<string, unknown>
): TrialResultPayload {
  return {
    trial_id: trialId,
    status: 'completed',
    metrics,
    duration,
    error_message: null,
    error_code: null,
    retryable: false,
    metadata,
  };
}

/**
 * Create a failed trial result.
 */
export function createFailureResult(
  trialId: string,
  errorMessage: string,
  errorCode: ErrorCode = 'INTERNAL_ERROR',
  retryable = false,
  duration = 0
): TrialResultPayload {
  return {
    trial_id: trialId,
    status: 'failed',
    metrics: {},
    duration,
    error_message: errorMessage,
    error_code: errorCode,
    retryable,
  };
}
