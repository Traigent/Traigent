/**
 * Trial DTOs with Zod validation schemas.
 * These mirror the Python SDK's traigent/cloud/dtos.py structures.
 */
import { z } from 'zod';

/**
 * Dataset subset information for indices-only data passing.
 */
export const DatasetSubsetSchema = z.object({
  /** Indices of examples to evaluate in this trial */
  indices: z.array(z.number().int().nonnegative()),
  /** Total number of examples in the full dataset */
  total: z.number().int().positive(),
  /** Optional hash for reproducibility verification */
  hash: z.string().optional(),
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
 */
export const ErrorCodeSchema = z.enum([
  'TIMEOUT',
  'VALIDATION_ERROR',
  'LLM_ERROR',
  'RATE_LIMIT',
  'BUDGET_EXCEEDED',
  'INTERNAL_ERROR',
  'USER_FUNCTION_ERROR',
]);

export type ErrorCode = z.infer<typeof ErrorCodeSchema>;

/**
 * Metrics dictionary with Python identifier key validation.
 * Keys must match: ^[a-zA-Z_][a-zA-Z0-9_]*$
 */
export const MetricsSchema = z.record(
  z.string().regex(/^[a-zA-Z_][a-zA-Z0-9_]*$/, {
    message: 'Metric keys must be valid Python identifiers',
  }),
  z.number().nullable()
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
