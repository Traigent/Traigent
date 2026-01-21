/**
 * NDJSON protocol types for Python-to-JS bridge communication.
 *
 * Protocol rules:
 * - Use newline-delimited JSON (NDJSON) framing
 * - stdout reserved for protocol messages
 * - stderr reserved for logs
 * - Each message is a single JSON object on one line
 */
import { z } from 'zod';
import { TrialConfigSchema, TrialResultPayloadSchema } from '../dtos/trial.js';

/** Current protocol version */
export const PROTOCOL_VERSION = '1.0';

/**
 * Action types supported by the protocol.
 */
export const ActionSchema = z.enum([
  'run_trial',
  'ping',
  'shutdown',
]);

export type Action = z.infer<typeof ActionSchema>;

/**
 * Request message from Python orchestrator to JS runtime.
 */
export const CLIRequestSchema = z.object({
  /** Protocol version for compatibility checking */
  version: z.literal(PROTOCOL_VERSION),
  /** Unique request ID for correlation */
  request_id: z.string().min(1),
  /** Action to perform */
  action: ActionSchema,
  /** Request payload (depends on action) */
  payload: z.unknown(),
});

export type CLIRequest = z.infer<typeof CLIRequestSchema>;

/**
 * Run trial request with validated payload.
 */
export const RunTrialRequestSchema = CLIRequestSchema.extend({
  action: z.literal('run_trial'),
  payload: TrialConfigSchema,
});

export type RunTrialRequest = z.infer<typeof RunTrialRequestSchema>;

/**
 * Ping request for health check / keepalive.
 */
export const PingRequestSchema = CLIRequestSchema.extend({
  action: z.literal('ping'),
  payload: z.object({}).optional(),
});

export type PingRequest = z.infer<typeof PingRequestSchema>;

/**
 * Shutdown request for graceful termination.
 */
export const ShutdownRequestSchema = CLIRequestSchema.extend({
  action: z.literal('shutdown'),
  payload: z.object({}).optional(),
});

export type ShutdownRequest = z.infer<typeof ShutdownRequestSchema>;

/**
 * Response status.
 */
export const ResponseStatusSchema = z.enum(['success', 'error']);

export type ResponseStatus = z.infer<typeof ResponseStatusSchema>;

/**
 * Base response message from JS runtime to Python orchestrator.
 */
export const CLIResponseSchema = z.object({
  /** Protocol version */
  version: z.literal(PROTOCOL_VERSION),
  /** Correlation ID (matches request_id) */
  request_id: z.string().min(1),
  /** Response status */
  status: ResponseStatusSchema,
  /** Response payload */
  payload: z.unknown(),
});

export type CLIResponse = z.infer<typeof CLIResponseSchema>;

/**
 * Successful trial response.
 */
export const TrialSuccessResponseSchema = CLIResponseSchema.extend({
  status: z.literal('success'),
  payload: TrialResultPayloadSchema,
});

export type TrialSuccessResponse = z.infer<typeof TrialSuccessResponseSchema>;

/**
 * Error response payload.
 */
export const ErrorPayloadSchema = z.object({
  error: z.string(),
  error_code: z.string().optional(),
  stack: z.string().optional(),
  retryable: z.boolean().optional().default(false),
});

export type ErrorPayload = z.infer<typeof ErrorPayloadSchema>;

/**
 * Error response.
 */
export const ErrorResponseSchema = CLIResponseSchema.extend({
  status: z.literal('error'),
  payload: ErrorPayloadSchema,
});

export type ErrorResponse = z.infer<typeof ErrorResponseSchema>;

/**
 * Ping response.
 */
export const PingResponseSchema = CLIResponseSchema.extend({
  status: z.literal('success'),
  payload: z.object({
    timestamp: z.string(),
    uptime_ms: z.number(),
  }),
});

export type PingResponse = z.infer<typeof PingResponseSchema>;

/**
 * Parse and validate an incoming request.
 */
export function parseRequest(line: string): CLIRequest {
  const json = JSON.parse(line) as unknown;
  return CLIRequestSchema.parse(json);
}

/**
 * Create a success response.
 */
export function createSuccessResponse<T>(
  requestId: string,
  payload: T
): CLIResponse {
  return {
    version: PROTOCOL_VERSION,
    request_id: requestId,
    status: 'success',
    payload,
  };
}

/**
 * Create an error response.
 */
export function createErrorResponse(
  requestId: string,
  error: Error | string,
  options: { errorCode?: string; retryable?: boolean } = {}
): CLIResponse {
  const errorMessage = error instanceof Error ? error.message : error;
  const stack = error instanceof Error ? error.stack : undefined;

  return {
    version: PROTOCOL_VERSION,
    request_id: requestId,
    status: 'error',
    payload: {
      error: errorMessage,
      error_code: options.errorCode,
      stack,
      retryable: options.retryable ?? false,
    },
  };
}

/**
 * Serialize a response to NDJSON format (single line).
 */
export function serializeResponse(response: CLIResponse): string {
  return JSON.stringify(response);
}
