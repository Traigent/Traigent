/**
 * NDJSON protocol types for Python-to-JS bridge communication.
 *
 * Protocol rules:
 * - Use newline-delimited JSON (NDJSON) framing
 * - stdout reserved for protocol messages
 * - stderr reserved for logs
 * - Each message is a single JSON object on one line
 *
 * Version compatibility:
 * - Major version mismatch: reject (breaking changes)
 * - Minor version mismatch: accept (additive features)
 */
import { z } from 'zod';
import { TrialConfigSchema, TrialResultPayloadSchema } from '../dtos/trial.js';

/** Current protocol version (runner's version) */
export const PROTOCOL_VERSION = '1.1';

/** Minimum supported protocol version */
export const MIN_PROTOCOL_VERSION = '1.0';

/**
 * Protocol version schema - accepts any 1.x version for forward compatibility.
 * Major version changes (2.x) would be rejected.
 */
export const ProtocolVersionSchema = z.string().regex(/^1\.\d+$/, {
  message: 'Protocol version must be 1.x (e.g., "1.0", "1.1")',
});

/**
 * Capabilities supported by this runner.
 * Python can query these via the 'capabilities' action.
 */
export const SUPPORTED_CAPABILITIES = [
  'validate_config',   // Config validation before trial
  'dataset_hash',      // Dataset hash verification
  'inline_rows',       // Inline data mode (vs indices)
  'warnings',          // Warnings array in response
  'error_details',     // Structured error details
] as const;

export type Capability = (typeof SUPPORTED_CAPABILITIES)[number];

/**
 * Action types supported by the protocol.
 */
export const ActionSchema = z.enum([
  'run_trial',
  'ping',
  'shutdown',
  'cancel',           // Cancel an in-flight trial
  'capabilities',     // Query supported capabilities (v1.1)
  'validate_config',  // Validate config without running (v1.1)
]);

export type Action = z.infer<typeof ActionSchema>;

/**
 * Request message from Python orchestrator to JS runtime.
 */
export const CLIRequestSchema = z.object({
  /** Protocol version for compatibility checking (accepts 1.x) */
  version: ProtocolVersionSchema,
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
 * Cancel request to abort an in-flight trial.
 */
export const CancelRequestSchema = CLIRequestSchema.extend({
  action: z.literal('cancel'),
  payload: z.object({
    /** Trial ID to cancel (optional - if not provided, cancels current trial) */
    trial_id: z.string().optional(),
  }).optional(),
});

export type CancelRequest = z.infer<typeof CancelRequestSchema>;

/**
 * Capabilities request to query supported features.
 */
export const CapabilitiesRequestSchema = CLIRequestSchema.extend({
  action: z.literal('capabilities'),
  payload: z.object({}).optional(),
});

export type CapabilitiesRequest = z.infer<typeof CapabilitiesRequestSchema>;

/**
 * Validate config request to check configuration before running trial.
 * Uses JSON Schema Draft 7 for cross-language validation.
 */
export const ValidateConfigRequestSchema = CLIRequestSchema.extend({
  action: z.literal('validate_config'),
  payload: z.object({
    /** Configuration to validate */
    config: z.record(z.string(), z.unknown()),
    /** Optional JSON Schema Draft 7 for validation */
    config_schema: z.record(z.string(), z.unknown()).optional(),
  }),
});

export type ValidateConfigRequest = z.infer<typeof ValidateConfigRequestSchema>;

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
 * Capabilities response payload.
 */
export const CapabilitiesPayloadSchema = z.object({
  protocol_version: z.string(),
  min_protocol_version: z.string(),
  capabilities: z.array(z.string()),
  supported_actions: z.array(z.string()),
});

export type CapabilitiesPayload = z.infer<typeof CapabilitiesPayloadSchema>;

/**
 * Capabilities response.
 */
export const CapabilitiesResponseSchema = CLIResponseSchema.extend({
  status: z.literal('success'),
  payload: CapabilitiesPayloadSchema,
});

export type CapabilitiesResponse = z.infer<typeof CapabilitiesResponseSchema>;

/**
 * Validate config response payload.
 */
export const ValidateConfigPayloadSchema = z.object({
  ok: z.boolean(),
  issues: z.array(z.unknown()).optional(),
  summary: z.string().optional(),
  truncated: z.boolean().optional(),
  total_issues: z.number().optional(),
});

export type ValidateConfigPayload = z.infer<typeof ValidateConfigPayloadSchema>;

/**
 * Validate config response.
 */
export const ValidateConfigResponseSchema = CLIResponseSchema.extend({
  status: z.literal('success'),
  payload: ValidateConfigPayloadSchema,
});

export type ValidateConfigResponse = z.infer<typeof ValidateConfigResponseSchema>;

/** Maximum JSON depth to prevent DoS attacks */
const MAX_JSON_DEPTH = 50;

/**
 * Check JSON depth to prevent stack overflow from deeply nested objects.
 */
function checkJsonDepth(obj: unknown, maxDepth = MAX_JSON_DEPTH, current = 0): boolean {
  if (current > maxDepth) return false;
  if (typeof obj === 'object' && obj !== null) {
    return Object.values(obj).every(v => checkJsonDepth(v, maxDepth, current + 1));
  }
  return true;
}

/**
 * Parse and validate an incoming request.
 * Includes JSON depth check to prevent DoS attacks.
 */
export function parseRequest(line: string): CLIRequest {
  const json = JSON.parse(line) as unknown;

  // Check depth to prevent DoS from deeply nested JSON
  if (!checkJsonDepth(json)) {
    throw new Error(`JSON depth exceeds maximum allowed (${MAX_JSON_DEPTH} levels)`);
  }

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
 * Sanitize stack trace for security.
 * In production, hide stack traces. Otherwise, truncate and strip absolute paths.
 */
function sanitizeStack(stack: string | undefined): string | undefined {
  if (!stack) return undefined;

  // In production, don't expose stack traces
  if (process.env['NODE_ENV'] === 'production') return undefined;

  // Keep first 5 lines, strip absolute paths to avoid leaking host layout
  return stack
    .split('\n')
    .slice(0, 5)
    .map(line => line.replaceAll(/\(\/[^)]+\//g, '('))
    .join('\n');
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
  const rawStack = error instanceof Error ? error.stack : undefined;

  return {
    version: PROTOCOL_VERSION,
    request_id: requestId,
    status: 'error',
    payload: {
      error: errorMessage,
      error_code: options.errorCode,
      stack: sanitizeStack(rawStack),
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
