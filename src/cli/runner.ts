#!/usr/bin/env node
/**
 * CLI runner for Python-to-JS bridge.
 *
 * This is the entry point that Python's JSBridge spawns.
 * It reads trial configs from stdin and writes results to stdout.
 *
 * Usage:
 *   node dist/cli/runner.js --module ./my-trial.js --function runTrial
 */

// IMPORTANT: Redirect console.log to stderr BEFORE any user code loads.
// This prevents user code from corrupting the NDJSON protocol on stdout.
console.log = (...args: unknown[]) => {
  console.error('[user:log]', ...args);
};
console.info = (...args: unknown[]) => {
  console.error('[user:info]', ...args);
};
console.debug = (...args: unknown[]) => {
  console.error('[user:debug]', ...args);
};
console.trace = (...args: unknown[]) => {
  console.error('[user:trace]', ...args);
};

process.on('unhandledRejection', (reason) => {
  console.error('[FATAL] Unhandled rejection:', reason);
  process.exit(1);
});

import { createInterface } from 'node:readline';
import { parseArgs } from 'node:util';
import { TrialContext } from '../core/context.js';
import {
  TimeoutError,
  CancelledError,
  BusyError,
  isTraigentError,
  getErrorCode,
  isRetryable,
} from '../core/errors.js';
import {
  type TrialConfig,
  TrialConfigSchema,
  createSuccessResult,
  createFailureResult,
} from '../dtos/trial.js';
import { sanitizeMeasures } from '../dtos/measures.js';
import {
  parseRequest,
  createSuccessResponse,
  createErrorResponse,
  serializeResponse,
  PROTOCOL_VERSION,
  MIN_PROTOCOL_VERSION,
  SUPPORTED_CAPABILITIES,
  ActionSchema,
  type CLIRequest,
  type CLIResponse,
} from './protocol.js';
import { validateConfigPayload } from './config-validation.js';

/** Start time for uptime calculation */
const startTime = Date.now();

/** Current trial state for cancellation support */
let currentTrialId: string | null = null;
let currentTrialAbortController: AbortController | null = null;
let currentTimeoutId: NodeJS.Timeout | undefined = undefined;

/**
 * Trial lock for race condition prevention.
 * Ensures only one trial can be starting at a time.
 */
let trialLockPromise: Promise<void> = Promise.resolve();

/**
 * Acquire the trial lock.
 * Returns an unlock function that MUST be called when done.
 */
async function acquireTrialLock(): Promise<() => void> {
  const previousLock = trialLockPromise;
  let unlock!: () => void;
  trialLockPromise = new Promise((resolve) => {
    unlock = resolve;
  });
  await previousLock;
  return unlock;
}

/** Maximum payload size (10MB) */
const MAX_PAYLOAD_SIZE = 10 * 1024 * 1024;

/** Maximum JSON depth to prevent DoS */
const MAX_JSON_DEPTH = 50;

/** Maximum inline rows */
const MAX_INLINE_ROWS = 100;

/** Maximum inline payload size (1MB) */
const MAX_INLINE_BYTES = 1024 * 1024;

/** Default timeout for trial execution (5 minutes) */
const DEFAULT_TIMEOUT_MS = 300_000;

/** Idle timeout - configurable via TRAIGENT_IDLE_TIMEOUT_MS (default 5 minutes) */
const IDLE_TIMEOUT_MS = Number.parseInt(process.env['TRAIGENT_IDLE_TIMEOUT_MS'] ?? '300000', 10);
let idleTimeoutId: NodeJS.Timeout | undefined = undefined;

/** Parent PID watcher interval (5 seconds) */
const PARENT_PID_CHECK_INTERVAL_MS = 5_000;
let parentPidIntervalId: NodeJS.Timeout | undefined = undefined;

/** Module-level readline reference for backpressure control */
let rl: ReturnType<typeof createInterface> | null = null;

/**
 * Backpressure state for stdout writes.
 * When stdout buffer is full, we queue writes and pause input.
 */
const writeQueue: string[] = [];
let draining = false;

/** Parsed CLI arguments */
interface RunnerArgs {
  module?: string;
  function?: string;
  help?: boolean;
}

/**
 * User's trial function signature.
 */
type UserTrialFunction = (config: TrialConfig) => Promise<{
  metrics: Record<string, unknown>;
  duration?: number;
  metadata?: Record<string, unknown>;
}>;

/**
 * Parse command line arguments.
 */
function parseCLIArgs(): RunnerArgs {
  const { values } = parseArgs({
    options: {
      module: { type: 'string', short: 'm' },
      function: { type: 'string', short: 'f', default: 'runTrial' },
      help: { type: 'boolean', short: 'h' },
    },
    strict: false,
  });
  return values as RunnerArgs;
}

/**
 * Print help message and exit.
 */
function printHelp(): void {
  console.error(`
Traigent JS SDK CLI Runner v${PROTOCOL_VERSION}

Usage:
  traigent-js --module <path> [--function <name>]

Options:
  -m, --module <path>    Path to the module containing the trial function
  -f, --function <name>  Name of the trial function to call (default: runTrial)
  -h, --help             Show this help message

The runner reads NDJSON requests from stdin and writes responses to stdout.
Logs should be written to stderr.
`);
  process.exit(0);
}

/**
 * Load the user's trial function from a module.
 */
async function loadTrialFunction(
  modulePath: string,
  functionName: string
): Promise<UserTrialFunction> {
  // Resolve relative paths
  const resolvedPath = modulePath.startsWith('.')
    ? new URL(modulePath, `file://${process.cwd()}/`).href
    : modulePath;

  // Dynamic import
  const module = (await import(resolvedPath)) as Record<string, unknown>;

  const fn = module[functionName];
  if (typeof fn !== 'function') {
    throw new TypeError(
      `Function "${functionName}" not found in module "${modulePath}". ` +
        `Available exports: ${Object.keys(module).join(', ')}`
    );
  }

  return fn as UserTrialFunction;
}

/**
 * Flush the write queue when stdout drains.
 */
function flushWriteQueue(): void {
  while (writeQueue.length > 0) {
    const next = writeQueue.shift()!;
    const ok = process.stdout.write(next);
    if (!ok) {
      // Still can't write, wait for another drain
      draining = true;
      process.stdout.once('drain', flushWriteQueue);
      return;
    }
  }
  // Queue fully flushed, resume input
  draining = false;
  if (rl) {
    rl.resume();
  }
}

/**
 * Send a response to stdout with backpressure handling.
 * When stdout buffer is full, queues writes and pauses input.
 */
function sendResponse(response: CLIResponse): void {
  const line = serializeResponse(response) + '\n';

  // If already draining, just queue
  if (draining) {
    writeQueue.push(line);
    return;
  }

  const ok = process.stdout.write(line);
  if (!ok) {
    // stdout buffer full - enable backpressure
    draining = true;
    if (rl) {
      rl.pause(); // INPUT BACKPRESSURE: Stop reading while we drain
    }
    process.stdout.once('drain', flushWriteQueue);
  }
}

/**
 * Log to stderr (not stdout, which is reserved for protocol).
 */
function log(message: string): void {
  console.error(`[traigent-js] ${message}`);
}

/**
 * Calculate normalized duration in seconds.
 * Supports duration_ms for migration from old code.
 */
function calculateNormalizedDuration(
  durationMs: number | undefined,
  durationSec: number | undefined,
  fallbackDuration: number,
  warnings: string[]
): number {
  if (durationMs !== undefined) {
    return durationMs / 1000;
  }
  if (durationSec !== undefined) {
    // Warn if it looks suspicious (> 1 hour suggests possible unit confusion)
    if (durationSec > 3600) {
      warnings.push(
        `Duration ${durationSec}s seems very long - ensure it's in seconds, not milliseconds`
      );
    }
    return durationSec;
  }
  return fallbackDuration;
}

/**
 * Handle a run_trial request.
 * Uses a lock to prevent race conditions when multiple requests arrive simultaneously.
 */
async function handleRunTrial(
  request: CLIRequest,
  trialFn: UserTrialFunction
): Promise<CLIResponse> {
  const trialStartTime = Date.now();

  // Acquire lock to prevent race condition
  const unlock = await acquireTrialLock();

  // CRITICAL: Check busy state AFTER acquiring lock, set flag BEFORE any async work
  if (currentTrialId !== null) {
    unlock();
    log(`Trial already running: ${currentTrialId}`);
    return createErrorResponse(
      request.request_id,
      new BusyError('Trial already running', currentTrialId),
      { errorCode: 'BUSY', retryable: true }
    );
  }

  // Validate payload FIRST with proper error code (not USER_FUNCTION_ERROR)
  const parseResult = TrialConfigSchema.safeParse(request.payload);
  if (!parseResult.success) {
    unlock();
    const errorMessage = `Invalid trial config: ${parseResult.error.message}`;
    log(`Validation error: ${errorMessage}`);

    const trialId = (request.payload as { trial_id?: string } | undefined)?.trial_id ?? 'unknown';

    // Include structured error details (truncate values, keep path + message only)
    const issues = parseResult.error.issues.slice(0, 10).map((issue) => ({
      path: issue.path.join('.'),
      message: issue.message,
      code: issue.code,
    }));
    const payload = createFailureResult(trialId, errorMessage, 'VALIDATION_ERROR', false, 0);
    // Add error details to metadata
    (payload as { metadata?: Record<string, unknown> }).metadata = {
      error_details: {
        issues,
        summary: parseResult.error.message,
        truncated: parseResult.error.issues.length > 10,
        total_issues: parseResult.error.issues.length,
      },
    };

    return createSuccessResponse(request.request_id, payload);
  }

  const config = parseResult.data;
  const timeoutMs =
    (request.payload as { timeout_ms?: number } | undefined)?.timeout_ms ?? DEFAULT_TIMEOUT_MS;

  // Set trial ID IMMEDIATELY after validation, before any async work
  currentTrialId = config.trial_id;
  currentTrialAbortController = new AbortController();
  const abortSignal = currentTrialAbortController.signal;

  log(`Running trial ${config.trial_id} (trial #${config.trial_number}, timeout: ${timeoutMs}ms)`);

  // Track warnings for the response
  const warnings: string[] = [];

  // Abort handler for cleanup
  let abortReject: ((error: Error) => void) | null = null;
  const abortHandler = () => {
    if (abortReject) abortReject(new CancelledError('Trial cancelled'));
  };

  try {
    // Execute trial within context with timeout and cancellation support
    const trialPromise = TrialContext.run(
      config,
      async () => {
        return await trialFn(config);
      },
      abortSignal
    );

    // Create timeout promise with proper cleanup
    const timeoutPromise = new Promise<never>((_, reject) => {
      currentTimeoutId = setTimeout(() => {
        reject(new TimeoutError('Trial timeout', timeoutMs));
      }, timeoutMs);
    });

    // Create abort promise - store reject for cleanup
    const abortPromise = new Promise<never>((_, reject) => {
      abortReject = reject;
      abortSignal.addEventListener('abort', abortHandler, { once: true });
    });

    const result = await Promise.race([trialPromise, timeoutPromise, abortPromise]);

    const duration = (Date.now() - trialStartTime) / 1000;

    // Guard metrics type before sanitizing
    let rawMetrics: Record<string, unknown> = result.metrics;
    if (rawMetrics === null || rawMetrics === undefined || typeof rawMetrics !== 'object') {
      warnings.push(`Invalid metrics type: ${typeof rawMetrics}, coerced to {}`);
      rawMetrics = {};
    }

    // Sanitize metrics and collect warnings
    const metrics = sanitizeMeasures(rawMetrics, {
      strict: false,
      warn: (msg) => {
        warnings.push(msg);
        log(`Warning: ${msg}`);
      },
    });

    // Use user-provided duration if available (must be in seconds)
    // Support duration_ms for migration from old code
    const resultWithDurationMs = result as { duration_ms?: number };
    const normalizedDuration = calculateNormalizedDuration(
      resultWithDurationMs.duration_ms,
      result.duration,
      duration,
      warnings
    );

    // Create success result with metadata if provided
    const payload = createSuccessResult(
      config.trial_id,
      metrics,
      normalizedDuration,
      result.metadata
    );

    // Add warnings if any
    if (warnings.length > 0) {
      (payload as { warnings?: string[]; metrics_sanitized?: boolean }).warnings = warnings;
      (payload as { warnings?: string[]; metrics_sanitized?: boolean }).metrics_sanitized = true;
    }

    log(`Trial ${config.trial_id} completed in ${duration.toFixed(2)}s`);

    return createSuccessResponse(request.request_id, payload);
  } catch (error) {
    const duration = (Date.now() - trialStartTime) / 1000;
    const errorMessage = error instanceof Error ? error.message : String(error);

    // Use typed error classification instead of string matching
    const errorCode = isTraigentError(error) ? getErrorCode(error) : 'USER_FUNCTION_ERROR';
    const retryable = isTraigentError(error) ? isRetryable(error) : false;
    const isCancelled = error instanceof CancelledError || abortSignal.aborted;

    log(`Trial failed: ${errorMessage}`);

    const payload = createFailureResult(
      config.trial_id,
      errorMessage,
      errorCode,
      retryable,
      duration
    );

    // Set status to 'cancelled' if aborted
    if (isCancelled) {
      (payload as { status: string }).status = 'cancelled';
    }

    return createSuccessResponse(request.request_id, payload);
  } finally {
    // Clean up abort listener (even with { once: true }, remove for older Node versions)
    abortSignal.removeEventListener('abort', abortHandler);

    // Clean up timeout
    if (currentTimeoutId !== undefined) {
      clearTimeout(currentTimeoutId);
      currentTimeoutId = undefined;
    }
    // Clear trial state
    currentTrialId = null;
    currentTrialAbortController = null;

    // Release lock
    unlock();
  }
}

/**
 * Handle a ping request.
 */
function handlePing(request: CLIRequest): CLIResponse {
  return createSuccessResponse(request.request_id, {
    timestamp: new Date().toISOString(),
    uptime_ms: Date.now() - startTime,
  });
}

/**
 * Handle a shutdown request.
 */
function handleShutdown(request: CLIRequest): CLIResponse {
  log('Shutdown requested');
  // Cancel any in-flight trial first
  if (currentTrialAbortController) {
    currentTrialAbortController.abort();
  }
  // Schedule exit after response is sent
  setImmediate(() => process.exit(0));
  return createSuccessResponse(request.request_id, { status: 'shutting_down' });
}

/**
 * Handle a cancel request.
 */
function handleCancel(request: CLIRequest): CLIResponse {
  const payload = request.payload as { trial_id?: string } | undefined;
  const requestedTrialId = payload?.trial_id;

  // If a specific trial ID is requested, check if it matches current
  if (requestedTrialId && currentTrialId !== requestedTrialId) {
    log(
      `Cancel requested for trial ${requestedTrialId} but current trial is ${currentTrialId ?? 'none'}`
    );
    return createSuccessResponse(request.request_id, {
      cancelled: false,
      reason: 'trial_not_found',
      requested_trial_id: requestedTrialId,
      current_trial_id: currentTrialId,
    });
  }

  if (currentTrialAbortController) {
    log(`Cancelling trial ${currentTrialId}`);
    currentTrialAbortController.abort();
    return createSuccessResponse(request.request_id, {
      cancelled: true,
      trial_id: currentTrialId,
    });
  }

  log('Cancel requested but no trial is running');
  return createSuccessResponse(request.request_id, {
    cancelled: false,
    reason: 'no_trial_running',
  });
}

/**
 * Handle a capabilities request.
 * Returns supported features and limits for protocol negotiation.
 */
function handleCapabilities(request: CLIRequest): CLIResponse {
  return createSuccessResponse(request.request_id, {
    protocol_version: PROTOCOL_VERSION,
    min_protocol_version: MIN_PROTOCOL_VERSION,
    capabilities: [...SUPPORTED_CAPABILITIES],
    supported_actions: ActionSchema.options,
    // Advertise limits to prevent guesswork
    limits: {
      max_line_bytes: MAX_PAYLOAD_SIZE,
      max_inline_rows: MAX_INLINE_ROWS,
      max_inline_bytes: MAX_INLINE_BYTES,
      max_json_depth: MAX_JSON_DEPTH,
    },
  });
}

/**
 * Handle a validate_config request.
 * Performs configuration validation without running a trial.
 * Supports optional JSON Schema Draft 7 validation when config_schema is provided.
 */
function handleValidateConfig(request: CLIRequest): CLIResponse {
  const payload = request.payload as
    | {
        config?: Record<string, unknown>;
        config_schema?: Record<string, unknown>;
      }
    | undefined;

  if (!payload?.config) {
    return createSuccessResponse(request.request_id, {
      ok: false,
      issues: [{ message: 'Missing required field: config' }],
      summary: 'Missing required field: config',
    });
  }

  return createSuccessResponse(
    request.request_id,
    validateConfigPayload(payload.config, payload.config_schema)
  );
}

/**
 * Reset idle timeout timer.
 * Called after each request is processed.
 */
function resetIdleTimer(): void {
  if (idleTimeoutId !== undefined) {
    clearTimeout(idleTimeoutId);
  }
  // Only set idle timeout if no trial is running
  if (currentTrialId === null) {
    idleTimeoutId = setTimeout(() => {
      log('Idle timeout - no requests received, exiting');
      process.exit(0);
    }, IDLE_TIMEOUT_MS);
  }
}

/**
 * Start parent PID watcher.
 * Exits if parent process dies (orphan detection).
 * Configurable via TRAIGENT_PARENT_PID env var for container environments.
 */
function startParentPidWatcher(): void {
  // Allow override for container environments where ppid is a supervisor
  const envPid = process.env['TRAIGENT_PARENT_PID'];
  const parentPid = envPid ? Number.parseInt(envPid, 10) : process.ppid;

  if (parentPid === undefined || parentPid <= 1 || Number.isNaN(parentPid)) {
    log('No valid parent PID, skipping orphan detection');
    return;
  }

  log(`Watching parent PID ${parentPid} for orphan detection`);

  parentPidIntervalId = setInterval(() => {
    try {
      // Signal 0 doesn't kill the process, just checks if it exists
      process.kill(parentPid, 0);
    } catch {
      log(`Parent process ${parentPid} died, exiting`);
      // Clean up before exit
      if (currentTrialAbortController) {
        currentTrialAbortController.abort();
      }
      process.exit(0);
    }
  }, PARENT_PID_CHECK_INTERVAL_MS);

  // Don't let this interval keep the process alive
  parentPidIntervalId.unref();
}

/**
 * Process a single request.
 * Handles payload size guard and dispatches to appropriate handler.
 */
async function processRequest(line: string, trialFn: UserTrialFunction): Promise<void> {
  // Payload size guard - use Buffer.byteLength for accurate UTF-8 byte count
  // (line.length undercounts multi-byte characters)
  const lineBytes = Buffer.byteLength(line, 'utf8');
  if (lineBytes > MAX_PAYLOAD_SIZE) {
    log(`Payload too large: ${lineBytes} bytes (max: ${MAX_PAYLOAD_SIZE})`);
    // Use 'unknown' request_id - don't parse to get it (defeats protection)
    const response = createErrorResponse(
      'unknown',
      `Payload too large: ${lineBytes} bytes exceeds ${MAX_PAYLOAD_SIZE} byte limit`,
      { errorCode: 'PAYLOAD_TOO_LARGE', retryable: false }
    );
    sendResponse(response);
    return;
  }

  let requestId = 'unknown';

  try {
    const request = parseRequest(line);
    requestId = request.request_id;

    let response: CLIResponse;

    switch (request.action) {
      case 'run_trial':
        response = await handleRunTrial(request, trialFn);
        break;
      case 'ping':
        response = handlePing(request);
        break;
      case 'shutdown':
        response = handleShutdown(request);
        break;
      case 'cancel':
        response = handleCancel(request);
        break;
      case 'capabilities':
        response = handleCapabilities(request);
        break;
      case 'validate_config':
        response = handleValidateConfig(request);
        break;
      default:
        response = createErrorResponse(requestId, `Unknown action: ${request.action}`, {
          errorCode: 'UNSUPPORTED_ACTION',
          retryable: false,
        });
    }

    sendResponse(response);
  } catch (error) {
    const response = createErrorResponse(
      requestId,
      error instanceof Error ? error : String(error),
      { errorCode: 'PROTOCOL_ERROR', retryable: false }
    );
    sendResponse(response);
  }
}

/**
 * Main entry point.
 */
async function main(): Promise<void> {
  const args = parseCLIArgs();

  if (args.help) {
    printHelp();
  }

  if (!args.module) {
    console.error('Error: --module is required');
    console.error('Run with --help for usage information');
    process.exit(1);
  }

  log(`Starting Traigent JS Runner v${PROTOCOL_VERSION}`);
  log(`Loading module: ${args.module}`);

  // Load user module with proper error reporting via NDJSON
  let trialFn: UserTrialFunction;
  try {
    trialFn = await loadTrialFunction(args.module, args.function ?? 'runTrial');
  } catch (error) {
    // MUST emit NDJSON error before exit so Python gets protocol response
    const errorMessage = error instanceof Error ? error.message : String(error);
    log(`Module load failed: ${errorMessage}`);
    const response = createErrorResponse(
      'init', // Explicit request_id for initialization errors
      `Module load failed: ${errorMessage}`,
      { errorCode: 'MODULE_LOAD_ERROR', retryable: false }
    );
    sendResponse(response);
    process.exit(1);
  }
  log(`Loaded function: ${args.function ?? 'runTrial'}`);

  // Start orphan detection (exit if parent dies)
  startParentPidWatcher();

  // Start idle timeout
  resetIdleTimer();

  log('Ready to receive requests');

  // Read NDJSON from stdin using event-based approach
  // CRITICAL: This allows cancel/ping to be processed while a trial is running
  // The old for-await approach blocked, preventing concurrent request handling
  // Assign to module-level variable for backpressure control
  rl = createInterface({
    input: process.stdin,
    crlfDelay: Infinity,
  });

  rl.on('line', (line: string) => {
    if (line.trim()) {
      // Process request without awaiting - allows concurrent handling
      // run_trial is still sequential (BUSY check), but cancel/ping can interrupt
      processRequest(line, trialFn)
        .then(() => {
          // Only reset idle timer on successful processing
          // This prevents malformed requests from keeping process alive indefinitely
          resetIdleTimer();
        })
        .catch((err) => {
          log(`Request processing error: ${err instanceof Error ? err.message : String(err)}`);
          // Still send a response on errors (processRequest already does this)
        });
    }
  });

  rl.on('close', () => {
    log('Stdin closed, exiting');
    // Clean up
    if (idleTimeoutId !== undefined) {
      clearTimeout(idleTimeoutId);
    }
    if (parentPidIntervalId !== undefined) {
      clearInterval(parentPidIntervalId);
    }
    process.exit(0);
  });

  rl.on('error', (err) => {
    log(`Readline error: ${err.message}`);
  });
}

// Run main with top-level await
try {
  await main();
} catch (error) {
  console.error('Fatal error:', error);
  process.exit(1);
}
