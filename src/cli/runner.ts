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

import { createInterface } from 'node:readline';
import { parseArgs } from 'node:util';
import { TrialContext } from '../core/context.js';
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
  type CLIRequest,
  type CLIResponse,
} from './protocol.js';

/** Start time for uptime calculation */
const startTime = Date.now();

/** Current trial state for cancellation support */
let currentTrialId: string | null = null;
let currentTrialAbortController: AbortController | null = null;

/** Parsed CLI arguments */
interface RunnerArgs {
  module?: string;
  function?: string;
  help?: boolean;
}

/**
 * User's trial function signature.
 */
type UserTrialFunction = (
  config: TrialConfig
) => Promise<{
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
    throw new Error(
      `Function "${functionName}" not found in module "${modulePath}". ` +
        `Available exports: ${Object.keys(module).join(', ')}`
    );
  }

  return fn as UserTrialFunction;
}

/**
 * Send a response to stdout.
 */
function sendResponse(response: CLIResponse): void {
  const line = serializeResponse(response);
  process.stdout.write(line + '\n');
}

/**
 * Log to stderr (not stdout, which is reserved for protocol).
 */
function log(message: string): void {
  console.error(`[traigent-js] ${message}`);
}

/** Default timeout for trial execution (5 minutes) */
const DEFAULT_TIMEOUT_MS = 300_000;

/**
 * Handle a run_trial request.
 */
async function handleRunTrial(
  request: CLIRequest,
  trialFn: UserTrialFunction
): Promise<CLIResponse> {
  const trialStartTime = Date.now();

  // Validate payload FIRST with proper error code (not USER_FUNCTION_ERROR)
  const parseResult = TrialConfigSchema.safeParse(request.payload);
  if (!parseResult.success) {
    const errorMessage = `Invalid trial config: ${parseResult.error.message}`;
    log(`Validation error: ${errorMessage}`);

    const trialId =
      (request.payload as { trial_id?: string } | undefined)?.trial_id ??
      'unknown';

    const payload = createFailureResult(
      trialId,
      errorMessage,
      'VALIDATION_ERROR',
      false,
      0
    );

    return createSuccessResponse(request.request_id, payload);
  }

  const config = parseResult.data;
  const timeoutMs =
    (request.payload as { timeout_ms?: number } | undefined)?.timeout_ms ??
    DEFAULT_TIMEOUT_MS;

  // Set up cancellation support
  currentTrialId = config.trial_id;
  currentTrialAbortController = new AbortController();
  const abortSignal = currentTrialAbortController.signal;

  log(`Running trial ${config.trial_id} (trial #${config.trial_number}, timeout: ${timeoutMs}ms)`);

  try {
    // Execute trial within context with timeout and cancellation support
    const trialPromise = TrialContext.run(config, async () => {
      return await trialFn(config);
    }, abortSignal);

    const timeoutPromise = new Promise<never>((_, reject) => {
      setTimeout(() => reject(new Error(`Trial timeout after ${timeoutMs}ms`)), timeoutMs);
    });

    const abortPromise = new Promise<never>((_, reject) => {
      abortSignal.addEventListener('abort', () => {
        reject(new Error('Trial cancelled'));
      });
    });

    const result = await Promise.race([trialPromise, timeoutPromise, abortPromise]);

    const duration = (Date.now() - trialStartTime) / 1000;

    // Sanitize metrics
    const metrics = sanitizeMeasures(result.metrics, {
      strict: false,
      warn: (msg) => log(`Warning: ${msg}`),
    });

    // Normalize duration: if user provides value > 1000, assume milliseconds and convert
    const userDuration = result.duration;
    const normalizedDuration =
      userDuration !== undefined
        ? userDuration > 1000
          ? userDuration / 1000
          : userDuration
        : duration;

    // Create success result with metadata if provided
    const payload = createSuccessResult(
      config.trial_id,
      metrics,
      normalizedDuration,
      result.metadata
    );

    log(`Trial ${config.trial_id} completed in ${duration.toFixed(2)}s`);

    return createSuccessResponse(request.request_id, payload);
  } catch (error) {
    const duration = (Date.now() - trialStartTime) / 1000;
    const errorMessage = error instanceof Error ? error.message : String(error);
    const isTimeout = errorMessage.includes('Trial timeout');
    const isCancelled = errorMessage.includes('Trial cancelled') || abortSignal.aborted;

    log(`Trial failed: ${errorMessage}`);

    const payload = createFailureResult(
      config.trial_id,
      errorMessage,
      isCancelled ? 'CANCELLED' : isTimeout ? 'TIMEOUT' : 'USER_FUNCTION_ERROR',
      isTimeout, // Timeouts are retryable, cancellations are not
      duration
    );

    // Set status to 'cancelled' if aborted
    if (isCancelled) {
      (payload as { status: string }).status = 'cancelled';
    }

    return createSuccessResponse(request.request_id, payload);
  } finally {
    // Clear trial state
    currentTrialId = null;
    currentTrialAbortController = null;
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
    log(`Cancel requested for trial ${requestedTrialId} but current trial is ${currentTrialId ?? 'none'}`);
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
 * Process a single request.
 */
async function processRequest(
  line: string,
  trialFn: UserTrialFunction
): Promise<void> {
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
      default:
        response = createErrorResponse(requestId, `Unknown action: ${request.action}`);
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

  const trialFn = await loadTrialFunction(args.module, args.function ?? 'runTrial');
  log(`Loaded function: ${args.function ?? 'runTrial'}`);
  log('Ready to receive requests');

  // Read NDJSON from stdin
  const rl = createInterface({
    input: process.stdin,
    crlfDelay: Infinity,
  });

  for await (const line of rl) {
    if (line.trim()) {
      await processRequest(line, trialFn);
    }
  }

  log('Stdin closed, exiting');
}

// Run main with top-level await
try {
  await main();
} catch (error) {
  console.error('Fatal error:', error);
  process.exit(1);
}
