/**
 * Custom error classes for Traigent SDK.
 *
 * These typed errors enable proper error classification without string matching.
 * All errors extend TraigentError and include a stable `code` property.
 */

import type { ErrorCode } from '../dtos/trial.js';

/**
 * Base error class for all Traigent SDK errors.
 * Includes a stable error code for wire-level classification.
 */
export class TraigentError extends Error {
  /** Stable error code for protocol responses */
  readonly code: ErrorCode;

  /** Whether this error is retryable */
  readonly retryable: boolean;

  constructor(
    message: string,
    code: ErrorCode,
    retryable = false
  ) {
    super(message);
    this.name = this.constructor.name;
    this.code = code;
    this.retryable = retryable;

    // Maintains proper stack trace for where error was thrown (V8 only)
    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, this.constructor);
    }
  }
}

/**
 * Thrown when a trial exceeds its timeout.
 * Retryable by default - transient failure.
 */
export class TimeoutError extends TraigentError {
  constructor(message = 'Trial timeout', timeoutMs?: number) {
    const fullMessage = timeoutMs
      ? `${message} after ${timeoutMs}ms`
      : message;
    super(fullMessage, 'TIMEOUT', true);
  }
}

/**
 * Thrown when a trial is explicitly cancelled.
 * Not retryable - intentional stop.
 */
export class CancelledError extends TraigentError {
  constructor(message = 'Trial cancelled') {
    super(message, 'CANCELLED', false);
  }
}

/**
 * Thrown when configuration or input validation fails.
 * Not retryable - same input would fail again.
 */
export class ValidationError extends TraigentError {
  /** Structured validation issues (from Zod or JSON Schema) */
  readonly issues?: unknown[];

  /** Human-readable summary */
  readonly summary?: string;

  constructor(
    message: string,
    options?: {
      issues?: unknown[];
      summary?: string;
    }
  ) {
    super(message, 'VALIDATION_ERROR', false);
    this.issues = options?.issues;
    this.summary = options?.summary;
  }
}

/**
 * Thrown when dataset hash verification fails.
 * Not retryable - data mismatch.
 */
export class DatasetMismatchError extends TraigentError {
  readonly expectedHash?: string;
  readonly actualHash?: string;

  constructor(
    message: string,
    expectedHash?: string,
    actualHash?: string
  ) {
    super(message, 'DATASET_MISMATCH', false);
    this.expectedHash = expectedHash;
    this.actualHash = actualHash;
  }
}

/**
 * Thrown when a trial is already running and another is requested.
 * Retryable - wait and try again.
 */
export class BusyError extends TraigentError {
  readonly currentTrialId?: string;

  constructor(message = 'Trial already running', currentTrialId?: string) {
    super(message, 'BUSY', true);
    this.currentTrialId = currentTrialId;
  }
}

/**
 * Thrown when an unsupported protocol action is received.
 * Not retryable - unknown action.
 */
export class UnsupportedActionError extends TraigentError {
  readonly action: string;

  constructor(action: string) {
    super(`Unsupported action: ${action}`, 'UNSUPPORTED_ACTION', false);
    this.action = action;
  }
}

/**
 * Thrown when a request payload exceeds size limits.
 * Not retryable - payload too large.
 */
export class PayloadTooLargeError extends TraigentError {
  readonly size?: number;
  readonly maxSize?: number;

  constructor(message = 'Payload too large', size?: number, maxSize?: number) {
    super(message, 'PAYLOAD_TOO_LARGE', false);
    this.size = size;
    this.maxSize = maxSize;
  }
}

/**
 * Check if an error is a Traigent error with a specific code.
 */
export function isTraigentError(error: unknown): error is TraigentError {
  return error instanceof TraigentError;
}

/**
 * Get the error code from any error.
 * Returns 'USER_FUNCTION_ERROR' for non-Traigent errors.
 */
export function getErrorCode(error: unknown): ErrorCode {
  if (isTraigentError(error)) {
    return error.code;
  }
  return 'USER_FUNCTION_ERROR';
}

/**
 * Check if an error is retryable.
 */
export function isRetryable(error: unknown): boolean {
  if (isTraigentError(error)) {
    return error.retryable;
  }
  return false;
}
