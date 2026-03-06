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
  readonly code: ErrorCode;
  readonly retryable: boolean;

  constructor(
    message: string,
    code: ErrorCode,
    retryable = false,
  ) {
    super(message);
    this.name = this.constructor.name;
    this.code = code;
    this.retryable = retryable;

    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, this.constructor);
    }
  }
}

export class TimeoutError extends TraigentError {
  constructor(message = 'Trial timeout', timeoutMs?: number) {
    const fullMessage = timeoutMs ? `${message} after ${timeoutMs}ms` : message;
    super(fullMessage, 'TIMEOUT', true);
  }
}

export class CancelledError extends TraigentError {
  constructor(message = 'Trial cancelled') {
    super(message, 'CANCELLED', false);
  }
}

export class ValidationError extends TraigentError {
  readonly issues?: unknown[];
  readonly summary?: string;

  constructor(
    message: string,
    options?: {
      issues?: unknown[];
      summary?: string;
    },
  ) {
    super(message, 'VALIDATION_ERROR', false);
    this.issues = options?.issues;
    this.summary = options?.summary;
  }
}

export class DatasetMismatchError extends TraigentError {
  readonly expectedHash?: string;
  readonly actualHash?: string;

  constructor(
    message: string,
    expectedHash?: string,
    actualHash?: string,
  ) {
    super(message, 'VALIDATION_ERROR', false);
    this.expectedHash = expectedHash;
    this.actualHash = actualHash;
  }
}

export class BusyError extends TraigentError {
  readonly currentTrialId?: string;

  constructor(message = 'Trial already running', currentTrialId?: string) {
    super(message, 'INTERNAL_ERROR', true);
    this.currentTrialId = currentTrialId;
  }
}

export class UnsupportedActionError extends TraigentError {
  readonly action: string;

  constructor(action: string) {
    super(`Unsupported action: ${action}`, 'VALIDATION_ERROR', false);
    this.action = action;
  }
}

export class PayloadTooLargeError extends TraigentError {
  readonly size?: number;
  readonly maxSize?: number;

  constructor(message = 'Payload too large', size?: number, maxSize?: number) {
    super(message, 'VALIDATION_ERROR', false);
    this.size = size;
    this.maxSize = maxSize;
  }
}

export function isTraigentError(error: unknown): error is TraigentError {
  return error instanceof TraigentError;
}

export function getErrorCode(error: unknown): ErrorCode {
  if (isTraigentError(error)) {
    return error.code;
  }
  return 'USER_FUNCTION_ERROR';
}

export function isRetryable(error: unknown): boolean {
  if (isTraigentError(error)) {
    return error.retryable;
  }
  return false;
}
