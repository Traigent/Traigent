/**
 * Unit tests for Traigent error classes and utilities.
 */
import { describe, it, expect } from 'vitest';
import {
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
} from '../../../src/core/errors.js';

describe('TraigentError', () => {
  it('should create error with message, code, and retryable flag', () => {
    const error = new TraigentError('Test error', 'TIMEOUT', true);

    expect(error.message).toBe('Test error');
    expect(error.code).toBe('TIMEOUT');
    expect(error.retryable).toBe(true);
    expect(error.name).toBe('TraigentError');
  });

  it('should default retryable to false', () => {
    const error = new TraigentError('Test error', 'CANCELLED');

    expect(error.retryable).toBe(false);
  });

  it('should be an instance of Error', () => {
    const error = new TraigentError('Test', 'TIMEOUT');

    expect(error).toBeInstanceOf(Error);
    expect(error).toBeInstanceOf(TraigentError);
  });

  it('should have a stack trace', () => {
    const error = new TraigentError('Test', 'TIMEOUT');

    expect(error.stack).toBeDefined();
    expect(error.stack).toContain('TraigentError');
  });
});

describe('TimeoutError', () => {
  it('should create with default message', () => {
    const error = new TimeoutError();

    expect(error.message).toBe('Trial timeout');
    expect(error.code).toBe('TIMEOUT');
    expect(error.retryable).toBe(true);
    expect(error.name).toBe('TimeoutError');
  });

  it('should create with custom message', () => {
    const error = new TimeoutError('Custom timeout message');

    expect(error.message).toBe('Custom timeout message');
  });

  it('should include timeout duration in message when provided', () => {
    const error = new TimeoutError('Trial timeout', 5000);

    expect(error.message).toBe('Trial timeout after 5000ms');
  });

  it('should be an instance of TraigentError', () => {
    const error = new TimeoutError();

    expect(error).toBeInstanceOf(TraigentError);
    expect(error).toBeInstanceOf(TimeoutError);
  });
});

describe('CancelledError', () => {
  it('should create with default message', () => {
    const error = new CancelledError();

    expect(error.message).toBe('Trial cancelled');
    expect(error.code).toBe('CANCELLED');
    expect(error.retryable).toBe(false);
    expect(error.name).toBe('CancelledError');
  });

  it('should create with custom message', () => {
    const error = new CancelledError('User requested cancellation');

    expect(error.message).toBe('User requested cancellation');
  });

  it('should be an instance of TraigentError', () => {
    const error = new CancelledError();

    expect(error).toBeInstanceOf(TraigentError);
  });
});

describe('ValidationError', () => {
  it('should create with message only', () => {
    const error = new ValidationError('Invalid config');

    expect(error.message).toBe('Invalid config');
    expect(error.code).toBe('VALIDATION_ERROR');
    expect(error.retryable).toBe(false);
    expect(error.issues).toBeUndefined();
    expect(error.summary).toBeUndefined();
    expect(error.name).toBe('ValidationError');
  });

  it('should create with issues and summary', () => {
    const issues = [
      { path: ['config', 'model'], message: 'Required' },
      { path: ['config', 'temperature'], message: 'Must be a number' },
    ];
    const error = new ValidationError('Validation failed', {
      issues,
      summary: '2 validation errors',
    });

    expect(error.issues).toEqual(issues);
    expect(error.summary).toBe('2 validation errors');
  });

  it('should be an instance of TraigentError', () => {
    const error = new ValidationError('Test');

    expect(error).toBeInstanceOf(TraigentError);
  });
});

describe('DatasetMismatchError', () => {
  it('should create with message only', () => {
    const error = new DatasetMismatchError('Dataset hash mismatch');

    expect(error.message).toBe('Dataset hash mismatch');
    expect(error.code).toBe('DATASET_MISMATCH');
    expect(error.retryable).toBe(false);
    expect(error.expectedHash).toBeUndefined();
    expect(error.actualHash).toBeUndefined();
    expect(error.name).toBe('DatasetMismatchError');
  });

  it('should create with expected and actual hashes', () => {
    const error = new DatasetMismatchError(
      'Dataset hash mismatch',
      'abc123',
      'def456'
    );

    expect(error.expectedHash).toBe('abc123');
    expect(error.actualHash).toBe('def456');
  });

  it('should be an instance of TraigentError', () => {
    const error = new DatasetMismatchError('Test');

    expect(error).toBeInstanceOf(TraigentError);
  });
});

describe('BusyError', () => {
  it('should create with default message', () => {
    const error = new BusyError();

    expect(error.message).toBe('Trial already running');
    expect(error.code).toBe('BUSY');
    expect(error.retryable).toBe(true);
    expect(error.currentTrialId).toBeUndefined();
    expect(error.name).toBe('BusyError');
  });

  it('should create with custom message and current trial ID', () => {
    const error = new BusyError('System busy', 'trial-123');

    expect(error.message).toBe('System busy');
    expect(error.currentTrialId).toBe('trial-123');
  });

  it('should be an instance of TraigentError', () => {
    const error = new BusyError();

    expect(error).toBeInstanceOf(TraigentError);
  });
});

describe('UnsupportedActionError', () => {
  it('should create with action name', () => {
    const error = new UnsupportedActionError('unknown_action');

    expect(error.message).toBe('Unsupported action: unknown_action');
    expect(error.code).toBe('UNSUPPORTED_ACTION');
    expect(error.retryable).toBe(false);
    expect(error.action).toBe('unknown_action');
    expect(error.name).toBe('UnsupportedActionError');
  });

  it('should be an instance of TraigentError', () => {
    const error = new UnsupportedActionError('test');

    expect(error).toBeInstanceOf(TraigentError);
  });
});

describe('PayloadTooLargeError', () => {
  it('should create with default message', () => {
    const error = new PayloadTooLargeError();

    expect(error.message).toBe('Payload too large');
    expect(error.code).toBe('PAYLOAD_TOO_LARGE');
    expect(error.retryable).toBe(false);
    expect(error.size).toBeUndefined();
    expect(error.maxSize).toBeUndefined();
    expect(error.name).toBe('PayloadTooLargeError');
  });

  it('should create with custom message, size, and maxSize', () => {
    const error = new PayloadTooLargeError('Request too big', 15_000_000, 10_000_000);

    expect(error.message).toBe('Request too big');
    expect(error.size).toBe(15_000_000);
    expect(error.maxSize).toBe(10_000_000);
  });

  it('should be an instance of TraigentError', () => {
    const error = new PayloadTooLargeError();

    expect(error).toBeInstanceOf(TraigentError);
  });
});

describe('isTraigentError()', () => {
  it('should return true for TraigentError instances', () => {
    expect(isTraigentError(new TraigentError('Test', 'TIMEOUT'))).toBe(true);
    expect(isTraigentError(new TimeoutError())).toBe(true);
    expect(isTraigentError(new CancelledError())).toBe(true);
    expect(isTraigentError(new ValidationError('Test'))).toBe(true);
    expect(isTraigentError(new DatasetMismatchError('Test'))).toBe(true);
    expect(isTraigentError(new BusyError())).toBe(true);
    expect(isTraigentError(new UnsupportedActionError('test'))).toBe(true);
    expect(isTraigentError(new PayloadTooLargeError())).toBe(true);
  });

  it('should return false for non-TraigentError values', () => {
    expect(isTraigentError(new Error('Regular error'))).toBe(false);
    expect(isTraigentError('string error')).toBe(false);
    expect(isTraigentError(null)).toBe(false);
    expect(isTraigentError(undefined)).toBe(false);
    expect(isTraigentError({ code: 'TIMEOUT', message: 'fake' })).toBe(false);
    expect(isTraigentError(42)).toBe(false);
  });
});

describe('getErrorCode()', () => {
  it('should return error code for TraigentError instances', () => {
    expect(getErrorCode(new TimeoutError())).toBe('TIMEOUT');
    expect(getErrorCode(new CancelledError())).toBe('CANCELLED');
    expect(getErrorCode(new ValidationError('Test'))).toBe('VALIDATION_ERROR');
    expect(getErrorCode(new DatasetMismatchError('Test'))).toBe('DATASET_MISMATCH');
    expect(getErrorCode(new BusyError())).toBe('BUSY');
    expect(getErrorCode(new UnsupportedActionError('test'))).toBe('UNSUPPORTED_ACTION');
    expect(getErrorCode(new PayloadTooLargeError())).toBe('PAYLOAD_TOO_LARGE');
  });

  it('should return USER_FUNCTION_ERROR for non-TraigentError values', () => {
    expect(getErrorCode(new Error('Regular error'))).toBe('USER_FUNCTION_ERROR');
    expect(getErrorCode('string error')).toBe('USER_FUNCTION_ERROR');
    expect(getErrorCode(null)).toBe('USER_FUNCTION_ERROR');
    expect(getErrorCode(undefined)).toBe('USER_FUNCTION_ERROR');
  });
});

describe('isRetryable()', () => {
  it('should return true for retryable TraigentError instances', () => {
    expect(isRetryable(new TimeoutError())).toBe(true);
    expect(isRetryable(new BusyError())).toBe(true);
  });

  it('should return false for non-retryable TraigentError instances', () => {
    expect(isRetryable(new CancelledError())).toBe(false);
    expect(isRetryable(new ValidationError('Test'))).toBe(false);
    expect(isRetryable(new DatasetMismatchError('Test'))).toBe(false);
    expect(isRetryable(new UnsupportedActionError('test'))).toBe(false);
    expect(isRetryable(new PayloadTooLargeError())).toBe(false);
  });

  it('should return false for non-TraigentError values', () => {
    expect(isRetryable(new Error('Regular error'))).toBe(false);
    expect(isRetryable('string error')).toBe(false);
    expect(isRetryable(null)).toBe(false);
    expect(isRetryable(undefined)).toBe(false);
  });
});
