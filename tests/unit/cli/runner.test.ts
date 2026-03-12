/**
 * Unit tests for CLI runner functions.
 *
 * Note: The runner.ts file has module-level side effects (console redirection)
 * and a main() function that's hard to test directly. We focus on testing
 * the exported helper functions and protocol handling logic.
 */
import { describe, it, expect } from 'vitest';

// We need to test the functions that can be imported
// Since runner.ts has side effects, we'll test the protocol integration
import {
  parseRequest,
  createSuccessResponse,
  createErrorResponse,
  serializeResponse,
  PROTOCOL_VERSION,
} from '../../../src/cli/protocol.js';
import {
  TimeoutError,
  CancelledError,
  BusyError,
  ValidationError,
  isTraigentError,
  getErrorCode,
  isRetryable,
} from '../../../src/core/errors.js';
import { TrialContext } from '../../../src/core/context.js';
import {
  TrialConfigSchema,
  createSuccessResult,
  createFailureResult,
} from '../../../src/dtos/trial.js';
import { sanitizeMeasures } from '../../../src/dtos/measures.js';
import type { TrialConfig } from '../../../src/dtos/trial.js';

// Mock trial config factory
const createMockTrialConfig = (overrides: Partial<TrialConfig> = {}): TrialConfig => ({
  trial_id: 'trial-123',
  trial_number: 1,
  experiment_run_id: 'exp-456',
  config: { model: 'gpt-4o-mini', temperature: 0.7 },
  dataset_subset: { indices: [0, 1, 2], total: 10 },
  ...overrides,
});

describe('Runner Integration Tests', () => {
  describe('Request Parsing', () => {
    it('should parse valid run_trial request', () => {
      const request = {
        version: '1.0',
        request_id: 'req-001',
        action: 'run_trial',
        payload: createMockTrialConfig(),
      };

      const parsed = parseRequest(JSON.stringify(request));
      expect(parsed.action).toBe('run_trial');
      expect(parsed.request_id).toBe('req-001');
    });

    it('should parse valid ping request', () => {
      const request = {
        version: '1.0',
        request_id: 'req-002',
        action: 'ping',
        payload: {},
      };

      const parsed = parseRequest(JSON.stringify(request));
      expect(parsed.action).toBe('ping');
    });

    it('should parse valid shutdown request', () => {
      const request = {
        version: '1.0',
        request_id: 'req-003',
        action: 'shutdown',
        payload: {},
      };

      const parsed = parseRequest(JSON.stringify(request));
      expect(parsed.action).toBe('shutdown');
    });

    it('should parse valid cancel request', () => {
      const request = {
        version: '1.0',
        request_id: 'req-004',
        action: 'cancel',
        payload: { trial_id: 'trial-123' },
      };

      const parsed = parseRequest(JSON.stringify(request));
      expect(parsed.action).toBe('cancel');
    });

    it('should parse valid capabilities request', () => {
      const request = {
        version: '1.0',
        request_id: 'req-005',
        action: 'capabilities',
        payload: {},
      };

      const parsed = parseRequest(JSON.stringify(request));
      expect(parsed.action).toBe('capabilities');
    });

    it('should parse valid validate_config request', () => {
      const request = {
        version: '1.0',
        request_id: 'req-006',
        action: 'validate_config',
        payload: {
          config: { model: 'gpt-4' },
        },
      };

      const parsed = parseRequest(JSON.stringify(request));
      expect(parsed.action).toBe('validate_config');
    });
  });

  describe('Trial Config Validation', () => {
    it('should validate valid trial config', () => {
      const config = createMockTrialConfig();
      const result = TrialConfigSchema.safeParse(config);

      expect(result.success).toBe(true);
    });

    it('should reject config missing trial_id', () => {
      const config = {
        trial_number: 1,
        experiment_run_id: 'exp-456',
        config: {},
        dataset_subset: { indices: [], total: 0 },
      };

      const result = TrialConfigSchema.safeParse(config);
      expect(result.success).toBe(false);
    });

    it('should reject config missing trial_number', () => {
      const config = {
        trial_id: 'trial-123',
        experiment_run_id: 'exp-456',
        config: {},
        dataset_subset: { indices: [], total: 0 },
      };

      const result = TrialConfigSchema.safeParse(config);
      expect(result.success).toBe(false);
    });

    it('should reject config missing dataset_subset', () => {
      const config = {
        trial_id: 'trial-123',
        trial_number: 1,
        experiment_run_id: 'exp-456',
        config: {},
      };

      const result = TrialConfigSchema.safeParse(config);
      expect(result.success).toBe(false);
    });
  });

  describe('Response Creation', () => {
    it('should create ping response', () => {
      const response = createSuccessResponse('req-001', {
        timestamp: new Date().toISOString(),
        uptime_ms: 1000,
      });

      expect(response.version).toBe(PROTOCOL_VERSION);
      expect(response.status).toBe('success');
      expect(response.payload).toHaveProperty('timestamp');
      expect(response.payload).toHaveProperty('uptime_ms');
    });

    it('should create shutdown response', () => {
      const response = createSuccessResponse('req-001', {
        status: 'shutting_down',
      });

      expect(response.status).toBe('success');
      expect((response.payload as { status: string }).status).toBe('shutting_down');
    });

    it('should create cancel response when no trial running', () => {
      const response = createSuccessResponse('req-001', {
        cancelled: false,
        reason: 'no_trial_running',
      });

      expect((response.payload as { cancelled: boolean }).cancelled).toBe(false);
      expect((response.payload as { reason: string }).reason).toBe('no_trial_running');
    });

    it('should create cancel response when trial cancelled', () => {
      const response = createSuccessResponse('req-001', {
        cancelled: true,
        trial_id: 'trial-123',
      });

      expect((response.payload as { cancelled: boolean }).cancelled).toBe(true);
      expect((response.payload as { trial_id: string }).trial_id).toBe('trial-123');
    });

    it('should create capabilities response', () => {
      const response = createSuccessResponse('req-001', {
        protocol_version: PROTOCOL_VERSION,
        min_protocol_version: '1.0',
        capabilities: ['warnings'],
        supported_actions: ['run_trial', 'ping', 'shutdown', 'cancel', 'capabilities', 'validate_config'],
      });

      expect(response.status).toBe('success');
      const payload = response.payload as { capabilities: string[] };
      expect(payload.capabilities).not.toContain('validate_config');
    });

    it('should create validate_config success response', () => {
      const response = createSuccessResponse('req-001', {
        ok: true,
        summary: 'Config validation passed',
      });

      expect((response.payload as { ok: boolean }).ok).toBe(true);
    });

    it('should create validate_config failure response', () => {
      const response = createSuccessResponse('req-001', {
        ok: false,
        issues: [{ message: 'config must be an object' }],
        summary: 'Validation failed: 1 issue(s)',
      });

      expect((response.payload as { ok: boolean }).ok).toBe(false);
      expect((response.payload as { issues: unknown[] }).issues).toHaveLength(1);
    });
  });

  describe('Trial Result Creation', () => {
    it('should create success result', () => {
      const result = createSuccessResult(
        'trial-123',
        { accuracy: 0.95, latency_ms: 100 },
        1.5,
        { model: 'gpt-4' }
      );

      expect(result.trial_id).toBe('trial-123');
      expect(result.status).toBe('completed');
      expect(result.metrics.accuracy).toBe(0.95);
      expect(result.duration).toBe(1.5);
      expect(result.metadata?.model).toBe('gpt-4');
    });

    it('should create failure result', () => {
      const result = createFailureResult(
        'trial-123',
        'Something went wrong',
        'USER_FUNCTION_ERROR',
        false,
        1.0
      );

      expect(result.trial_id).toBe('trial-123');
      expect(result.status).toBe('failed');
      expect(result.error_message).toBe('Something went wrong');
      expect(result.error_code).toBe('USER_FUNCTION_ERROR');
      expect(result.retryable).toBe(false);
    });

    it('should create failure result with retryable flag', () => {
      const result = createFailureResult(
        'trial-123',
        'Timeout',
        'TIMEOUT',
        true,
        5.0
      );

      expect(result.retryable).toBe(true);
      expect(result.error_code).toBe('TIMEOUT');
    });
  });

  describe('Error Classification', () => {
    it('should classify TimeoutError correctly', () => {
      const error = new TimeoutError('Trial timeout', 5000);

      expect(isTraigentError(error)).toBe(true);
      expect(getErrorCode(error)).toBe('TIMEOUT');
      expect(isRetryable(error)).toBe(true);
    });

    it('should classify CancelledError correctly', () => {
      const error = new CancelledError();

      expect(isTraigentError(error)).toBe(true);
      expect(getErrorCode(error)).toBe('CANCELLED');
      expect(isRetryable(error)).toBe(false);
    });

    it('should classify BusyError correctly', () => {
      const error = new BusyError('Trial running', 'trial-123');

      expect(isTraigentError(error)).toBe(true);
      expect(getErrorCode(error)).toBe('BUSY');
      expect(isRetryable(error)).toBe(true);
    });

    it('should classify ValidationError correctly', () => {
      const error = new ValidationError('Invalid config');

      expect(isTraigentError(error)).toBe(true);
      expect(getErrorCode(error)).toBe('VALIDATION_ERROR');
      expect(isRetryable(error)).toBe(false);
    });

    it('should classify regular Error as USER_FUNCTION_ERROR', () => {
      const error = new Error('User code failed');

      expect(isTraigentError(error)).toBe(false);
      expect(getErrorCode(error)).toBe('USER_FUNCTION_ERROR');
      expect(isRetryable(error)).toBe(false);
    });
  });

  describe('Metrics Sanitization', () => {
    it('should pass through valid metrics', () => {
      const metrics = {
        accuracy: 0.95,
        latency_ms: 100,
        cost: 0.01,
      };

      const sanitized = sanitizeMeasures(metrics);
      expect(sanitized).toEqual(metrics);
    });

    it('should filter out non-numeric values', () => {
      const warnings: string[] = [];
      const metrics = {
        accuracy: 0.95,
        invalid: 'string value',
        nested: { value: 1 },
      };

      const sanitized = sanitizeMeasures(metrics, {
        strict: false,
        warn: (msg) => warnings.push(msg),
      });

      expect(sanitized.accuracy).toBe(0.95);
      expect(sanitized['invalid']).toBeUndefined();
      expect(warnings.length).toBeGreaterThan(0);
    });

    it('should handle NaN values', () => {
      const warnings: string[] = [];
      const metrics = {
        valid: 1.0,
        nan_value: NaN,
      };

      const sanitized = sanitizeMeasures(metrics, {
        strict: false,
        warn: (msg) => warnings.push(msg),
      });

      expect(sanitized.valid).toBe(1.0);
      expect(sanitized['nan_value']).toBeUndefined();
    });

    it('should handle Infinity values', () => {
      const warnings: string[] = [];
      const metrics = {
        valid: 1.0,
        inf_value: Infinity,
        neg_inf_value: -Infinity,
      };

      const sanitized = sanitizeMeasures(metrics, {
        strict: false,
        warn: (msg) => warnings.push(msg),
      });

      expect(sanitized.valid).toBe(1.0);
      expect(sanitized['inf_value']).toBeUndefined();
      expect(sanitized['neg_inf_value']).toBeUndefined();
      expect(warnings.length).toBeGreaterThan(0);
    });
  });

  describe('Trial Context Integration', () => {
    it('should run trial function within context', async () => {
      const config = createMockTrialConfig();

      const result = await TrialContext.run(config, async () => {
        expect(TrialContext.isInTrial()).toBe(true);
        expect(TrialContext.getTrialId()).toBe('trial-123');
        return { metrics: { accuracy: 0.95 } };
      });

      expect(result.metrics.accuracy).toBe(0.95);
    });

    it('should support abort signal in context', async () => {
      const config = createMockTrialConfig();
      const abortController = new AbortController();

      // Abort immediately
      abortController.abort();

      await expect(
        TrialContext.run(
          config,
          async () => {
            // Check if aborted
            if (abortController.signal.aborted) {
              throw new CancelledError();
            }
            return { metrics: { accuracy: 0.95 } };
          },
          abortController.signal
        )
      ).rejects.toBeInstanceOf(CancelledError);
    });

    it('should propagate context through async operations', async () => {
      const config = createMockTrialConfig({ trial_id: 'async-trial' });

      await TrialContext.run(config, async () => {
        // Simulate async operation
        await new Promise((resolve) => setTimeout(resolve, 10));
        expect(TrialContext.getTrialId()).toBe('async-trial');

        // Another level of async
        await Promise.resolve().then(() => {
          expect(TrialContext.getTrialId()).toBe('async-trial');
        });
      });
    });
  });

  describe('Error Response Generation', () => {
    it('should generate error response for payload too large', () => {
      const response = createErrorResponse(
        'unknown',
        'Payload too large: 15000000 bytes exceeds 10000000 byte limit',
        { errorCode: 'PAYLOAD_TOO_LARGE', retryable: false }
      );

      expect(response.status).toBe('error');
      const payload = response.payload as { error: string; error_code: string };
      expect(payload.error_code).toBe('PAYLOAD_TOO_LARGE');
    });

    it('should generate error response for protocol error', () => {
      const response = createErrorResponse(
        'req-001',
        'Invalid JSON',
        { errorCode: 'PROTOCOL_ERROR', retryable: false }
      );

      expect(response.status).toBe('error');
      const payload = response.payload as { error_code: string };
      expect(payload.error_code).toBe('PROTOCOL_ERROR');
    });

    it('should generate error response for unsupported action', () => {
      const response = createErrorResponse(
        'req-001',
        'Unknown action: invalid_action',
        { errorCode: 'UNSUPPORTED_ACTION', retryable: false }
      );

      expect(response.status).toBe('error');
      const payload = response.payload as { error_code: string };
      expect(payload.error_code).toBe('UNSUPPORTED_ACTION');
    });

    it('should generate error response for busy state', () => {
      const error = new BusyError('Trial already running', 'trial-123');
      const response = createErrorResponse(
        'req-001',
        error,
        { errorCode: 'BUSY', retryable: true }
      );

      expect(response.status).toBe('error');
      const payload = response.payload as { error_code: string; retryable: boolean };
      expect(payload.error_code).toBe('BUSY');
      expect(payload.retryable).toBe(true);
    });
  });

  describe('Response Serialization', () => {
    it('should serialize success response to NDJSON', () => {
      const response = createSuccessResponse('req-001', { data: 'test' });
      const serialized = serializeResponse(response);

      expect(serialized).not.toContain('\n');
      expect(JSON.parse(serialized)).toEqual(response);
    });

    it('should serialize error response to NDJSON', () => {
      const response = createErrorResponse('req-001', 'Error occurred');
      const serialized = serializeResponse(response);

      expect(serialized).not.toContain('\n');
      const parsed = JSON.parse(serialized);
      expect(parsed.status).toBe('error');
    });

    it('should handle special characters in payloads', () => {
      const response = createSuccessResponse('req-001', {
        message: 'Test with "quotes" and \\ backslash',
        unicode: '\u0000\u001f',
      });
      const serialized = serializeResponse(response);

      // Should be valid JSON
      const parsed = JSON.parse(serialized);
      expect(parsed.payload.message).toContain('quotes');
    });
  });

  describe('Validation Error Details', () => {
    it('should include structured error details for validation failures', () => {
      const config = {
        trial_id: 'trial-123',
        // Missing required fields
      };

      const parseResult = TrialConfigSchema.safeParse(config);
      expect(parseResult.success).toBe(false);

      if (!parseResult.success) {
        const issues = parseResult.error.issues.slice(0, 10).map(issue => ({
          path: issue.path.join('.'),
          message: issue.message,
          code: issue.code,
        }));

        expect(issues.length).toBeGreaterThan(0);
        expect(issues[0]).toHaveProperty('path');
        expect(issues[0]).toHaveProperty('message');
        expect(issues[0]).toHaveProperty('code');
      }
    });
  });

  describe('Duration Handling', () => {
    it('should use user-provided duration in seconds', () => {
      const result = createSuccessResult(
        'trial-123',
        { accuracy: 0.95 },
        2.5 // 2.5 seconds
      );

      expect(result.duration).toBe(2.5);
    });

    it('should handle zero duration', () => {
      const result = createSuccessResult(
        'trial-123',
        { accuracy: 0.95 },
        0
      );

      expect(result.duration).toBe(0);
    });
  });
});
