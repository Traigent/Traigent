/**
 * Unit tests for trial DTOs.
 */
import { describe, it, expect } from 'vitest';
import {
  TrialConfigSchema,
  TrialResultPayloadSchema,
  MetricsSchema,
  createSuccessResult,
  createFailureResult,
} from '../../../src/dtos/trial.js';

describe('TrialConfigSchema', () => {
  it('should validate a valid trial config', () => {
    const config = {
      trial_id: 'trial-123',
      trial_number: 1,
      experiment_run_id: 'exp-456',
      config: { model: 'gpt-4', temperature: 0.7 },
      dataset_subset: { indices: [0, 1, 2], total: 100 },
    };

    const result = TrialConfigSchema.safeParse(config);
    expect(result.success).toBe(true);
  });

  it('should reject missing required fields', () => {
    const config = {
      trial_id: 'trial-123',
      // Missing trial_number
    };

    const result = TrialConfigSchema.safeParse(config);
    expect(result.success).toBe(false);
  });

  it('should accept optional metadata', () => {
    const config = {
      trial_id: 'trial-123',
      trial_number: 1,
      experiment_run_id: 'exp-456',
      config: {},
      dataset_subset: { indices: [], total: 0 },
      metadata: { custom: 'value' },
    };

    // total must be positive
    const result = TrialConfigSchema.safeParse({
      ...config,
      dataset_subset: { indices: [], total: 1 },
    });
    expect(result.success).toBe(true);
  });

  it('should validate dataset_subset with hash', () => {
    const config = {
      trial_id: 'trial-123',
      trial_number: 1,
      experiment_run_id: 'exp-456',
      config: {},
      dataset_subset: { indices: [0], total: 10, hash: 'sha256:abc123' },
    };

    const result = TrialConfigSchema.safeParse(config);
    expect(result.success).toBe(true);
  });
});

describe('MetricsSchema', () => {
  it('should validate valid Python identifier keys', () => {
    const metrics = {
      accuracy: 0.95,
      latency_ms: 120.5,
      total_cost: 0.002,
    };

    const result = MetricsSchema.safeParse(metrics);
    expect(result.success).toBe(true);
  });

  it('should reject invalid keys with hyphens', () => {
    const metrics = {
      'accuracy-score': 0.95, // Invalid: hyphen
    };

    const result = MetricsSchema.safeParse(metrics);
    expect(result.success).toBe(false);
  });

  it('should reject keys starting with numbers', () => {
    const metrics = {
      '1st_place': 0.95, // Invalid: starts with number
    };

    const result = MetricsSchema.safeParse(metrics);
    expect(result.success).toBe(false);
  });

  it('should accept null values', () => {
    const metrics = {
      accuracy: 0.95,
      missing_value: null,
    };

    const result = MetricsSchema.safeParse(metrics);
    expect(result.success).toBe(true);
  });

  it('should reject infinite metric values', () => {
    const metrics = {
      accuracy: Number.POSITIVE_INFINITY,
    };

    const result = MetricsSchema.safeParse(metrics);
    expect(result.success).toBe(false);
  });

  it('should accept underscore-prefixed keys', () => {
    const metrics = {
      _private_metric: 0.5,
      __double_underscore: 0.3,
    };

    const result = MetricsSchema.safeParse(metrics);
    expect(result.success).toBe(true);
  });
});

describe('TrialResultPayloadSchema', () => {
  it('should validate a successful result', () => {
    const result = {
      trial_id: 'trial-123',
      status: 'completed',
      metrics: { accuracy: 0.95 },
      duration: 1.234,
      error_message: null,
    };

    const parsed = TrialResultPayloadSchema.safeParse(result);
    expect(parsed.success).toBe(true);
  });

  it('should validate a failed result with error code', () => {
    const result = {
      trial_id: 'trial-123',
      status: 'failed',
      metrics: {},
      duration: 0.5,
      error_message: 'Something went wrong',
      error_code: 'TIMEOUT',
      retryable: true,
    };

    const parsed = TrialResultPayloadSchema.safeParse(result);
    expect(parsed.success).toBe(true);
  });

  it('should reject invalid status', () => {
    const result = {
      trial_id: 'trial-123',
      status: 'invalid_status',
      metrics: {},
      duration: 0,
      error_message: null,
    };

    const parsed = TrialResultPayloadSchema.safeParse(result);
    expect(parsed.success).toBe(false);
  });
});

describe('createSuccessResult()', () => {
  it('should create a success result with correct structure', () => {
    const result = createSuccessResult('trial-123', { accuracy: 0.95, latency: 100 }, 1.5);

    expect(result).toEqual({
      trial_id: 'trial-123',
      status: 'completed',
      metrics: { accuracy: 0.95, latency: 100 },
      duration: 1.5,
      error_message: null,
      error_code: null,
      retryable: false,
      metadata: undefined,
    });
  });

  it('should include metadata when provided', () => {
    const result = createSuccessResult('trial-123', { accuracy: 0.95 }, 1.0, { custom: 'data' });

    expect(result.metadata).toEqual({ custom: 'data' });
  });
});

describe('createFailureResult()', () => {
  it('should create a failure result with error details', () => {
    const result = createFailureResult('trial-123', 'Connection timeout', 'TIMEOUT', true, 0.5);

    expect(result).toEqual({
      trial_id: 'trial-123',
      status: 'failed',
      metrics: {},
      duration: 0.5,
      error_message: 'Connection timeout',
      error_code: 'TIMEOUT',
      retryable: true,
    });
  });

  it('should use defaults when optional params not provided', () => {
    const result = createFailureResult('trial-123', 'Error occurred');

    expect(result.error_code).toBe('INTERNAL_ERROR');
    expect(result.retryable).toBe(false);
    expect(result.duration).toBe(0);
  });
});
