/**
 * Unit tests for CLI protocol types and functions.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import {
  PROTOCOL_VERSION,
  MIN_PROTOCOL_VERSION,
  SUPPORTED_CAPABILITIES,
  ProtocolVersionSchema,
  ActionSchema,
  CLIRequestSchema,
  RunTrialRequestSchema,
  PingRequestSchema,
  ShutdownRequestSchema,
  CancelRequestSchema,
  CapabilitiesRequestSchema,
  ValidateConfigRequestSchema,
  ResponseStatusSchema,
  CLIResponseSchema,
  ErrorPayloadSchema,
  PingResponseSchema,
  CapabilitiesPayloadSchema,
  ValidateConfigPayloadSchema,
  parseRequest,
  createSuccessResponse,
  createErrorResponse,
  serializeResponse,
} from '../../../src/cli/protocol.js';

describe('Protocol Constants', () => {
  it('should have valid protocol version', () => {
    expect(PROTOCOL_VERSION).toMatch(/^1\.\d+$/);
  });

  it('should have valid minimum protocol version', () => {
    expect(MIN_PROTOCOL_VERSION).toMatch(/^1\.\d+$/);
  });

  it('should have supported capabilities', () => {
    expect(SUPPORTED_CAPABILITIES).toContain('validate_config');
    expect(SUPPORTED_CAPABILITIES).toContain('dataset_hash');
    expect(SUPPORTED_CAPABILITIES).toContain('inline_rows');
    expect(SUPPORTED_CAPABILITIES).toContain('warnings');
    expect(SUPPORTED_CAPABILITIES).toContain('error_details');
  });
});

describe('ProtocolVersionSchema', () => {
  it('should accept valid 1.x versions', () => {
    expect(ProtocolVersionSchema.parse('1.0')).toBe('1.0');
    expect(ProtocolVersionSchema.parse('1.1')).toBe('1.1');
    expect(ProtocolVersionSchema.parse('1.99')).toBe('1.99');
  });

  it('should reject invalid versions', () => {
    expect(() => ProtocolVersionSchema.parse('2.0')).toThrow();
    expect(() => ProtocolVersionSchema.parse('0.1')).toThrow();
    expect(() => ProtocolVersionSchema.parse('1')).toThrow();
    expect(() => ProtocolVersionSchema.parse('1.0.0')).toThrow();
    expect(() => ProtocolVersionSchema.parse('v1.0')).toThrow();
  });
});

describe('ActionSchema', () => {
  it('should accept valid actions', () => {
    expect(ActionSchema.parse('run_trial')).toBe('run_trial');
    expect(ActionSchema.parse('ping')).toBe('ping');
    expect(ActionSchema.parse('shutdown')).toBe('shutdown');
    expect(ActionSchema.parse('cancel')).toBe('cancel');
    expect(ActionSchema.parse('capabilities')).toBe('capabilities');
    expect(ActionSchema.parse('validate_config')).toBe('validate_config');
  });

  it('should reject invalid actions', () => {
    expect(() => ActionSchema.parse('invalid')).toThrow();
    expect(() => ActionSchema.parse('')).toThrow();
  });
});

describe('CLIRequestSchema', () => {
  it('should accept valid requests', () => {
    const request = {
      version: '1.0',
      request_id: 'req-001',
      action: 'ping',
      payload: {},
    };

    const result = CLIRequestSchema.parse(request);
    expect(result.version).toBe('1.0');
    expect(result.request_id).toBe('req-001');
    expect(result.action).toBe('ping');
  });

  it('should reject requests with empty request_id', () => {
    const request = {
      version: '1.0',
      request_id: '',
      action: 'ping',
      payload: {},
    };

    expect(() => CLIRequestSchema.parse(request)).toThrow();
  });

  it('should reject requests with invalid version', () => {
    const request = {
      version: '2.0',
      request_id: 'req-001',
      action: 'ping',
      payload: {},
    };

    expect(() => CLIRequestSchema.parse(request)).toThrow();
  });
});

describe('RunTrialRequestSchema', () => {
  it('should accept valid run_trial requests', () => {
    const request = {
      version: '1.0',
      request_id: 'req-001',
      action: 'run_trial',
      payload: {
        trial_id: 'trial-123',
        trial_number: 1,
        experiment_run_id: 'exp-456',
        config: { model: 'gpt-4' },
        dataset_subset: { indices: [0, 1, 2], total: 10 },
      },
    };

    const result = RunTrialRequestSchema.parse(request);
    expect(result.action).toBe('run_trial');
    expect(result.payload.trial_id).toBe('trial-123');
  });

  it('should reject run_trial with invalid payload', () => {
    const request = {
      version: '1.0',
      request_id: 'req-001',
      action: 'run_trial',
      payload: { invalid: 'payload' },
    };

    expect(() => RunTrialRequestSchema.parse(request)).toThrow();
  });
});

describe('PingRequestSchema', () => {
  it('should accept valid ping requests', () => {
    const request = {
      version: '1.0',
      request_id: 'req-001',
      action: 'ping',
      payload: {},
    };

    const result = PingRequestSchema.parse(request);
    expect(result.action).toBe('ping');
  });

  it('should accept ping without payload', () => {
    const request = {
      version: '1.0',
      request_id: 'req-001',
      action: 'ping',
      payload: undefined,
    };

    expect(() => PingRequestSchema.parse(request)).not.toThrow();
  });
});

describe('ShutdownRequestSchema', () => {
  it('should accept valid shutdown requests', () => {
    const request = {
      version: '1.0',
      request_id: 'req-001',
      action: 'shutdown',
      payload: {},
    };

    const result = ShutdownRequestSchema.parse(request);
    expect(result.action).toBe('shutdown');
  });
});

describe('CancelRequestSchema', () => {
  it('should accept cancel with trial_id', () => {
    const request = {
      version: '1.0',
      request_id: 'req-001',
      action: 'cancel',
      payload: { trial_id: 'trial-123' },
    };

    const result = CancelRequestSchema.parse(request);
    expect(result.action).toBe('cancel');
    expect(result.payload?.trial_id).toBe('trial-123');
  });

  it('should accept cancel without trial_id', () => {
    const request = {
      version: '1.0',
      request_id: 'req-001',
      action: 'cancel',
      payload: {},
    };

    const result = CancelRequestSchema.parse(request);
    expect(result.payload?.trial_id).toBeUndefined();
  });
});

describe('CapabilitiesRequestSchema', () => {
  it('should accept valid capabilities requests', () => {
    const request = {
      version: '1.0',
      request_id: 'req-001',
      action: 'capabilities',
      payload: {},
    };

    const result = CapabilitiesRequestSchema.parse(request);
    expect(result.action).toBe('capabilities');
  });
});

describe('ValidateConfigRequestSchema', () => {
  it('should accept valid validate_config requests', () => {
    const request = {
      version: '1.0',
      request_id: 'req-001',
      action: 'validate_config',
      payload: {
        config: { model: 'gpt-4', temperature: 0.7 },
      },
    };

    const result = ValidateConfigRequestSchema.parse(request);
    expect(result.action).toBe('validate_config');
    expect(result.payload.config).toEqual({ model: 'gpt-4', temperature: 0.7 });
  });

  it('should accept validate_config with config_schema', () => {
    const request = {
      version: '1.0',
      request_id: 'req-001',
      action: 'validate_config',
      payload: {
        config: { model: 'gpt-4' },
        config_schema: { type: 'object', properties: {} },
      },
    };

    const result = ValidateConfigRequestSchema.parse(request);
    expect(result.payload.config_schema).toBeDefined();
  });
});

describe('ResponseStatusSchema', () => {
  it('should accept valid statuses', () => {
    expect(ResponseStatusSchema.parse('success')).toBe('success');
    expect(ResponseStatusSchema.parse('error')).toBe('error');
  });

  it('should reject invalid statuses', () => {
    expect(() => ResponseStatusSchema.parse('pending')).toThrow();
    expect(() => ResponseStatusSchema.parse('')).toThrow();
  });
});

describe('CLIResponseSchema', () => {
  it('should accept valid responses', () => {
    const response = {
      version: PROTOCOL_VERSION,
      request_id: 'req-001',
      status: 'success',
      payload: { data: 'test' },
    };

    const result = CLIResponseSchema.parse(response);
    expect(result.status).toBe('success');
  });

  it('should reject responses with wrong version', () => {
    const response = {
      version: '2.0',
      request_id: 'req-001',
      status: 'success',
      payload: {},
    };

    expect(() => CLIResponseSchema.parse(response)).toThrow();
  });
});

describe('ErrorPayloadSchema', () => {
  it('should accept valid error payloads', () => {
    const payload = {
      error: 'Something went wrong',
      error_code: 'TIMEOUT',
      stack: 'Error: ...\n  at test',
      retryable: true,
    };

    const result = ErrorPayloadSchema.parse(payload);
    expect(result.error).toBe('Something went wrong');
    expect(result.error_code).toBe('TIMEOUT');
    expect(result.retryable).toBe(true);
  });

  it('should default retryable to false', () => {
    const payload = {
      error: 'Test error',
    };

    const result = ErrorPayloadSchema.parse(payload);
    expect(result.retryable).toBe(false);
  });
});

describe('PingResponseSchema', () => {
  it('should accept valid ping responses', () => {
    const response = {
      version: PROTOCOL_VERSION,
      request_id: 'req-001',
      status: 'success',
      payload: {
        timestamp: '2024-01-01T00:00:00.000Z',
        uptime_ms: 1000,
      },
    };

    const result = PingResponseSchema.parse(response);
    expect(result.payload.timestamp).toBeDefined();
    expect(result.payload.uptime_ms).toBe(1000);
  });
});

describe('CapabilitiesPayloadSchema', () => {
  it('should accept valid capabilities payloads', () => {
    const payload = {
      protocol_version: '1.1',
      min_protocol_version: '1.0',
      capabilities: ['validate_config', 'warnings'],
      supported_actions: ['run_trial', 'ping'],
    };

    const result = CapabilitiesPayloadSchema.parse(payload);
    expect(result.capabilities).toContain('validate_config');
  });
});

describe('ValidateConfigPayloadSchema', () => {
  it('should accept valid validation response', () => {
    const payload = {
      ok: true,
    };

    const result = ValidateConfigPayloadSchema.parse(payload);
    expect(result.ok).toBe(true);
  });

  it('should accept validation response with issues', () => {
    const payload = {
      ok: false,
      issues: [{ path: 'config.model', message: 'Required' }],
      summary: '1 issue found',
      truncated: false,
      total_issues: 1,
    };

    const result = ValidateConfigPayloadSchema.parse(payload);
    expect(result.ok).toBe(false);
    expect(result.issues).toHaveLength(1);
  });
});

describe('parseRequest()', () => {
  it('should parse valid JSON request', () => {
    const line = JSON.stringify({
      version: '1.0',
      request_id: 'req-001',
      action: 'ping',
      payload: {},
    });

    const request = parseRequest(line);
    expect(request.action).toBe('ping');
    expect(request.request_id).toBe('req-001');
  });

  it('should throw on invalid JSON', () => {
    expect(() => parseRequest('not valid json')).toThrow();
  });

  it('should throw on invalid request schema', () => {
    const line = JSON.stringify({ invalid: 'data' });
    expect(() => parseRequest(line)).toThrow();
  });

  it('should throw on deeply nested JSON (DoS prevention)', () => {
    // Create deeply nested object (> 50 levels)
    let nested: Record<string, unknown> = { value: 'deep' };
    for (let i = 0; i < 55; i++) {
      nested = { nested };
    }

    const request = {
      version: '1.0',
      request_id: 'req-001',
      action: 'ping',
      payload: nested,
    };

    const line = JSON.stringify(request);
    expect(() => parseRequest(line)).toThrow(/depth/i);
  });

  it('should accept moderately nested JSON', () => {
    // Create nested object with 10 levels (should pass)
    let nested: Record<string, unknown> = { value: 'moderate' };
    for (let i = 0; i < 10; i++) {
      nested = { nested };
    }

    const request = {
      version: '1.0',
      request_id: 'req-001',
      action: 'ping',
      payload: nested,
    };

    const line = JSON.stringify(request);
    const result = parseRequest(line);
    expect(result.action).toBe('ping');
  });
});

describe('createSuccessResponse()', () => {
  it('should create success response with payload', () => {
    const response = createSuccessResponse('req-001', { data: 'test' });

    expect(response.version).toBe(PROTOCOL_VERSION);
    expect(response.request_id).toBe('req-001');
    expect(response.status).toBe('success');
    expect(response.payload).toEqual({ data: 'test' });
  });

  it('should create success response with empty payload', () => {
    const response = createSuccessResponse('req-002', {});

    expect(response.payload).toEqual({});
  });

  it('should create success response with complex payload', () => {
    const payload = {
      metrics: { accuracy: 0.95, latency_ms: 100 },
      metadata: { model: 'gpt-4' },
    };

    const response = createSuccessResponse('req-003', payload);
    expect(response.payload).toEqual(payload);
  });
});

describe('createErrorResponse()', () => {
  let originalEnv: string | undefined;

  beforeEach(() => {
    originalEnv = process.env['NODE_ENV'];
  });

  afterEach(() => {
    if (originalEnv !== undefined) {
      process.env['NODE_ENV'] = originalEnv;
    } else {
      delete process.env['NODE_ENV'];
    }
  });

  it('should create error response with Error object', () => {
    const error = new Error('Something failed');
    const response = createErrorResponse('req-001', error);

    expect(response.version).toBe(PROTOCOL_VERSION);
    expect(response.request_id).toBe('req-001');
    expect(response.status).toBe('error');
    expect((response.payload as { error: string }).error).toBe('Something failed');
  });

  it('should create error response with string message', () => {
    const response = createErrorResponse('req-002', 'String error message');

    expect((response.payload as { error: string }).error).toBe('String error message');
  });

  it('should include error code and retryable when provided', () => {
    const response = createErrorResponse('req-003', 'Test error', {
      errorCode: 'TIMEOUT',
      retryable: true,
    });

    const payload = response.payload as {
      error: string;
      error_code?: string;
      retryable?: boolean;
    };
    expect(payload.error_code).toBe('TIMEOUT');
    expect(payload.retryable).toBe(true);
  });

  it('should default retryable to false', () => {
    const response = createErrorResponse('req-004', 'Test error');

    const payload = response.payload as { retryable?: boolean };
    expect(payload.retryable).toBe(false);
  });

  it('should include sanitized stack in non-production', () => {
    process.env['NODE_ENV'] = 'development';
    const error = new Error('Dev error');
    const response = createErrorResponse('req-005', error);

    const payload = response.payload as { stack?: string };
    // Stack should exist in development
    expect(payload.stack).toBeDefined();
  });

  it('should hide stack in production', () => {
    process.env['NODE_ENV'] = 'production';
    const error = new Error('Prod error');
    const response = createErrorResponse('req-006', error);

    const payload = response.payload as { stack?: string };
    expect(payload.stack).toBeUndefined();
  });
});

describe('serializeResponse()', () => {
  it('should serialize response to JSON string', () => {
    const response = createSuccessResponse('req-001', { data: 'test' });
    const serialized = serializeResponse(response);

    expect(typeof serialized).toBe('string');
    expect(JSON.parse(serialized)).toEqual(response);
  });

  it('should produce single-line JSON (NDJSON format)', () => {
    const response = createSuccessResponse('req-001', {
      nested: { data: 'value' },
    });
    const serialized = serializeResponse(response);

    expect(serialized).not.toContain('\n');
  });
});
