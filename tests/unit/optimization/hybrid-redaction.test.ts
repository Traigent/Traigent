import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import {
  createOptimizationSession,
  getOptimizationSessionStatus,
  listOptimizationSessions,
} from '../../../src/index.js';
import { param } from '../../../src/optimization/spec.js';

function jsonResponse(status: number, payload: unknown) {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => payload,
    text: async () => JSON.stringify(payload),
  };
}

function textResponse(status: number, body: string) {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => {
      throw new Error('no json body');
    },
    text: async () => body,
  };
}

describe('hybrid redaction', () => {
  const fetchMock = vi.fn();

  beforeEach(() => {
    fetchMock.mockReset();
    vi.stubGlobal('fetch', fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.unstubAllEnvs();
  });

  it('does not leak api keys or backend urls in thrown request errors', async () => {
    vi.stubEnv('TRAIGENT_BACKEND_URL', 'http://localhost:5000');
    vi.stubEnv('TRAIGENT_API_KEY', 'super-secret-key');

    fetchMock.mockResolvedValueOnce(
      textResponse(
        401,
        JSON.stringify({
          error: 'Authentication required',
          message:
            'Invalid key: Bearer super-secret-key. Tried http://localhost:5000/api/v1/sessions with header X-API-Key: super-secret-key',
        })
      )
    );

    let errorMessage = '';
    try {
      await createOptimizationSession({
        functionName: 'agent_fn',
        configurationSpace: {
          model: param.enum(['gpt-4o-mini']),
        },
        objectives: ['accuracy'],
      });
    } catch (error) {
      errorMessage = error instanceof Error ? error.message : String(error);
    }

    expect(errorMessage).toContain('Authentication required');
    expect(errorMessage).not.toContain('super-secret-key');
    expect(errorMessage).not.toContain('Authorization');
    expect(errorMessage).not.toContain('X-API-Key');
    expect(errorMessage).not.toContain('http://localhost:5000/api/v1/sessions');
    expect(errorMessage).toContain('[REDACTED]');
  });

  it('does not echo auth credentials in normalized helper responses', async () => {
    vi.stubEnv('TRAIGENT_BACKEND_URL', 'http://localhost:5000');
    vi.stubEnv('TRAIGENT_API_KEY', 'super-secret-key');

    fetchMock
      .mockResolvedValueOnce(
        jsonResponse(200, {
          session_id: 'session-redaction',
          status: 'ACTIVE',
          metadata: { owner: 'js' },
        })
      )
      .mockResolvedValueOnce(
        jsonResponse(200, {
          sessions: [
            {
              session_id: 'session-redaction',
              status: 'ACTIVE',
              metadata: { owner: 'js' },
            },
          ],
          total: 1,
        })
      );

    const status = await getOptimizationSessionStatus('session-redaction');
    const listed = await listOptimizationSessions();

    expect(JSON.stringify(status)).not.toContain('super-secret-key');
    expect(JSON.stringify(listed)).not.toContain('super-secret-key');
    expect(status).not.toHaveProperty('Authorization');
    expect(status).not.toHaveProperty('X-API-Key');
    expect(listed).not.toHaveProperty('Authorization');
    expect(listed).not.toHaveProperty('X-API-Key');
  });
});
