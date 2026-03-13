/**
 * Integration tests for the CLI runner.
 *
 * Spawns the runner as a subprocess and communicates via NDJSON protocol,
 * exercising the full request/response lifecycle.
 */
import { spawn, type ChildProcess } from 'node:child_process';
import path from 'node:path';
import { createInterface } from 'node:readline';
import { describe, it, expect, afterEach } from 'vitest';

const ROOT = path.resolve(__dirname, '../..');
const RUNNER_PATH = path.join(ROOT, 'src/cli/runner.ts');
const MOCK_TRIAL = path.join(ROOT, 'tests/integration/fixtures/mock-trial.ts');

interface NDJSONResponse {
  version: string;
  request_id: string;
  status: 'success' | 'error';
  payload: Record<string, unknown>;
}

/**
 * Spawn the runner subprocess using tsx for TypeScript support.
 */
function spawnRunner(modulePath = MOCK_TRIAL): ChildProcess {
  return spawn('npx', ['tsx', RUNNER_PATH, '--module', modulePath], {
    stdio: ['pipe', 'pipe', 'pipe'],
    cwd: ROOT,
    env: {
      ...process.env,
      // Short idle timeout to prevent test hangs
      TRAIGENT_IDLE_TIMEOUT_MS: '10000',
      // Prevent parent PID watcher from interfering
      TRAIGENT_PARENT_PID: String(process.pid),
    },
  });
}

/**
 * Send an NDJSON request to the runner's stdin.
 */
function sendRequest(proc: ChildProcess, request: Record<string, unknown>): void {
  proc.stdin!.write(JSON.stringify(request) + '\n');
}

/**
 * Read a single NDJSON response from the runner's stdout.
 */
function readResponse(proc: ChildProcess, timeoutMs = 15_000): Promise<NDJSONResponse> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      reject(new Error(`Timed out waiting for response after ${timeoutMs}ms`));
    }, timeoutMs);

    const rl = createInterface({ input: proc.stdout! });
    rl.once('line', (line) => {
      clearTimeout(timer);
      rl.close();
      try {
        resolve(JSON.parse(line) as NDJSONResponse);
      } catch {
        reject(new Error(`Invalid JSON response: ${line}`));
      }
    });
  });
}

/**
 * Wait for the runner to be ready by reading stderr for the "Ready" message.
 */
function waitForReady(proc: ChildProcess, timeoutMs = 10_000): Promise<void> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      reject(new Error('Runner did not become ready in time'));
    }, timeoutMs);

    const rl = createInterface({ input: proc.stderr! });
    rl.on('line', (line) => {
      if (line.includes('Ready to receive requests')) {
        clearTimeout(timer);
        rl.close();
        resolve();
      }
    });
  });
}

/**
 * Create a standard trial request payload.
 */
function makeTrialRequest(
  requestId: string,
  config: Record<string, unknown> = {},
  overrides: Record<string, unknown> = {}
) {
  return {
    version: '1.0',
    request_id: requestId,
    action: 'run_trial',
    payload: {
      trial_id: `trial-${requestId}`,
      trial_number: 1,
      experiment_run_id: 'exp-integration',
      config: { behavior: 'success', ...config },
      dataset_subset: { indices: [0, 1, 2], total: 10 },
      ...overrides,
    },
  };
}

// Track spawned processes for cleanup
const processes: ChildProcess[] = [];

afterEach(() => {
  for (const proc of processes) {
    try {
      proc.kill('SIGKILL');
    } catch {
      // already dead
    }
  }
  processes.length = 0;
});

describe(
  'Runner Integration',
  () => {
    it('handles ping request', async () => {
      const proc = spawnRunner();
      processes.push(proc);

      await waitForReady(proc);

      sendRequest(proc, {
        version: '1.0',
        request_id: 'ping-001',
        action: 'ping',
        payload: {},
      });

      const response = await readResponse(proc);

      expect(response.version).toBe('1.1');
      expect(response.request_id).toBe('ping-001');
      expect(response.status).toBe('success');
      expect(response.payload).toHaveProperty('timestamp');
      expect(response.payload).toHaveProperty('uptime_ms');
      expect(typeof response.payload['uptime_ms']).toBe('number');
    });

    it('handles capabilities request', async () => {
      const proc = spawnRunner();
      processes.push(proc);

      await waitForReady(proc);

      sendRequest(proc, {
        version: '1.0',
        request_id: 'cap-001',
        action: 'capabilities',
        payload: {},
      });

      const response = await readResponse(proc);

      expect(response.status).toBe('success');
      const payload = response.payload as Record<string, unknown>;
      expect(payload['protocol_version']).toBe('1.1');
      expect(payload['min_protocol_version']).toBe('1.0');
      expect(payload['capabilities']).toContain('json_schema_validation');
      expect(payload['capabilities']).toContain('dataset_hash');
      expect(payload['supported_actions']).toContain('run_trial');
      expect(payload['supported_actions']).toContain('validate_config');
      expect(payload['limits']).toHaveProperty('max_line_bytes');
    });

    it('handles successful trial execution', async () => {
      const proc = spawnRunner();
      processes.push(proc);

      await waitForReady(proc);

      sendRequest(proc, makeTrialRequest('trial-001', { behavior: 'success' }));

      const response = await readResponse(proc);

      expect(response.status).toBe('success');
      const payload = response.payload as Record<string, unknown>;
      expect(payload['status']).toBe('completed');
      expect(payload['trial_id']).toBe('trial-trial-001');
      const metrics = payload['metrics'] as Record<string, number>;
      expect(metrics['accuracy']).toBe(0.95);
      expect(metrics['latency_ms']).toBe(100);
      expect(typeof payload['duration']).toBe('number');
    });

    it('handles trial function error', async () => {
      const proc = spawnRunner();
      processes.push(proc);

      await waitForReady(proc);

      sendRequest(proc, makeTrialRequest('trial-err', { behavior: 'error' }));

      const response = await readResponse(proc);

      expect(response.status).toBe('success'); // protocol success, trial failed
      const payload = response.payload as Record<string, unknown>;
      expect(payload['status']).toBe('failed');
      expect(payload['error_code']).toBe('USER_FUNCTION_ERROR');
      expect(payload['error_message']).toBe('User function error');
      expect(payload['retryable']).toBe(false);
    });

    it('handles validate_config request', async () => {
      const proc = spawnRunner();
      processes.push(proc);

      await waitForReady(proc);

      sendRequest(proc, {
        version: '1.0',
        request_id: 'val-001',
        action: 'validate_config',
        payload: {
          config: { model: 'gpt-4o', temperature: 0.7 },
          config_schema: {
            type: 'object',
            required: ['model'],
            properties: {
              model: { type: 'string' },
              temperature: { type: 'number', minimum: 0, maximum: 1 },
            },
          },
        },
      });

      const response = await readResponse(proc);

      expect(response.status).toBe('success');
      const payload = response.payload as Record<string, unknown>;
      expect(payload['ok']).toBe(true);
    });

    it('handles validate_config with schema violations', async () => {
      const proc = spawnRunner();
      processes.push(proc);

      await waitForReady(proc);

      sendRequest(proc, {
        version: '1.0',
        request_id: 'val-002',
        action: 'validate_config',
        payload: {
          config: { model: 123, temperature: 'hot' },
          config_schema: {
            type: 'object',
            properties: {
              model: { type: 'string' },
              temperature: { type: 'number' },
            },
          },
        },
      });

      const response = await readResponse(proc);

      expect(response.status).toBe('success');
      const payload = response.payload as Record<string, unknown>;
      expect(payload['ok']).toBe(false);
      expect((payload['issues'] as unknown[]).length).toBeGreaterThan(0);
    });

    it('handles cancel when no trial is running', async () => {
      const proc = spawnRunner();
      processes.push(proc);

      await waitForReady(proc);

      sendRequest(proc, {
        version: '1.0',
        request_id: 'cancel-001',
        action: 'cancel',
        payload: {},
      });

      const response = await readResponse(proc);

      expect(response.status).toBe('success');
      const payload = response.payload as Record<string, unknown>;
      expect(payload['cancelled']).toBe(false);
      expect(payload['reason']).toBe('no_trial_running');
    });

    it('handles shutdown gracefully', async () => {
      const proc = spawnRunner();
      processes.push(proc);

      await waitForReady(proc);

      sendRequest(proc, {
        version: '1.0',
        request_id: 'shutdown-001',
        action: 'shutdown',
        payload: {},
      });

      const response = await readResponse(proc);

      expect(response.status).toBe('success');
      const payload = response.payload as Record<string, unknown>;
      expect(payload['status']).toBe('shutting_down');

      // Process should exit after shutdown
      await new Promise<void>((resolve) => {
        proc.on('exit', () => resolve());
        // Ensure we don't wait forever
        setTimeout(() => resolve(), 3000);
      });
    });

    it('returns PROTOCOL_ERROR for invalid JSON', async () => {
      const proc = spawnRunner();
      processes.push(proc);

      await waitForReady(proc);

      proc.stdin!.write('this is not valid json\n');

      const response = await readResponse(proc);

      expect(response.status).toBe('error');
      const payload = response.payload as Record<string, unknown>;
      expect(payload['error_code']).toBe('PROTOCOL_ERROR');
      expect(payload['retryable']).toBe(false);
    });

    it('returns PROTOCOL_ERROR for malformed request', async () => {
      const proc = spawnRunner();
      processes.push(proc);

      await waitForReady(proc);

      // Valid JSON but invalid request structure
      sendRequest(proc, { some: 'garbage' });

      const response = await readResponse(proc);

      expect(response.status).toBe('error');
      const payload = response.payload as Record<string, unknown>;
      expect(payload['error_code']).toBe('PROTOCOL_ERROR');
    });

    it('handles invalid trial config with VALIDATION_ERROR', async () => {
      const proc = spawnRunner();
      processes.push(proc);

      await waitForReady(proc);

      sendRequest(proc, {
        version: '1.0',
        request_id: 'bad-config-001',
        action: 'run_trial',
        payload: {
          // Missing required fields
          trial_id: 'trial-bad',
        },
      });

      const response = await readResponse(proc);

      expect(response.status).toBe('success'); // protocol-level success
      const payload = response.payload as Record<string, unknown>;
      expect(payload['status']).toBe('failed');
      expect(payload['error_code']).toBe('VALIDATION_ERROR');
    });

    it('handles multiple sequential requests', async () => {
      const proc = spawnRunner();
      processes.push(proc);

      await waitForReady(proc);

      // Ping
      sendRequest(proc, {
        version: '1.0',
        request_id: 'seq-ping',
        action: 'ping',
        payload: {},
      });
      const pingResp = await readResponse(proc);
      expect(pingResp.request_id).toBe('seq-ping');
      expect(pingResp.status).toBe('success');

      // Trial
      sendRequest(proc, makeTrialRequest('seq-trial'));
      const trialResp = await readResponse(proc);
      expect(trialResp.request_id).toBe('seq-trial');
      expect(trialResp.status).toBe('success');

      // Capabilities
      sendRequest(proc, {
        version: '1.0',
        request_id: 'seq-caps',
        action: 'capabilities',
        payload: {},
      });
      const capsResp = await readResponse(proc);
      expect(capsResp.request_id).toBe('seq-caps');
      expect(capsResp.status).toBe('success');
    });

    it('sanitizes bad metrics with warnings', async () => {
      const proc = spawnRunner();
      processes.push(proc);

      await waitForReady(proc);

      sendRequest(proc, makeTrialRequest('trial-bad-metrics', { behavior: 'bad_metrics' }));

      const response = await readResponse(proc);

      expect(response.status).toBe('success');
      const payload = response.payload as Record<string, unknown>;
      expect(payload['status']).toBe('completed');
      const metrics = payload['metrics'] as Record<string, unknown>;
      // Valid metric should be preserved
      expect(metrics['valid']).toBe(1.0);
      // Invalid metrics should be filtered out
      expect(metrics['nan_value']).toBeUndefined();
      expect(metrics['inf_value']).toBeUndefined();
      expect(metrics['string_value']).toBeUndefined();
      // Warnings should be present
      expect(payload['warnings']).toBeDefined();
      expect(payload['metrics_sanitized']).toBe(true);
    });

    it('emits MODULE_LOAD_ERROR for non-existent module', async () => {
      const proc = spawnRunner('/tmp/nonexistent-module.ts');
      processes.push(proc);

      // The runner emits an NDJSON error before exiting
      const response = await readResponse(proc, 10_000);

      expect(response.status).toBe('error');
      const payload = response.payload as Record<string, unknown>;
      expect(payload['error_code']).toBe('MODULE_LOAD_ERROR');
      expect(payload['retryable']).toBe(false);
    });
  },
  { timeout: 30_000 }
);
