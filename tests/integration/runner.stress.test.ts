import { spawn, type ChildProcess } from 'node:child_process';
import path from 'node:path';
import { createInterface } from 'node:readline';
import { afterEach, describe, expect, it } from 'vitest';

const ROOT = path.resolve(__dirname, '../..');
const RUNNER_PATH = path.join(ROOT, 'src/cli/runner.ts');
const MOCK_TRIAL = path.join(ROOT, 'tests/integration/fixtures/mock-trial.ts');

interface NDJSONResponse {
  request_id: string;
  status: 'success' | 'error';
  payload: Record<string, unknown>;
}

function spawnRunner(): ChildProcess {
  return spawn('npx', ['tsx', RUNNER_PATH, '--module', MOCK_TRIAL], {
    stdio: ['pipe', 'pipe', 'pipe'],
    cwd: ROOT,
    env: {
      ...process.env,
      TRAIGENT_IDLE_TIMEOUT_MS: '10000',
      TRAIGENT_PARENT_PID: String(process.pid),
    },
  });
}

function waitForReady(proc: ChildProcess, timeoutMs = 10_000): Promise<void> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(
      () => reject(new Error('Runner did not become ready in time')),
      timeoutMs
    );
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

function sendRequest(proc: ChildProcess, request: Record<string, unknown>): void {
  proc.stdin!.write(JSON.stringify(request) + '\n');
}

function readResponse(proc: ChildProcess, timeoutMs = 15_000): Promise<NDJSONResponse> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(
      () => reject(new Error(`Timed out waiting for response after ${timeoutMs}ms`)),
      timeoutMs
    );
    const rl = createInterface({ input: proc.stdout! });
    rl.once('line', (line) => {
      clearTimeout(timer);
      rl.close();
      resolve(JSON.parse(line) as NDJSONResponse);
    });
  });
}

const processes: ChildProcess[] = [];

afterEach(() => {
  for (const proc of processes) {
    try {
      proc.kill('SIGKILL');
    } catch {
      // already exited
    }
  }
  processes.length = 0;
});

describe(
  'runner protocol stress',
  () => {
    it('handles 100 sequential protocol exchanges without hanging', async () => {
      const proc = spawnRunner();
      processes.push(proc);

      await waitForReady(proc);

      for (let index = 0; index < 50; index += 1) {
        sendRequest(proc, {
          version: '1.0',
          request_id: `ping-${index}`,
          action: 'ping',
          payload: {},
        });
        const pingResponse = await readResponse(proc);
        expect(pingResponse.request_id).toBe(`ping-${index}`);
        expect(pingResponse.status).toBe('success');

        sendRequest(proc, {
          version: '1.0',
          request_id: `validate-${index}`,
          action: 'validate_config',
          payload: {
            config: { model: 'gpt-4o-mini', temperature: 0.5 },
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
        const validateResponse = await readResponse(proc);
        expect(validateResponse.request_id).toBe(`validate-${index}`);
        expect(validateResponse.status).toBe('success');
        expect(validateResponse.payload['ok']).toBe(true);
      }

      sendRequest(proc, {
        version: '1.0',
        request_id: 'shutdown-stress',
        action: 'shutdown',
        payload: {},
      });
      const shutdownResponse = await readResponse(proc);
      expect(shutdownResponse.status).toBe('success');
    });
  },
  { timeout: 30_000 }
);
