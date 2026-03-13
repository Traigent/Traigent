import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { describe, expect, it } from 'vitest';

import { detectTunedVariablesInFiles } from '../../../src/cli/detect.js';

describe('detectTunedVariablesInFiles', () => {
  it('detects tuned-variable candidates from files with function filtering', () => {
    const dir = mkdtempSync(join(tmpdir(), 'traigent-detect-'));
    const file = join(dir, 'agent.mjs');
    writeFileSync(
      file,
      `
        export function helper() {
          const label = "hello";
          return label.toUpperCase();
        }

        export async function answerQuestion(input) {
          const model = "gpt-4o-mini";
          const temperature = 0.4;
          return client.chat.completions.create({
            model,
            temperature,
            messages: [{ role: "user", content: input }],
          });
        }
      `,
      'utf8'
    );

    const [result] = detectTunedVariablesInFiles([file], {
      functionName: 'answerQuestion',
    });

    expect(result.file).toBe(file);
    expect(result.results).toHaveLength(1);
    expect(result.results[0]?.functionName).toBe('answerQuestion');
    expect(result.results[0]?.candidates.map((candidate) => candidate.name)).toEqual([
      'model',
      'temperature',
    ]);
  });
});
