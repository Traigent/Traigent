import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { describe, expect, it } from 'vitest';

import { inspectTvlFiles } from '../../../src/cli/tvl.js';

describe('inspectTvlFiles', () => {
  it('loads TVL files and reports native compatibility and artifact usage', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'traigent-tvl-'));
    const file = join(dir, 'spec.yml');
    writeFileSync(
      file,
      `
spec:
  id: cli-demo
tvars:
  - name: model
    type: enum[str]
    domain: ["gpt-4o-mini", "gpt-4o"]
  - name: temperature
    type: float
    domain:
      range: [0.0, 0.8]
      step: 0.2

objectives:
  - name: accuracy
    direction: maximize
  - name: latency
    direction: minimize

promotion_policy:
  min_effect:
    accuracy: 0.01
`,
      'utf8',
    );

    const [result] = await inspectTvlFiles([file]);

    expect(result.file).toBe(file);
    expect(result.configurationKeys).toEqual(['model', 'temperature']);
    expect(result.objectiveMetrics).toEqual(['accuracy', 'latency']);
    expect(result.usedFeatures).toContain('tvars');
    expect(result.usedFeatures).toContain('promotion-policy');
    expect(result.nativeCompatibility.items.some((item) => item.feature === 'promotion-policy')).toBe(
      true,
    );
  });
});
