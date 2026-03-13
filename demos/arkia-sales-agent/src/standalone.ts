/**
 * Standalone Arkia demo for local development.
 *
 * This runner keeps application scoring inside the Arkia agent implementation,
 * while delegating configuration search to the Traigent JS SDK.
 */

import 'dotenv/config';

import { DEFAULT_CONFIG, runSalesAgent, type AgentConfig } from './agent.js';
import { SALES_DATASET, getDatasetStats } from './dataset.js';
import { REAL_MODE } from './real-llm.js';
import {
  deriveDeterministicSeed,
  resolveDemoDatasetSize,
  runArkiaOptimization,
} from './trial.js';

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function formatUsd(value: number): string {
  return `$${value.toFixed(6)}`;
}

function toAgentConfig(config: Record<string, unknown> | null): AgentConfig | null {
  if (!config) {
    return null;
  }

  return {
    model: String(config.model ?? DEFAULT_CONFIG.model) as AgentConfig['model'],
    temperature: Number(config.temperature ?? DEFAULT_CONFIG.temperature),
    system_prompt: String(
      config.system_prompt ?? DEFAULT_CONFIG.system_prompt,
    ) as AgentConfig['system_prompt'],
    memory_turns: Number(config.memory_turns ?? DEFAULT_CONFIG.memory_turns),
    tool_set: String(config.tool_set ?? DEFAULT_CONFIG.tool_set) as AgentConfig['tool_set'],
    random_seed: deriveDeterministicSeed(config),
  };
}

async function evaluateNamedConfig(
  name: string,
  examples: typeof SALES_DATASET,
  config: AgentConfig,
) {
  console.log(`\n${'#'.repeat(70)}`);
  console.log(`# ${name}`);
  console.log(`${'#'.repeat(70)}`);
  return runSalesAgent(examples, config);
}

async function main() {
  const startTime = Date.now();
  const datasetSize = resolveDemoDatasetSize();
  const demoExamples = SALES_DATASET.slice(0, datasetSize);

  console.log(`\n${'*'.repeat(70)}`);
  console.log('*  ARKIA SALES AGENT - TRAIGENT JS SDK DEMO');
  console.log('*  SDK selects configurations; Arkia code scores each chosen config.');
  if (REAL_MODE) {
    console.log('*  >>> REAL MODE - actual LLM/eval calls may incur cost <<<');
  } else {
    console.log('*  (Mock mode - deterministic per-config scoring, no API costs)');
  }
  console.log(`${'*'.repeat(70)}`);

  if (demoExamples.length === 0) {
    console.error('\nERROR: Dataset not loaded.');
    process.exit(1);
  }

  const stats = getDatasetStats();
  console.log('\nDataset Statistics:');
  console.log(`  Total examples: ${stats.total}`);
  console.log(`  Demo subset: ${datasetSize}`);
  console.log('  By intent:', stats.by_intent);

  const baselineConfig: AgentConfig = {
    ...DEFAULT_CONFIG,
    random_seed: deriveDeterministicSeed(
      DEFAULT_CONFIG as unknown as Record<string, unknown>,
    ),
  };
  const baseline = await evaluateNamedConfig(
    'BASELINE: Current default Arkia config',
    demoExamples,
    baselineConfig,
  );

  console.log(`\n${'='.repeat(70)}`);
  console.log('RUNNING TRAIGENT NATIVE OPTIMIZATION');
  console.log('='.repeat(70));
  console.log('Search: random optimization over the Arkia configuration space');

  const optimizationResult = await runArkiaOptimization({
    algorithm: REAL_MODE ? 'random' : 'random',
    maxTrials: REAL_MODE ? 6 : Number(process.env.ARKIA_MAX_TRIALS ?? 12),
    randomSeed: 7,
  });

  const bestConfig = toAgentConfig(optimizationResult.bestConfig);
  if (!bestConfig) {
    throw new Error('Optimization completed without a bestConfig.');
  }

  const best = await evaluateNamedConfig(
    'BEST CONFIG REPLAY: SDK-selected Arkia config',
    demoExamples,
    bestConfig,
  );

  console.log(`\n${'='.repeat(70)}`);
  console.log('SUMMARY');
  console.log('='.repeat(70));
  console.log(`Trials run: ${optimizationResult.trials.length}`);
  console.log(`Stop reason: ${optimizationResult.stopReason}`);
  console.log(`Optimizer best config: ${JSON.stringify(optimizationResult.bestConfig)}`);
  console.log('\nBaseline vs Best:');
  console.log(
    `  Conversion: ${formatPercent(baseline.avg_conversion_score)} -> ${formatPercent(best.avg_conversion_score)}`,
  );
  console.log(
    `  Margin efficiency: ${baseline.margin_efficiency.toFixed(2)} -> ${best.margin_efficiency.toFixed(2)}`,
  );
  console.log(
    `  Total cost: ${formatUsd(baseline.total_cost)} -> ${formatUsd(best.total_cost)}`,
  );
  console.log(
    `  Avg latency: ${baseline.avg_latency_ms.toFixed(0)}ms -> ${best.avg_latency_ms.toFixed(0)}ms`,
  );

  const elapsed = Date.now() - startTime;
  console.log(`\n${'*'.repeat(70)}`);
  console.log('*  DEMO COMPLETE');
  console.log(`*  Total time: ${elapsed}ms`);
  console.log('*');
  console.log('*  Native mode: npm run dev');
  console.log('*  Python bridge: TRAIGENT_COST_APPROVED=true python run_with_python.py');
  console.log(`${'*'.repeat(70)}\n`);
}

main().catch((error) => {
  console.error('\n' + '='.repeat(70));
  console.error('DEMO FAILED');
  console.error('='.repeat(70));
  console.error(`\nError: ${error instanceof Error ? error.message : error}`);
  if (error instanceof Error && error.stack) {
    console.error('\nStack trace:');
    console.error(error.stack);
  }
  process.exit(1);
});
