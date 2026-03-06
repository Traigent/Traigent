#!/usr/bin/env node
/**
 * Real LLM Optimization Demo
 *
 * Demonstrates Traigent optimization with REAL API calls to various LLMs
 * via OpenRouter. This shows actual accuracy, cost, and latency differences
 * between models and configurations.
 *
 * Results are submitted to the Traigent backend for visualization in the FE.
 */

import { SENTIMENT_DATASET, getDatasetSubset } from './dataset.js';
import {
  runRealSentimentAgent,
  REAL_CONFIGURATION_SPACE,
  type AgentConfig,
} from './real-agent.js';

// Get API keys from environment
const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY;
const TRAIGENT_API_URL = process.env.TRAIGENT_API_URL ?? 'http://localhost:5000/api/v1';
const TRAIGENT_API_KEY = process.env.TRAIGENT_API_KEY ?? '';

if (!OPENROUTER_API_KEY) {
  console.error('\x1b[31mError: OPENROUTER_API_KEY environment variable is required\x1b[0m');
  console.error('Run: source .env.local  (from project root)');
  process.exit(1);
}

interface SessionInfo {
  session_id: string;
  experiment_id: string;
  experiment_run_id: string;
}

interface TrialResult {
  trial_id: string;
  trial_number: number;
  config: AgentConfig;
  metrics: {
    accuracy: number;
    cost: number;
    latency_ms: number;
    input_tokens: number;
    output_tokens: number;
  };
  duration: number;
}

/**
 * Create a Traigent optimization session.
 */
async function createSession(maxTrials: number): Promise<SessionInfo | null> {
  const url = `${TRAIGENT_API_URL}/sessions`;

  const payload = {
    problem_statement: 'sentiment_classifier',
    dataset: {
      examples: [],  // Privacy mode - no actual data sent
      metadata: {
        size: SENTIMENT_DATASET.length,
        privacy_mode: true,
        type: 'sentiment_classification',
      },
    },
    search_space: REAL_CONFIGURATION_SPACE,
    optimization_config: {
      algorithm: 'random',
      max_trials: maxTrials,
      optimization_goal: 'maximize',
      objectives: ['accuracy', 'cost'],
    },
    metadata: {
      function_name: 'sentiment_classifier',
      runtime: 'node',
      demo: true,
      agent: 'real_llm_sentiment_classifier',
      llm_provider: 'openrouter',
    },
  };

  console.log(`\n\x1b[36m[SESSION]\x1b[0m Creating Traigent session at ${url}`);

  if (!TRAIGENT_API_KEY) {
    console.log('\x1b[33m[SESSION]\x1b[0m No API key configured - running in local-only mode');
    const mockId = `local_${Date.now()}`;
    return {
      session_id: mockId,
      experiment_id: mockId,
      experiment_run_id: mockId,
    };
  }

  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': TRAIGENT_API_KEY,
        'Authorization': `Bearer ${TRAIGENT_API_KEY}`,
      },
      body: JSON.stringify(payload),
    });

    if (response.status === 201 || response.status === 200) {
      const result = await response.json();
      const sessionInfo: SessionInfo = {
        session_id: result.session_id,
        experiment_id: result.metadata?.experiment_id ?? result.session_id,
        experiment_run_id: result.metadata?.experiment_run_id ?? result.session_id,
      };
      console.log(`\x1b[32m[SESSION]\x1b[0m Created session: ${sessionInfo.session_id}`);
      console.log(`\x1b[32m[SESSION]\x1b[0m Experiment run: ${sessionInfo.experiment_run_id}`);
      return sessionInfo;
    } else {
      const text = await response.text();
      console.log(`\x1b[31m[SESSION]\x1b[0m Failed to create session: ${response.status} - ${text}`);
      // Return mock session for offline mode
      const mockId = `local_${Date.now()}`;
      return {
        session_id: mockId,
        experiment_id: mockId,
        experiment_run_id: mockId,
      };
    }
  } catch (error) {
    console.log(`\x1b[31m[SESSION]\x1b[0m Error creating session: ${error}`);
    const mockId = `local_${Date.now()}`;
    return {
      session_id: mockId,
      experiment_id: mockId,
      experiment_run_id: mockId,
    };
  }
}

/**
 * Submit trial result to Traigent backend.
 */
async function submitTrialResult(
  sessionInfo: SessionInfo,
  result: TrialResult,
  totalExamples: number
): Promise<boolean> {
  const url = `${TRAIGENT_API_URL}/sessions/${sessionInfo.session_id}/results`;

  // Calculate weighted score (accuracy is primary, cost is secondary penalty)
  const weightedScore = result.metrics.accuracy - (result.metrics.cost * 100);

  const payload = {
    trial_id: result.trial_id,
    config: result.config,
    status: 'COMPLETED',
    metrics: {
      accuracy: result.metrics.accuracy,
      cost: result.metrics.cost,
      latency_ms: result.metrics.latency_ms,
      input_tokens: result.metrics.input_tokens,
      output_tokens: result.metrics.output_tokens,
    },
    summary_stats: {
      metrics: {
        accuracy: {
          count: totalExamples,
          mean: result.metrics.accuracy,
          std: 0,
          min: result.metrics.accuracy,
          max: result.metrics.accuracy,
        },
        cost: {
          count: totalExamples,
          mean: result.metrics.cost / totalExamples,
          std: 0,
          min: result.metrics.cost / totalExamples,
          max: result.metrics.cost / totalExamples,
        },
        latency_ms: {
          count: totalExamples,
          mean: result.metrics.latency_ms,
          std: 0,
          min: result.metrics.latency_ms,
          max: result.metrics.latency_ms,
        },
      },
      execution_time: result.duration,
      total_examples: totalExamples,
      weighted_score: weightedScore,
    },
    metadata: {
      mode: 'privacy',
      sdk_version: '1.0.0',
      experiment_run_id: sessionInfo.experiment_run_id,
      model: result.config.model,
      temperature: result.config.temperature,
      prompt_type: result.config.system_prompt,
    },
  };

  console.log(`\x1b[36m[BACKEND]\x1b[0m Submitting trial ${result.trial_id}`);

  if (!TRAIGENT_API_KEY) {
    console.log('\x1b[33m[BACKEND]\x1b[0m No API key - skipping submission');
    return true;
  }

  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': TRAIGENT_API_KEY,
        'Authorization': `Bearer ${TRAIGENT_API_KEY}`,
      },
      body: JSON.stringify(payload),
    });

    if (response.ok) {
      console.log(`\x1b[32m[BACKEND]\x1b[0m Trial ${result.trial_number} submitted successfully`);
      return true;
    } else {
      const text = await response.text();
      console.log(`\x1b[31m[BACKEND]\x1b[0m Failed to submit: ${response.status} - ${text}`);
      return false;
    }
  } catch (error) {
    console.log(`\x1b[31m[BACKEND]\x1b[0m Error submitting trial: ${error}`);
    return false;
  }
}

/**
 * Sample specific configurations to compare across models
 */
function getComparisonConfigs(): AgentConfig[] {
  // Test configurations that showcase different trade-offs
  return [
    // Fast & cheap models
    { model: 'openai/gpt-4o-mini', temperature: 0.0, system_prompt: 'concise' },
    { model: 'openai/gpt-4o-mini', temperature: 0.0, system_prompt: 'cot' },

    // Open source model
    { model: 'meta-llama/llama-3.1-8b-instruct', temperature: 0.0, system_prompt: 'concise' },
    { model: 'meta-llama/llama-3.1-8b-instruct', temperature: 0.0, system_prompt: 'cot' },

    // Claude comparison
    { model: 'anthropic/claude-3-haiku', temperature: 0.0, system_prompt: 'concise' },
    { model: 'anthropic/claude-3-haiku', temperature: 0.0, system_prompt: 'cot' },

    // Legacy model
    { model: 'openai/gpt-3.5-turbo', temperature: 0.0, system_prompt: 'concise' },
  ];
}

/**
 * Run a single trial with the given configuration.
 */
async function runTrial(
  trialNumber: number,
  config: AgentConfig,
  datasetIndices: number[]
): Promise<TrialResult> {
  const trialId = `trial_${Date.now()}_${trialNumber}`;

  console.log(`\n${'#'.repeat(80)}`);
  console.log(`# TRIAL ${trialNumber} (${trialId})`);
  console.log(`${'#'.repeat(80)}`);

  const startTime = Date.now();

  // Get the subset of examples for this trial
  const examples = getDatasetSubset(datasetIndices);
  console.log(`[TRIAL] Evaluating ${examples.length} examples from dataset`);

  // Run the REAL agent with actual API calls
  const result = await runRealSentimentAgent(examples, config, OPENROUTER_API_KEY!, console.log);

  const duration = (Date.now() - startTime) / 1000;

  return {
    trial_id: trialId,
    trial_number: trialNumber,
    config,
    metrics: {
      accuracy: result.accuracy,
      cost: result.total_cost,
      latency_ms: result.avg_latency_ms,
      input_tokens: result.total_input_tokens,
      output_tokens: result.total_output_tokens,
    },
    duration,
  };
}

/**
 * Main optimization loop.
 */
async function runOptimization(): Promise<void> {
  console.log('\n');
  console.log('\x1b[1m\x1b[36m' + '*'.repeat(80) + '\x1b[0m');
  console.log('\x1b[1m\x1b[36m*  TRAIGENT REAL LLM OPTIMIZATION DEMO                                         *\x1b[0m');
  console.log('\x1b[1m\x1b[36m' + '*'.repeat(80) + '\x1b[0m');
  console.log('\n\x1b[1mConfiguration Space:\x1b[0m');
  console.log(`  Models:      ${REAL_CONFIGURATION_SPACE.model.join(', ')}`);
  console.log(`  Temperature: ${REAL_CONFIGURATION_SPACE.temperature.join(', ')}`);
  console.log(`  Prompts:     ${REAL_CONFIGURATION_SPACE.system_prompt.join(', ')}`);
  console.log(`\n\x1b[1mDataset:\x1b[0m ${SENTIMENT_DATASET.length} sentiment classification examples`);
  console.log(`\x1b[1mBackend:\x1b[0m ${TRAIGENT_API_URL}`);
  console.log(`\x1b[1mAPI Key:\x1b[0m ${TRAIGENT_API_KEY ? '✓ configured' : '✗ not configured'}`);

  const configs = getComparisonConfigs();
  console.log(`\x1b[1mTrials:\x1b[0m ${configs.length} configurations to compare`);

  // Create Traigent session
  const sessionInfo = await createSession(configs.length);
  if (!sessionInfo) {
    console.error('\x1b[31m[ERROR]\x1b[0m Failed to create session');
    return;
  }

  console.log(`\n\x1b[1mSession ID:\x1b[0m ${sessionInfo.session_id}`);
  console.log(`\x1b[1mExperiment Run:\x1b[0m ${sessionInfo.experiment_run_id}`);

  const results: TrialResult[] = [];

  // Use all examples for fair comparison
  const indices = Array.from({ length: SENTIMENT_DATASET.length }, (_, i) => i);

  for (let i = 0; i < configs.length; i++) {
    const config = configs[i];

    try {
      const result = await runTrial(i + 1, config, indices);
      results.push(result);

      // Submit to Traigent backend
      await submitTrialResult(sessionInfo, result, indices.length);
    } catch (error) {
      console.error(`\x1b[31m[ERROR] Trial ${i + 1} failed: ${error}\x1b[0m`);
    }

    // Pause between trials to avoid rate limiting
    if (i < configs.length - 1) {
      console.log('\n\x1b[90m[Waiting 2s before next trial...]\x1b[0m');
      await new Promise(resolve => setTimeout(resolve, 2000));
    }
  }

  // Print summary
  printSummary(results, sessionInfo);
}

function printSummary(results: TrialResult[], sessionInfo: SessionInfo): void {
  console.log('\n\n');
  console.log('\x1b[1m\x1b[36m' + '*'.repeat(80) + '\x1b[0m');
  console.log('\x1b[1m\x1b[36m*  OPTIMIZATION COMPLETE - RESULTS SUMMARY                                     *\x1b[0m');
  console.log('\x1b[1m\x1b[36m' + '*'.repeat(80) + '\x1b[0m');

  // Sort by accuracy descending
  const sorted = [...results].sort((a, b) => b.metrics.accuracy - a.metrics.accuracy);

  console.log('\n\x1b[1mAll Trial Results (sorted by accuracy):\x1b[0m');
  console.log('─'.repeat(115));
  console.log('| \x1b[1mTrial\x1b[0m | \x1b[1mModel\x1b[0m                              | \x1b[1mTemp\x1b[0m | \x1b[1mPrompt\x1b[0m   | \x1b[1mAccuracy\x1b[0m | \x1b[1mCost\x1b[0m       | \x1b[1mLatency\x1b[0m  | \x1b[1mTokens\x1b[0m |');
  console.log('─'.repeat(115));

  for (const r of sorted) {
    const model = r.config.model.padEnd(36);
    const temp = r.config.temperature.toFixed(1);
    const prompt = r.config.system_prompt.padEnd(8);
    const acc = (r.metrics.accuracy * 100).toFixed(1).padStart(5) + '%';
    const accColor = r.metrics.accuracy >= 0.9 ? '\x1b[32m' : r.metrics.accuracy >= 0.7 ? '\x1b[33m' : '\x1b[31m';
    const cost = ('$' + r.metrics.cost.toFixed(6)).padStart(10);
    const lat = r.metrics.latency_ms.toFixed(0).padStart(6) + 'ms';
    const tokens = (r.metrics.input_tokens + r.metrics.output_tokens).toString().padStart(6);
    console.log(`|   ${r.trial_number.toString().padStart(2)}  | ${model} | ${temp} | ${prompt} | ${accColor}${acc}\x1b[0m | ${cost} | ${lat} | ${tokens} |`);
  }
  console.log('─'.repeat(115));

  // Find best configurations
  const best = sorted[0];
  const cheapest = [...results].sort((a, b) => a.metrics.cost - b.metrics.cost)[0];
  const fastest = [...results].sort((a, b) => a.metrics.latency_ms - b.metrics.latency_ms)[0];

  if (best) {
    console.log('\n\x1b[1m\x1b[32m═══ BEST ACCURACY ═══\x1b[0m');
    console.log(`  Model:       ${best.config.model}`);
    console.log(`  Temperature: ${best.config.temperature}`);
    console.log(`  Prompt:      ${best.config.system_prompt}`);
    console.log(`  Accuracy:    \x1b[1m${(best.metrics.accuracy * 100).toFixed(1)}%\x1b[0m`);
    console.log(`  Cost:        $${best.metrics.cost.toFixed(6)}`);
    console.log(`  Latency:     ${best.metrics.latency_ms.toFixed(0)}ms avg`);
  }

  if (cheapest && cheapest.metrics.cost > 0) {
    console.log('\n\x1b[1m\x1b[33m═══ CHEAPEST ═══\x1b[0m');
    console.log(`  Model:       ${cheapest.config.model}`);
    console.log(`  Cost:        \x1b[1m$${cheapest.metrics.cost.toFixed(6)}\x1b[0m`);
    console.log(`  Accuracy:    ${(cheapest.metrics.accuracy * 100).toFixed(1)}%`);
  }

  if (fastest && fastest.metrics.latency_ms > 0) {
    console.log('\n\x1b[1m\x1b[34m═══ FASTEST ═══\x1b[0m');
    console.log(`  Model:       ${fastest.config.model}`);
    console.log(`  Latency:     \x1b[1m${fastest.metrics.latency_ms.toFixed(0)}ms\x1b[0m avg`);
    console.log(`  Accuracy:    ${(fastest.metrics.accuracy * 100).toFixed(1)}%`);
  }

  // Calculate trade-offs
  console.log('\n\x1b[1m═══ KEY INSIGHTS ═══\x1b[0m');

  // Accuracy improvement from prompt engineering
  const conciseResults = results.filter(r => r.config.system_prompt === 'concise');
  const cotResults = results.filter(r => r.config.system_prompt === 'cot');

  if (conciseResults.length > 0 && cotResults.length > 0) {
    const avgConcise = conciseResults.reduce((sum, r) => sum + r.metrics.accuracy, 0) / conciseResults.length;
    const avgCot = cotResults.reduce((sum, r) => sum + r.metrics.accuracy, 0) / cotResults.length;
    const improvement = ((avgCot - avgConcise) / avgConcise * 100);
    console.log(`\n  Chain-of-Thought vs Concise Prompts:`);
    console.log(`     Concise avg accuracy: ${(avgConcise * 100).toFixed(1)}%`);
    console.log(`     CoT avg accuracy:     ${(avgCot * 100).toFixed(1)}%`);
    console.log(`     \x1b[1mDifference: ${improvement >= 0 ? '+' : ''}${improvement.toFixed(1)}%\x1b[0m`);
  }

  // Model comparison
  const byModel = new Map<string, TrialResult[]>();
  for (const r of results) {
    const existing = byModel.get(r.config.model) ?? [];
    existing.push(r);
    byModel.set(r.config.model, existing);
  }

  console.log(`\n  Model Performance Summary:`);
  for (const [model, modelResults] of byModel.entries()) {
    const avgAcc = modelResults.reduce((sum, r) => sum + r.metrics.accuracy, 0) / modelResults.length;
    const avgCost = modelResults.reduce((sum, r) => sum + r.metrics.cost, 0) / modelResults.length;
    const avgLat = modelResults.reduce((sum, r) => sum + r.metrics.latency_ms, 0) / modelResults.length;
    console.log(`     ${model.padEnd(38)} - Acc: ${(avgAcc * 100).toFixed(1).padStart(5)}%, Cost: $${avgCost.toFixed(6)}, Lat: ${avgLat.toFixed(0)}ms`);
  }

  console.log(`\n\x1b[1m═══ VIEW IN FRONTEND ═══\x1b[0m`);
  console.log(`  Session ID: ${sessionInfo.session_id}`);
  console.log(`  URL: http://localhost:3000/experiments`);

  console.log('\n\x1b[32mDemo complete!\x1b[0m\n');
}

// Run the optimization
runOptimization().catch(error => {
  console.error('\x1b[31mFatal error:\x1b[0m', error);
  process.exit(1);
});
