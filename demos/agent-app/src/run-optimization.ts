#!/usr/bin/env node
/**
 * Traigent Optimization Demo Runner
 *
 * This script demonstrates how a JS agent can be optimized using Traigent.
 * It creates a session via the Traigent API and submits trial results.
 */

import { SENTIMENT_DATASET, getDatasetSubset } from './dataset.js';
import { runSentimentAgent, CONFIGURATION_SPACE, type AgentConfig } from './agent.js';

// Load environment variables
const TRAIGENT_API_URL = process.env.TRAIGENT_API_URL ?? 'http://localhost:5000/api/v1';
const TRAIGENT_API_KEY = process.env.TRAIGENT_API_KEY ?? '';

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
  };
  duration: number;
}

/**
 * Create a Traigent optimization session.
 * This must be called before submitting trial results.
 */
async function createSession(maxTrials: number): Promise<SessionInfo | null> {
  const url = `${TRAIGENT_API_URL}/sessions`;

  const payload = {
    problem_statement: 'sentiment_agent_optimization',
    dataset: {
      size: SENTIMENT_DATASET.length,
      evaluation_set: 'sentiment_test',
    },
    search_space: CONFIGURATION_SPACE,
    optimization_config: {
      algorithm: 'random',
      max_trials: maxTrials,
      optimization_goal: 'maximize',
      objectives: ['accuracy', 'cost'],
    },
    metadata: {
      runtime: 'node',
      demo: true,
      agent: 'sentiment_classifier',
    },
  };

  console.log(`\n[SESSION] Creating Traigent session at ${url}`);

  if (!TRAIGENT_API_KEY) {
    console.log('[SESSION] No API key configured - running in local-only mode');
    // Return mock session for local-only mode
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
        Authorization: `Bearer ${TRAIGENT_API_KEY}`,
      },
      body: JSON.stringify(payload),
    });

    if (response.status === 201) {
      const result = await response.json();
      const sessionInfo: SessionInfo = {
        session_id: result.session_id,
        experiment_id: result.metadata?.experiment_id ?? result.session_id,
        experiment_run_id: result.metadata?.experiment_run_id ?? result.session_id,
      };
      console.log(`[SESSION] Created session: ${sessionInfo.session_id}`);
      console.log(`[SESSION] Experiment run: ${sessionInfo.experiment_run_id}`);
      return sessionInfo;
    } else {
      const text = await response.text();
      console.log(`[SESSION] Failed to create session: ${response.status} - ${text}`);
      return null;
    }
  } catch (error) {
    console.log(`[SESSION] Error creating session: ${error}`);
    return null;
  }
}

/**
 * Generate a random configuration from the configuration space.
 */
function sampleRandomConfig(): AgentConfig {
  const models = CONFIGURATION_SPACE.model;
  const temps = CONFIGURATION_SPACE.temperature;
  const prompts = CONFIGURATION_SPACE.system_prompt;

  return {
    model: models[Math.floor(Math.random() * models.length)],
    temperature: temps[Math.floor(Math.random() * temps.length)],
    system_prompt: prompts[Math.floor(Math.random() * prompts.length)],
  };
}

/**
 * Generate a random subset of dataset indices.
 */
function sampleDatasetIndices(subsetSize: number): number[] {
  const allIndices = Array.from({ length: SENTIMENT_DATASET.length }, (_, i) => i);
  const shuffled = allIndices.sort(() => Math.random() - 0.5);
  return shuffled.slice(0, subsetSize);
}

/**
 * Submit trial result to Traigent backend via session endpoint.
 */
async function submitTrialResult(sessionInfo: SessionInfo, result: TrialResult): Promise<boolean> {
  // Use the session results endpoint (same as Python SDK)
  const url = `${TRAIGENT_API_URL}/sessions/${sessionInfo.session_id}/results`;

  const payload = {
    trial_id: result.trial_id,
    trial_number: result.trial_number,
    status: 'COMPLETED',
    config: result.config,
    metrics: result.metrics,
    execution_time: result.duration,
    metadata: {
      experiment_run_id: sessionInfo.experiment_run_id,
    },
  };

  console.log(
    `\n[BACKEND] Submitting trial ${result.trial_id} to session ${sessionInfo.session_id}`
  );

  if (!TRAIGENT_API_KEY) {
    console.log('[BACKEND] No API key configured - skipping actual submission');
    console.log(`[BACKEND] Would submit: ${JSON.stringify(payload, null, 2)}`);
    return true;
  }

  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': TRAIGENT_API_KEY,
        Authorization: `Bearer ${TRAIGENT_API_KEY}`,
      },
      body: JSON.stringify(payload),
    });

    if (response.ok) {
      console.log(`[BACKEND] Trial ${result.trial_id} submitted successfully`);
      return true;
    } else {
      const text = await response.text();
      console.log(`[BACKEND] Failed to submit: ${response.status} - ${text}`);
      return false;
    }
  } catch (error) {
    console.log(`[BACKEND] Error submitting trial: ${error}`);
    return false;
  }
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

  console.log(`\n${'#'.repeat(70)}`);
  console.log(`# TRIAL ${trialNumber} (${trialId})`);
  console.log(`${'#'.repeat(70)}`);

  const startTime = Date.now();

  // Get the subset of examples for this trial
  const examples = getDatasetSubset(datasetIndices);
  console.log(`[TRIAL] Evaluating ${examples.length} examples from dataset`);

  // Run the agent with this configuration
  const result = await runSentimentAgent(examples, config, console.log);

  const duration = (Date.now() - startTime) / 1000;

  return {
    trial_id: trialId,
    trial_number: trialNumber,
    config,
    metrics: {
      accuracy: result.accuracy,
      cost: result.total_cost,
      latency_ms: result.avg_latency_ms,
    },
    duration,
  };
}

/**
 * Main optimization loop.
 */
async function runOptimization(maxTrials: number = 6): Promise<void> {
  console.log('\n');
  console.log('*'.repeat(70));
  console.log('*  TRAIGENT OPTIMIZATION DEMO - JS AGENT                              *');
  console.log('*'.repeat(70));
  console.log('\nConfiguration Space:');
  console.log(`  Models:      ${CONFIGURATION_SPACE.model.join(', ')}`);
  console.log(`  Temperature: ${CONFIGURATION_SPACE.temperature.join(', ')}`);
  console.log(`  Prompts:     ${CONFIGURATION_SPACE.system_prompt.join(', ')}`);
  console.log(`\nDataset: ${SENTIMENT_DATASET.length} sentiment classification examples`);
  console.log(`Max Trials: ${maxTrials}`);
  console.log(`Backend: ${TRAIGENT_API_URL}`);

  // Create a Traigent session first
  const sessionInfo = await createSession(maxTrials);
  if (!sessionInfo) {
    console.log('\n[ERROR] Failed to create session - running in offline mode');
    // Create a fallback local session
    const fallbackId = `local_${Date.now()}`;
    console.log(`[FALLBACK] Using local session ID: ${fallbackId}`);
  }
  console.log(`\nSession ID: ${sessionInfo?.session_id ?? 'local'}`);
  console.log(`Experiment Run ID: ${sessionInfo?.experiment_run_id ?? 'local'}`);

  const results: TrialResult[] = [];
  const subsetSize = 10; // Use 10 examples per trial

  for (let i = 0; i < maxTrials; i++) {
    // Sample a random configuration
    const config = sampleRandomConfig();

    // Sample random dataset indices
    const indices = sampleDatasetIndices(subsetSize);

    // Run the trial
    const result = await runTrial(i + 1, config, indices);
    results.push(result);

    // Submit to backend if we have a valid session
    if (sessionInfo) {
      await submitTrialResult(sessionInfo, result);
    }
  }

  // Print summary
  console.log('\n');
  console.log('*'.repeat(70));
  console.log('*  OPTIMIZATION COMPLETE - SUMMARY                                    *');
  console.log('*'.repeat(70));
  console.log('\nAll Trial Results:');
  console.log('-'.repeat(90));
  console.log('| Trial | Model          | Temp | Prompt   | Accuracy | Cost      | Latency |');
  console.log('-'.repeat(90));

  for (const r of results) {
    const model = r.config.model.padEnd(14);
    const temp = r.config.temperature.toFixed(1);
    const prompt = r.config.system_prompt.padEnd(8);
    const acc = (r.metrics.accuracy * 100).toFixed(1).padStart(6) + '%';
    const cost = ('$' + r.metrics.cost.toFixed(6)).padStart(9);
    const lat = r.metrics.latency_ms.toFixed(0).padStart(5) + 'ms';
    console.log(
      `|   ${r.trial_number}   | ${model} | ${temp} | ${prompt} | ${acc} | ${cost} | ${lat} |`
    );
  }
  console.log('-'.repeat(90));

  // Find best configuration (maximize accuracy, minimize cost)
  const best = results.reduce((a, b) => {
    // Simple scoring: accuracy * 100 - cost * 1000000
    const scoreA = a.metrics.accuracy * 100 - a.metrics.cost * 1000000;
    const scoreB = b.metrics.accuracy * 100 - b.metrics.cost * 1000000;
    return scoreA > scoreB ? a : b;
  });

  console.log('\nBest Configuration Found:');
  console.log(`  Model:       ${best.config.model}`);
  console.log(`  Temperature: ${best.config.temperature}`);
  console.log(`  Prompt:      ${best.config.system_prompt}`);
  console.log(`  Accuracy:    ${(best.metrics.accuracy * 100).toFixed(1)}%`);
  console.log(`  Cost:        $${best.metrics.cost.toFixed(6)}`);
  console.log('\nDemo complete!');
}

// Run the optimization
runOptimization(6).catch(console.error);
