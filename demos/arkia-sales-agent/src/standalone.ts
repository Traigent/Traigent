/**
 * Standalone demo runner for local development.
 *
 * Run this to test the Arkia sales agent without the Python orchestrator.
 * Demonstrates the margin optimization opportunity.
 *
 * Usage:
 *   npm run dev              # Mock mode (no API costs)
 *   REAL_MODE=true npm run dev   # Real LLM calls (requires API keys)
 *
 * For real mode, set environment variables:
 *   OPENAI_API_KEY - For OpenAI models
 *   GROQ_API_KEY   - For Groq models
 */

import 'dotenv/config';  // Load .env file
import { runSalesAgent, CONFIGURATION_SPACE, type AgentConfig } from './agent.js';
import { SALES_DATASET, getDatasetStats } from './dataset.js';
import { REAL_MODE } from './real-llm.js';

async function main() {
  const startTime = Date.now();

  console.log('\n' + '*'.repeat(70));
  console.log('*  ARKIA SALES AGENT - LOCAL DEMO');
  console.log('*  Margin Optimization Demonstration');
  if (REAL_MODE) {
    console.log('*  >>> REAL MODE - Making actual LLM API calls! <<<');
    console.log('*  (Using OpenAI/Groq APIs - costs will be incurred)');
  } else {
    console.log('*  (Mock mode - no API costs)');
  }
  console.log('*'.repeat(70));

  // Verify dataset is available
  if (!SALES_DATASET || SALES_DATASET.length === 0) {
    console.error('\nERROR: Dataset not loaded!');
    console.error('The SALES_DATASET is empty. Check dataset.ts for issues.');
    process.exit(1);
  }

  // Show dataset stats
  const stats = getDatasetStats();
  console.log('\nDataset Statistics:');
  console.log(`  Total examples: ${stats.total}`);
  console.log('  By intent:', stats.by_intent);

  console.log('\nConfiguration Space:');
  console.log(`  Models:       ${CONFIGURATION_SPACE.model.join(', ')}`);
  console.log(`  Temperature:  ${CONFIGURATION_SPACE.temperature.join(', ')}`);
  console.log(`  Prompts:      ${CONFIGURATION_SPACE.system_prompt.join(', ')}`);
  console.log(`  Memory Turns: ${CONFIGURATION_SPACE.memory_turns.join(', ')}`);
  console.log(`  Tool Sets:    ${CONFIGURATION_SPACE.tool_set.join(', ')}`);

  // Run comparison: OpenAI SOTA vs OpenAI Optimized vs Groq
  console.log('\n' + '='.repeat(70));
  console.log('COMPARISON: OpenAI SOTA vs OpenAI Optimized vs Groq');
  console.log('='.repeat(70));

  // Config 1: Current default (expensive SOTA - OpenAI)
  const sotaConfig: AgentConfig = {
    model: 'gpt-4o',
    temperature: 0.7,
    system_prompt: 'informative',
    memory_turns: 10,  // High memory = high cost
    tool_set: 'full',  // All tools = high cost
  };

  // Config 2: OpenAI optimized for margins
  const openaiOptimizedConfig: AgentConfig = {
    model: 'gpt-4o-mini',
    temperature: 0.3,
    system_prompt: 'consultative',
    memory_turns: 5,  // Reduced memory = lower cost
    tool_set: 'standard',  // Balanced tools
  };

  // Config 3: Groq - high quality + cheap + fast
  const groqConfig: AgentConfig = {
    model: 'groq/llama-3.3-70b-versatile',
    temperature: 0.3,
    system_prompt: 'consultative',
    memory_turns: 5,
    tool_set: 'standard',
  };

  // Config 4: Groq ultra-cheap for high-volume
  const groqCheapConfig: AgentConfig = {
    model: 'groq/llama-3.1-8b-instant',
    temperature: 0.3,
    system_prompt: 'consultative',
    memory_turns: 5,
    tool_set: 'minimal',  // Minimal tools for cost
  };

  // Use a subset for demo
  const demoExamples = SALES_DATASET.slice(0, 10);

  if (demoExamples.length === 0) {
    console.error('\nERROR: No examples available for demo!');
    console.error('Check that dataset.ts exports SALES_DATASET correctly.');
    process.exit(1);
  }

  console.log(`\nRunning 4 configurations on ${demoExamples.length} examples...`);

  // Run trials with error handling
  let sotaResult, openaiOptResult, groqResult, groqCheapResult;

  try {
    console.log('\n' + '#'.repeat(70));
    console.log('# TRIAL 1: OpenAI SOTA (gpt-4o, 10 memory turns, full tools)');
    console.log('#'.repeat(70));
    sotaResult = await runSalesAgent(demoExamples, sotaConfig);
  } catch (error) {
    console.error(`\nTRIAL 1 FAILED: ${error instanceof Error ? error.message : error}`);
    process.exit(1);
  }

  try {
    console.log('\n' + '#'.repeat(70));
    console.log('# TRIAL 2: OpenAI Optimized (gpt-4o-mini, 5 memory turns)');
    console.log('#'.repeat(70));
    openaiOptResult = await runSalesAgent(demoExamples, openaiOptimizedConfig);
  } catch (error) {
    console.error(`\nTRIAL 2 FAILED: ${error instanceof Error ? error.message : error}`);
    process.exit(1);
  }

  try {
    console.log('\n' + '#'.repeat(70));
    console.log('# TRIAL 3: Groq Llama 3.3 70B (high quality + cheap + FAST)');
    console.log('#'.repeat(70));
    groqResult = await runSalesAgent(demoExamples, groqConfig);
  } catch (error) {
    console.error(`\nTRIAL 3 FAILED: ${error instanceof Error ? error.message : error}`);
    process.exit(1);
  }

  try {
    console.log('\n' + '#'.repeat(70));
    console.log('# TRIAL 4: Groq Llama 3.1 8B (ultra-cheap for volume)');
    console.log('#'.repeat(70));
    groqCheapResult = await runSalesAgent(demoExamples, groqCheapConfig);
  } catch (error) {
    console.error(`\nTRIAL 4 FAILED: ${error instanceof Error ? error.message : error}`);
    process.exit(1);
  }

  // Comparison summary
  console.log('\n' + '='.repeat(70));
  console.log('COMPARISON SUMMARY - OpenAI vs Groq');
  console.log('='.repeat(70));
  console.log('\n                      GPT-4o      GPT-4o-mini   Groq 70B    Groq 8B');
  console.log('-'.repeat(70));
  console.log(`Conversion:           ${(sotaResult.avg_conversion_score * 100).toFixed(1)}%        ${(openaiOptResult.avg_conversion_score * 100).toFixed(1)}%          ${(groqResult.avg_conversion_score * 100).toFixed(1)}%        ${(groqCheapResult.avg_conversion_score * 100).toFixed(1)}%`);
  console.log(`Cost:                 $${sotaResult.total_cost.toFixed(5)}    $${openaiOptResult.total_cost.toFixed(5)}      $${groqResult.total_cost.toFixed(5)}    $${groqCheapResult.total_cost.toFixed(5)}`);
  console.log(`Latency (avg):        ${sotaResult.avg_latency_ms.toFixed(0)}ms         ${openaiOptResult.avg_latency_ms.toFixed(0)}ms          ${groqResult.avg_latency_ms.toFixed(0)}ms         ${groqCheapResult.avg_latency_ms.toFixed(0)}ms`);
  console.log(`Margin Efficiency:    ${sotaResult.margin_efficiency.toFixed(2)}          ${openaiOptResult.margin_efficiency.toFixed(2)}           ${groqResult.margin_efficiency.toFixed(2)}         ${groqCheapResult.margin_efficiency.toFixed(2)}`);
  console.log('-'.repeat(70));

  // Find best margin efficiency
  const configs = [
    { name: 'GPT-4o (SOTA)', result: sotaResult },
    { name: 'GPT-4o-mini', result: openaiOptResult },
    { name: 'Groq Llama 70B', result: groqResult },
    { name: 'Groq Llama 8B', result: groqCheapResult },
  ];
  const best = configs.reduce((a, b) =>
    a.result.margin_efficiency > b.result.margin_efficiency ? a : b
  );

  console.log(`\n[WINNER] ${best.name} has the best margin efficiency: ${best.result.margin_efficiency.toFixed(2)}`);

  const sotaVsBest = ((best.result.margin_efficiency - sotaResult.margin_efficiency) / sotaResult.margin_efficiency) * 100;
  console.log(`         ${sotaVsBest.toFixed(0)}% better than GPT-4o SOTA!`);

  // Groq insight
  if (groqResult.margin_efficiency > openaiOptResult.margin_efficiency) {
    console.log('\n[INSIGHT] Groq Llama 3.3 70B offers BETTER margins than OpenAI!');
    console.log('          Similar quality, lower cost, 5x faster response times.');
  }

  const elapsed = Date.now() - startTime;

  console.log('\n' + '*'.repeat(70));
  console.log('*  DEMO COMPLETE - SUCCESS');
  console.log(`*  Total time: ${elapsed}ms`);
  console.log('*');
  console.log('*  Run with Python orchestrator for full optimization:');
  console.log('*  TRAIGENT_COST_APPROVED=true python run_with_python.py');
  console.log('*'.repeat(70) + '\n');
}

// Run with error handling
main().catch((error) => {
  console.error('\n' + '='.repeat(70));
  console.error('DEMO FAILED');
  console.error('='.repeat(70));
  console.error(`\nError: ${error instanceof Error ? error.message : error}`);

  if (error instanceof Error && error.stack) {
    console.error('\nStack trace:');
    console.error(error.stack);
  }

  console.error('\nTroubleshooting:');
  console.error('1. Make sure you built the project: npm run build');
  console.error('2. Check that dataset.ts exports SALES_DATASET');
  console.error('3. Verify all dependencies are installed: npm install');

  process.exit(1);
});
