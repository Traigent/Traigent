/**
 * Real Sentiment Classification Agent using OpenRouter API
 *
 * This agent makes actual LLM API calls to demonstrate real-world
 * hyperparameter optimization with Traigent.
 */

import type { DatasetExample } from './dataset.js';

// Verbose mode - set VERBOSE=1 to see full API requests/responses
const VERBOSE = process.env.VERBOSE === '1' || process.env.VERBOSE === 'true';

/** Configuration space for the agent */
export interface AgentConfig {
  model: string;
  temperature: number;
  system_prompt: 'concise' | 'detailed' | 'cot';
}

/** Cost per 1M tokens [input, output] - OpenRouter pricing */
const MODEL_COSTS: Record<string, [number, number]> = {
  'openai/gpt-3.5-turbo': [0.5, 1.5],
  'openai/gpt-4o-mini': [0.15, 0.6],
  'openai/gpt-4o': [2.5, 10.0],
  'anthropic/claude-3-haiku': [0.25, 1.25],
  'anthropic/claude-3-5-sonnet': [3.0, 15.0],
  'google/gemini-flash-1.5': [0.075, 0.3],
  'meta-llama/llama-3.1-8b-instruct': [0.06, 0.06],
};

/** System prompts for different strategies */
const SYSTEM_PROMPTS: Record<string, string> = {
  concise: `Classify the sentiment of the following text as exactly one of: positive, negative, or neutral. Reply with only that single word.`,
  detailed: `You are a sentiment analysis expert. Analyze the following text and classify its sentiment.
Consider the overall tone, word choice, and emotional expression.
Your response must be exactly one word: positive, negative, or neutral.`,
  cot: `Classify the sentiment of the following text. Think step by step:
1. Identify key emotional words or phrases
2. Consider the overall tone
3. Determine if the sentiment is positive, negative, or neutral

After your analysis, your final answer must be on its own line as exactly one word: positive, negative, or neutral.`,
};

interface LLMResponse {
  prediction: string;
  correct: boolean;
  latency: number;
  inputTokens: number;
  outputTokens: number;
  cost: number;
  rawResponse: string;
}

/**
 * Make a real LLM API call via OpenRouter
 */
async function callLLM(
  text: string,
  expectedOutput: string,
  config: AgentConfig,
  apiKey: string
): Promise<LLMResponse> {
  const startTime = Date.now();

  const systemPrompt = SYSTEM_PROMPTS[config.system_prompt] ?? SYSTEM_PROMPTS.concise;

  const requestBody = {
    model: config.model,
    temperature: config.temperature,
    max_tokens: 50,
    messages: [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: `Text to classify:\n"${text}"` },
    ],
  };

  // VERBOSE: Show the actual API request
  if (VERBOSE) {
    console.log(
      '\n\x1b[35m┌──────────────────────────────────────────────────────────────────────┐\x1b[0m'
    );
    console.log(
      '\x1b[35m│ 📤 OPENROUTER API REQUEST                                            │\x1b[0m'
    );
    console.log(
      '\x1b[35m├──────────────────────────────────────────────────────────────────────┤\x1b[0m'
    );
    console.log(`\x1b[35m│\x1b[0m URL: https://openrouter.ai/api/v1/chat/completions`);
    console.log(`\x1b[35m│\x1b[0m Model: ${config.model}`);
    console.log(`\x1b[35m│\x1b[0m Temperature: ${config.temperature}`);
    console.log(`\x1b[35m│\x1b[0m System Prompt: "${systemPrompt.substring(0, 60)}..."`);
    console.log(`\x1b[35m│\x1b[0m User Input: "${text.substring(0, 50)}..."`);
    console.log(
      '\x1b[35m└──────────────────────────────────────────────────────────────────────┘\x1b[0m'
    );
  }

  const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
      'HTTP-Referer': 'https://traigent.io',
      'X-Title': 'Traigent Demo',
    },
    body: JSON.stringify(requestBody),
  });

  const latency = Date.now() - startTime;

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`API error ${response.status}: ${errorText}`);
  }

  const data = await response.json();

  // VERBOSE: Show the actual API response
  if (VERBOSE) {
    console.log(
      '\n\x1b[36m┌──────────────────────────────────────────────────────────────────────┐\x1b[0m'
    );
    console.log(
      '\x1b[36m│ 📥 OPENROUTER API RESPONSE                                           │\x1b[0m'
    );
    console.log(
      '\x1b[36m├──────────────────────────────────────────────────────────────────────┤\x1b[0m'
    );
    console.log(`\x1b[36m│\x1b[0m ID: ${data.id}`);
    console.log(`\x1b[36m│\x1b[0m Model: ${data.model}`);
    console.log(`\x1b[36m│\x1b[0m Response: "${data.choices?.[0]?.message?.content}"`);
    console.log(`\x1b[36m│\x1b[0m Finish Reason: ${data.choices?.[0]?.finish_reason}`);
    console.log(
      `\x1b[36m│\x1b[0m Tokens - Prompt: ${data.usage?.prompt_tokens}, Completion: ${data.usage?.completion_tokens}`
    );
    console.log(`\x1b[36m│\x1b[0m Latency: ${latency}ms`);
    console.log(
      '\x1b[36m└──────────────────────────────────────────────────────────────────────┘\x1b[0m'
    );
  }

  // Extract response
  const rawResponse = data.choices?.[0]?.message?.content ?? '';
  const usage = data.usage ?? { prompt_tokens: 0, completion_tokens: 0 };

  // Parse prediction from response
  const prediction = extractPrediction(rawResponse);
  const correct = prediction === expectedOutput.toLowerCase();

  // Calculate cost
  const costs = MODEL_COSTS[config.model] ?? [1, 3];
  const cost = (usage.prompt_tokens * costs[0] + usage.completion_tokens * costs[1]) / 1_000_000;

  return {
    prediction,
    correct,
    latency,
    inputTokens: usage.prompt_tokens,
    outputTokens: usage.completion_tokens,
    cost,
    rawResponse: rawResponse.trim(),
  };
}

/**
 * Extract sentiment prediction from LLM response
 */
function extractPrediction(response: string): string {
  const normalized = response.toLowerCase().trim();

  // Check for exact match
  if (['positive', 'negative', 'neutral'].includes(normalized)) {
    return normalized;
  }

  // Look for the sentiment word in the response
  const lines = normalized.split('\n');
  for (const line of lines.reverse()) {
    if (line.includes('positive')) return 'positive';
    if (line.includes('negative')) return 'negative';
    if (line.includes('neutral')) return 'neutral';
  }

  // Fallback - look anywhere in response
  if (normalized.includes('positive')) return 'positive';
  if (normalized.includes('negative')) return 'negative';
  if (normalized.includes('neutral')) return 'neutral';

  return 'unknown';
}

export interface AgentResult {
  accuracy: number;
  total_cost: number;
  avg_latency_ms: number;
  total_input_tokens: number;
  total_output_tokens: number;
  predictions: string[];
  details: Array<{
    text: string;
    expected: string;
    predicted: string;
    correct: boolean;
    latency: number;
    rawResponse: string;
  }>;
}

/**
 * Real Sentiment Classification Agent
 *
 * Processes examples using actual LLM API calls and returns accuracy metrics.
 */
export async function runRealSentimentAgent(
  examples: DatasetExample[],
  config: AgentConfig,
  apiKey: string,
  logger: (msg: string) => void = console.log
): Promise<AgentResult> {
  logger(`\n${'='.repeat(70)}`);
  logger(`REAL LLM AGENT CONFIGURATION:`);
  logger(`  Model:       ${config.model}`);
  logger(`  Temperature: ${config.temperature}`);
  logger(`  Prompt Type: ${config.system_prompt}`);
  logger(`${'='.repeat(70)}`);

  let correctCount = 0;
  let totalCost = 0;
  let totalLatency = 0;
  let totalInputTokens = 0;
  let totalOutputTokens = 0;
  const predictions: string[] = [];
  const details: AgentResult['details'] = [];

  for (let i = 0; i < examples.length; i++) {
    const example = examples[i];

    try {
      const result = await callLLM(example.input.text, example.output, config, apiKey);

      predictions.push(result.prediction);
      if (result.correct) correctCount++;
      totalCost += result.cost;
      totalLatency += result.latency;
      totalInputTokens += result.inputTokens;
      totalOutputTokens += result.outputTokens;

      details.push({
        text: example.input.text,
        expected: example.output,
        predicted: result.prediction,
        correct: result.correct,
        latency: result.latency,
        rawResponse: result.rawResponse,
      });

      // Log each prediction
      const status = result.correct ? '\x1b[32m[OK]\x1b[0m' : '\x1b[31m[X]\x1b[0m ';
      const textPreview = example.input.text.substring(0, 45).padEnd(45);
      logger(
        `  ${status} "${textPreview}..." => ${result.prediction.padEnd(8)} (expected: ${example.output}) [${result.latency}ms]`
      );

      // Add small delay to avoid rate limiting
      if (i < examples.length - 1) {
        await new Promise((resolve) => setTimeout(resolve, 100));
      }
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : String(error);
      logger(`  \x1b[31m[ERR]\x1b[0m "${example.input.text.substring(0, 40)}..." - ${errorMsg}`);
      predictions.push('error');
      details.push({
        text: example.input.text,
        expected: example.output,
        predicted: 'error',
        correct: false,
        latency: 0,
        rawResponse: errorMsg,
      });
    }
  }

  const accuracy = examples.length > 0 ? correctCount / examples.length : 0;
  const avgLatency = examples.length > 0 ? totalLatency / examples.length : 0;

  logger(`\n${'─'.repeat(70)}`);
  logger(`RESULTS:`);
  logger(
    `  Accuracy:      \x1b[1m${(accuracy * 100).toFixed(1)}%\x1b[0m (${correctCount}/${examples.length})`
  );
  logger(`  Total Cost:    \x1b[1m$${totalCost.toFixed(6)}\x1b[0m`);
  logger(`  Avg Latency:   \x1b[1m${avgLatency.toFixed(0)}ms\x1b[0m`);
  logger(`  Input Tokens:  ${totalInputTokens}`);
  logger(`  Output Tokens: ${totalOutputTokens}`);
  logger(`${'='.repeat(70)}\n`);

  return {
    accuracy,
    total_cost: totalCost,
    avg_latency_ms: avgLatency,
    total_input_tokens: totalInputTokens,
    total_output_tokens: totalOutputTokens,
    predictions,
    details,
  };
}

/** Available models for optimization */
export const REAL_CONFIGURATION_SPACE = {
  model: [
    'openai/gpt-4o-mini',
    'openai/gpt-3.5-turbo',
    'anthropic/claude-3-haiku',
    'google/gemini-flash-1.5',
    'meta-llama/llama-3.1-8b-instruct',
  ] as const,
  temperature: [0.0, 0.3, 0.7] as const,
  system_prompt: ['concise', 'detailed', 'cot'] as const,
};
