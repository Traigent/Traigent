/**
 * Curated dataset for Arkia travel sales agent evaluation.
 *
 * These examples represent real customer interactions that Arkia's team
 * has manually curated for quality evaluation. Each example includes:
 * - Customer query and conversation history
 * - Expected intent classification
 * - Ground truth quality expectations
 * - Expected output for accuracy measurement
 */

import { readFileSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';

export interface Message {
  role: 'customer' | 'agent';
  content: string;
}

export interface ConversationExample {
  id: string;
  customer_query: string;           // The main query being evaluated
  messages: Message[];              // Conversation history (for memory)
  intent: 'flight_inquiry' | 'price_negotiation' | 'booking_intent' | 'support' | 'complaint';
  destination: string;
  /**
   * Difficulty level for optimization testing:
   * - easy: Simple queries, high-intent customers, single-turn
   * - medium: Multi-turn conversations, moderate complexity
   * - hard: Complex negotiations, complaints, edge cases
   */
  difficulty: 'easy' | 'medium' | 'hard';
  expected_quality: {
    min_relevancy: number;          // Minimum acceptable relevancy
    min_completeness: number;       // Minimum acceptable completeness
    should_upsell: boolean;         // Should agent try to upsell?
    urgency_level: 'low' | 'medium' | 'high';
  };
  /** Expected output for accuracy measurement (used by SDK for scoring) */
  output?: string;
}

/** Raw JSONL entry format from dataset.jsonl */
interface DatasetEntry {
  id: string;
  input: { text: string };
  intent: string;
  destination: string;
  difficulty: string;
  expected_quality: {
    min_relevancy: number;
    min_completeness: number;
    should_upsell: boolean;
    urgency_level: string;
  };
  output?: string;
}

/**
 * Load dataset from JSONL file.
 * Creates synthetic conversation history for each example.
 */
function loadDatasetFromJsonl(): ConversationExample[] {
  const __filename = fileURLToPath(import.meta.url);
  const __dirname = dirname(__filename);
  const datasetPath = resolve(__dirname, '..', 'dataset.jsonl');

  try {
    const content = readFileSync(datasetPath, 'utf-8');
    const lines = content.trim().split('\n').filter(line => line.trim());

    return lines.map((line): ConversationExample => {
      const entry: DatasetEntry = JSON.parse(line);
      const customerQuery = entry.input.text;

      // Create synthetic conversation history based on intent
      const messages = createConversationHistory(customerQuery, entry.intent);

      return {
        id: entry.id,
        customer_query: customerQuery,
        messages,
        intent: entry.intent as ConversationExample['intent'],
        destination: entry.destination,
        difficulty: entry.difficulty as ConversationExample['difficulty'],
        expected_quality: {
          min_relevancy: entry.expected_quality.min_relevancy,
          min_completeness: entry.expected_quality.min_completeness,
          should_upsell: entry.expected_quality.should_upsell,
          urgency_level: entry.expected_quality.urgency_level as 'low' | 'medium' | 'high',
        },
        output: entry.output,
      };
    });
  } catch (error) {
    console.error(`[DATASET] Failed to load dataset.jsonl: ${error}`);
    return [];
  }
}

/**
 * Create synthetic conversation history for an example.
 * Adds context turns based on the intent type.
 */
function createConversationHistory(
  customerQuery: string,
  intent: string
): Message[] {
  const messages: Message[] = [];

  // Add contextual opening based on intent
  switch (intent) {
    case 'flight_inquiry':
      messages.push(
        { role: 'customer', content: 'Hi, I am looking for flight information' },
        { role: 'agent', content: 'Hello! I would be happy to help you find the perfect flight. What destination are you interested in?' }
      );
      break;
    case 'price_negotiation':
      messages.push(
        { role: 'customer', content: 'I saw your prices online' },
        { role: 'agent', content: 'Thank you for considering Arkia! How can I assist you with pricing today?' }
      );
      break;
    case 'booking_intent':
      messages.push(
        { role: 'customer', content: 'I am ready to book a trip' },
        { role: 'agent', content: 'Wonderful! I will help you complete your booking. What would you like to book?' }
      );
      break;
    case 'support':
      messages.push(
        { role: 'customer', content: 'I have a question about my travel' },
        { role: 'agent', content: 'Of course! I am here to help. What would you like to know?' }
      );
      break;
    case 'complaint':
      messages.push(
        { role: 'customer', content: 'I need to speak to someone about a problem' },
        { role: 'agent', content: 'I am sorry to hear you had an issue. Please tell me what happened and I will do my best to help.' }
      );
      break;
    default:
      messages.push(
        { role: 'customer', content: 'Hello' },
        { role: 'agent', content: 'Hello! How can I assist you today?' }
      );
  }

  // Add the actual customer query
  messages.push({ role: 'customer', content: customerQuery });

  return messages;
}

/**
 * Loaded dataset from JSONL file.
 * Contains 50 curated examples covering various intents and difficulty levels.
 */
export const SALES_DATASET: ConversationExample[] = loadDatasetFromJsonl();

/**
 * Get a subset of the dataset by indices.
 * Used by the trial runner when Python orchestrator sends specific indices.
 *
 * @param indices - Array of dataset indices to retrieve
 * @returns Array of conversation examples at the specified indices.
 *          Out-of-bounds indices are silently filtered out.
 *
 * @example
 * ```ts
 * const subset = getDatasetSubset([0, 5, 10]);
 * // Returns examples at indices 0, 5, and 10
 *
 * const empty = getDatasetSubset([999]); // Out of bounds
 * // Returns [] (empty array)
 * ```
 */
export function getDatasetSubset(indices: number[]): ConversationExample[] {
  return indices.map(i => SALES_DATASET[i]).filter(Boolean);
}

/**
 * Get dataset statistics for logging and analysis.
 *
 * @returns Statistics including:
 *   - total: Number of examples in the dataset
 *   - by_intent: Count of examples per intent type
 *   - by_difficulty: Count of examples per difficulty level
 *   - by_destination: Count of examples per destination
 */
export function getDatasetStats(): {
  total: number;
  by_intent: Record<string, number>;
  by_difficulty: Record<string, number>;
  by_destination: Record<string, number>;
} {
  const byIntent: Record<string, number> = {};
  const byDifficulty: Record<string, number> = {};
  const byDestination: Record<string, number> = {};

  for (const example of SALES_DATASET) {
    byIntent[example.intent] = (byIntent[example.intent] ?? 0) + 1;
    byDifficulty[example.difficulty] = (byDifficulty[example.difficulty] ?? 0) + 1;
    byDestination[example.destination] = (byDestination[example.destination] ?? 0) + 1;
  }

  return {
    total: SALES_DATASET.length,
    by_intent: byIntent,
    by_difficulty: byDifficulty,
    by_destination: byDestination,
  };
}
