/**
 * Tools for Arkia Sales Agent
 *
 * These are the tools that Arkia provides to their sales agents.
 * Tool selection is a KEY optimization dimension because:
 *
 * - More tools = more capable agent BUT higher token cost (tool descriptions)
 * - Some tools are essential, others are "nice to have"
 * - Tool choice affects latency (multiple tool calls = slower)
 *
 * Optimization question: Which tool set gives best conversion per dollar?
 */

import { z } from 'zod';

/**
 * Tool definition interface (Mastra-compatible)
 */
export interface Tool {
  name: string;
  description: string;
  parameters: z.ZodType;
  execute: (params: unknown) => Promise<unknown>;
}

// ============================================================================
// CORE TOOLS - Essential for sales (always included)
// ============================================================================

export const searchFlights: Tool = {
  name: 'search_flights',
  description:
    'Search for available flights between two cities on specific dates. Returns flight options with prices.',
  parameters: z.object({
    origin: z.string().describe('Origin airport code (e.g., TLV)'),
    destination: z.string().describe('Destination airport code (e.g., BCN)'),
    departure_date: z.string().describe('Departure date (YYYY-MM-DD)'),
    return_date: z.string().optional().describe('Return date for round trip'),
    passengers: z.number().default(1).describe('Number of passengers'),
    cabin_class: z.enum(['economy', 'business', 'first']).default('economy'),
  }),
  execute: async () => {
    // Mock implementation
    return {
      flights: [
        { flight: 'IZ301', price: 1200, departure: '08:00', arrival: '12:30' },
        { flight: 'IZ305', price: 950, departure: '14:00', arrival: '18:30' },
      ],
    };
  },
};

export const getFlightPrice: Tool = {
  name: 'get_flight_price',
  description: 'Get detailed pricing for a specific flight including taxes and fees breakdown.',
  parameters: z.object({
    flight_number: z.string().describe('Flight number (e.g., IZ301)'),
    passengers: z.number().default(1),
    cabin_class: z.enum(['economy', 'business', 'first']).default('economy'),
  }),
  execute: async () => {
    return {
      base_fare: 800,
      taxes: 150,
      fees: 50,
      total: 1000,
      currency: 'ILS',
    };
  },
};

export const createBooking: Tool = {
  name: 'create_booking',
  description: 'Create a flight booking reservation. Requires passenger details and payment.',
  parameters: z.object({
    flight_number: z.string(),
    passengers: z.array(
      z.object({
        first_name: z.string(),
        last_name: z.string(),
        passport_number: z.string(),
        date_of_birth: z.string(),
      })
    ),
    contact_email: z.string().email(),
    contact_phone: z.string(),
  }),
  execute: async () => {
    return {
      booking_reference: 'ARK-' + Math.random().toString(36).substring(7).toUpperCase(),
      status: 'confirmed',
      total_paid: 1000,
    };
  },
};

// ============================================================================
// ENHANCED TOOLS - Improve conversion but add cost
// ============================================================================

export const checkAvailability: Tool = {
  name: 'check_availability',
  description: 'Check real-time seat availability for a specific flight.',
  parameters: z.object({
    flight_number: z.string(),
    date: z.string(),
    cabin_class: z.enum(['economy', 'business', 'first']).default('economy'),
  }),
  execute: async () => {
    return {
      available_seats: Math.floor(Math.random() * 20) + 5,
      total_capacity: 180,
      availability: 'available',
    };
  },
};

export const getCustomerHistory: Tool = {
  name: 'get_customer_history',
  description: 'Look up customer booking history and preferences for personalized recommendations.',
  parameters: z.object({
    email: z.string().email().optional(),
    phone: z.string().optional(),
    loyalty_number: z.string().optional(),
  }),
  execute: async () => {
    return {
      customer_type: 'returning',
      total_bookings: 5,
      preferred_destinations: ['Paris', 'Barcelona'],
      loyalty_tier: 'gold',
      preferred_seat: 'aisle',
    };
  },
};

export const getCurrentPromotions: Tool = {
  name: 'get_current_promotions',
  description: 'Get current deals and promotions that can be applied to bookings.',
  parameters: z.object({
    destination: z.string().optional(),
    travel_dates: z.string().optional(),
  }),
  execute: async () => {
    return {
      promotions: [
        { code: 'SUMMER25', discount: '25%', valid_until: '2025-08-31' },
        { code: 'EARLYBIRD', discount: '15%', conditions: 'Book 30+ days ahead' },
      ],
    };
  },
};

export const getDestinationInfo: Tool = {
  name: 'get_destination_info',
  description:
    'Get information about a destination including weather, attractions, and travel requirements.',
  parameters: z.object({
    destination: z.string().describe('City or country name'),
    travel_date: z.string().optional(),
  }),
  execute: async () => {
    return {
      weather: { temp: '25°C', conditions: 'Sunny' },
      attractions: ['Sagrada Familia', 'Park Güell', 'La Rambla'],
      visa_required: false,
      covid_requirements: 'None',
    };
  },
};

// ============================================================================
// PREMIUM TOOLS - High value but expensive (more tokens)
// ============================================================================

export const buildPackage: Tool = {
  name: 'build_vacation_package',
  description:
    'Build a complete vacation package with flights, hotels, and activities. Higher margin product.',
  parameters: z.object({
    destination: z.string(),
    departure_date: z.string(),
    return_date: z.string(),
    passengers: z.number(),
    hotel_stars: z.enum(['3', '4', '5']).default('4'),
    include_transfers: z.boolean().default(true),
    include_activities: z.boolean().default(false),
  }),
  execute: async () => {
    return {
      package_id: 'PKG-' + Math.random().toString(36).substring(7).toUpperCase(),
      flights: { outbound: 'IZ301', return: 'IZ302' },
      hotel: { name: 'Hotel Barcelona Plaza', stars: 4, nights: 4 },
      transfers: 'Airport transfers included',
      total_price: 3500,
      savings: 450,
    };
  },
};

export const calculateInstallments: Tool = {
  name: 'calculate_installments',
  description: 'Calculate payment installment options for a booking.',
  parameters: z.object({
    total_amount: z.number(),
    num_installments: z.enum(['3', '6', '12']),
  }),
  execute: async () => {
    const total = (params as { total_amount: number }).total_amount;
    const num = parseInt((params as { num_installments: string }).num_installments);
    return {
      monthly_payment: Math.round(total / num),
      total_with_interest: Math.round(total * (1 + num * 0.005)),
      first_payment_date: new Date().toISOString().split('T')[0],
    };
  },
};

export const upgradeOptions: Tool = {
  name: 'get_upgrade_options',
  description: 'Get available upgrade options for a booking (seats, class, baggage).',
  parameters: z.object({
    booking_reference: z.string().optional(),
    flight_number: z.string().optional(),
  }),
  execute: async () => {
    return {
      seat_upgrades: [
        { type: 'Extra legroom', price: 150 },
        { type: 'Exit row', price: 200 },
      ],
      class_upgrade: { to: 'business', price: 800 },
      baggage: [
        { type: 'Extra 23kg bag', price: 100 },
        { type: 'Sports equipment', price: 150 },
      ],
    };
  },
};

// ============================================================================
// TOOL SETS - Different configurations to optimize
// ============================================================================

/** Minimal tool set - lowest cost, basic functionality */
export const TOOLS_MINIMAL = [searchFlights, getFlightPrice, createBooking];

/** Standard tool set - balanced cost/capability */
export const TOOLS_STANDARD = [
  searchFlights,
  getFlightPrice,
  createBooking,
  checkAvailability,
  getCurrentPromotions,
];

/** Enhanced tool set - better conversion, higher cost */
export const TOOLS_ENHANCED = [
  searchFlights,
  getFlightPrice,
  createBooking,
  checkAvailability,
  getCurrentPromotions,
  getCustomerHistory,
  getDestinationInfo,
];

/** Full tool set - maximum capability, highest cost */
export const TOOLS_FULL = [
  searchFlights,
  getFlightPrice,
  createBooking,
  checkAvailability,
  getCurrentPromotions,
  getCustomerHistory,
  getDestinationInfo,
  buildPackage,
  calculateInstallments,
  upgradeOptions,
];

/** Tool set options for optimization */
export const TOOL_SETS = {
  minimal: TOOLS_MINIMAL, // 3 tools, ~300 tokens in system prompt
  standard: TOOLS_STANDARD, // 5 tools, ~500 tokens in system prompt
  enhanced: TOOLS_ENHANCED, // 7 tools, ~700 tokens in system prompt
  full: TOOLS_FULL, // 10 tools, ~1000 tokens in system prompt
} as const;

export type ToolSetName = keyof typeof TOOL_SETS;

/**
 * Estimated token cost for each tool set (in system prompt)
 */
export const TOOL_SET_TOKEN_COSTS: Record<ToolSetName, number> = {
  minimal: 300,
  standard: 500,
  enhanced: 700,
  full: 1000,
};

/**
 * Estimated conversion boost for each tool set
 * (More tools = better agent = higher conversion, but diminishing returns)
 */
export const TOOL_SET_CONVERSION_BOOST: Record<ToolSetName, number> = {
  minimal: 1.0, // Baseline
  standard: 1.08, // +8% conversion
  enhanced: 1.12, // +12% conversion
  full: 1.15, // +15% conversion (diminishing returns)
};
