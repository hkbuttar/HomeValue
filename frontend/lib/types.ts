export type ValueDriver = {
  component: "property" | "place" | "time_market" | "other";
  feature: string;
  dollar_contribution: number;
};

export type Comparable = {
  sale_id: string;
  sale_price: number;
  distance_miles: number;
  sale_date: string;
};

export type ValuationResponse = {
  estimated_value: number;
  lower_interval: number;
  upper_interval: number;
  baseline_market_value: number;
  property_component: number;
  location_component: number;
  time_market_component: number;
  other_component: number;
  confidence: number;
  model_name: string;
  value_drivers: ValueDriver[];
  comparables: Comparable[];
};

export type RecordCollection = {
  count: number;
  offset: number;
  limit: number;
  records: Record<string, unknown>[];
};
