"use client";

import { FormEvent, useEffect, useState } from "react";
import { getRecords, predictValue } from "@/lib/api";
import type { ValuationResponse } from "@/lib/types";

const money = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 });
type NeighborhoodOption = { neighborhood_id: string; label: string };

export default function ValuationPage() {
  const [result, setResult] = useState<ValuationResponse | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [neighborhoods, setNeighborhoods] = useState<NeighborhoodOption[]>([]);
  const [neighborhoodError, setNeighborhoodError] = useState("");
  useEffect(() => {
    getRecords("/valuation/neighborhoods")
      .then((value) => setNeighborhoods(value.records as NeighborhoodOption[]))
      .catch((reason: Error) => setNeighborhoodError(reason.message));
  }, []);
  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault(); setLoading(true); setError("");
    const values = Object.fromEntries(new FormData(event.currentTarget));
    const numeric = ["building_sqft", "land_sqft", "bedrooms", "bathrooms", "building_age", "garage_spaces", "latitude", "longitude"];
    numeric.forEach((key) => { if (values[key]) values[key] = Number(values[key]) as never; else delete values[key]; });
    try { setResult(await predictValue(values)); } catch (reason) { setError((reason as Error).message); } finally { setLoading(false); }
  }
  return (
    <section className="shell page-grid">
      <div className="page-intro"><p className="eyebrow">Property valuation</p><h1>Tell us about the home.</h1><p>We’ll connect its physical character to the market around it—and show the evidence behind the estimate.</p></div>
      <form className="valuation-form" onSubmit={submit}>
        <label className="field-wide">Neighborhood
          <select name="neighborhood" defaultValue="">
            <option value="">Use countywide context</option>
            {neighborhoods.map((option) => <option key={option.neighborhood_id} value={option.neighborhood_id}>{option.label}</option>)}
          </select>
        </label>
        {neighborhoodError && <p className="form-error field-wide">Neighborhood list unavailable: {neighborhoodError}</p>}
        <label>Building area<input required name="building_sqft" type="number" min="1" placeholder="1,850 sq ft" /></label>
        <label>Land area<input name="land_sqft" type="number" min="1" placeholder="3,125 sq ft" /></label>
        <label>Bedrooms<input name="bedrooms" type="number" min="0" step="1" placeholder="3" /></label>
        <label>Bathrooms<input name="bathrooms" type="number" min="0" step="0.5" placeholder="2.5" /></label>
        <label>Building age<input name="building_age" type="number" min="0" placeholder="74 years" /></label>
        <label>Garage spaces<input name="garage_spaces" type="number" min="0" placeholder="2" /></label>
        <label>Latitude<input name="latitude" type="number" step="any" placeholder="41.878" /></label>
        <label>Longitude<input name="longitude" type="number" step="any" placeholder="-87.630" /></label>
        <label className="field-wide">Property type<input name="residence_type" placeholder="Single Family" /></label>
        <button className="button field-wide" disabled={loading}>{loading ? "Reading the market…" : "Estimate market value →"}</button>
        {error && <p className="form-error field-wide">{error}</p>}
      </form>
      {result && <section className="valuation-result">
        <p className="eyebrow">Estimated market value</p><h2>{money.format(result.estimated_value)}</h2>
        <div className="range"><span>{money.format(result.lower_interval)}</span><i/><span>{money.format(result.upper_interval)}</span></div>
        <p className="range-label">{Math.round(result.confidence * 100)}% calibrated likely range · {result.model_name.replaceAll("_", " ")}</p>
        <div className="component-list">
          {[['Property', result.property_component], ['Place', result.location_component], ['Time / market', result.time_market_component]].map(([label, value]) =>
            <div key={String(label)}><span>{label}</span><strong>{Number(value) >= 0 ? "+" : ""}{money.format(Number(value))}</strong></div>)}
        </div>
        <h3>Strongest value drivers</h3>
        {result.value_drivers.slice(0, 5).map((driver) => <div className="driver" key={driver.feature}><span>{driver.feature.replaceAll("_", " ")}</span><b>{driver.dollar_contribution >= 0 ? "+" : ""}{money.format(driver.dollar_contribution)}</b></div>)}
      </section>}
    </section>
  );
}
