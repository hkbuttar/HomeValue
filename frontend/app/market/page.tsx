"use client";

import { useEffect, useState } from "react";
import { MarketMap } from "@/components/market-map";
import { apiFetch } from "@/lib/api";
import { ErrorState, LoadingState } from "@/components/api-state";

type Summary = { latest_year: number; geography_count: number; transaction_count: number; median_sale_price: number; median_ppsf: number };
const money = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 });

export default function MarketPage() {
  const [summary, setSummary] = useState<Summary | null>(null); const [error, setError] = useState("");
  useEffect(() => { apiFetch<Summary>("/market/summary").then(setSummary).catch((reason: Error) => setError(reason.message)); }, []);
  return <section className="shell market-page"><div className="page-intro"><p className="eyebrow">Chicago housing market</p><h1>One city.<br/>Many market rhythms.</h1><p>Explore price levels, transaction activity, and neighborhood trajectories without flattening Chicago into one average.</p></div>
    {error ? <ErrorState message={error}/> : !summary ? <LoadingState/> : <div className="stat-strip"><div><span>Market areas</span><strong>{summary.geography_count}</strong></div><div><span>Median sale</span><strong>{money.format(summary.median_sale_price)}</strong></div><div><span>Median PPSF</span><strong>{money.format(summary.median_ppsf)}</strong></div><div><span>{summary.latest_year} sales</span><strong>{summary.transaction_count.toLocaleString()}</strong></div></div>}
    <MarketMap/><p className="caption">Map context uses open vector tiles. Neighborhood analytical overlays are served by FastAPI.</p>
  </section>;
}
