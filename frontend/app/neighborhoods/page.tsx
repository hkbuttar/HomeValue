"use client";

import { useEffect, useState } from "react";
import { getRecords } from "@/lib/api";
import { ErrorState, LoadingState } from "@/components/api-state";

export default function NeighborhoodsPage() {
  const [records, setRecords] = useState<Record<string, unknown>[] | null>(null); const [error, setError] = useState("");
  useEffect(() => { getRecords("/neighborhoods/segments?limit=100").then((value) => setRecords(value.records)).catch((reason: Error) => setError(reason.message)); }, []);
  return <section className="shell"><div className="page-intro"><p className="eyebrow">Neighborhood dynamics</p><h1>Market archetypes,<br/>not stereotypes.</h1><p>Segments are named only after their measured price, growth, turnover, housing, income, and accessibility profiles are fitted.</p></div>
    {error ? <ErrorState message={error}/> : !records ? <LoadingState/> : <div className="segment-grid">{records.map((record, index) => <article key={index}><span>{String(record.label ?? `Area ${record.nbhd ?? index + 1}`)}</span><h2>{String(record.archetype ?? "Market segment")}</h2><dl><div><dt>Median sale</dt><dd>{record.median_sale_price ? `$${Number(record.median_sale_price).toLocaleString()}` : "—"}</dd></div><div><dt>Annual velocity</dt><dd>{record.annual_sale_velocity ? Number(record.annual_sale_velocity).toFixed(1) : "—"}</dd></div></dl></article>)}</div>}
  </section>;
}
