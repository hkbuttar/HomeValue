"use client";

import { useEffect, useState } from "react";
import { getRecords } from "@/lib/api";
import { ErrorState, LoadingState } from "@/components/api-state";

export default function ResearchPage() {
  const [rows, setRows] = useState<Record<string, unknown>[] | null>(null); const [error, setError] = useState("");
  useEffect(() => { getRecords("/models/performance").then((value) => setRows(value.records)).catch((reason: Error) => setError(reason.message)); }, []);
  return <section className="shell"><div className="page-intro"><p className="eyebrow">Model evidence</p><h1>Prediction and explanation<br/>are different jobs.</h1><p>The strongest predictor is not automatically the clearest explanation. We report temporal and spatial tests separately.</p></div>
    {error ? <ErrorState message={error}/> : !rows ? <LoadingState/> : <div className="table-wrap"><table><thead><tr><th>Model</th><th>MAE</th><th>RMSE</th><th>MdAPE</th><th>Temporal</th><th>Spatial</th></tr></thead><tbody>{rows.map((row, index) => <tr key={index}><td>{String(row.model)}</td><td>{row.mae ? `$${Number(row.mae).toLocaleString()}` : "—"}</td><td>{row.rmse ? `$${Number(row.rmse).toLocaleString()}` : "—"}</td><td>{row.mdape ? `${(Number(row.mdape) * 100).toFixed(1)}%` : "—"}</td><td>{row.temporal_test ? "✓" : "—"}</td><td>{row.spatial_test ? "✓" : "—"}</td></tr>)}</tbody></table></div>}
    <div className="method-note"><strong>Our rule:</strong><p>No model “wins” without surviving the validation design relevant to the claim being made.</p></div>
  </section>;
}
