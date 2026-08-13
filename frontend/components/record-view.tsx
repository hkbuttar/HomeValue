"use client";

import { useEffect, useState } from "react";
import { apiFetch } from "@/lib/api";
import { ErrorState, LoadingState } from "./api-state";

export function ResearchPayload({ path, title }: { path: string; title: string }) {
  const [data, setData] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    apiFetch<{ results: Record<string, unknown> }>(path)
      .then((value) => setData(value.results)).catch((reason: Error) => setError(reason.message));
  }, [path]);
  if (error) return <ErrorState message={error} />;
  if (!data) return <LoadingState label={title} />;
  const entries = Object.entries(data).filter(([, value]) => ["string", "number", "boolean"].includes(typeof value)).slice(0, 8);
  return (
    <article className="evidence-card">
      <p className="eyebrow">Live research artifact</p><h3>{title}</h3>
      <dl>{entries.map(([key, value]) => <div key={key}><dt>{key.replaceAll("_", " ")}</dt><dd>{String(value)}</dd></div>)}</dl>
    </article>
  );
}
