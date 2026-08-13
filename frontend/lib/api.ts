import type { RecordCollection, ValuationResponse } from "./types";

export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
    cache: init?.method === "POST" ? "no-store" : "no-store",
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail ?? `HomeValue API returned ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export const getRecords = (path: string) => apiFetch<RecordCollection>(path);
export const predictValue = (payload: Record<string, unknown>) =>
  apiFetch<ValuationResponse>("/valuation/predict", { method: "POST", body: JSON.stringify(payload) });
