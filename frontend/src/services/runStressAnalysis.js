// Service for running stress analysis by sending holdings, date range, and shock scenario to backend API.
import { buildAnalyzePayloadFromLots } from "../lib/portfolioPayload";
import { apiUrl } from "../config/api";

export async function analyzeWithStress({ holdings, startDate, endDate, shock }) {
  const payload = buildAnalyzePayloadFromLots(holdings, startDate, endDate);

  const res = await fetch(apiUrl("/api/analyze_shock"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      ...payload,
      shock, // <-- scenario config object
    }),
  });

  if (!res.ok) {
    let msg = "Stress analysis failed.";
    try {
      const err = await res.json();
      msg = err?.error || err?.message || msg;
    } catch {}
    throw new Error(msg);
  }

  return res.json();
}
