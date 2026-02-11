// src/services/runShockAnalysis.js
import { buildAnalyzePayloadFromLots } from "../lib/portfolioPayload";

/**
 * Expected backend route examples (pick the one you implemented):
 * - POST /api/stress/analyze_with_shock
 * - POST /api/stress/analyze-with-shock
 * - POST /api/analyze_with_shock
 *
 * Adjust URL below to match your Flask route.
 */
export async function analyzeWithStress({ holdings, startDate, endDate, shock }) {
  const payload = buildAnalyzePayloadFromLots(holdings, startDate, endDate);

  const res = await fetch("http://localhost:5000/api/analyze_shock", {
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
