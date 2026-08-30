// Frontend service for running stress tests. Two shock mechanisms, both
// forward Monte Carlo simulations from a chosen as-of date, both built off
// an existing analysis_id -- matches runForecast.js's pattern, not the old
// raw-holdings pattern in the now-deleted runStressAnalysis.js.
import { apiUrl } from "../config/api";

async function postJson(path, body) {
  const res = await fetch(apiUrl(path), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  const data = await res.json();
  if (!res.ok) throw new Error(data?.error || "Request failed.");
  return data;
}

export async function runDeterministicRegimeStress({
  analysisId,
  source,
  asOfDate,
  days,
  simulations,
  driftShift,
  volMult,
}) {
  if (!analysisId) throw new Error("analysisId is required.");
  if (!days || days <= 0) throw new Error("days must be > 0.");
  if (!simulations || simulations <= 0) {
    throw new Error("simulations must be > 0.");
  }
  if (volMult != null && volMult <= 0) {
    throw new Error("volMult must be > 0.");
  }

  const body = { analysis_id: analysisId, days, simulations };

  if (source) body.source = source;
  if (asOfDate) body.as_of_date = asOfDate;
  if (driftShift != null) body.drift_shift = driftShift;
  if (volMult != null) body.vol_mult = volMult;

  return postJson("/api/stress/deterministic_regime", body);
}

export async function previewCalibratedRegimeStates({
  analysisId,
  source,
  asOfDate,
  nRegimes,
}) {
  if (!analysisId) throw new Error("analysisId is required.");

  const body = { analysis_id: analysisId };

  if (source) body.source = source;
  if (asOfDate) body.as_of_date = asOfDate;
  if (nRegimes != null) body.n_regimes = nRegimes;

  return postJson("/api/stress/calibrated_regime/preview", body);
}

export async function runCalibratedRegimeStress({
  regimeId,
  selectedState,
  days,
  simulations,
}) {
  if (!regimeId) throw new Error("regimeId is required.");
  if (selectedState == null) throw new Error("selectedState is required.");
  if (!days || days <= 0) throw new Error("days must be > 0.");
  if (!simulations || simulations <= 0) {
    throw new Error("simulations must be > 0.");
  }

  return postJson("/api/stress/calibrated_regime/stress", {
    regime_id: regimeId,
    selected_state: selectedState,
    days,
    simulations,
  });
}
