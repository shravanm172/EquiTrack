// Frontend service for running deterministic or stochastic forecasts.
import { apiUrl } from "../config/api";

export async function runForecast({
  analysisId,
  source,
  type = "deterministic", // "deterministic" | "stochastic"
  days,
  driftMode = "mean",     // "mean" | "rolling" | "ewma"
  volMode = "historical", // stochastic only: "historical" | "rolling" | "ewma"
  simulations,
  window,
  alpha,
  lambda: lam,
}) {
  if (!analysisId) throw new Error("analysisId is required.");
  if (!days || days <= 0) throw new Error("days must be > 0");

  if (!["deterministic", "stochastic"].includes(type)) {
    throw new Error("type must be 'deterministic' or 'stochastic'.");
  }

  if (!["mean", "rolling", "ewma"].includes(driftMode)) {
    throw new Error("driftMode must be 'mean', 'rolling', or 'ewma'.");
  }

  if (type === "stochastic") {
    if (!["historical", "rolling", "ewma"].includes(volMode)) {
      throw new Error("volMode must be 'historical', 'rolling', or 'ewma'.");
    }
    if (!simulations || simulations <= 0) {
      throw new Error("simulations must be > 0 for stochastic forecasts.");
    }
  }

  const usesRolling =
    driftMode === "rolling" || (type === "stochastic" && volMode === "rolling");

  const usesEwma =
    driftMode === "ewma" || (type === "stochastic" && volMode === "ewma");

  const forecast = {
    type,
    days,
    drift_mode: driftMode,
  };

  if (type === "stochastic") {
    forecast.vol_mode = volMode;
    forecast.simulations = simulations;
  }

  if (usesRolling) {
    if (!window || window < 2) {
      throw new Error("window must be >= 2 for rolling mode.");
    }
    forecast.window = window;
  }

  if (usesEwma) {
    if (alpha != null) {
      if (!(alpha > 0 && alpha < 1)) {
        throw new Error("alpha must be in (0,1)");
      }
      forecast.alpha = alpha;
    } else {
      if (!(lam > 0 && lam < 1)) {
        throw new Error("lambda must be in (0,1)");
      }
      forecast.lambda = lam;
    }
  }

  const body = {
    analysis_id: analysisId,
    forecast,
  };

  if (source) body.source = source;

  const res = await fetch(apiUrl("/api/forecast"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  const data = await res.json();
  if (!res.ok) throw new Error(data?.error || "Forecast failed.");
  return data;
}