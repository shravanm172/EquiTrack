// Service for running forecasts by sending analysis ID and parameters to backend API.
import { apiUrl } from "../config/api";

export async function runForecast({ analysisId, days, source, mode = "mean", window, alpha, lambda: lam }) {
    if (!analysisId) throw new Error("analysisId is required.");
    if (!days || days <= 0) throw new Error("days must be > 0");

    const forecast = { days, mode };

    if (mode === "rolling") { 
        if (!window || window < 2) throw new Error("window must be >= 2 for rolling mode.");
        forecast.window = window;
    }

  
    if (mode === "ewma") { 
        if (alpha != null) {
            if (!(alpha > 0 && alpha < 1)) throw new Error("alpha must be in (0,1)");
            forecast.alpha = alpha;        
        } else {
            if (!(lam > 0 && lam < 1)) throw new Error("lambda must be in (0,1)");
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
