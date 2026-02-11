export async function runForecast({ analysisId, days }) {
  if (!analysisId) throw new Error("analysisId is required.");
  if (!days || days <= 0) throw new Error("days must be > 0");

  const res = await fetch("http://localhost:5000/api/forecast", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      analysis_id: analysisId,
      forecast: { days, mode: "mean" },
      // source optional: defaults to baseline
    }),
  });

  const data = await res.json();

  if (!res.ok) {
    throw new Error(data?.error || "Forecast failed.");
  }

  return data;
}
