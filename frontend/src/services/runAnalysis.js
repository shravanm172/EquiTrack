import { buildAnalyzePayloadFromLots } from "../lib/portfolioPayload";

function todayYYYYMMDD() {
  return new Date().toISOString().slice(0, 10);
}

export async function runPortfolioAnalysis({ holdings, startDate, endDate }) {
  const today = todayYYYYMMDD();

  if (!startDate) throw new Error("Start date is required.");
  if (!endDate) throw new Error("End date is required.");
  if (endDate > today) throw new Error("End date cannot be in the future.");
  if (endDate < startDate) throw new Error("End date cannot be before start date.");

  const payload = buildAnalyzePayloadFromLots(holdings, startDate, endDate);

  if (!payload.portfolio.holdings.length) {
    throw new Error("Add at least one holding before running analysis.");
  }

  const res = await fetch("http://localhost:5000/api/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  const data = await res.json();

  if (!res.ok) {
    throw new Error(data.error || "Analysis failed.");
  }

  return data;
}
