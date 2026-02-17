// Builds the payload for the /analyze endpoint from an array of lots, start/end dates, and optional starting cash.
function normalizeTicker(t) {
  return (t || "").trim().toUpperCase();
}

export function buildAnalyzePayloadFromLots(lots, start, end, startingCash) {
  const sharesByTicker = new Map();

  for (const lot of lots) {
    const ticker = normalizeTicker(lot.ticker);
    const shares = Number(lot.shares);

    if (!ticker) continue;
    if (!Number.isFinite(shares) || shares <= 0) continue;

    sharesByTicker.set(ticker, (sharesByTicker.get(ticker) || 0) + shares);
  }

  const holdings = Array.from(sharesByTicker.entries()).map(([ticker, shares]) => ({
    ticker,
    shares,
  }));

  return {
    date_range: { start, end },
    portfolio: {
      holdings,
      ...(startingCash != null ? { starting_cash: Number(startingCash) } : {}),
    },
  };
}
