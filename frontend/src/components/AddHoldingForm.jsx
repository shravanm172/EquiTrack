import { useMemo, useState } from "react";
import { apiUrl } from "../config/api";

function normalizeTicker(t) {
  return (t || "").trim().toUpperCase();
}

function isYYYYMMDD(s) {
  return /^\d{4}-\d{2}-\d{2}$/.test(s);
}

function todayYYYYMMDD() {
  return new Date().toISOString().slice(0, 10);
}

function isFutureDate(dateStr) {
  // Compare by date-only (local). "Today" is allowed.
  const [y, m, d] = dateStr.split("-").map(Number);
  const picked = new Date(y, m - 1, d);
  const now = new Date();
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  return picked.getTime() > today.getTime();
}

export default function AddHoldingForm({ holdings, setHoldings }) {
  const [ticker, setTicker] = useState("");
  const [shares, setShares] = useState("");
  const [buyDate, setBuyDate] = useState(todayYYYYMMDD());
  const [error, setError] = useState("");
  const [note, setNote] = useState("");
  const [loading, setLoading] = useState(false);

  const canSubmit = useMemo(() => {
    const sym = normalizeTicker(ticker);
    const sh = Number(shares);
    return sym.length > 0 && Number.isFinite(sh) && sh > 0 && !!buyDate;
  }, [ticker, shares, buyDate]);

  async function validateWithBackend(symbol, date) {
    const res = await fetch(apiUrl("/api/holdings/validate"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        ticker: symbol,
        buy_date: date,
        lookahead_days: 7,
      }),
    });

    if (res.status === 400) {
      const data = await res.json().catch(() => ({}));
      throw new Error(
        data.reason || data.error || "Validation request invalid.",
      );
    }

    if (!res.ok) {
      throw new Error(`Validation failed (${res.status}).`);
    }

    const data = await res.json();
    return data;
  }

  async function handleSubmit(e) {
    e.preventDefault();
    setError("");
    setNote("");

    const symbol = normalizeTicker(ticker);
    const sh = Number(shares);
    const date = (buyDate || "").trim();

    // ---------- Frontend validation ----------
    if (!symbol) {
      setError("Ticker is required.");
      return;
    }

    if (!Number.isFinite(sh) || sh <= 0) {
      setError("Shares must be a number greater than 0.");
      return;
    }

    if (!date || !isYYYYMMDD(date)) {
      setError("Buy date must be in YYYY-MM-DD format.");
      return;
    }

    if (isFutureDate(date)) {
      setError("Buy date cannot be in the future.");
      return;
    }

    // ---------- Backend validation (ticker + next trading day) ----------
    setLoading(true);
    try {
      const result = await validateWithBackend(symbol, date);

      if (!result.valid) {
        setError(
          result.reason ||
            `No price data found for ${symbol} on/after ${date} (within lookahead window).`,
        );
        return;
      }

      if (result.note) setNote(result.note);

      // Creating new holding with backend-validated price and date info
      const newHolding = {
        id: crypto.randomUUID(),
        ticker: symbol,
        shares: sh,

        requestedDate: result.requested_date,
        asOfDate: result.as_of,
        buyPrice: result.price,

        weight: null,
      };

      setHoldings([...holdings, newHolding]);

      // Reset form fields
      setTicker("");
      setShares("");
      setBuyDate(todayYYYYMMDD());
    } catch (err) {
      setError(err.message || "Validation failed.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <form onSubmit={handleSubmit} className="panel-block">
      <h4 style={{ marginTop: 0 }}>Add Holding</h4>

      <div className="add-holding-form">
        <input
          placeholder="Ticker (e.g. AAPL)"
          value={ticker}
          onChange={(e) => setTicker(e.target.value)}
        />

        <input
          type="number"
          placeholder="Shares"
          value={shares}
          onChange={(e) => setShares(e.target.value)}
          min="0"
          step="any"
        />

        <input
          type="date"
          value={buyDate}
          onChange={(e) => setBuyDate(e.target.value)}
        />

        <button
          type="submit"
          disabled={!canSubmit || loading}
          className="secondary-btn"
        >
          {loading ? "Validating..." : "Add Holding"}
        </button>

        {note ? <div className="text-secondary">{note}</div> : null}
        {error ? <div className="text-danger">{error}</div> : null}
      </div>
    </form>
  );
}
