import { useState } from "react";
import Portfolio from "../components/Portfolio";
import AddHoldingForm from "../components/AddHoldingForm";
import EquityCurveCard from "../components/EquityCurveCard";
import MetricsCard from "../components/MetricsCard";
import { runPortfolioAnalysis } from "../services/runAnalysis";
import { analyzeWithStress } from "../services/runStressAnalysis";
import ScenarioDeltasCard from "../components/ScenarioDeltasCard";

function todayYYYYMMDD() {
  return new Date().toISOString().slice(0, 10);
}

export default function SimulatorPage() {
  const [holdings, setHoldings] = useState([]);
  const [startDate, setStartDate] = useState(""); // add this
  const [endDate, setEndDate] = useState("");
  const [analysis, setAnalysis] = useState(null);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const today = todayYYYYMMDD();

  const [stress, setStress] = useState(null);
  const [stressLoading, setStressLoading] = useState(false);
  const [shock, setShock] = useState({
    type: "permanent", // "permanent" | "linear_rebound" | "regime_shift"
    date: "", // YYYY-MM-DD (required)
    pct: -0.2, // e.g. -0.2 = -20%
    rebound_days: 10, // only used for linear_rebound
    vol_mult: 1.5, // only used for regime_shift
    drift_shift: -0.0005, // only used for regime_shift
  });

  async function handleRunAnalysis() {
    // For basic analysis
    setError("");
    setLoading(true);
    try {
      const result = await runPortfolioAnalysis({
        holdings,
        startDate,
        endDate,
      });
      setAnalysis(result);
    } catch (err) {
      setError(err.message || "Analysis failed.");
    } finally {
      setLoading(false);
    }
  }

  async function handleRunStress() {
    // For stress testing
    setError("");
    setStress(null); // Clear previous results
    setStressLoading(true);
    try {
      const result = await analyzeWithStress({
        holdings,
        startDate,
        endDate,
        shock,
      });
      setStress(result);
    } catch (err) {
      setError(err.message || "Stress analysis failed.");
    } finally {
      setStressLoading(false);
    }
  }

  return (
    <div className="simulator-page">
      <div className="simulator-grid">
        {/* Left Panel: Portfolio / Controls */}
        <section className="simulator-panel input-panel">
          <h2>Portfolio</h2>
          <p className="text-muted">Build your portfolio and run analysis.</p>

          {/* Placeholder blocks */}
          <Portfolio holdings={holdings} />
          <AddHoldingForm holdings={holdings} setHoldings={setHoldings} />
          {/* Dates (start pinned) */}

          <label style={{ display: "block", marginBottom: "0.4rem" }}>
            Start date
          </label>
          <div className="panel-block">
            <label style={{ display: "block", marginBottom: "0.4rem" }}>
              Start date
            </label>
            <input
              type="date"
              max={today}
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
            />

            <label
              style={{
                display: "block",
                marginTop: "0.8rem",
                marginBottom: "0.4rem",
              }}
            >
              End date
            </label>
            <input
              type="date"
              min={startDate || undefined}
              max={today}
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
            />
          </div>
          <button onClick={handleRunAnalysis} disabled={loading}>
            {loading ? "Running..." : "Run Analysis"}
          </button>
          {error && <div className="text-danger">{error}</div>}

          {/* Stress test controls */}
          <div className="panel-block">
            <h3 style={{ marginTop: 0 }}>Stress test controls</h3>

            {/* Shock date (always required) */}
            <label style={{ display: "block", marginBottom: "0.4rem" }}>
              Shock date
            </label>
            <input
              type="date"
              min={startDate || undefined}
              max={endDate || today}
              value={shock.date}
              onChange={(e) =>
                setShock((s) => ({ ...s, date: e.target.value }))
              }
            />

            {/* Shock type (always) */}
            <label
              style={{
                display: "block",
                marginTop: "0.8rem",
                marginBottom: "0.4rem",
              }}
            >
              Type
            </label>
            <select
              value={shock.type}
              onChange={(e) =>
                setShock((s) => ({ ...s, type: e.target.value }))
              }
            >
              <option value="permanent">Permanent</option>
              <option value="linear_rebound">Linear rebound</option>
              <option value="regime_shift">Regime shift</option>
            </select>

            {/* Shock % (only for permanent + linear_rebound) */}
            {shock.type !== "regime_shift" && (
              <>
                <label
                  style={{
                    display: "block",
                    marginTop: "0.8rem",
                    marginBottom: "0.4rem",
                  }}
                >
                  Shock %
                </label>
                <input
                  type="number"
                  step="0.01"
                  value={shock.pct}
                  onChange={(e) =>
                    setShock((s) => ({ ...s, pct: Number(e.target.value) }))
                  }
                />
              </>
            )}

            {/* Linear rebound controls */}
            {shock.type === "linear_rebound" && (
              <>
                <label
                  style={{
                    display: "block",
                    marginTop: "0.8rem",
                    marginBottom: "0.4rem",
                  }}
                >
                  Rebound days
                </label>
                <input
                  type="number"
                  min={1}
                  step={1}
                  value={shock.rebound_days}
                  onChange={(e) =>
                    setShock((s) => ({
                      ...s,
                      rebound_days: Number(e.target.value),
                    }))
                  }
                />
              </>
            )}

            {/* Regime shift controls */}
            {shock.type === "regime_shift" && (
              <>
                <label
                  style={{
                    display: "block",
                    marginTop: "0.8rem",
                    marginBottom: "0.4rem",
                  }}
                >
                  Volatility multiplier
                </label>
                <input
                  type="number"
                  min={0.1}
                  step="0.1"
                  value={shock.vol_mult}
                  onChange={(e) =>
                    setShock((s) => ({
                      ...s,
                      vol_mult: Number(e.target.value),
                    }))
                  }
                />

                <label
                  style={{
                    display: "block",
                    marginTop: "0.8rem",
                    marginBottom: "0.4rem",
                  }}
                >
                  Drift shift
                </label>
                <input
                  type="number"
                  step="0.0001"
                  value={shock.drift_shift}
                  onChange={(e) =>
                    setShock((s) => ({
                      ...s,
                      drift_shift: Number(e.target.value),
                    }))
                  }
                />
              </>
            )}

            {/* Run button */}
            <button
              style={{ marginTop: "1rem" }}
              onClick={() => {
                if (!shock.date) {
                  setError("Shock date is required.");
                  return;
                }
                handleRunStress();
              }}
              disabled={stressLoading}
            >
              {stressLoading ? "Running..." : "Run Stress Test"}
            </button>
          </div>
        </section>

        {/* Right Panel: Charts and Metrics */}
        <section className="simulator-panel results-panel">
          <h2>Results</h2>
          <p className="text-muted">
            Equity curve and risk metrics will appear here.
          </p>

          {/* Placeholder blocks */}
          {analysis || stress ? (
            <EquityCurveCard analysis={analysis} stress={stress} />
          ) : (
            <div className="panel-block">Equity curve</div>
          )}
          {analysis ? (
            <MetricsCard metrics={analysis.metrics} />
          ) : (
            <div className="panel-block">Metrics</div>
          )}
          {/* NEW: scenario deltas card */}
          {stress ? (
            <ScenarioDeltasCard stress={stress} />
          ) : (
            <div className="panel-block">Scenario deltas</div>
          )}
        </section>
      </div>
    </div>
  );
}
