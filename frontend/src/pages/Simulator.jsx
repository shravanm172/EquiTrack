import { useState } from "react";
import Portfolio from "../components/Portfolio";
import AddHoldingForm from "../components/AddHoldingForm";
import EquityCurveCard from "../components/EquityCurveCard";
import MetricsCard from "../components/MetricsCard";
import { runPortfolioAnalysis } from "../services/runAnalysis";

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

  async function handleRunAnalysis() {
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

          <div className="panel-block">Stress test controls</div>
        </section>

        {/* Right Panel: Charts and Metrics */}
        <section className="simulator-panel results-panel">
          <h2>Results</h2>
          <p className="text-muted">
            Equity curve and risk metrics will appear here.
          </p>

          {/* Placeholder blocks */}
          {analysis ? (
            <EquityCurveCard analysis={analysis} />
          ) : (
            <div className="panel-block">Equity curve</div>
          )}
          {analysis ? (
            <MetricsCard metrics={analysis.metrics} />
          ) : (
            <div className="panel-block">Metrics</div>
          )}
          <div className="panel-block">Scenario deltas</div>
        </section>
      </div>
    </div>
  );
}
