// Main page for portfolio simulation. Allows users to input holdings, run analysis, stress tests, and forecasts, and view results in charts and metrics.

import { useState } from "react";
import Portfolio from "../components/Portfolio";
import AddHoldingForm from "../components/AddHoldingForm";
import EquityCurveCard from "../components/EquityCurveCard";
import MetricsCard from "../components/MetricsCard";
import ForecastSummaryCard from "../components/ForecastSummaryCard";
import { runPortfolioAnalysis } from "../services/runAnalysis";
import { analyzeWithStress } from "../services/runStressAnalysis";
import { runForecast } from "../services/runForecast";
import StressControls from "../components/StressControls";
import ForecastControls from "../components/ForecastControls";
import AnalyticsPanel from "../components/AnalyticsPanel";
import useBootstrapTooltip from "../hooks/useBootstrapTooltip";

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

  const [forecast, setForecast] = useState(null);
  const [stressForecast, setStressForecast] = useState(null);
  const [forecastDays, setForecastDays] = useState(30);
  const [forecastMode, setForecastMode] = useState("mean"); // "mean" | "rolling" | "ewma"
  const [rollingWindow, setRollingWindow] = useState(60);
  const [ewmaLambda, setEwmaLambda] = useState(0.94);

  function removeHolding(id) {
    // for portfolio
    setHoldings((prev) => prev.filter((h) => h.id !== id));

    // Optional but recommended: clear results
    setAnalysis(null);
    setStress(null);
    setForecast(null);
    setStressForecast(null);
  }

  async function handleRunAnalysis() {
    // For basic analysis
    setError("");
    setLoading(true);
    setAnalysis(null);
    setForecast(null); // Clear previous forecast results
    setStress(null);
    setStressForecast(null);
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
    setStressForecast(null);
    setForecast(null); // Clear previous forecast results
    setAnalysis(null);
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

  async function handleRunForecast() {
    setError("");
    setStressForecast(null);
    if (!analysis?.analysis_id) {
      setError("Run analysis first.");
      return;
    }
    try {
      const result = await runForecast({
        analysisId: analysis.analysis_id,
        days: forecastDays,
        mode: forecastMode,
        window: forecastMode === "rolling" ? rollingWindow : undefined,
        lambda: forecastMode === "ewma" ? ewmaLambda : undefined,
      });
      setForecast(result);
      console.log("analysis curve", analysis.equity_curve?.length);
      console.log("forecast curve", result.equity_curve?.length);
      console.log("forecast hist", result.historical_equity_curve?.length);
      console.log("forecast fc", result.forecast_equity_curve?.length);
    } catch (err) {
      setError(err.message || "Forecast failed.");
    }
  }

  async function handleRunStressForecast() {
    setError("");
    setForecast(null);

    try {
      // IMPORTANT: use the stress analysis_id (the one stored in analysis_store for kind="analyze_shock")
      const stressId = stress?.analysis_id || stress?.inputs?.analysis_id;
      if (!stressId)
        throw new Error("Stress analysis_id missing. Re-run stress test.");

      const [baselineFc, scenarioFc] = await Promise.all([
        runForecast({
          analysisId: stressId,
          days: forecastDays,
          source: "baseline",
          mode: forecastMode,
          window: forecastMode === "rolling" ? rollingWindow : undefined,
          lambda: forecastMode === "ewma" ? ewmaLambda : undefined,
        }),
        runForecast({
          analysisId: stressId,
          days: forecastDays,
          source: "scenario",
          mode: forecastMode,
          window: forecastMode === "rolling" ? rollingWindow : undefined,
          lambda: forecastMode === "ewma" ? ewmaLambda : undefined,
        }),
      ]);

      setStressForecast({ baseline: baselineFc, scenario: scenarioFc });
    } catch (err) {
      setError(err.message || "Stress forecast failed.");
    }
  }

  useBootstrapTooltip([analysis, stress, forecast, stressForecast]); // reinitialize tooltips when results change

  return (
    <div className="simulator-page">
      <div className="simulator-grid">
        {/* Left Panel: Portfolio / Controls */}
        <section className="simulator-panel input-panel">
          <h2>Portfolio</h2>
          <Portfolio holdings={holdings} onRemove={removeHolding} />
          <AddHoldingForm holdings={holdings} setHoldings={setHoldings} />

          <label style={{ display: "block", marginBottom: "0.4rem" }}>
            Portfolio Analysis
          </label>
          <div className="panel-block">
            <label style={{ display: "block", marginBottom: "0.4rem" }}>
              Analysis Window
            </label>
            <div className="date-inputs">
              <label style={{ display: "block", marginBottom: "0.4rem" }}>
                Start
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
                End{" "}
              </label>
              <input
                type="date"
                min={startDate || undefined}
                max={today}
                value={endDate}
                onChange={(e) => setEndDate(e.target.value)}
              />
            </div>
            <button
              onClick={handleRunAnalysis}
              disabled={loading}
              className="primary-btn"
            >
              {loading ? "Running..." : "Analyze Portfolio"}
            </button>
            <span
              className="my-text-muted"
              style={{ fontSize: "1rem", marginLeft: "1rem" }}
            >
              Run baseline analysis on portfolio
            </span>
            {error && <div className="text-danger">{error}</div>}
          </div>

          {/* Render forecast controls only if we have analysis results (need analysis_id) */}
          {analysis && (
            <ForecastControls
              forecastDays={forecastDays}
              setForecastDays={setForecastDays}
              forecastMode={forecastMode}
              setForecastMode={setForecastMode}
              rollingWindow={rollingWindow}
              setRollingWindow={setRollingWindow}
              ewmaLambda={ewmaLambda}
              setEwmaLambda={setEwmaLambda}
              onRun={handleRunForecast}
              buttonLabel="Run Forecast"
            />
          )}

          {/* Stress test controls */}
          <StressControls
            shock={shock}
            setShock={setShock}
            startDate={startDate}
            endDate={endDate}
            today={today}
            stressLoading={stressLoading}
            onRunStress={handleRunStress}
            onError={setError}
          />

          {stress && (
            <ForecastControls
              forecastDays={forecastDays}
              setForecastDays={setForecastDays}
              forecastMode={forecastMode}
              setForecastMode={setForecastMode}
              rollingWindow={rollingWindow}
              setRollingWindow={setRollingWindow}
              ewmaLambda={ewmaLambda}
              setEwmaLambda={setEwmaLambda}
              onRun={handleRunStressForecast}
              buttonLabel="Run Forecast"
            />
          )}
        </section>

        {/* ------------Right Panel: Charts and Metrics--------------- */}
        <section className="simulator-panel results-panel">
          <h2>Results</h2>
          <p className="my-text-muted">
            Add holdings and run analysis to see equity curve and metrics here.
          </p>

          {/* Equity Curve */}
          {analysis || stress ? (
            <EquityCurveCard
              analysis={analysis}
              stress={stress}
              forecast={forecast}
              stressForecast={stressForecast}
            />
          ) : (
            <div className="panel-block">Equity curve</div>
          )}

          {/* Analytics Metrics */}
          {analysis || stress || forecast || stressForecast ? (
            <AnalyticsPanel
              analysis={analysis}
              stress={stress}
              forecast={forecast}
              stressForecast={stressForecast}
            />
          ) : (
            <div className="panel-block">Analytics</div>
          )}
        </section>
      </div>
    </div>
  );
}
