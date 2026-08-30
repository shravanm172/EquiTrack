// Main page for portfolio simulation. Allows users to input holdings, run analysis, stress tests, and forecasts, and view results in charts and metrics.

import { useState } from "react";
import Portfolio from "../components/Portfolio";
import AddHoldingForm from "../components/AddHoldingForm";
import EquityCurveCard from "../components/EquityCurveCard";
import MetricsCard from "../components/MetricsCard";
import ForecastSummaryCard from "../components/ForecastSummaryCard";
import { runPortfolioAnalysis } from "../services/runAnalysis";
import {
  runDeterministicRegimeStress,
  previewCalibratedRegimeStates,
  runCalibratedRegimeStress,
} from "../services/runStressService";
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

  // Stress testing: two shock mechanisms, both forward Monte Carlo
  // simulations from an as-of date, both requiring a prior analysis
  // (analysis_id-based, same precondition as forecasting).
  const [stressResult, setStressResult] = useState(null);
  const [stressLoading, setStressLoading] = useState(false);
  const [regimePreview, setRegimePreview] = useState(null); // calibrated_regime step 1 result
  const [previewLoading, setPreviewLoading] = useState(false);
  const [shock, setShock] = useState({
    type: "deterministic_regime", // "deterministic_regime" | "calibrated_regime"
    as_of_date: "", // YYYY-MM-DD, optional -- blank means end of analysis window
    drift_shift: -0.0005, // deterministic_regime only
    vol_mult: 1.5, // deterministic_regime only
    n_regimes: 3, // calibrated_regime only
    selected_state: null, // calibrated_regime only, set by RegimeStatesTable
  });

  const [forecast, setForecast] = useState(null);
  const [forecastDays, setForecastDays] = useState(30);
  const [forecastType, setForecastType] = useState("deterministic");
  const [model, setModel] = useState("gbm");
  const [driftMode, setDriftMode] = useState("mean");
  const [volMode, setVolMode] = useState("historical");
  const [simulations, setSimulations] = useState(1000);
  const [rollingWindow, setRollingWindow] = useState(60);
  const [ewmaLambda, setEwmaLambda] = useState(0.94);

  function removeHolding(id) {
    // for portfolio
    setHoldings((prev) => prev.filter((h) => h.id !== id));

    // Optional but recommended: clear results
    setAnalysis(null);
    setForecast(null);
    setStressResult(null);
    setRegimePreview(null);
  }

  async function handleRunAnalysis() {
    // For basic analysis
    setError("");
    setLoading(true);
    setAnalysis(null);
    setForecast(null); // Clear previous forecast results
    setStressResult(null);
    setRegimePreview(null);
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

  async function handlePreviewRegimeStates() {
    setError("");

    if (!analysis?.analysis_id) {
      setError("Run analysis first.");
      return;
    }

    setPreviewLoading(true);
    setRegimePreview(null);
    setStressResult(null);
    setShock((s) => ({ ...s, selected_state: null }));

    try {
      const result = await previewCalibratedRegimeStates({
        analysisId: analysis.analysis_id,
        source: "baseline",
        asOfDate: shock.as_of_date || undefined,
        nRegimes: shock.n_regimes,
      });
      setRegimePreview(result);
    } catch (err) {
      setError(err.message || "Regime preview failed.");
    } finally {
      setPreviewLoading(false);
    }
  }

  async function handleRunStress() {
    setError("");

    if (!analysis?.analysis_id) {
      setError("Run analysis first.");
      return;
    }

    setStressResult(null);
    setForecast(null); // only one active result on screen at a time
    setStressLoading(true);

    try {
      let result;

      if (shock.type === "calibrated_regime") {
        if (!regimePreview || shock.selected_state == null) {
          throw new Error("Preview regime states and select one first.");
        }
        result = await runCalibratedRegimeStress({
          regimeId: regimePreview.regime_id,
          selectedState: shock.selected_state,
          days: forecastDays,
          simulations,
        });
      } else {
        result = await runDeterministicRegimeStress({
          analysisId: analysis.analysis_id,
          source: "baseline",
          asOfDate: shock.as_of_date || undefined,
          days: forecastDays,
          simulations,
          driftShift: shock.drift_shift,
          volMult: shock.vol_mult,
        });
      }

      setStressResult(result);
    } catch (err) {
      setError(err.message || "Stress test failed.");
    } finally {
      setStressLoading(false);
    }
  }

  async function handleRunForecast() {
    setError("");

    if (!analysis?.analysis_id) {
      setError("Run analysis first.");
      return;
    }

    setStressResult(null); // only one active result on screen at a time
    setRegimePreview(null);

    try {
      const usesRolling =
        driftMode === "rolling" ||
        (forecastType === "stochastic" && volMode === "rolling");

      const usesEwma =
        driftMode === "ewma" ||
        (forecastType === "stochastic" && volMode === "ewma");

      const result = await runForecast({
        analysisId: analysis.analysis_id,
        source: "baseline",
        type: forecastType,
        model: forecastType === "stochastic" ? model : undefined,
        days: forecastDays,
        driftMode,
        volMode: forecastType === "stochastic" ? volMode : undefined,
        simulations: forecastType === "stochastic" ? simulations : undefined,
        window: usesRolling ? rollingWindow : undefined,
        lambda: usesEwma ? ewmaLambda : undefined,
      });

      setForecast(result);
    } catch (err) {
      setError(err.message || "Forecast failed.");
    }
  }

  useBootstrapTooltip([analysis, stressResult, forecast, regimePreview]); // reinitialize tooltips when results change

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

          {/* Forecast and stress controls both require a prior analysis
              (analysis_id-based). */}
          {analysis && (
            <ForecastControls
              forecastType={forecastType}
              setForecastType={setForecastType}
              forecastDays={forecastDays}
              setForecastDays={setForecastDays}
              model={model}
              setModel={setModel}
              driftMode={driftMode}
              setDriftMode={setDriftMode}
              volMode={volMode}
              setVolMode={setVolMode}
              simulations={simulations}
              setSimulations={setSimulations}
              rollingWindow={rollingWindow}
              setRollingWindow={setRollingWindow}
              ewmaLambda={ewmaLambda}
              setEwmaLambda={setEwmaLambda}
              onRun={handleRunForecast}
              buttonLabel="Run Forecast"
            />
          )}

          {analysis && (
            <StressControls
              shock={shock}
              setShock={setShock}
              startDate={startDate}
              endDate={endDate}
              today={today}
              stressLoading={stressLoading}
              previewLoading={previewLoading}
              regimePreview={regimePreview}
              onPreviewRegimeStates={handlePreviewRegimeStates}
              onRunStress={handleRunStress}
              onError={setError}
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
          {analysis || stressResult ? (
            <EquityCurveCard
              analysis={analysis}
              forecast={forecast}
              stressResult={stressResult}
            />
          ) : (
            <div className="panel-block">Equity curve</div>
          )}

          {/* Analytics Metrics */}
          {analysis || forecast || stressResult || regimePreview ? (
            <AnalyticsPanel
              analysis={analysis}
              forecast={forecast}
              stressResult={stressResult}
              regimePreview={regimePreview}
              selectedState={shock.selected_state}
              onSelectState={(s) =>
                setShock((prev) => ({ ...prev, selected_state: s }))
              }
            />
          ) : (
            <div className="panel-block">Analytics</div>
          )}
        </section>
      </div>
    </div>
  );
}
