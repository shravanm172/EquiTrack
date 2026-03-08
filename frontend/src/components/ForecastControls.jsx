export default function ForecastControls({
  forecastType,
  setForecastType,
  forecastDays,
  setForecastDays,
  driftMode,
  setDriftMode,
  volMode,
  setVolMode,
  simulations,
  setSimulations,
  rollingWindow,
  setRollingWindow,
  ewmaLambda,
  setEwmaLambda,
  onRun,
  buttonLabel = "Run Forecast",
}) {
  const usesRolling =
    driftMode === "rolling" ||
    (forecastType === "stochastic" && volMode === "rolling");

  const usesEwma =
    driftMode === "ewma" ||
    (forecastType === "stochastic" && volMode === "ewma");

  return (
    <div className="panel-block forecast-controls">
      <label className="form-label">Forecast type</label>
      <select
        className="form-input"
        value={forecastType}
        onChange={(e) => setForecastType(e.target.value)}
      >
        <option value="deterministic">Deterministic</option>
        <option value="stochastic">Stochastic (Monte Carlo)</option>
      </select>

      <label className="form-label form-label--spaced">Forecast days</label>
      <input
        className="form-input"
        type="number"
        min={1}
        value={forecastDays}
        onChange={(e) => setForecastDays(Number(e.target.value))}
      />

      {forecastType === "stochastic" && (
        <>
          <label className="form-label form-label--spaced">Simulations</label>
          <input
            className="form-input"
            type="number"
            min={1}
            step={100}
            value={simulations}
            onChange={(e) => setSimulations(Number(e.target.value))}
          />
        </>
      )}

      <label className="form-label form-label--spaced">Drift mode</label>
      <select
        className="form-input"
        value={driftMode}
        onChange={(e) => setDriftMode(e.target.value)}
      >
        <option value="mean">Mean (full sample)</option>
        <option value="rolling">Rolling mean</option>
        <option value="ewma">EWMA</option>
      </select>

      {forecastType === "stochastic" && (
        <>
          <label className="form-label form-label--spaced">
            Volatility mode
          </label>
          <select
            className="form-input"
            value={volMode}
            onChange={(e) => setVolMode(e.target.value)}
          >
            <option value="historical">Historical volatility</option>
            <option value="rolling">Rolling volatility</option>
            <option value="ewma">EWMA volatility</option>
          </select>
        </>
      )}

      {usesRolling && (
        <>
          <label className="form-label form-label--spaced">
            Rolling window (days)
          </label>
          <input
            className="form-input"
            type="number"
            min={2}
            step={1}
            value={rollingWindow}
            onChange={(e) => setRollingWindow(Number(e.target.value))}
          />
        </>
      )}

      {usesEwma && (
        <>
          <label className="form-label form-label--spaced">
            EWMA lambda (0–1)
          </label>
          <input
            className="form-input"
            type="number"
            min={0.001}
            max={0.999}
            step={0.01}
            value={ewmaLambda}
            onChange={(e) => setEwmaLambda(Number(e.target.value))}
          />
          <div className="form-help">
            Higher λ = more weight on recent observations. Default 0.94.
          </div>
        </>
      )}

      <button className="secondary-btn" onClick={onRun}>
        {buttonLabel}
      </button>
    </div>
  );
}
