export default function ForecastControls({
  forecastDays,
  setForecastDays,
  forecastMode,
  setForecastMode,
  rollingWindow,
  setRollingWindow,
  ewmaLambda,
  setEwmaLambda,
  onRun,
  buttonLabel = "Run Forecast",
}) {
  return (
    <div className="panel-block forecast-controls">
      <label className="form-label">Forecast days</label>
      <input
        className="form-input"
        type="number"
        min={1}
        value={forecastDays}
        onChange={(e) => setForecastDays(Number(e.target.value))}
      />

      <label className="form-label form-label--spaced">Mode</label>
      <select
        value={forecastMode}
        onChange={(e) => setForecastMode(e.target.value)}
      >
        <option value="mean">Mean (full sample)</option>
        <option value="rolling">Rolling mean</option>
        <option value="ewma">EWMA</option>
      </select>

      {forecastMode === "rolling" && (
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

      {forecastMode === "ewma" && (
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
            Higher λ = more weight on recent returns. Default 0.94.
          </div>
        </>
      )}

      <button className="secondary-btn" onClick={onRun}>
        {buttonLabel}
      </button>
    </div>
  );
}
