// Inputs for stress testing. Two shock mechanisms, both forward Monte Carlo
// simulations from a chosen as-of date (bounded by the analysis window):
//   - deterministic_regime: one call, drift_shift/vol_mult applied directly.
//   - calibrated_regime: two calls -- "Preview States" fits the GMM and
//     hands the fitted states to AnalyticsPanel's RegimeStatesTable for the
//     user to pick a state from (shock.selected_state), THEN "Run Stress
//     Test" simulates forced-start from that state. The selection UI lives
//     in the results panel (RegimeStatesTable), not here -- this component
//     only reads shock.selected_state to know whether step 2 is unlocked.
export default function StressControls({
  shock,
  setShock,
  startDate,
  endDate,
  today,
  stressLoading,
  previewLoading,
  regimePreview,
  onPreviewRegimeStates,
  onRunStress,
  onError,
}) {
  const maxAsOfDate = endDate || today;
  const isCalibrated = shock.type === "calibrated_regime";

  const canRunCalibrated = !!regimePreview && shock.selected_state != null;

  return (
    <div className="panel-block stress-controls">
      <h3 className="panel-block__title">Stress Testing</h3>

      <label className="form-label">As-of date</label>
      <input
        className="form-input"
        type="date"
        min={startDate || undefined}
        max={maxAsOfDate}
        value={shock.as_of_date}
        onChange={(e) =>
          setShock((s) => ({ ...s, as_of_date: e.target.value }))
        }
      />
      <p className="my-text-muted" style={{ fontSize: "1.2rem" }}>
        Leave blank to simulate forward from the end of the analysis window.
        Pick an earlier date to also see what was actually realized, as a
        baseline.
      </p>

      <label className="form-label form-label--spaced">Type</label>
      <select
        value={shock.type}
        onChange={(e) =>
          setShock((s) => ({
            ...s,
            type: e.target.value,
            selected_state: null, // switching mechanisms invalidates any prior selection
          }))
        }
      >
        <option value="deterministic_regime">
          Regime shift (deterministic)
        </option>
        <option value="calibrated_regime">Regime-switching (calibrated)</option>
      </select>

      {!isCalibrated && (
        <>
          <label className="form-label form-label--spaced">
            Volatility multiplier
          </label>
          <input
            className="form-input"
            type="number"
            min={0.1}
            step="0.1"
            value={shock.vol_mult}
            onChange={(e) =>
              setShock((s) => ({ ...s, vol_mult: Number(e.target.value) }))
            }
          />

          <label className="form-label form-label--spaced">Drift shift</label>
          <input
            className="form-input"
            type="number"
            step="0.0001"
            value={shock.drift_shift}
            onChange={(e) =>
              setShock((s) => ({ ...s, drift_shift: Number(e.target.value) }))
            }
          />
        </>
      )}

      {isCalibrated && (
        <>
          <label className="form-label form-label--spaced">
            Number of regimes
          </label>
          <input
            className="form-input"
            type="number"
            min={2}
            max={8}
            step={1}
            value={shock.n_regimes}
            onChange={(e) =>
              setShock((s) => ({
                ...s,
                n_regimes: Number(e.target.value),
                selected_state: null, // stale once n_regimes changes
              }))
            }
          />

          <button
            className="secondary-btn"
            onClick={onPreviewRegimeStates}
            disabled={previewLoading}
          >
            {previewLoading ? "Fitting..." : "Preview Regime States"}
          </button>

          <p className="my-text-muted" style={{ fontSize: "1.2rem" }}>
            {!regimePreview
              ? "Preview the fitted regime states, then select one below to stress test from."
              : shock.selected_state == null
                ? "Select a state in the Regime States table to run the stress test."
                : `Selected state: ${shock.selected_state}`}
          </p>
        </>
      )}

      <button
        className="secondary-btn"
        onClick={() => {
          if (isCalibrated && !canRunCalibrated) {
            onError?.("Preview regime states and select one first.");
            return;
          }
          onRunStress();
        }}
        disabled={stressLoading || (isCalibrated && !canRunCalibrated)}
      >
        {stressLoading ? "Running..." : "Run Stress Test"}
      </button>
    </div>
  );
}
