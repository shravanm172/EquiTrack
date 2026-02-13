// src/components/StressControls.jsx

export default function StressControls({
  shock,
  setShock,
  startDate,
  endDate,
  today,
  stressLoading,
  onRunStress,
  onError,
}) {
  const maxShockDate = endDate || today;

  return (
    <div className="panel-block stress-controls">
      <h3 className="panel-block__title">Stress Testing</h3>

      <label className="form-label">Shock date</label>
      <input
        className="form-input"
        type="date"
        min={startDate || undefined}
        max={maxShockDate}
        value={shock.date}
        onChange={(e) => setShock((s) => ({ ...s, date: e.target.value }))}
      />

      <label className="form-label form-label--spaced">Type</label>
      <select
        className="form-select"
        value={shock.type}
        onChange={(e) => setShock((s) => ({ ...s, type: e.target.value }))}
      >
        <option value="permanent">Permanent</option>
        <option value="linear_rebound">Linear rebound</option>
        <option value="regime_shift">Regime shift</option>
      </select>

      {shock.type !== "regime_shift" && (
        <>
          <label className="form-label form-label--spaced">Shock %</label>
          <input
            className="form-input"
            type="number"
            step="0.01"
            value={shock.pct}
            onChange={(e) =>
              setShock((s) => ({ ...s, pct: Number(e.target.value) }))
            }
          />
        </>
      )}

      {shock.type === "linear_rebound" && (
        <>
          <label className="form-label form-label--spaced">Rebound days</label>
          <input
            className="form-input"
            type="number"
            min={1}
            step={1}
            value={shock.rebound_days}
            onChange={(e) =>
              setShock((s) => ({ ...s, rebound_days: Number(e.target.value) }))
            }
          />
        </>
      )}

      {shock.type === "regime_shift" && (
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
              setShock((s) => ({
                ...s,
                drift_shift: Number(e.target.value),
              }))
            }
          />
        </>
      )}

      <button
        className="secondary-btn"
        onClick={() => {
          if (!shock.date) {
            onError?.("Shock date is required.");
            return;
          }
          onRunStress();
        }}
        disabled={stressLoading}
      >
        {stressLoading ? "Running..." : "Run Stress Test"}
      </button>
    </div>
  );
}
