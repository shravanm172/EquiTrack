// Displays the fitted regime states from a calibrated_regime preview call
// (see services/runStressService.js -> previewCalibratedRegimeStates).
// No auto-labeling (calm/crisis/etc, deliberately, matching the backend) --
// just each state's raw numbers, plus a way to pick one to stress test.
//
// Meant to be rendered as the children of an AnalyticsCard in
// AnalyticsPanel, e.g.:
//   <AnalyticsCard title="Regime States (fitted)">
//     <RegimeStatesTable regimeStats={...} selectedState={...} onSelectState={...} />
//   </AnalyticsCard>
import { formatPct } from "../lib/formatters";

export default function RegimeStatesTable({
  regimeStats,
  selectedState,
  onSelectState,
}) {
  if (!regimeStats || !regimeStats.length) return null;

  return (
    <div className="regime-states-table-wrap">
      <table className="holdings-table regime-states-table">
        <thead>
          <tr>
            <th></th>
            <th>State</th>
            <th>Annualized Return</th>
            <th>Annualized Volatility</th>
            <th>Daily Mean</th>
            <th>Daily Volatility</th>
          </tr>
        </thead>
        <tbody>
          {regimeStats.map((row) => {
            const isSelected = selectedState === row.state;
            return (
              <tr
                key={row.state}
                className={
                  "regime-states-table__row" +
                  (isSelected ? " regime-states-table__row--selected" : "")
                }
                onClick={() => onSelectState?.(row.state)}
              >
                <td onClick={(e) => e.stopPropagation()}>
                  <input
                    type="radio"
                    name="regime-selected-state"
                    checked={isSelected}
                    onChange={() => onSelectState?.(row.state)}
                  />
                </td>
                <td>State {row.state}</td>
                <td>{formatPct(row.annualized_return)}</td>
                <td>{formatPct(row.annualized_volatility)}</td>
                <td>{formatPct(row.mean_daily_return, 3)}</td>
                <td>{formatPct(row.daily_volatility, 3)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="regime-states-table__hint">
        Click a row to select the initial state to stress test from.
      </p>
    </div>
  );
}
