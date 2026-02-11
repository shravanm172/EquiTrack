// src/components/ScenarioDeltasCard.jsx
function fmtPct(x) {
  const n = Number(x);
  if (!Number.isFinite(n)) return "-";
  return `${(n * 100).toFixed(2)}%`;
}

function fmtNum(x) {
  const n = Number(x);
  if (!Number.isFinite(n)) return "-";
  return n.toFixed(2);
}

function fmtDelta(key, x) {
  // Metrics that are naturally percentages
  const pctKeys = new Set([
    "annualized_return",
    "annualized_volatility",
    "max_drawdown",
  ]);

  if (pctKeys.has(key)) return fmtPct(x);
  return fmtNum(x); // sharpe_ratio etc.
}

function labelFor(key) {
  switch (key) {
    case "annualized_return":
      return "Annualized Return";
    case "annualized_volatility":
      return "Annualized Volatility";
    case "max_drawdown":
      return "Maximum Drawdown";
    case "sharpe_ratio":
      return "Sharpe Ratio";
    default:
      return key;
  }
}

export default function ScenarioDeltasCard({ stress }) {
  if (!stress) return null;

  const baselineM = stress?.baseline?.metrics || null;
  const scenarioM = stress?.scenario?.metrics || null;
  const deltaM = stress?.delta?.metrics || null;

  if (!baselineM || !scenarioM || !deltaM) {
    return (
      <div className="panel-block">
        <h3 style={{ marginTop: 0 }}>Scenario Deltas</h3>
        <div className="text-muted">No stress test results to display.</div>
      </div>
    );
  }

  const shockInfo = stress?.inputs?.shock || {};
  const requested = shockInfo?.date_requested;
  const applied = shockInfo?.date_applied;
  const note = shockInfo?.note;

  const type = shockInfo?.type;
  const pct = shockInfo?.pct;
  const reboundDays = shockInfo?.rebound_days;

  // Display order (matches your MetricsCard)
  const rows = [
    "annualized_return",
    "annualized_volatility",
    "max_drawdown",
    "sharpe_ratio",
  ];

  return (
    <div className="panel-block">
      <h3 style={{ marginTop: 0 }}>Scenario Deltas</h3>

      {/* Shock summary */}
      <div style={{ marginBottom: "0.75rem" }}>
        <div className="text-muted">
          <strong>Shock:</strong> {type ? type.replaceAll("_", " ") : "unknown"}
          {typeof pct === "number" ? ` (${(pct * 100).toFixed(1)}%)` : ""}
          {type === "linear_rebound" && reboundDays
            ? ` • rebound ${reboundDays}d`
            : ""}
        </div>

        {requested || applied ? (
          <div className="text-muted">
            <strong>Date:</strong> {requested ? requested : "-"}
            {applied && applied !== requested ? ` → ${applied}` : ""}
          </div>
        ) : null}

        {note ? <div className="text-muted">{note}</div> : null}
      </div>

      {/* Delta table */}
      <div style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse" }}>
          <thead>
            <tr style={{ textAlign: "left" }}>
              <th style={{ padding: "0.35rem 0.4rem" }}>Metric</th>
              <th style={{ padding: "0.35rem 0.4rem" }}>Baseline</th>
              <th style={{ padding: "0.35rem 0.4rem" }}>Scenario</th>
              <th style={{ padding: "0.35rem 0.4rem" }}>Delta</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((k) => {
              const b = baselineM[k];
              const s = scenarioM[k];
              const d = deltaM[k];

              const bFmt = k === "sharpe_ratio" ? fmtNum(b) : fmtPct(b);
              const sFmt = k === "sharpe_ratio" ? fmtNum(s) : fmtPct(s);

              const dFmt = fmtDelta(k, d);

              // Optional: simple visual cue for positive/negative delta
              const dNum = Number(d);
              const deltaStyle =
                Number.isFinite(dNum) && dNum !== 0
                  ? { fontWeight: 600 }
                  : undefined;

              return (
                <tr key={k} style={{ borderTop: "1px solid rgba(0,0,0,0.08)" }}>
                  <td style={{ padding: "0.35rem 0.4rem" }}>{labelFor(k)}</td>
                  <td style={{ padding: "0.35rem 0.4rem" }}>{bFmt}</td>
                  <td style={{ padding: "0.35rem 0.4rem" }}>{sFmt}</td>
                  <td style={{ padding: "0.35rem 0.4rem", ...deltaStyle }}>
                    {dFmt}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Extra: show starting cash / mode if you want */}
      {/* <div className="text-muted" style={{ marginTop: "0.6rem" }}>
        Mode: {stress?.inputs?.mode} • Starting cash: {stress?.inputs?.starting_cash}
      </div> */}
    </div>
  );
}
