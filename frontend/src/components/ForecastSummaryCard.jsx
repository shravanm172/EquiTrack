export default function ForecastSummaryCard({
  title = "Forecast Summary",
  summary,
}) {
  if (!summary) return null;

  const fmtPct = (x) => (x == null ? "—" : `${(x * 100).toFixed(2)}%`);
  const fmtNum = (x) =>
    x == null
      ? "—"
      : Number(x).toLocaleString(undefined, { maximumFractionDigits: 2 });

  return (
    <div className="panel-block">
      <h3 style={{ marginTop: 0 }}>{title}</h3>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: "0.6rem",
        }}
      >
        <div>
          <div className="text-muted">Last historical value</div>
          <div>{fmtNum(summary.last_historical_value)}</div>
        </div>

        <div>
          <div className="text-muted">Forecast end value</div>
          <div>{fmtNum(summary.forecast_end_value)}</div>
        </div>

        <div>
          <div className="text-muted">Abs change</div>
          <div>{fmtNum(summary.forecast_abs_change)}</div>
        </div>

        <div>
          <div className="text-muted">Total return</div>
          <div>{fmtPct(summary.forecast_total_return)}</div>
        </div>

        <div>
          <div className="text-muted">Avg daily return</div>
          <div>{fmtPct(summary.forecast_avg_daily_return)}</div>
        </div>

        <div>
          <div className="text-muted">
            Days to target (×{summary.target_multiple})
          </div>
          <div>{summary.days_to_target_multiple ?? "—"}</div>
        </div>
      </div>

      {summary.trend && (
        <div style={{ marginTop: "0.8rem" }} className="text-muted">
          Trend: {summary.trend.mode}
          {summary.trend.window ? ` (window=${summary.trend.window})` : ""}
          {summary.trend.lambda ? ` (λ=${summary.trend.lambda})` : ""}
          {summary.trend.alpha ? ` (α=${summary.trend.alpha})` : ""}
        </div>
      )}
    </div>
  );
}
