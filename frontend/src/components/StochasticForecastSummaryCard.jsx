export default function StochasticForecastSummaryCard({
  title = "Stochastic Forecast Summary",
  terminal,
  drawdown,
  trend,
  volatility,
}) {
  if (!terminal && !drawdown) return null;

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
          <div className="my-text-muted">Mean terminal value</div>
          <div>{fmtNum(terminal?.mean_terminal_value)}</div>
        </div>

        <div>
          <div className="my-text-muted">Median terminal value</div>
          <div>{fmtNum(terminal?.median_terminal_value)}</div>
        </div>

        <div>
          <div className="my-text-muted">Bear case</div>
          <div>{fmtNum(terminal?.bear_case)}</div>
        </div>

        <div>
          <div className="my-text-muted">Bull case</div>
          <div>{fmtNum(terminal?.bull_case)}</div>
        </div>

        <div>
          <div className="my-text-muted">Probability of loss</div>
          <div>{fmtPct(terminal?.probability_of_loss)}</div>
        </div>

        <div>
          <div className="my-text-muted">Median max drawdown</div>
          <div>{fmtPct(drawdown?.median_max_drawdown)}</div>
        </div>

        <div>
          <div className="my-text-muted">Prob. drawdown &gt; 20%</div>
          <div>{fmtPct(drawdown?.prob_drawdown_gt_20)}</div>
        </div>

        <div>
          <div className="my-text-muted">Annualized volatility</div>
          <div>{fmtPct(volatility?.annualized_volatility)}</div>
        </div>
      </div>

      {(trend || volatility) && (
        <div style={{ marginTop: "0.8rem" }} className="my-text-muted">
          {trend && (
            <>
              Drift: {trend.mode}
              {trend.window ? ` (window=${trend.window})` : ""}
              {trend.lambda ? ` (λ=${trend.lambda})` : ""}
              {trend.alpha ? ` (α=${trend.alpha})` : ""}
            </>
          )}

          {trend && volatility ? " • " : ""}

          {volatility && (
            <>
              Volatility: {volatility.mode}
              {volatility.window ? ` (window=${volatility.window})` : ""}
              {volatility.lambda ? ` (λ=${volatility.lambda})` : ""}
              {volatility.alpha ? ` (α=${volatility.alpha})` : ""}
            </>
          )}
        </div>
      )}
    </div>
  );
}
