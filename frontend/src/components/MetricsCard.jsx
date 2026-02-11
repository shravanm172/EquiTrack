// components/MetricsCard.jsx
export default function MetricsCard({ metrics }) {
  if (!metrics) return null;

  const fmtPct = (x) => `${(x * 100).toFixed(2)}%`;
  const fmtNum = (x) => Number(x).toFixed(2);

  return (
    <div className="panel-block">
      <h3 style={{ marginTop: 0 }}>Risk Metrics</h3>

      <div>
        <strong>Annualized Return:</strong> {fmtPct(metrics.annualized_return)}
      </div>

      <div>
        <strong>Annualized Volatility:</strong>{" "}
        {fmtPct(metrics.annualized_volatility)}
      </div>

      <div>
        <strong>Maximum Drawdown:</strong> {fmtPct(metrics.max_drawdown)}
      </div>

      <div>
        <strong>Sharpe Ratio:</strong> {fmtNum(metrics.sharpe_ratio)}
      </div>
    </div>
  );
}
