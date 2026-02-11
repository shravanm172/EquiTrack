import LineChartCard from "./LineChartCard";

function money(v) {
  const n = Number(v);
  if (!Number.isFinite(n)) return "-";
  return n.toLocaleString(undefined, {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  });
}

export default function EquityCurveCard({ analysis }) {
  const curve = analysis?.equity_curve || [];
  const asOf = analysis?.holdings_breakdown?.as_of;

  return (
    <LineChartCard
      title="Equity Curve"
      subtitle={asOf ? `Valued as of ${asOf}` : undefined}
      data={curve}
      yKey="value"
      yLabel="Equity"
      yTickFormatter={(v) => (Number(v) / 1000).toFixed(0) + "k"}
      tooltipValueFormatter={(v) => [money(v), "Equity"]}
    />
  );
}
