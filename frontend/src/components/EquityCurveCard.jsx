import { useMemo } from "react";
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

function mergeCurves(base = [], scen = []) {
  const m = new Map();

  for (const p of base) m.set(p.date, { date: p.date, baseline: p.value });
  for (const p of scen) {
    const row = m.get(p.date) || { date: p.date };
    row.scenario = p.value;
    m.set(p.date, row);
  }

  return Array.from(m.values()).sort((a, b) =>
    String(a.date).localeCompare(String(b.date)),
  );
}

export default function EquityCurveCard({ analysis, stress }) {
  // Mode 1: stress overlay (preferred if present)
  const hasStress =
    stress?.baseline?.equity_curve?.length &&
    stress?.scenario?.equity_curve?.length;

  const hasAnalysis = analysis?.equity_curve?.length;

  const data = useMemo(() => {
    if (hasStress) {
      return mergeCurves(
        stress.baseline.equity_curve,
        stress.scenario.equity_curve,
      );
    }
    if (hasAnalysis) {
      // adapt existing shape {date, value} -> {date, baseline}
      return analysis.equity_curve.map((p) => ({
        date: p.date,
        baseline: p.value,
      }));
    }
    return [];
  }, [hasStress, hasAnalysis, stress, analysis]);

  if (!data.length) return null;

  const shockApplied = stress?.inputs?.shock?.date_applied;
  const asOf = analysis?.holdings_breakdown?.as_of;

  const subtitle = hasStress
    ? shockApplied
      ? `Shock applied on ${shockApplied}`
      : "Baseline vs stress"
    : asOf
      ? `Valued as of ${asOf}`
      : undefined;

  const series = hasStress // <-- here is the series
    ? [
        { key: "baseline", label: "Baseline", color: "#4ea1ff" },
        { key: "scenario", label: "Stressed", color: "#ff6b6b" },
      ]
    : [{ key: "baseline", label: "Equity", color: "#4ea1ff" }];

  return (
    <LineChartCard
      title={hasStress ? "Equity Curve (Baseline vs Stress)" : "Equity Curve"}
      subtitle={subtitle}
      data={data}
      series={series}
      yLabel="Equity"
      yTickFormatter={(v) => (Number(v) / 1000).toFixed(0) + "k"}
      tooltipValueFormatter={(v) => [money(v), "Equity"]}
    />
  );
}
