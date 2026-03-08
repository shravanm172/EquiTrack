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

function mergeSeries(seriesMap) {
  const m = new Map();

  Object.entries(seriesMap).forEach(([key, arr]) => {
    (arr || []).forEach((p) => {
      const row = m.get(p.date) || { date: p.date };
      row[key] = p.value;
      m.set(p.date, row);
    });
  });

  return Array.from(m.values()).sort((a, b) =>
    String(a.date).localeCompare(String(b.date)),
  );
}

function getForecastType(fc) {
  return fc?.inputs?.forecast?.type || "deterministic";
}

export default function EquityCurveCard({
  analysis,
  stress,
  forecast,
  stressForecast,
}) {
  const hasStress =
    stress?.baseline?.equity_curve?.length &&
    stress?.scenario?.equity_curve?.length;

  const hasAnalysis = analysis?.equity_curve?.length;

  const forecastType = getForecastType(forecast);
  const stressBaselineForecastType = getForecastType(stressForecast?.baseline);
  const stressScenarioForecastType = getForecastType(stressForecast?.scenario);

  const isSingleStochastic =
    !hasStress &&
    forecastType === "stochastic" &&
    forecast?.historical_equity_curve?.length &&
    forecast?.forecast_paths?.p50?.length;

  const isStressStochastic =
    hasStress &&
    stressBaselineForecastType === "stochastic" &&
    stressScenarioForecastType === "stochastic" &&
    stressForecast?.baseline?.forecast_paths?.p50?.length &&
    stressForecast?.scenario?.forecast_paths?.p50?.length;

  const data = useMemo(() => {
    // Stress + stochastic forecast: compare median paths only for now
    if (isStressStochastic) {
      return mergeSeries({
        baseline: stressForecast?.baseline?.historical_equity_curve,
        baseline_p50: stressForecast?.baseline?.forecast_paths?.p50,
        scenario: stressForecast?.scenario?.historical_equity_curve,
        scenario_p50: stressForecast?.scenario?.forecast_paths?.p50,
      });
    }

    // Stress + deterministic (existing behavior)
    if (hasStress) {
      const baseCurve =
        stressForecast?.baseline?.equity_curve || stress.baseline.equity_curve;

      const scenCurve =
        stressForecast?.scenario?.equity_curve || stress.scenario.equity_curve;

      return mergeCurves(baseCurve, scenCurve);
    }

    // Single + stochastic forecast
    if (isSingleStochastic) {
      return mergeSeries({
        historical: forecast?.historical_equity_curve,
        p10: forecast?.forecast_paths?.p10,
        p50: forecast?.forecast_paths?.p50,
        p90: forecast?.forecast_paths?.p90,
      });
    }

    // Single + deterministic (existing behavior)
    if (hasAnalysis) {
      const curve = forecast?.equity_curve || analysis.equity_curve;
      return curve.map((p) => ({
        date: p.date,
        baseline: p.value,
      }));
    }

    return [];
  }, [
    hasStress,
    hasAnalysis,
    isSingleStochastic,
    isStressStochastic,
    stress,
    analysis,
    forecast,
    stressForecast,
  ]);

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

  let title = "Equity Curve";
  let series = [{ key: "baseline", label: "Equity", color: "#4ea1ff" }];

  if (isStressStochastic) {
    title = "Stochastic Forecast (Baseline vs Stress)";
    series = [
      { key: "baseline", label: "Baseline Hist", color: "#4ea1ff" },
      { key: "baseline_p50", label: "Baseline Median", color: "#1f78ff" },
      { key: "scenario", label: "Stress Hist", color: "#ff9a9a" },
      { key: "scenario_p50", label: "Stress Median", color: "#ff6b6b" },
    ];
  } else if (hasStress) {
    title = "Equity Curve (Baseline vs Stress)";
    series = [
      { key: "baseline", label: "Baseline", color: "#4ea1ff" },
      { key: "scenario", label: "Stressed", color: "#ff6b6b" },
    ];
  } else if (isSingleStochastic) {
    title = "Stochastic Forecast";
    series = [
      { key: "historical", label: "Historical", color: "#4ea1ff" },
      { key: "p10", label: "Bear", color: "#ff9a9a" },
      { key: "p50", label: "Median", color: "#1f78ff" },
      { key: "p90", label: "Bull", color: "#7bd88f" },
    ];
  }

  return (
    <LineChartCard
      title={title}
      subtitle={subtitle}
      data={data}
      series={series}
      yLabel="Equity"
      yTickFormatter={(v) => (Number(v) / 1000).toFixed(0) + "k"}
      tooltipValueFormatter={(v) => [money(v), "Equity"]}
    />
  );
}
