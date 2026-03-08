import { useEffect, useMemo, useState } from "react";
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

  for (const p of base) {
    m.set(p.date, { date: p.date, baseline: p.value });
  }

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

function mergeSeriesWithBands(seriesMap, bandMap) {
  const m = new Map();

  Object.entries(seriesMap).forEach(([key, arr]) => {
    (arr || []).forEach((p) => {
      const row = m.get(p.date) || { date: p.date };
      row[key] = p.value;
      m.set(p.date, row);
    });
  });

  Object.entries(bandMap).forEach(([rangeKey, cfg]) => {
    const lowerByDate = new Map(
      (cfg.lower || []).map((p) => [p.date, p.value]),
    );
    const upperByDate = new Map(
      (cfg.upper || []).map((p) => [p.date, p.value]),
    );

    const allDates = new Set([...lowerByDate.keys(), ...upperByDate.keys()]);

    allDates.forEach((date) => {
      const lower = lowerByDate.get(date);
      const upper = upperByDate.get(date);
      const row = m.get(date) || { date };

      if (Number.isFinite(lower) && Number.isFinite(upper) && upper >= lower) {
        row[rangeKey] = [lower, upper];
      }

      m.set(date, row);
    });
  });

  return Array.from(m.values()).sort((a, b) =>
    String(a.date).localeCompare(String(b.date)),
  );
}

function getForecastType(fc) {
  return fc?.inputs?.forecast?.type || "deterministic";
}

function BandToggles({
  showBaselineBand,
  setShowBaselineBand,
  showScenarioBand,
  setShowScenarioBand,
  hasScenario,
}) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
      <label
        style={{
          display: "flex",
          alignItems: "center",
          gap: "0.35rem",
          fontSize: "0.9rem",
        }}
      >
        <input
          type="checkbox"
          checked={showBaselineBand}
          onChange={(e) => setShowBaselineBand(e.target.checked)}
        />
        Baseline band
      </label>

      {hasScenario && (
        <label
          style={{
            display: "flex",
            alignItems: "center",
            gap: "0.35rem",
            fontSize: "0.9rem",
          }}
        >
          <input
            type="checkbox"
            checked={showScenarioBand}
            onChange={(e) => setShowScenarioBand(e.target.checked)}
          />
          Scenario band
        </label>
      )}
    </div>
  );
}

export default function EquityCurveCard({
  analysis,
  stress,
  forecast,
  stressForecast,
}) {
  const [showBaselineBand, setShowBaselineBand] = useState(true);
  const [showScenarioBand, setShowScenarioBand] = useState(true);

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

  useEffect(() => {
    if (isStressStochastic) {
      setShowBaselineBand(false);
      setShowScenarioBand(true);
    } else if (isSingleStochastic) {
      setShowBaselineBand(true);
      setShowScenarioBand(true);
    }
  }, [isSingleStochastic, isStressStochastic]);

  const data = useMemo(() => {
    if (isStressStochastic) {
      return mergeSeriesWithBands(
        {
          baseline: stressForecast?.baseline?.historical_equity_curve,
          baseline_p50: stressForecast?.baseline?.forecast_paths?.p50,
          scenario: stressForecast?.scenario?.historical_equity_curve,
          scenario_p50: stressForecast?.scenario?.forecast_paths?.p50,
        },
        {
          baseline_band: {
            lower: stressForecast?.baseline?.forecast_paths?.p10,
            upper: stressForecast?.baseline?.forecast_paths?.p90,
          },
          scenario_band: {
            lower: stressForecast?.scenario?.forecast_paths?.p10,
            upper: stressForecast?.scenario?.forecast_paths?.p90,
          },
        },
      );
    }

    if (hasStress) {
      const baseCurve =
        stressForecast?.baseline?.equity_curve ||
        stress?.baseline?.equity_curve;

      const scenCurve =
        stressForecast?.scenario?.equity_curve ||
        stress?.scenario?.equity_curve;

      return mergeCurves(baseCurve, scenCurve);
    }

    if (isSingleStochastic) {
      return mergeSeriesWithBands(
        {
          historical: forecast?.historical_equity_curve,
          p50: forecast?.forecast_paths?.p50,
        },
        {
          band: {
            lower: forecast?.forecast_paths?.p10,
            upper: forecast?.forecast_paths?.p90,
          },
        },
      );
    }

    if (hasAnalysis) {
      const curve = forecast?.equity_curve || analysis?.equity_curve || [];
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
    analysis,
    stress,
    forecast,
    stressForecast,
  ]);

  if (!data.length) return null;

  const shockApplied = stress?.inputs?.shock?.date_applied;
  const asOf =
    analysis?.holdings_breakdown?.as_of ||
    forecast?.holdings_breakdown?.as_of ||
    stress?.baseline?.holdings_breakdown?.as_of;

  const subtitle = hasStress
    ? shockApplied
      ? `Shock applied on ${shockApplied}`
      : "Baseline vs stress"
    : asOf
      ? `Valued as of ${asOf}`
      : undefined;

  let title = "Equity Curve";
  let series = [{ key: "baseline", label: "Equity", color: "#4ea1ff" }];
  let bands = [];
  let extraControls = null;

  if (isStressStochastic) {
    title = "Stochastic Forecast (Baseline vs Stress)";
    series = [
      { key: "baseline", label: "Baseline Hist", color: "#4ea1ff" },
      { key: "baseline_p50", label: "Baseline Median", color: "#1f78ff" },
      { key: "scenario", label: "Stress Hist", color: "#ff9a9a" },
      { key: "scenario_p50", label: "Stress Median", color: "#ff6b6b" },
    ];

    bands = [
      {
        key: "baselineBand",
        rangeKey: "baseline_band",
        color: "#4ea1ff",
        visible: showBaselineBand,
      },
      {
        key: "scenarioBand",
        rangeKey: "scenario_band",
        color: "#ff6b6b",
        visible: showScenarioBand,
      },
    ];

    extraControls = (
      <BandToggles
        showBaselineBand={showBaselineBand}
        setShowBaselineBand={setShowBaselineBand}
        showScenarioBand={showScenarioBand}
        setShowScenarioBand={setShowScenarioBand}
        hasScenario={true}
      />
    );
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
      { key: "p50", label: "Median", color: "#1f78ff" },
    ];

    bands = [
      {
        key: "baselineBand",
        rangeKey: "band",
        color: "#4ea1ff",
        visible: showBaselineBand,
      },
    ];

    extraControls = (
      <BandToggles
        showBaselineBand={showBaselineBand}
        setShowBaselineBand={setShowBaselineBand}
        showScenarioBand={showScenarioBand}
        setShowScenarioBand={setShowScenarioBand}
        hasScenario={false}
      />
    );
  }

  return (
    <LineChartCard
      title={title}
      subtitle={subtitle}
      data={data}
      series={series}
      bands={bands}
      extraControls={extraControls}
      yLabel="Equity"
      yTickFormatter={(v) => (Number(v) / 1000).toFixed(0) + "k"}
      tooltipValueFormatter={(v) => [money(v), "Equity"]}
    />
  );
}
