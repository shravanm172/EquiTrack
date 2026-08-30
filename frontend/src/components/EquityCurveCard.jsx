import { useMemo, useState } from "react";
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

function getShockType(sr) {
  return sr?.inputs?.shock?.type;
}

function shockTitle(shockType) {
  if (shockType === "calibrated_regime") {
    return "Calibrated Regime-Switching Stress Test";
  }
  if (shockType === "deterministic_regime") {
    return "Deterministic Regime-Shift Stress Test";
  }
  return "Stress Test";
}

// Single checkbox now -- there's only ever one simulated distribution (one
// p10-p90 fan) on screen at a time, unlike the old paired baseline/scenario
// comparison which had two.
function BandToggle({ showBand, setShowBand }) {
  return (
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
        checked={showBand}
        onChange={(e) => setShowBand(e.target.checked)}
      />
      Uncertainty Fan
    </label>
  );
}

export default function EquityCurveCard({ analysis, forecast, stressResult }) {
  const [showBand, setShowBand] = useState(true);

  const hasAnalysis = analysis?.equity_curve?.length;

  const forecastType = getForecastType(forecast);

  const isSingleStochastic =
    forecastType === "stochastic" &&
    forecast?.historical_equity_curve?.length &&
    forecast?.forecast_paths?.p50?.length;

  const isRegimeStress =
    stressResult?.historical_equity_curve?.length &&
    stressResult?.forecast_paths?.p50?.length;

  // Reset the band toggle to visible whenever a genuinely new result comes
  // in (object identity, so re-running with identical params still counts),
  // while still letting the user manually hide it for the result on screen.
  // Adjusted during render rather than in a useEffect -- React's documented
  // pattern for "reset state when derived data changes" -- so there's no
  // extra render-then-setState-then-rerender cascade.
  const activeResult = isRegimeStress
    ? stressResult
    : isSingleStochastic
      ? forecast
      : null;

  const [prevActiveResult, setPrevActiveResult] = useState(null);
  if (activeResult !== prevActiveResult) {
    setPrevActiveResult(activeResult);
    if (activeResult != null) {
      setShowBand(true);
    }
  }

  const hasActualRealized = !!stressResult?.actual_realized_curve?.length;

  const data = useMemo(() => {
    if (isRegimeStress) {
      return mergeSeriesWithBands(
        {
          historical: stressResult?.historical_equity_curve,
          actual: stressResult?.actual_realized_curve,
          p50: stressResult?.forecast_paths?.p50,
        },
        {
          band: {
            lower: stressResult?.forecast_paths?.p10,
            upper: stressResult?.forecast_paths?.p90,
          },
        },
      );
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
    hasAnalysis,
    isSingleStochastic,
    isRegimeStress,
    analysis,
    forecast,
    stressResult,
  ]);

  if (!data.length) return null;

  const asOf =
    analysis?.holdings_breakdown?.as_of || forecast?.holdings_breakdown?.as_of;

  let title = "Equity Curve";
  let subtitle = asOf ? `Valued as of ${asOf}` : undefined;
  let series = [{ key: "baseline", label: "Equity", color: "#4ea1ff" }];
  let bands = [];
  let extraControls = null;
  let linePropsByKey = {};

  if (isRegimeStress) {
    const shockType = getShockType(stressResult);
    const asOfApplied = stressResult?.inputs?.as_of_date?.applied;
    const asOfNote = stressResult?.inputs?.as_of_date?.note;

    title = shockTitle(shockType);
    subtitle = asOfApplied
      ? `As of ${asOfApplied}${asOfNote ? ` — ${asOfNote}` : ""}`
      : undefined;

    series = [
      { key: "historical", label: "Historical", color: "#4ea1ff" },
      ...(hasActualRealized
        ? [{ key: "actual", label: "Actual (realized)", color: "#8f8f8f" }]
        : []),
      { key: "p50", label: "Median (simulated)", color: "#ff6b6b" },
    ];

    bands = [
      { key: "band", rangeKey: "band", color: "#ff6b6b", visible: showBand },
    ];

    // Dashed so "what actually happened" reads visually distinct from the
    // solid historical line and the solid simulated median.
    if (hasActualRealized) {
      linePropsByKey = { actual: { strokeDasharray: "5 3" } };
    }

    extraControls = (
      <BandToggle showBand={showBand} setShowBand={setShowBand} />
    );
  } else if (isSingleStochastic) {
    title = "Stochastic Forecast";
    series = [
      { key: "historical", label: "Historical", color: "#4ea1ff" },
      { key: "p50", label: "Median", color: "#1f78ff" },
    ];

    bands = [
      { key: "band", rangeKey: "band", color: "#4ea1ff", visible: showBand },
    ];

    extraControls = (
      <BandToggle showBand={showBand} setShowBand={setShowBand} />
    );
  }

  return (
    <LineChartCard
      title={title}
      subtitle={subtitle}
      data={data}
      series={series}
      bands={bands}
      linePropsByKey={linePropsByKey}
      extraControls={extraControls}
      yLabel="Equity"
      yTickFormatter={(v) => (Number(v) / 1000).toFixed(0) + "k"}
      tooltipValueFormatter={(v) => [money(v), "Equity"]}
    />
  );
}
