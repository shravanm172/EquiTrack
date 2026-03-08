import React, { useEffect, useMemo, useState } from "react";
import {
  ResponsiveContainer,
  ComposedChart,
  Line,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Brush,
} from "recharts";

function parseISODate(s) {
  const [y, m, d] = String(s).split("-").map(Number);
  return new Date(y, m - 1, d);
}

function defaultXTick(iso) {
  const dt = parseISODate(iso);
  return dt.toLocaleDateString(undefined, { year: "2-digit", month: "short" });
}

function defaultTooltipLabel(iso) {
  const dt = parseISODate(iso);
  return dt.toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "2-digit",
  });
}

function DefaultTooltipContent({
  active,
  label,
  payload,
  labelFormatter,
  valueFormatter,
  series,
}) {
  if (!active || !payload || payload.length === 0) return null;

  const formattedLabel = labelFormatter ? labelFormatter(label) : String(label);
  const allowedKeys = new Set((series || []).map((s) => s.key));

  return (
    <div className="chart-tooltip">
      <div className="chart-tooltip__label">{formattedLabel}</div>

      <div className="chart-tooltip__rows">
        {payload
          .filter((p) => allowedKeys.has(p.dataKey))
          .map((p) => {
            const key = p.dataKey;
            const s = Array.isArray(series)
              ? series.find((it) => it.key === key)
              : null;
            const displayName = s?.label || p.name || key;

            const out = valueFormatter
              ? valueFormatter(p.value, key)
              : [p.value];
            const displayValue = Array.isArray(out) ? out[0] : out;

            return (
              <div className="chart-tooltip__row" key={key}>
                <span className="chart-tooltip__name">{displayName}</span>
                <span className="chart-tooltip__value">{displayValue}</span>
              </div>
            );
          })}
      </div>
    </div>
  );
}

function resample(points, mode, xKey = "date") {
  if (!Array.isArray(points) || points.length === 0) return [];
  if (mode === "daily" || mode === "max") return points;

  const keyFn =
    mode === "monthly"
      ? (p) => String(p[xKey]).slice(0, 7)
      : (p) => {
          const d = parseISODate(p[xKey]);
          const y = d.getFullYear();
          const m = d.getMonth() + 1;
          const wom = Math.floor((d.getDate() - 1) / 7) + 1;
          return `${y}-${String(m).padStart(2, "0")}-W${wom}`;
        };

  const out = [];
  let lastKey = null;
  for (const p of points) {
    const k = keyFn(p);
    if (k !== lastKey) {
      out.push(p);
      lastKey = k;
    } else {
      out[out.length - 1] = p;
    }
  }
  return out;
}

export default function LineChartCard({
  title,
  subtitle,
  data,
  xKey = "date",
  yKey = "value",
  yLabel = "Value",
  series = [],
  bands = [],
  linePropsByKey = {},
  xTickFormatter = defaultXTick,
  tooltipLabelFormatter = defaultTooltipLabel,
  tooltipValueFormatter = (v) => [v, yLabel],
  yTickFormatter = (v) => v,
  defaultView = "daily",
  allowViews = ["daily", "weekly", "monthly", "max"],
  extraControls = null,
}) {
  const [view, setView] = useState(defaultView);
  const [brush, setBrush] = useState({ startIndex: 0, endIndex: 0 });

  const hasSeries = Array.isArray(series) && series.length > 0;
  const hasBands = Array.isArray(bands) && bands.length > 0;

  const chartData = useMemo(() => {
    const pts = Array.isArray(data) ? data : [];
    return view === "max" ? pts : resample(pts, view, xKey);
  }, [data, view, xKey]);

  useEffect(() => {
    const n = chartData.length;
    if (!n) return;
    setBrush({ startIndex: 0, endIndex: n - 1 });
  }, [chartData.length]);

  const chartKey = useMemo(() => {
    if (!chartData.length) return "empty";
    const first = String(chartData[0]?.[xKey] ?? "");
    const last = String(chartData[chartData.length - 1]?.[xKey] ?? "");
    return `${view}__${first}__${last}__${chartData.length}`;
  }, [chartData, xKey, view]);

  return (
    <div className="chart-card">
      <div className="chart-header">
        <div>
          <h3 className="chart-title">{title}</h3>
          {subtitle ? <div className="my-text-muted">{subtitle}</div> : null}
        </div>

        <div
          className="chart-header-right"
          style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}
        >
          {extraControls}
          <div className="chart-toggle">
            {allowViews.map((k) => (
              <button
                key={k}
                type="button"
                className={`toggle-btn ${view === k ? "active" : ""}`}
                onClick={() => setView(k)}
              >
                {k === "max" ? "Max" : k[0].toUpperCase() + k.slice(1)}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="chart-area">
        <ResponsiveContainer width="100%" height={320}>
          <ComposedChart
            key={chartKey}
            data={chartData}
            margin={{ top: 10, right: 20, left: 10, bottom: 0 }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              dataKey={xKey}
              tickFormatter={xTickFormatter}
              minTickGap={30}
            />
            <YAxis tickFormatter={yTickFormatter} />

            <Tooltip
              content={
                <DefaultTooltipContent
                  labelFormatter={tooltipLabelFormatter}
                  valueFormatter={(v, key) => {
                    const out = tooltipValueFormatter(v, key);
                    return [out?.[0] ?? v];
                  }}
                  series={hasSeries ? series : []}
                />
              }
            />

            {hasBands &&
              bands.map((b) =>
                b.visible ? (
                  <Area
                    key={b.key}
                    type="natural"
                    dataKey={b.rangeKey}
                    isRange
                    stroke="none"
                    fill={b.color}
                    fillOpacity={0.22}
                    connectNulls={false}
                    dot={false}
                    activeDot={false}
                    isAnimationActive={false}
                  />
                ) : null,
              )}

            {hasSeries ? (
              series.map((s) => (
                <Line
                  key={s.key}
                  type="monotone"
                  dataKey={s.key}
                  dot={false}
                  strokeWidth={2}
                  stroke={s.color}
                  name={s.label}
                  connectNulls={false}
                  isAnimationActive={false}
                  {...(linePropsByKey?.[s.key] || {})}
                />
              ))
            ) : (
              <Line
                type="monotone"
                dataKey={yKey}
                dot={false}
                strokeWidth={2}
                connectNulls={false}
                isAnimationActive={false}
              />
            )}

            <Brush
              dataKey={xKey}
              startIndex={brush.startIndex}
              endIndex={brush.endIndex}
              onChange={(r) => {
                if (!r) return;
                setBrush({
                  startIndex: r.startIndex ?? 0,
                  endIndex: r.endIndex ?? chartData.length - 1,
                });
              }}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
