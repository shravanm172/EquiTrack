// Base Template for Line Charts (supports 1+ series)

import { useEffect, useMemo, useState } from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
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

// Keep last point per bucket
function resample(points, mode, xKey = "date") {
  if (!Array.isArray(points) || points.length === 0) return [];
  if (mode === "daily" || mode === "max") return points;

  const keyFn =
    mode === "monthly"
      ? (p) => String(p[xKey]).slice(0, 7) // YYYY-MM
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
      out[out.length - 1] = p; // keep last point in bucket
    }
  }
  return out;
}

/**
 * Props:
 * - data: array of points
 * - xKey: key for x axis (default "date")
 *
 * Single-series (legacy):
 * - yKey: key for y values (default "value")
 * - yLabel: label for tooltip (default "Value")
 *
 * Multi-series (new):
 * - series: [{ key: "baseline", label: "Baseline" }, ...]
 */
export default function LineChartCard({
  title,
  subtitle,
  data,
  xKey = "date",

  // legacy single-series support
  yKey = "value",
  yLabel = "Value",

  // multi-series support
  series = [],
  linePropsByKey = {},

  // formatting hooks
  xTickFormatter = defaultXTick,
  tooltipLabelFormatter = defaultTooltipLabel,
  tooltipValueFormatter = (v) => [v, yLabel],
  yTickFormatter = (v) => v,

  // view controls
  defaultView = "daily", // daily | weekly | monthly | max
  allowViews = ["daily", "weekly", "monthly", "max"],
}) {
  const [view, setView] = useState(defaultView);
  const [brush, setBrush] = useState({ startIndex: 0, endIndex: 0 });

  const hasSeries = Array.isArray(series) && series.length > 0;

  const chartData = useMemo(() => {
    const pts = Array.isArray(data) ? data : [];
    if (view === "max") return pts;
    return resample(pts, view, xKey);
  }, [data, view, xKey]);

  useEffect(() => {
    const n = chartData.length;
    if (!n) return;
    setBrush({ startIndex: 0, endIndex: n - 1 });
  }, [chartData.length]); // IMPORTANT: depend on length

  // ✅ Force FULL remount when the data range changes (fixes stale domain/brush)
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
          {subtitle ? <div className="text-muted">{subtitle}</div> : null}
        </div>

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

      <div className="chart-area">
        <ResponsiveContainer width="100%" height={320}>
          <LineChart
            key={chartKey} // ✅ THIS is the important part
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
              labelFormatter={tooltipLabelFormatter}
              formatter={(value, name, ctx) => {
                if (hasSeries) {
                  const key = ctx?.dataKey ?? name;
                  const s = series.find((it) => it.key === key);
                  const label = s?.label || name;

                  const out = tooltipValueFormatter(value);
                  return [out?.[0] ?? value, label];
                }
                return tooltipValueFormatter(value);
              }}
            />

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
                  {...(linePropsByKey?.[s.key] || {})}
                />
              ))
            ) : (
              <Line
                type="monotone"
                dataKey={yKey}
                dot={false}
                strokeWidth={2}
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
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
