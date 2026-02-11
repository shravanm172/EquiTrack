// Base Template for Line Charts

import { useMemo, useState } from "react";
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
function resample(points, mode) {
  if (!Array.isArray(points) || points.length === 0) return [];
  if (mode === "daily" || mode === "max") return points;

  const keyFn =
    mode === "monthly"
      ? (p) => String(p.date).slice(0, 7) // YYYY-MM
      : (p) => {
          const d = parseISODate(p.date);
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

export default function LineChartCard({
  title,
  subtitle,
  data,
  xKey = "date",
  yKey = "value",
  yLabel = "Value",

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

  const chartData = useMemo(() => {
    const pts = Array.isArray(data) ? data : [];
    if (view === "max") return pts;
    return resample(pts, view);
  }, [data, view]);

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
              formatter={(value) => tooltipValueFormatter(value)}
            />
            <Line type="monotone" dataKey={yKey} dot={false} strokeWidth={2} />
            <Brush dataKey={xKey} height={25} travellerWidth={10} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
