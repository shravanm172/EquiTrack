export default function MetricRow({ label, value }) {
  return (
    <div className="metric-row">
      <div className="metric-row__label">{label}</div>
      <div className="metric-row__value">{value ?? "—"}</div>
    </div>
  );
}
